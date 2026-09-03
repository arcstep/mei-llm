/**
 * Browser worker: load WASM + quantized package and run complete(max_new>=8).
 * Imports wasm-abi only (no Node builtins).
 */
import { loadModel } from "./wasm-abi.mjs";

async function run() {
  const report = { refuse_float: false, complete_ran: false, ok: false, browser_worker: true };
  try {
    try {
      loadModel({ manifest: { weights: { format: "mlx-npz" } }, weights: new Uint8Array([1]) });
    } catch (err) {
      report.refuse_float = err && err.code === "engine_unavailable";
    }
    const query = new URL(import.meta.url).searchParams;
    const wasmUrl = query.get("wasm")
      ? new URL(`/${query.get("wasm").replace(/^\/+/, "")}`, self.location.origin)
      : new URL("../_shared/target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm", import.meta.url);
    const pkg = query.get("package")
      ? new URL(`/${query.get("package").replace(/^\/+/, "")}/`, self.location.origin)
      : new URL("../../../.local/artifacts/mei-1.0-51m/exp-000300m/packages/mei-1.0-51m-base-scratch300m-qat-q4-v1/", import.meta.url);
    const goldenUrl = query.get("golden")
      ? new URL(`/${query.get("golden").replace(/^\/+/, "")}`, self.location.origin)
      : new URL("../../../.local/artifacts/_legacy/notebook/evaluation/jobs/mei-1.0-51m/mlx-qat-q4-golden.json", import.meta.url);
    const manifestText = await fetch(new URL("mei-model.json", pkg)).then((r) => {
      if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
      return r.text();
    });
    const manifest = JSON.parse(manifestText);
    const isV2 = manifest.package_format === "mei-model-package-v2";
    const weightsFile = isV2 ? manifest.tensor_container?.file : manifest.weights?.file;
    const tokenizerFile = manifest.tokenizer?.vocab_file
      || manifest.tokenizer?.file
      || "tokenizer.model";
    const toolIndexFile = isV2
      ? manifest.files?.find((row) => row.role === "tool_index")?.path
      : null;
    const [wasmBuf, weightsBuf, vocabBuf, toolIndexBuf, golden] = await Promise.all([
      fetch(wasmUrl).then((r) => {
        if (!r.ok) throw new Error(`wasm fetch ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(new URL(weightsFile, pkg)).then((r) => {
        if (!r.ok) throw new Error(`weights fetch ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(new URL(tokenizerFile, pkg)).then((r) => {
        if (!r.ok) throw new Error(`vocab fetch ${r.status}`);
        return r.arrayBuffer();
      }),
      toolIndexFile
        ? fetch(new URL(toolIndexFile, pkg)).then((r) => {
          if (!r.ok) throw new Error(`tool index fetch ${r.status}`);
          return r.arrayBuffer();
        })
        : Promise.resolve(null),
      isV2
        ? Promise.resolve({ token_ids: [2] })
        : fetch(goldenUrl).then((r) => {
          if (!r.ok) throw new Error(`golden fetch ${r.status}`);
          return r.json();
        }),
    ]);
    const { instance } = await WebAssembly.instantiate(wasmBuf, {});
    let model = loadModel({
      manifest,
      weights: new Uint8Array(weightsBuf),
      vocab: new Uint8Array(vocabBuf),
      toolIndex: toolIndexBuf ? new Uint8Array(toolIndexBuf) : undefined,
      wasm: instance,
    });
    if (isV2 && toolIndexBuf) {
      const index = JSON.parse(new TextDecoder().decode(toolIndexBuf));
      model.registerTools(index.records.map((row) => row.schema));
    }
    const memory = instance.exports.memory;
    const heapSamples = [Number(memory?.buffer?.byteLength || 0)];
    const requestFor = (maxNew) => isV2 ? {
      wire_version: "mei-runtime-wire-v2",
      query: "厨房灯打开",
      context: {},
      evidence: [],
      history: [],
      tool_results: [],
      permissions: {},
      state: {},
      decode_mode: "constrained",
      max_new: maxNew,
    } : {
      query: "厨房灯打开",
      decode_mode: "raw",
      max_new: maxNew,
      token_ids: golden.token_ids,
      catalog: [],
    };
    const runProbe = (maxNew) => {
      const session = model.createSession();
      const started = performance.now();
      try {
        return { complete: session.complete(requestFor(maxNew)), elapsed_ms: performance.now() - started };
      } finally {
        session.close();
      }
    };
    const candidateProbe = runProbe(isV2 ? 1 : 8);
    let shortProbe = candidateProbe;
    heapSamples.push(Number(memory?.buffer?.byteLength || 0));
    let longProbe = null;
    let performanceProbeReleaseOverride = false;
    if (isV2) {
      model.unload();
      model = loadModel({
        manifest: { ...manifest, release_class: "experimental" },
        weights: new Uint8Array(weightsBuf),
        vocab: new Uint8Array(vocabBuf),
        toolIndex: toolIndexBuf ? new Uint8Array(toolIndexBuf) : undefined,
        wasm: instance,
      });
      const runRawProbe = (maxNew) => {
        const session = model.createSession();
        const started = performance.now();
        try {
          const complete = session.complete({
            wire_version: "mei-runtime-wire-v2",
            query: "kernel-throughput-diagnostic",
            context: {}, evidence: [], history: [], tool_results: [], permissions: {}, state: {},
            decode_mode: "raw", max_new: maxNew, token_ids: golden.token_ids || [2],
          });
          return { complete, elapsed_ms: performance.now() - started };
        } finally {
          session.close();
        }
      };
      shortProbe = runRawProbe(1);
      longProbe = runRawProbe(32);
      performanceProbeReleaseOverride = true;
    }
    heapSamples.push(Number(memory?.buffer?.byteLength || 0));
    const complete = shortProbe.complete;
    const candidateComplete = candidateProbe.complete;
    const nNew = Number(complete?.stats?.output_tokens || 0);
    const longNew = Number(longProbe?.complete?.stats?.output_tokens || 0);
    const steadyDecodeTokens = longProbe ? longNew - nNew : 0;
    const steadyDecodeMs = longProbe ? longProbe.elapsed_ms - shortProbe.elapsed_ms : 0;
    const steadyDecodeTokS = steadyDecodeTokens > 0 && steadyDecodeMs > 0
      ? (steadyDecodeTokens / steadyDecodeMs) * 1000
      : null;
    const shortGenerated = complete?.generated_token_ids || [];
    const longGenerated = longProbe?.complete?.generated_token_ids || [];
    const steadyDecodeRan = !longProbe || (
      steadyDecodeTokens >= 16
      && Number.isFinite(steadyDecodeTokS)
      && shortGenerated.every((token, index) => longGenerated[index] === token)
    );
    const prefillTopk = complete?.prefill_topk_ids || [];
    const numericForwardRan = Array.isArray(prefillTopk) && prefillTopk.length === 5;
    const runtimeCache = complete?.runtime_cache || {};
    const boundedInt8CacheRan = runtimeCache.kv_storage_dtype === "int8"
      && runtimeCache.cache_growth_bounded === true;
    const candidatePrefillTopk = candidateComplete?.prefill_topk_ids || [];
    const candidateRuntimeCache = candidateComplete?.runtime_cache || {};
    const candidateFunctionalRan = Array.isArray(candidatePrefillTopk)
      && candidatePrefillTopk.length === 5
      && candidateRuntimeCache.kv_storage_dtype === "int8"
      && candidateRuntimeCache.cache_growth_bounded === true
      && ["call", "refuse", "respond", "error"].includes(candidateComplete?.kind);
    report.complete_ran = true;
    report.n_new = nNew;
    report.raw_text = complete?.raw_text || "";
    report.prompt_token_ids = golden.token_ids || [];
    report.generated_token_ids = shortGenerated;
    report.prefill_topk_ids = prefillTopk;
    report.quantized_only = true;
    report.package_id = model.packageId;
    report.package_format = manifest.package_format;
    report.numeric_forward_ran = numericForwardRan;
    report.bounded_int8_cache_ran = boundedInt8CacheRan;
    report.candidate_constrained_functional_ran = candidateFunctionalRan;
    report.candidate_result_kind = candidateComplete?.kind ?? null;
    report.candidate_error = candidateComplete?.error ?? null;
    report.candidate_selected_tools = candidateComplete?.selected_tools ?? [];
    report.candidate_schema_budget = candidateComplete?.schema_budget ?? null;
    report.candidate_input_budget = candidateComplete?.input_budget ?? null;
    report.candidate_prefill_topk_ids = candidatePrefillTopk;
    report.candidate_runtime_cache = candidateRuntimeCache;
    report.runtime_cache = runtimeCache;
    report.wasm_heap_samples_bytes = heapSamples;
    report.wasm_heap_peak_bytes = Math.max(...heapSamples);
    report.prefill_plus_one_ms = isV2 ? shortProbe.elapsed_ms : null;
    report.prefill_plus_32_ms = longProbe?.elapsed_ms ?? null;
    report.steady_decode_tokens = steadyDecodeTokens;
    report.steady_decode_ms = steadyDecodeMs;
    report.steady_decode_tok_s = steadyDecodeTokS;
    report.steady_decode_ran = steadyDecodeRan;
    report.steady_generated_token_ids = longGenerated;
    report.performance_probe_release_override = performanceProbeReleaseOverride;
    report.performance_probe_contract = performanceProbeReleaseOverride
      ? "same-package-bytes-experimental-raw-1-vs-32-token-kernel-diagnostic-v1"
      : "candidate-request";
    report.measurement_profile = longProbe
      ? "browser-worker-split-prefill-plus-steady-decode-v1"
      : "browser-worker-prefill-plus-bounded-decode-diagnostic-v1";
    report.ok = report.refuse_float && candidateFunctionalRan && numericForwardRan
      && boundedInt8CacheRan && steadyDecodeRan;
  } catch (err) {
    report.error = String(err && err.message ? err.message : err);
  }
  return report;
}

run().then((report) => self.postMessage(report));
self.onmessage = () => {
  run().then((report) => self.postMessage(report));
};
