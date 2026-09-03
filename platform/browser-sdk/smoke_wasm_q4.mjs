/**
 * Instantiate mei-sdk-wasm, load a quantized 51M package, run short complete().
 * Float npz is refused. Prefer a Worker when document is present.
 */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadModel } from "./browser.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const sdkRoot = join(here, "..");
const cargoTarget = process.env.CARGO_TARGET_DIR || join(sdkRoot, "../../.local/cache/cargo-sdk-target");
const wasmCandidates = [
  join(cargoTarget, "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"),
  join(cargoTarget, "wasm32-unknown-unknown/debug/mei_sdk_wasm.wasm"),
];
const wasmPath = wasmCandidates.find((p) => existsSync(p));
const envPkg = process.env.MEI_51M_PACKAGE_DIR;
const packageRoot = join(sdkRoot, "../../.local/artifacts/mei-1.0-51m/exp-000300m/packages");
const qatPkg = join(packageRoot, "mei-1.0-51m-base-scratch300m-qat-q4-v1");
const ptqPkg = join(packageRoot, "mei-1.0-51m-base-scratch300m-q4-v1");
const pkgDir = envPkg || (existsSync(join(qatPkg, "weights.q4")) ? qatPkg : ptqPkg);
const jobsDir = process.env.MEI_51M_JOBS_DIR || join(sdkRoot, "../../.local/artifacts/_legacy/notebook/evaluation/jobs/mei-1.0-51m");
const reportPath = process.env.MEI_WASM_REPORT_PATH || join(jobsDir, "wasm-q4-smoke.json");

function writeReport(report) {
  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
}

let refuseFloat = false;
try {
  loadModel({
    manifest: { weights: { format: "mlx-npz" } },
    weights: new Uint8Array([1]),
  });
} catch (err) {
  refuseFloat = err && err.code === "engine_unavailable";
}

if (!wasmPath) {
  writeReport({
    ok: false,
    compiled: false,
    loaded: false,
    complete_ran: false,
    refuse_float: refuseFloat,
    quantized_only: true,
    note: `missing wasm artifact; looked in ${wasmCandidates.join(", ")}`,
  });
  process.exit(2);
}

const wasmBytes = readFileSync(wasmPath);
const tLoad0 = Date.now();
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const heapSamples = [Number(exp.memory?.buffer?.byteLength || 0)];
if (typeof exp.mei_sdk_wasm_tier !== "function" || exp.mei_sdk_wasm_tier() !== 1) {
  writeReport({
    ok: false,
    compiled: true,
    loaded: false,
    complete_ran: false,
    refuse_float: refuseFloat,
    quantized_only: true,
    note: "mei_sdk_wasm_tier is not 1",
  });
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(join(pkgDir, "mei-model.json"), "utf8"));
const isV2 = manifest.package_format === "mei-model-package-v2";
const weightName = isV2 ? manifest.tensor_container?.file : manifest.weights?.file;
const vocabName = manifest.tokenizer?.vocab_file || manifest.tokenizer?.file || "tokenizer.model";
const toolIndexName = isV2
  ? manifest.files?.find((row) => row.role === "tool_index")?.path
  : null;
const weights = new Uint8Array(readFileSync(join(pkgDir, weightName)));
const vocab = new Uint8Array(readFileSync(join(pkgDir, vocabName)));
const toolIndex = toolIndexName
  ? new Uint8Array(readFileSync(join(pkgDir, toolIndexName)))
  : undefined;
const expectedMagic = isV2 ? "MEICQ201" : "MEIQPK01";
if (weights.subarray(0, 8).toString() !== new TextEncoder().encode(expectedMagic).toString()) {
  writeReport({
    ok: false,
    compiled: true,
    loaded: false,
    complete_ran: false,
    refuse_float: refuseFloat,
    quantized_only: true,
    note: `${weightName} magic mismatch`,
  });
  process.exit(2);
}

let model = loadModel({ manifest, weights, vocab, toolIndex, wasm: instance });
heapSamples.push(Number(exp.memory?.buffer?.byteLength || 0));
const loaded = typeof exp.mei_sdk_wasm_loaded === "function" ? exp.mei_sdk_wasm_loaded() === 1 : true;
const cold_start_ms = Date.now() - tLoad0;

if (isV2) {
  const frozenIndex = JSON.parse(new TextDecoder().decode(toolIndex));
  model.registerTools(frozenIndex.records.map((row) => row.schema));
}

const goldenPath = join(jobsDir, "mlx-qat-q4-golden.json");
let tokenIds = [2];
if (existsSync(goldenPath)) {
  const golden = JSON.parse(readFileSync(goldenPath, "utf8"));
  if (Array.isArray(golden.token_ids) && golden.token_ids.length) {
    tokenIds = golden.token_ids;
  }
}
function requestFor(maxNew) {
  return isV2 ? {
    wire_version: "mei-runtime-wire-v2",
    query: "厨房灯打开",
    context: {}, evidence: [], history: [], tool_results: [], permissions: {}, state: {},
    decode_mode: "constrained",
    max_new: maxNew,
  } : {
    query: "厨房灯打开",
    decode_mode: "raw",
    max_new: maxNew,
    token_ids: tokenIds,
    catalog: [],
  };
}

function runProbe(maxNew) {
  const session = model.createSession();
  const started = Date.now();
  try {
    const complete = session.complete(requestFor(maxNew));
    return { complete, elapsed_ms: Date.now() - started };
  } finally {
    session.close();
  }
}

let shortProbe;
let longProbe = null;
let candidateProbe;
let performanceProbeReleaseOverride = false;
try {
  const requestedMaxNew = Number(process.env.MEI_WASM_MAX_NEW || 0);
  candidateProbe = runProbe(requestedMaxNew > 0 ? requestedMaxNew : (isV2 ? 1 : 8));
  shortProbe = candidateProbe;
  heapSamples.push(Number(exp.memory?.buffer?.byteLength || 0));
  // Performance is a raw numerical-kernel diagnostic on the exact same
  // tensor bytes. Candidate behavior is tested above with constrained decode;
  // only the in-memory manifest class is relaxed after unloading so EOS/JSON
  // completion cannot truncate a 32-token throughput sample.
  if (isV2 && requestedMaxNew === 0) {
    model.unload();
    model = loadModel({
      manifest: { ...manifest, release_class: "experimental" },
      weights, vocab, toolIndex, wasm: instance,
    });
    const runRawProbe = (maxNew) => {
      const session = model.createSession();
      const started = Date.now();
      try {
        const complete = session.complete({
          wire_version: "mei-runtime-wire-v2",
          query: "kernel-throughput-diagnostic",
          context: {}, evidence: [], history: [], tool_results: [], permissions: {}, state: {},
          decode_mode: "raw", max_new: maxNew, token_ids: tokenIds,
        });
        return { complete, elapsed_ms: Date.now() - started };
      } finally {
        session.close();
      }
    };
    shortProbe = runRawProbe(1);
    longProbe = runRawProbe(32);
    performanceProbeReleaseOverride = true;
    heapSamples.push(Number(exp.memory?.buffer?.byteLength || 0));
  }
} catch (err) {
  writeReport({
    ok: false,
    compiled: true,
    loaded,
    complete_ran: false,
    refuse_float: refuseFloat,
    quantized_only: true,
    error: String(err && err.message ? err.message : err),
    cold_start_ms,
  });
  process.exit(2);
}
const complete = shortProbe.complete;
const candidateComplete = candidateProbe.complete;
const infer_ms = shortProbe.elapsed_ms;
const nNew = Number(complete?.stats?.output_tokens || 0);
const longNew = Number(longProbe?.complete?.stats?.output_tokens || 0);
const steadyDecodeTokens = longProbe ? longNew - nNew : 0;
const steadyDecodeMs = longProbe ? longProbe.elapsed_ms - infer_ms : 0;
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
const raw = complete?.raw_text || "";
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
writeReport({
  ok: loaded && model.quantizedOnly === true && refuseFloat && candidateFunctionalRan && numericForwardRan
    && boundedInt8CacheRan && steadyDecodeRan,
  compiled: true,
  loaded,
  refuse_float: refuseFloat,
  quantized_only: true,
  wasm_tier: 1,
  wasm_required_features: ["simd128"],
  weights_bytes: weights.length,
  package_format: manifest.package_format,
  tensor_file: weightName,
  vocab_file: vocabName,
  tool_index_file: toolIndexName,
  package_dir: pkgDir,
  complete_ran: true,
  n_new: nNew,
  raw_text: raw,
  prompt_token_ids: tokenIds,
  generated_token_ids: shortGenerated,
  prefill_topk_ids: prefillTopk,
  numeric_forward_ran: numericForwardRan,
  bounded_int8_cache_ran: boundedInt8CacheRan,
  candidate_constrained_functional_ran: candidateFunctionalRan,
  candidate_result_kind: candidateComplete?.kind ?? null,
  candidate_error: candidateComplete?.error ?? null,
  candidate_selected_tools: candidateComplete?.selected_tools ?? [],
  candidate_schema_budget: candidateComplete?.schema_budget ?? null,
  candidate_input_budget: candidateComplete?.input_budget ?? null,
  candidate_prefill_topk_ids: candidatePrefillTopk,
  candidate_runtime_cache: candidateRuntimeCache,
  runtime_cache: runtimeCache,
  wasm_heap_samples_bytes: heapSamples,
  wasm_heap_peak_bytes: Math.max(...heapSamples),
  cold_start_ms,
  infer_ms,
  prefill_plus_one_ms: isV2 && longProbe ? infer_ms : null,
  prefill_plus_32_ms: longProbe?.elapsed_ms ?? null,
  steady_decode_tokens: steadyDecodeTokens,
  steady_decode_ms: steadyDecodeMs,
  steady_decode_tok_s: steadyDecodeTokS,
  steady_decode_ran: steadyDecodeRan,
  decode_tok_s: steadyDecodeTokS ?? (nNew > 0 ? (nNew / Math.max(infer_ms, 1)) * 1000 : 0),
  steady_generated_token_ids: longGenerated,
  performance_probe_release_override: performanceProbeReleaseOverride,
  performance_probe_contract: performanceProbeReleaseOverride
    ? "same-package-bytes-experimental-raw-1-vs-32-token-kernel-diagnostic-v1"
    : "candidate-request",
  measurement_profile: longProbe
    ? "split-prefill-plus-steady-decode-v2"
    : "prefill-plus-bounded-decode-diagnostic-v1",
  worker_note: "Node smoke runs complete() off the browser main thread; see wasm-q4-smoke.html for Worker.",
  note: numericForwardRan ? "WASM load, prefill, and constrained decode ran." : "WASM prefill evidence is incomplete",
});
process.exit(candidateFunctionalRan && numericForwardRan && boundedInt8CacheRan
  && steadyDecodeRan && refuseFloat && loaded ? 0 : 2);
