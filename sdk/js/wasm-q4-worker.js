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
      : new URL("../target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm", import.meta.url);
    const pkg = query.get("package")
      ? new URL(`/${query.get("package").replace(/^\/+/, "")}/`, self.location.origin)
      : new URL("../packages/mei-1.0-51m-base-scratch300m-qat-q4-v1/", import.meta.url);
    const goldenUrl = query.get("golden")
      ? new URL(`/${query.get("golden").replace(/^\/+/, "")}`, self.location.origin)
      : new URL("../../notebook/evaluation/jobs/mei-1.0-51m/mlx-qat-q4-golden.json", import.meta.url);
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
      fetch(goldenUrl).then((r) => {
        if (!r.ok) throw new Error(`golden fetch ${r.status}`);
        return r.json();
      }),
    ]);
    const { instance } = await WebAssembly.instantiate(wasmBuf, {});
    const model = loadModel({
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
    const session = model.createSession();
    const complete = session.complete({
      query: "厨房灯打开",
      decode_mode: "raw",
      max_new: 8,
      token_ids: golden.token_ids,
      catalog: [],
    });
    report.complete_ran = true;
    report.n_new = Number(complete?.stats?.output_tokens || 0);
    report.raw_text = complete?.raw_text || "";
    report.prompt_token_ids = golden.token_ids || [];
    report.generated_token_ids = complete?.generated_token_ids || [];
    report.prefill_topk_ids = complete?.prefill_topk_ids || [];
    report.quantized_only = true;
    report.package_id = model.packageId;
    report.ok = report.refuse_float && report.n_new >= 8;
  } catch (err) {
    report.error = String(err && err.message ? err.message : err);
  }
  return report;
}

run().then((report) => self.postMessage(report));
self.onmessage = () => {
  run().then((report) => self.postMessage(report));
};
