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
    const wasmUrl = new URL("../../target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm", import.meta.url);
    const pkg = new URL("../packages/mei-1.0-51m-base-scratch300m-qat-q4-v1/", import.meta.url);
    const [wasmBuf, manifestText, weightsBuf, vocabBuf] = await Promise.all([
      fetch(wasmUrl).then((r) => {
        if (!r.ok) throw new Error(`wasm fetch ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(new URL("mei-model.json", pkg)).then((r) => {
        if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
        return r.text();
      }),
      fetch(new URL("weights.q4", pkg)).then((r) => {
        if (!r.ok) throw new Error(`weights fetch ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(new URL("tokenizer.vocab.json", pkg)).then((r) => {
        if (!r.ok) throw new Error(`vocab fetch ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    const { instance } = await WebAssembly.instantiate(wasmBuf, {});
    const model = loadModel({
      manifest: JSON.parse(manifestText),
      weights: new Uint8Array(weightsBuf),
      vocab: new Uint8Array(vocabBuf),
      wasm: instance,
    });
    const session = model.createSession();
    const complete = session.complete({
      query: "厨房灯打开",
      decode_mode: "raw",
      max_new: 8,
      catalog: [],
    });
    report.complete_ran = true;
    report.n_new = Number(complete?.stats?.output_tokens || 0);
    report.raw_text = complete?.raw_text || "";
    report.quantized_only = true;
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
