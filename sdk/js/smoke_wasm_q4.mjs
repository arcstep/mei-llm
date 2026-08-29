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
const cargoTarget = process.env.CARGO_TARGET_DIR || join(sdkRoot, "target");
const wasmCandidates = [
  join(cargoTarget, "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"),
  join(cargoTarget, "wasm32-unknown-unknown/debug/mei_sdk_wasm.wasm"),
  join(sdkRoot, "target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"),
  join(sdkRoot, "target/wasm32-unknown-unknown/debug/mei_sdk_wasm.wasm"),
];
const wasmPath = wasmCandidates.find((p) => existsSync(p));
const envPkg = process.env.MEI_51M_PACKAGE_DIR;
const qatPkg = join(sdkRoot, "packages/mei-1.0-51m-base-scratch300m-qat-q4-v1");
const ptqPkg = join(sdkRoot, "packages/mei-1.0-51m-base-scratch300m-q4-v1");
const pkgDir = envPkg || (existsSync(join(qatPkg, "weights.q4")) ? qatPkg : ptqPkg);
const jobsDir = join(sdkRoot, "../notebook/evaluation/jobs/mei-1.0-51m");

function writeReport(report) {
  mkdirSync(jobsDir, { recursive: true });
  writeFileSync(join(jobsDir, "wasm-q4-smoke.json"), JSON.stringify(report, null, 2) + "\n");
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
const weights = new Uint8Array(readFileSync(join(pkgDir, "weights.q4")));
const vocab = new Uint8Array(readFileSync(join(pkgDir, "tokenizer.vocab.json")));
if (weights.subarray(0, 8).toString() !== new TextEncoder().encode("MEIQPK01").toString()) {
  writeReport({
    ok: false,
    compiled: true,
    loaded: false,
    complete_ran: false,
    refuse_float: refuseFloat,
    quantized_only: true,
    note: "weights.q4 magic mismatch",
  });
  process.exit(2);
}

const model = loadModel({ manifest, weights, vocab, wasm: instance });
const loaded = typeof exp.mei_sdk_wasm_loaded === "function" ? exp.mei_sdk_wasm_loaded() === 1 : true;
const cold_start_ms = Date.now() - tLoad0;

const session = model.createSession();
const goldenPath = join(jobsDir, "mlx-qat-q4-golden.json");
let tokenIds = [2];
if (existsSync(goldenPath)) {
  const golden = JSON.parse(readFileSync(goldenPath, "utf8"));
  if (Array.isArray(golden.token_ids) && golden.token_ids.length) {
    tokenIds = golden.token_ids;
  }
}
const tInf0 = Date.now();
let complete;
try {
  complete = session.complete({
    query: "厨房灯打开",
    decode_mode: "raw",
    max_new: 8,
    token_ids: tokenIds,
    catalog: [],
  });
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
const infer_ms = Date.now() - tInf0;
const nNew = Number(complete?.stats?.output_tokens || 0);
const raw = complete?.raw_text || "";
writeReport({
  ok: loaded && model.quantizedOnly === true && refuseFloat && nNew >= 8,
  compiled: true,
  loaded,
  refuse_float: refuseFloat,
  quantized_only: true,
  wasm_tier: 1,
  weights_bytes: weights.length,
  package_dir: pkgDir,
  complete_ran: true,
  n_new: nNew,
  raw_text: raw,
  cold_start_ms,
  infer_ms,
  decode_tok_s: nNew > 0 ? (nNew / Math.max(infer_ms, 1)) * 1000 : 0,
  worker_note: "Node smoke runs complete() off the browser main thread; see wasm-q4-smoke.html for Worker.",
  note: nNew >= 8 ? "WASM short decode ran (max_new>=8)." : "complete() returned fewer than 8 tokens",
});
process.exit(nNew >= 8 && refuseFloat && loaded ? 0 : 2);
