/** Node binding for the same Rust/WASM numerical core used by the browser. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadModel as loadInMemoryWasm } from "./wasm-abi.mjs";
import { loadPackage } from "./package.mjs";
import { sdkVersions } from "./version.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const sdkRoot = resolve(here, "..");

function fail(id, message) {
  const error = new Error(message);
  error.code = id;
  throw error;
}

function defaultWasmPath() {
  const cargoTarget = process.env.CARGO_TARGET_DIR || join(sdkRoot, "../../.local/cache/cargo-sdk-target");
  return [
    process.env.MEI_SDK_WASM_PATH,
    join(cargoTarget, "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"),
    join(cargoTarget, "wasm32-unknown-unknown/debug/mei_sdk_wasm.wasm"),
  ].find((candidate) => candidate && existsSync(candidate));
}

export function loadNodeWasmModel(packageDir, options = {}) {
  const pkg = loadPackage(packageDir, { verifyHashes: options.verifyHashes !== false });
  if (pkg.manifest.package_format !== "mei-model-package-v2") {
    fail("capability_missing", "Node WASM numerical inference requires a native v2 package");
  }
  const wasmPath = options.wasmPath ? resolve(options.wasmPath) : defaultWasmPath();
  if (!wasmPath) {
    fail("engine_unavailable", "compiled mei-sdk-wasm artifact was not found");
  }
  const indexRows = pkg.manifest.files.filter((row) => row.role === "tool_index");
  if (indexRows.length !== 1) {
    fail("capability_missing", "native v2 package requires exactly one frozen tool index");
  }
  const module = new WebAssembly.Module(readFileSync(wasmPath));
  const instance = new WebAssembly.Instance(module, {});
  const model = loadInMemoryWasm({
    manifest: pkg.manifest,
    weights: new Uint8Array(readFileSync(join(pkg.path, pkg.manifest.tensor_container.file))),
    vocab: new Uint8Array(readFileSync(join(
      pkg.path,
      pkg.manifest.tokenizer.vocab_file || pkg.manifest.tokenizer.file,
    ))),
    toolIndex: new Uint8Array(readFileSync(join(pkg.path, indexRows[0].path))),
    wasm: instance,
  });
  const inMemoryCapabilities = model.capabilities.bind(model);
  model.capabilities = () => {
    const verified = pkg.capabilities(sdkVersions());
    const executed = inMemoryCapabilities();
    const headsReady = ["contrastive", "mw_disposition", "confidence", "narration_adapter"]
      .every((name) => verified.heads?.[name]?.status === "ready");
    const mechanismComplete = verified.hash_verified === true
      && verified.training_receipts_verified === true
      && verified.tool_index_verified === true
      && verified.tensor_identity_complete === true
      && verified.portable_quantization_policy_complete === true
      && verified.runtime_quantization_complete === true
      && verified.narration === true
      && headsReady
      && executed.backend_loaded === true;
    return {
      ...verified,
      inference: mechanismComplete,
      backend_loaded: executed.backend_loaded === true,
      backend: "rust-wasm-portable",
      diagnostic_inference: true,
      quantized_only: true,
      implementation_complete: mechanismComplete,
      product_ready: mechanismComplete,
      degraded: !mechanismComplete,
      release_eligible: false,
      runtime_head_execution: {
        retrieval: "independent-contrastive-head+frozen-tool-index",
        mw_disposition: "independent-20class-sidecar",
        confidence: "independent-calibrated-sidecar",
        narration: "independent-frozen-backbone-rank16-generation-sidecar",
      },
      open_capabilities: mechanismComplete ? ["trusted-resource-qualification"] : verified.open_capabilities,
      wasm_tier: 1,
      wasm_sha256_verified_by_package: false,
    };
  };
  model.nodeWasmPath = wasmPath;
  return model;
}

export const load_model_wasm = loadNodeWasmModel;
