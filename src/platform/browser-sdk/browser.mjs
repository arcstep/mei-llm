/**
 * Browser API: quantized 51M WASM tier-1 only. Float packages are refused.
 *
 * This entry never imports Node builtins. Protocol parsing and SHA validation
 * are performed by the shared Rust/WASM core.
 */
export { loadModel, load_model, version, wasmTier } from "./wasm-abi.mjs";

export const apiSurface = "browser";
