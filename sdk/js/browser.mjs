/**
 * Browser API: quantized 51M WASM tier-1 only. Float packages are refused.
 *
 * Node re-exports protocol helpers. WASM Workers must import `./wasm-abi.mjs`
 * so they never load `node:fs` / `node:crypto`.
 */
import { complete, parseV2Text } from "./protocol.mjs";
import { sdkVersions } from "./version.mjs";
export { loadModel, wasmTier } from "./wasm-abi.mjs";

export { complete, parseV2Text, sdkVersions };

export const apiSurface = "browser";
