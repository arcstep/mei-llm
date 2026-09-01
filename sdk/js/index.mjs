/**
 * MEI Runtime Node SDK (experimental, protocol layer).
 * Node local API. Browser API is in browser.mjs and is declared separately.
 */
export { sdkVersions, PROTOCOL } from "./version.mjs";
export {
  catalogFingerprint, complete as completeProtocol, parseV2Text, renderRequest, schemaFingerprint,
  validateGeneratedCall, validateJsonValue, validateToolSchema,
} from "./protocol.mjs";
export { loadPackage, validatePackageManifest } from "./package.mjs";
export { Engine, Session } from "./engine.mjs";
export { loadNodeWasmModel, load_model_wasm } from "./node-wasm.mjs";
export { quantizeCq2, dequantizeCq2, parseCq2Container, CQ2_QUANT_MATH_ID } from "./cq2.mjs";

import { Engine } from "./engine.mjs";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadNodeWasmModel } from "./node-wasm.mjs";
import { sdkVersions } from "./version.mjs";

// Portable v2 façade. Camel-case class methods remain available, while these
// operation names match spec/api.md and the Rust/Python/C bindings.
export function version() {
  return sdkVersions();
}

export function load_model(packageDir, options) {
  const manifest = JSON.parse(readFileSync(join(packageDir, "mei-model.json"), "utf8"));
  return manifest.package_format === "mei-model-package-v2"
    ? loadNodeWasmModel(packageDir, options)
    : Engine.load(packageDir, options);
}

export function register_tools(engine, tools) {
  return engine.registerTools(tools);
}

export function create_session(engine, options = {}) {
  return engine.createSession(options);
}

export function complete(session, request) {
  return session.complete(request);
}

export function submit_tool_result(session, result) {
  return session.submitToolResult(result);
}

export function run(session, request, executors = {}) {
  return session.run(request, executors);
}

export function narrate(session, options = {}) {
  return session.narrate(options);
}

export function cancel(session) {
  return session.cancel();
}

export function close_session(session) {
  return session.close();
}

export function close_engine(engine) {
  return engine.close();
}

export const loadModel = load_model;
export const registerTools = register_tools;
export const createSession = create_session;
export const submitToolResult = submit_tool_result;
export const closeSession = close_session;
export const closeEngine = close_engine;
