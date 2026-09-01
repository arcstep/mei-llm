import { readFileSync } from "node:fs";
import { load_model } from "./index.mjs";

const packageDir = process.env.MEI_51M_PACKAGE_DIR;
if (!packageDir) throw new Error("MEI_51M_PACKAGE_DIR is required");
const manifest = JSON.parse(readFileSync(`${packageDir}/mei-model.json`, "utf8"));
if (manifest.package_format !== "mei-model-package-v2") throw new Error("native v2 package required");
const engine = load_model(packageDir);
const caps = engine.capabilities();
const runtimeContract = engine.runtimeContract();
if (!caps.implementation_complete || !caps.inference) {
  throw new Error(`Node v2 mechanisms incomplete: ${JSON.stringify(caps)}`);
}
const catalog = JSON.parse(readFileSync(`${packageDir}/tool-index.json`, "utf8"))
  .records.map((row) => row.schema);
engine.register_tools(catalog);
const headDiagnostics = engine.diagnoseHeads("打开厨房灯并确认状态");
const narrationResidual = headDiagnostics.narration_residual;
if (!Array.isArray(narrationResidual) || narrationResidual.length !== 24_000
    || narrationResidual.some((value) => !Number.isFinite(value))) {
  throw new Error("Node/WASM did not execute the rank-16 narration residual");
}
let narrationArgmax = 0;
for (let index = 1; index < narrationResidual.length; index += 1) {
  if (narrationResidual[index] > narrationResidual[narrationArgmax]) {
    narrationArgmax = index;
  }
}
const session = engine.create_session({ max_steps: 1 });
const turn = session.complete({
  wire_version: "mei-runtime-wire-v2",
  query: "打开厨房灯",
  context: {}, evidence: [], history: [], tool_results: [], permissions: {}, state: {},
  decode_mode: "constrained", max_new: 1,
});
console.log(JSON.stringify({
  ok: ["call", "respond", "refuse", "error"].includes(turn.kind),
  inference: caps.inference,
  implementation_complete: caps.implementation_complete,
  backend: caps.backend,
  runtime_head_execution: caps.runtime_head_execution,
  runtime_contract: runtimeContract,
  narration_numeric_runtime: {
    residual_width: narrationResidual.length,
    residual_argmax: narrationArgmax,
    semantic_boundary: headDiagnostics.semantic_boundaries?.narration || null,
  },
  runtime_cache: turn.runtime_cache || null,
  mw_disposition: turn.mw_disposition || null,
}));
session.close_session();
engine.close_engine();
