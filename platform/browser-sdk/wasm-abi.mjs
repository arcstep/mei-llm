/**
 * Quantized WASM C ABI loader. No Node builtins — safe in a browser Worker.
 */
export const wasmTier = 1;

const WASM_TIER1_VERSIONS = {
  sdk_semver: "0.2.0-experimental",
  wire_version: "mei-runtime-wire-v2",
  model_package_version: "mei-model-package-v2",
  runtime_abi_version: "mei-runtime-abi-2",
  protocol_id: "mei-tool-call-protocol-v2",
  serializer_id: "mei-tool-call-serializer-v2",
  release_class: "experimental",
  product: "mei-1.0-51m Runtime",
};

const ABI_ERROR_IDS = Object.freeze([
  "ok", "invalid_argument", "invalid_json", "package_invalid", "package_hash_mismatch",
  "head_missing", "engine_unavailable", "session_closed", "too_many_tools", "gold_leak",
  "protocol_violation", "cancelled", "not_implemented", "abi_version_mismatch", "file_not_found",
  "buffer_too_small", "unsupported_schema", "tool_result_required", "stale_call_id",
  "tool_result_too_large", "max_steps", "executor_missing", "package_path_unsafe",
  "package_range_invalid", "duplicate_tensor", "capability_missing", "decode_mode_forbidden",
  "tool_schema_budget_exceeded",
]);

export function version() {
  return { ...WASM_TIER1_VERSIONS };
}

function fail(code, message) {
  const err = new Error(message);
  err.code = code;
  throw err;
}

function failAbiCode(code, locus) {
  const id = ABI_ERROR_IDS[Number(code)] || "engine_unavailable";
  fail(id, `${locus} failed code=${code} (${id})`);
}

function wasmExports(wasm) {
  if (wasm && wasm.exports && wasm.exports.memory) return wasm.exports;
  return wasm;
}

function encodeCString(text) {
  const body = new TextEncoder().encode(String(text));
  const out = new Uint8Array(body.length + 1);
  out.set(body, 0);
  return out;
}

function allocBytes(exp, bytes) {
  if (typeof exp.mei_sdk_wasm_alloc !== "function") {
    fail("engine_unavailable", "WASM module missing mei_sdk_wasm_alloc");
  }
  const ptr = exp.mei_sdk_wasm_alloc(bytes.length);
  if (!ptr) {
    fail("engine_unavailable", "WASM alloc returned null");
  }
  new Uint8Array(exp.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function freeBytes(exp, slot) {
  if (slot && slot.ptr && typeof exp.mei_sdk_wasm_free === "function") {
    exp.mei_sdk_wasm_free(slot.ptr, slot.len);
  }
}

function readCString(exp, ptr) {
  if (!ptr) return "";
  const mem = new Uint8Array(exp.memory.buffer);
  let end = ptr;
  while (end < mem.length && mem[end] !== 0) end += 1;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}

function loadQuantizedCAbi(exp, manifestText, weightBytes, vocabBytes, toolIndexBytes, v2) {
  const man = allocBytes(exp, encodeCString(manifestText));
  const weights = allocBytes(exp, weightBytes);
  const vocab = allocBytes(exp, vocabBytes);
  const toolIndex = v2 ? allocBytes(exp, toolIndexBytes) : null;
  let payloadOwnershipTransferred = false;
  try {
    if (v2) {
      if (typeof exp.mei_sdk_wasm_load_package_v2_owned === "function") {
        payloadOwnershipTransferred = true;
        return exp.mei_sdk_wasm_load_package_v2_owned(
          man.ptr, weights.ptr, weights.len, vocab.ptr, vocab.len, toolIndex.ptr, toolIndex.len,
        );
      }
      if (typeof exp.mei_sdk_wasm_load_package_v2 !== "function") {
        fail("engine_unavailable", "WASM module missing mei_sdk_wasm_load_package_v2");
      }
      return exp.mei_sdk_wasm_load_package_v2(
        man.ptr, weights.ptr, weights.len, vocab.ptr, vocab.len, toolIndex.ptr, toolIndex.len,
      );
    }
    return exp.mei_sdk_wasm_load_quantized(man.ptr, weights.ptr, weights.len, vocab.ptr, vocab.len);
  } finally {
    freeBytes(exp, man);
    if (!payloadOwnershipTransferred) {
    freeBytes(exp, weights);
    freeBytes(exp, vocab);
    freeBytes(exp, toolIndex);
  }
}
}

function invokeNoArgJsonCAbi(exp, functionName) {
  const fn = exp[functionName];
  if (typeof fn !== "function") fail("engine_unavailable", `WASM module missing ${functionName}`);
  const outSlot = allocBytes(exp, new Uint8Array(4));
  try {
    const rc = fn(outSlot.ptr);
    if (rc !== 0) failAbiCode(rc, functionName);
    const strPtr = new DataView(exp.memory.buffer).getUint32(outSlot.ptr, true);
    const text = readCString(exp, strPtr);
    if (typeof exp.mei_sdk_wasm_string_free === "function" && strPtr) {
      exp.mei_sdk_wasm_string_free(strPtr);
    }
    return JSON.parse(text);
  } finally {
    freeBytes(exp, outSlot);
  }
}

function invokeJsonCAbi(exp, functionName, payload, leadingArgs = []) {
  const fn = exp[functionName];
  if (typeof fn !== "function") fail("engine_unavailable", `WASM module missing ${functionName}`);
  const req = allocBytes(exp, encodeCString(JSON.stringify(payload)));
  const outSlot = allocBytes(exp, new Uint8Array(4));
  try {
    const rc = fn(...leadingArgs, req.ptr, outSlot.ptr);
    if (rc !== 0) {
      failAbiCode(rc, functionName);
    }
    const strPtr = new DataView(exp.memory.buffer).getUint32(outSlot.ptr, true);
    const text = readCString(exp, strPtr);
    if (typeof exp.mei_sdk_wasm_string_free === "function" && strPtr) {
      exp.mei_sdk_wasm_string_free(strPtr);
    }
    return JSON.parse(text);
  } finally {
    freeBytes(exp, req);
    freeBytes(exp, outSlot);
  }
}

/**
 * Load a quantized model into the WASM engine.
 * `weights` must be a portable CQ2 v2 container or legacy q4 bytes.
 */
export function loadModel(options = {}) {
  const { manifest, weights, vocab, toolIndex, wasm } = options;
  if (!manifest || !weights) {
    fail(
      "engine_unavailable",
      "browser loadModel requires a quantized package (manifest + tensor container); float checkpoints are not loaded in WASM",
    );
  }
  const fmt = manifest.package_format === "mei-model-package-v2"
    ? manifest.tensor_container?.format
    : manifest.weights?.format;
  const v2 = manifest.package_format === "mei-model-package-v2" && fmt === "mei-cq-tensor-v2";
  const v1 = manifest.package_format === "mei-model-package-v1" && fmt === "mei-q4-packed-v1";
  if (!v2 && !v1) {
    fail("engine_unavailable", "WASM tier-1 loads CQ2 v2 or legacy q4 read-only packages");
  }
  if (!wasm) {
    fail(
      "engine_unavailable",
      "WASM module not provided. Instantiate mei-sdk-wasm and pass { wasm } to loadModel",
    );
  }
  const exp = wasmExports(wasm);
  const manifestText = typeof manifest === "string" ? manifest : JSON.stringify(manifest);
  const weightBytes = weights instanceof Uint8Array ? weights : new Uint8Array(weights);
  const vocabBytes = vocab ? (vocab instanceof Uint8Array ? vocab : new Uint8Array(vocab)) : new Uint8Array();
  const toolIndexBytes = toolIndex
    ? (toolIndex instanceof Uint8Array ? toolIndex : new Uint8Array(toolIndex))
    : new Uint8Array();
  if (v2 && toolIndexBytes.length === 0) {
    fail("capability_missing", "native v2 WASM load requires the frozen tool-index payload");
  }

  let rc;
  if (typeof exp.mei_sdk_wasm_alloc === "function" && typeof exp.mei_sdk_wasm_load_quantized === "function") {
    rc = loadQuantizedCAbi(exp, manifestText, weightBytes, vocabBytes, toolIndexBytes, v2);
  } else if (typeof exp.mei_sdk_wasm_load_quantized === "function") {
    rc = exp.mei_sdk_wasm_load_quantized(manifestText, weightBytes, vocabBytes);
  } else {
    fail(
      "engine_unavailable",
      "WASM module not provided. Instantiate mei-sdk-wasm and pass { wasm } to loadModel",
    );
  }
  if (rc !== 0) {
    if (typeof exp.mei_sdk_wasm_last_error === "function") {
      const detail = invokeNoArgJsonCAbi(exp, "mei_sdk_wasm_last_error");
      fail(detail.id || ABI_ERROR_IDS[Number(rc)] || "engine_unavailable",
        detail.message || `WASM package load failed code=${rc}`);
    }
    failAbiCode(rc, "mei_sdk_wasm_load_quantized");
  }
  const model = {
    packageId: manifest.package_id,
    wasmTier: 1,
    quantizedOnly: true,
    registerTools(tools) {
      return invokeJsonCAbi(exp, "mei_sdk_wasm_register_tools", tools);
    },
    createSession(options = {}) {
      const opened = invokeJsonCAbi(exp, "mei_sdk_wasm_session_open", options);
      const sessionId = Number(opened.session_id);
      if (!Number.isSafeInteger(sessionId) || sessionId <= 0) fail("engine_unavailable", "invalid WASM session id");
      const maxSteps = Number(options.max_steps || 4);
      let closed = false;
      const ensureOpen = () => { if (closed) fail("session_closed", "WASM session is closed"); };
      const session = {
        complete(request) {
          ensureOpen();
          if (typeof exp.mei_sdk_wasm_alloc === "function") {
            return invokeJsonCAbi(exp, "mei_sdk_wasm_complete", request, [sessionId]);
          }
          return JSON.parse(exp.mei_sdk_wasm_complete(sessionId, JSON.stringify(request)));
        },
        submitToolResult(result) {
          ensureOpen();
          if (typeof exp.mei_sdk_wasm_alloc === "function") {
            return invokeJsonCAbi(exp, "mei_sdk_wasm_submit_tool_result", result, [sessionId]);
          }
          return JSON.parse(exp.mei_sdk_wasm_submit_tool_result(sessionId, JSON.stringify(result)));
        },
        narrate(options = {}) {
          ensureOpen();
          if (typeof exp.mei_sdk_wasm_narrate !== "function") {
            fail("engine_unavailable", "WASM module missing mei_sdk_wasm_narrate");
          }
          return invokeJsonCAbi(exp, "mei_sdk_wasm_narrate", options, [sessionId]);
        },
        cancel() {
          ensureOpen();
          if (typeof exp.mei_sdk_wasm_cancel === "function" && exp.mei_sdk_wasm_cancel(sessionId) !== 0) {
            fail("invalid_argument", "unknown WASM session");
          }
        },
        async run(request, executors = {}) {
          ensureOpen();
          const turns = [];
          const toolResults = [];
          let stoppedReason = "max_steps";
          for (let step = 0; step < maxSteps; step += 1) {
            const turn = this.complete(request);
            turns.push(turn);
            if (turn.kind === "error") {
              stoppedReason = turn.error?.id === "cancelled" ? "cancelled" : "error";
              break;
            }
            if (turn.kind === "refuse") { stoppedReason = "refuse"; break; }
            if (turn.kind === "respond") { stoppedReason = "respond"; break; }
            const executor = executors instanceof Map ? executors.get(turn.call.name) : executors[turn.call.name];
            if (typeof executor !== "function") {
              const terminal = {
                wire_version: "mei-runtime-wire-v2", call_id: turn.call.call_id,
                status: "error", payload: null,
                error: { code: "executor_missing", message: `no executor for ${turn.call.name}` },
                provenance: { source: "host-runtime", verified: true },
              };
              this.submitToolResult(terminal);
              toolResults.push(terminal);
              turns.push({
                wire_version: "mei-runtime-wire-v2", kind: "error", ok: false, refuse: true,
                call: null, refusal: null,
                error: { code: 21, id: "executor_missing", message: `no executor for ${turn.call.name}` },
              });
              stoppedReason = "error";
              break;
            }
            let result;
            try {
              result = {
                wire_version: "mei-runtime-wire-v2", call_id: turn.call.call_id, status: "ok",
                payload: await executor(turn.call.arguments, turn.call),
                provenance: { source: `host-executor:${turn.call.name}`, verified: true },
              };
            } catch (error) {
              result = {
                wire_version: "mei-runtime-wire-v2", call_id: turn.call.call_id, status: "error", payload: null,
                error: { code: String(error?.code || "executor_error"), message: String(error?.message || error) },
                provenance: { source: `host-executor:${turn.call.name}`, verified: true },
              };
            }
            this.submitToolResult(result);
            toolResults.push(result);
            if (result.status !== "ok") { stoppedReason = "error"; break; }
          }
          return {
            wire_version: "mei-runtime-wire-v2",
            ok: stoppedReason === "respond" && turns.length > 0 && turns.every((turn) => turn.kind !== "error"),
            turns, tool_results: toolResults, stopped_reason: stoppedReason,
          };
        },
        close() {
          if (!closed && typeof exp.mei_sdk_wasm_session_close === "function") {
            exp.mei_sdk_wasm_session_close(sessionId);
          }
          closed = true;
        },
      };
      session.submit_tool_result = session.submitToolResult;
      session.close_session = session.close;
      return session;
    },
    capabilities() {
      const caps = invokeNoArgJsonCAbi(exp, "mei_sdk_wasm_capabilities");
      return { ...caps, wasm_tier: 1, in_memory_unverified: true };
    },
    runtimeContract() {
      return invokeNoArgJsonCAbi(exp, "mei_sdk_wasm_runtime_contract");
    },
    diagnoseHeads(text) {
      const report = invokeJsonCAbi(
        exp,
        "mei_sdk_wasm_diagnose_heads",
        { text: String(text) },
      );
      if (!Array.isArray(report.narration_residual)
          || report.narration_residual.length !== 24_000) {
        fail("protocol_violation", "WASM narration residual diagnostic is incomplete");
      }
      return report;
    },
    unload() {
      if (typeof exp.mei_sdk_wasm_unload === "function") {
        exp.mei_sdk_wasm_unload();
      }
    },
  };
  model.version = version;
  model.register_tools = model.registerTools;
  model.create_session = model.createSession;
  model.diagnose_heads = model.diagnoseHeads;
  model.runtime_contract = model.runtimeContract;
  model.close_engine = model.unload;
  return model;
}

export const load_model = loadModel;
