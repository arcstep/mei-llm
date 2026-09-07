import { createHash } from "node:crypto";
import { catalogFingerprint, complete, dumpsCanonical, validateSemanticJson, validateToolSchema } from "./protocol.mjs";
import { loadPackage } from "./package.mjs";
import { ERRORS, sdkVersions } from "./version.mjs";

function fail(id, message) {
  const error = new Error(message || ERRORS[id]?.message || id);
  error.code = id;
  error.abiCode = ERRORS[id]?.code;
  throw error;
}

function errorObject(id, message) {
  return { code: ERRORS[id]?.code ?? 1, id, message: message || ERRORS[id]?.message || id };
}

function isSuccessfulRespond(turn, toolResults) {
  if (!toolResults.length || toolResults.some((row) => row.status !== "ok")) return false;
  if (turn.error || turn.function_calls?.length) return false;
  if ((turn.provenance?.gates || []).some((gate) => gate.ok === false)) return false;
  try {
    const parsed = JSON.parse(String(turn.raw_text || ""));
    return Array.isArray(parsed) && parsed.length === 0;
  } catch {
    return false;
  }
}

function trustedCallHistory(toolResults, completedCalls) {
  const history = [];
  for (const result of toolResults) {
    const callId = String(result.call_id || "");
    const call = completedCalls.get(callId);
    if (!call) continue;
    history.push({
      role: "assistant",
      call_id: callId,
      content: dumpsCanonical({
        call_id: callId,
        name: String(call.name || ""),
        arguments: call.arguments || {},
      }),
    });
  }
  return history;
}

function compactNarrationValue(value, limit = 240) {
  let text = typeof value === "string" ? value.trim() : dumpsCanonical(value);
  const chars = Array.from(text);
  if (chars.length > limit) text = `${chars.slice(0, limit - 1).join("")}…`;
  return text;
}

function deterministicNarration(call, result) {
  const toolName = String(call?.name || "该工具");
  const args = call?.arguments && typeof call.arguments === "object" ? call.arguments : {};
  const payload = result.payload;
  const object = payload && typeof payload === "object" && !Array.isArray(payload) ? payload : {};
  const device = String(args.device || "设备");
  const zone = String(args.zone || "");
  const door = String(args.door || "门");
  if (result.status === "error") {
    const detail = String(result.error?.message || compactNarrationValue(result.error || payload));
    if (toolName === "start_device") return `${device}启动失败：${detail}。`;
    if (toolName === "stop_device") return `${device}关闭失败：${detail}。`;
    if (toolName === "unlock_door") return `${door}解锁失败：${detail}。`;
    return `${toolName}执行未成功：${detail}。`;
  }
  if (result.status === "cancelled") {
    return toolName === "start_device" ? `${device}的启动操作已取消。` : `${toolName}已取消，未继续执行。`;
  }
  if (result.status !== "ok") return `${toolName}执行未成功：结果状态无效。`;
  if (toolName === "get_temperature" && Object.hasOwn(object, "temperature_c")) {
    return `${zone ? `${zone}当前` : "当前"}温度为${compactNarrationValue(object.temperature_c)}℃。`;
  }
  if (toolName === "set_temperature" && Object.hasOwn(object, "temperature_c")) {
    const area = zone || "目标区域";
    if (object.applied === false) return `${area}已经是${compactNarrationValue(object.temperature_c)}℃，无需调整。`;
    return `已将${area}温度设为${compactNarrationValue(object.temperature_c)}℃。`;
  }
  if (toolName === "adjust_temperature" && Object.hasOwn(object, "temperature_c")) {
    return `已将${zone || "目标区域"}温度${Number(args.delta_c) < 0 ? "调低" : "调高"}到${compactNarrationValue(object.temperature_c)}℃。`;
  }
  if (toolName === "get_humidity" && Object.hasOwn(object, "humidity_percent")) {
    return `${zone ? `${zone}当前` : "当前"}湿度为${compactNarrationValue(object.humidity_percent)}%。`;
  }
  if (toolName === "start_device" && object.state === "on") return `${device}已启动。`;
  if (toolName === "stop_device" && object.state === "off") return `${device}已关闭。`;
  if (toolName === "set_brightness" && Object.hasOwn(object, "brightness_percent")) return `已将${String(args.light || "灯光")}亮度调到${compactNarrationValue(object.brightness_percent)}%。`;
  if (toolName === "lock_door" && object.locked === true) return `${door}已锁定。`;
  if (toolName === "set_fan_speed" && Object.hasOwn(object, "level")) return `已将${device}调到${compactNarrationValue(object.level)}档。`;
  if (toolName === "create_timer" && Object.hasOwn(object, "timer_id")) return `已设置${compactNarrationValue(object.minutes ?? args.minutes)}分钟计时器，编号为${compactNarrationValue(object.timer_id)}。`;
  if (toolName === "cancel_timer" && object.cancelled === true) return `计时器${compactNarrationValue(object.timer_id ?? args.timer_id)}已取消。`;
  if (toolName === "activate_scene") {
    const scene = String(object.scene || args.scene || "场景");
    if (object.status === "partial") return `${scene}模式已部分启动：${compactNarrationValue(object.completed)}项完成，${compactNarrationValue(object.failed)}项失败。`;
    if (object.active === true) return `${scene}模式已启动。`;
  }
  if (toolName === "play_music" && object.playing === true) return `已开始播放${compactNarrationValue(object.playlist ?? args.playlist)}。`;
  if (payload == null || (Array.isArray(payload) && payload.length === 0)
      || (payload && typeof payload === "object" && !Array.isArray(payload) && Object.keys(payload).length === 0)) {
    return `${toolName}已执行完成。`;
  }
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    const pairs = Object.keys(payload).sort()
      .map((key) => `${key}为${compactNarrationValue(payload[key], 96)}`);
    return `${toolName}已执行完成，${pairs.join("；")}。`;
  }
  return `${toolName}已执行完成，结果为${compactNarrationValue(payload)}。`;
}

let nextSessionNonce = 1;

function allocateSessionNonce() {
  if (nextSessionNonce > 0xffffffff) fail("engine_unavailable", "session nonce space exhausted");
  const nonce = `s${nextSessionNonce.toString(16).padStart(8, "0")}`;
  nextSessionNonce += 1;
  return nonce;
}

export class Engine {
  constructor(pkg) {
    this.package = pkg;
    this.closed = false;
    this.registeredTools = [];
  }

  static load(packageDir, options) {
    return new Engine(loadPackage(packageDir, options));
  }

  capabilities() {
    return this.package.capabilities(sdkVersions());
  }

  registerTools(tools) {
    if (this.closed) fail("session_closed", "engine is closed");
    if (!Array.isArray(tools)) fail("invalid_argument", "tools must be an array");
    const names = new Set();
    for (const tool of tools) {
      validateToolSchema(tool);
      if (names.has(tool.name)) fail("invalid_argument", `duplicate tool ${tool.name}`);
      names.add(tool.name);
    }
    const fingerprint = catalogFingerprint(tools);
    if (this.package.tool_index && this.package.tool_index.catalog_sha256 !== fingerprint) {
      fail("package_hash_mismatch", "registered catalog does not match the package tool index");
    }
    this.registeredTools = JSON.parse(dumpsCanonical(tools));
    return { registered: tools.length, catalog_fingerprint: fingerprint };
  }

  createSession(options = {}) {
    if (this.closed) fail("session_closed", "engine is closed");
    return new Session(this.capabilities(), this.registeredTools, options);
  }

  close() {
    this.closed = true;
    this.registeredTools = [];
  }
}

export class Session {
  constructor(capabilities, registeredTools, options = {}) {
    this.sessionNonce = allocateSessionNonce();
    this.capabilities = capabilities;
    this.registeredTools = JSON.parse(dumpsCanonical(registeredTools || []));
    this.closed = false;
    this.cancelled = false;
    this.pendingCall = null;
    this.toolResults = [];
    this.completedCalls = new Map();
    this.responded = false;
    this.narrationQuery = "";
    this.step = 0;
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      fail("invalid_argument", "session options must be an object");
    }
    // 选项面与 Rust core create_session_with_options 对齐：
    // 检索候选扫描的 topN 池上限（= 每批 5 × 批数）与双阈值随会话可配置
    const unknownOption = Object.keys(options).find(
      (key) => ![
        "max_steps",
        "max_tool_result_bytes",
        "runtime_profile",
        "retrieval_discard_threshold",
        "retrieval_expand_threshold",
        "max_candidate_batches",
      ].includes(key),
    );
    if (unknownOption) fail("invalid_argument", `unknown session option ${unknownOption}`);
    this.maxSteps = Number(options.max_steps ?? 4);
    if (!Number.isInteger(this.maxSteps) || this.maxSteps < 1 || this.maxSteps > 8) {
      fail("invalid_argument", "max_steps must be between 1 and 8");
    }
    this.maxToolResultBytes = Number(options.max_tool_result_bytes ?? 65_536);
    if (!Number.isInteger(this.maxToolResultBytes) || this.maxToolResultBytes < 1 || this.maxToolResultBytes > 1_048_576) {
      fail("invalid_argument", "max_tool_result_bytes must be between 1 and 1048576");
    }
    this.runtimeProfile = String(options.runtime_profile ?? "standard");
    if (!["compact", "standard"].includes(this.runtimeProfile)) {
      fail("invalid_argument", "runtime_profile must be compact or standard");
    }
    const threshold = (value, name) => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
        fail("invalid_argument", `${name} must be a finite number in [0,1]`);
      }
      return parsed;
    };
    this.retrievalDiscardThreshold = options.retrieval_discard_threshold == null
      ? 0
      : threshold(options.retrieval_discard_threshold, "retrieval_discard_threshold");
    this.retrievalExpandThreshold = options.retrieval_expand_threshold == null
      ? 0
      : threshold(options.retrieval_expand_threshold, "retrieval_expand_threshold");
    if (this.retrievalExpandThreshold < this.retrievalDiscardThreshold) {
      fail("invalid_argument", "retrieval_expand_threshold must be >= retrieval_discard_threshold");
    }
    this.maxCandidateBatches = options.max_candidate_batches == null
      ? null
      : Number(options.max_candidate_batches);
    if (this.maxCandidateBatches != null
        && (!Number.isInteger(this.maxCandidateBatches) || this.maxCandidateBatches < 1)) {
      fail("invalid_argument", "max_candidate_batches must be >= 1 or null");
    }
  }

  state() {
    return { step: this.step, pending_call_id: this.pendingCall?.call_id || null, cancelled: this.cancelled };
  }

  errorTurn(id, message) {
    return {
      wire_version: sdkVersions().wire_version,
      kind: "error",
      ok: false,
      error: errorObject(id, message),
      refuse: true,
      call: null,
      refusal: null,
      selected_tools: [],
      schema_fingerprint: null,
      function_calls: [],
      raw_text: null,
      confidence: { available: false, value: null, source: null },
      provenance: { validated: false, ok: false, detail: id },
      capabilities: this.capabilities,
      state: this.state(),
      stats: { backend: "state-machine", wall_ms: 0, decode_mode: null },
    };
  }

  effectiveRequest(request) {
    if (!request || typeof request !== "object" || Array.isArray(request)) fail("invalid_json", "request must be an object");
    const effective = JSON.parse(dumpsCanonical(request));
    const nativeV2 = this.capabilities.compatibility_mode === "v2-native";
    if (nativeV2 && !Object.hasOwn(effective, "wire_version")) {
      fail("invalid_argument", "CompleteRequestV2.wire_version is required");
    }
    if (!nativeV2 && !Object.hasOwn(effective, "wire_version")) effective.wire_version = "mei-runtime-wire-v1";
    if (!["mei-runtime-wire-v1", "mei-runtime-wire-v2"].includes(effective.wire_version)) {
      fail("invalid_argument", "unsupported wire_version");
    }
    if (nativeV2 && effective.wire_version !== "mei-runtime-wire-v2") {
      fail("abi_version_mismatch", "native v2 sessions cannot be downgraded to wire v1");
    }
    if (effective.wire_version === "mei-runtime-wire-v2") validateCompleteRequestV2(effective);
    if (effective.oracle_tools == null && effective.catalog == null && this.registeredTools.length) {
      effective.catalog = this.registeredTools;
    }
    if (this.toolResults.length) {
      effective.history = [...(effective.history || []), ...trustedCallHistory(this.toolResults, this.completedCalls)];
      effective.tool_results = this.toolResults;
    }
    return effective;
  }

  complete(request) {
    if (this.closed) fail("session_closed", "session is closed");
    if (this.pendingCall) return this.errorTurn("tool_result_required");
    if (this.step >= this.maxSteps) return this.errorTurn("max_steps");
    if (this.cancelled) return this.errorTurn("cancelled");
    const effective = this.effectiveRequest(request);
    if (!this.narrationQuery) this.narrationQuery = String(effective.query || "");
    const releaseClass = this.capabilities.release_class;
    if (["candidate", "release"].includes(releaseClass)
        && (effective.candidate_text != null || Object.hasOwn(effective, "mw")
          || Object.hasOwn(effective, "mw_disposition") || Object.hasOwn(effective, "confidence")
          || effective.enforce_confidence === false)) {
      return this.errorTurn(
        "decode_mode_forbidden",
        "candidate/release sessions forbid protocol-test and learned-head overrides",
      );
    }
    if (["candidate", "release"].includes(releaseClass) && effective.decode_mode === "raw") {
      return this.errorTurn("decode_mode_forbidden", "candidate/release sessions require constrained decode");
    }
    const turn = complete(effective, this.capabilities, this.toolResults);
    this.step += 1;
    const legacyCall = turn.function_calls?.[0];
    if (legacyCall) {
      const digest = createHash("sha256")
        .update(`${this.sessionNonce}:${this.step}:${dumpsCanonical(legacyCall)}`)
        .digest("hex").slice(0, 12);
      const call = { ...legacyCall, call_id: `call-${this.sessionNonce}-${this.step}-${digest}` };
      this.pendingCall = call;
      turn.kind = "call";
      turn.call = call;
      turn.function_calls = [call];
      turn.refusal = null;
    } else if (turn.error) {
      turn.kind = "error";
      turn.call = null;
      turn.refusal = null;
    } else if (isSuccessfulRespond(turn, this.toolResults)) {
      turn.kind = "respond";
      turn.ok = true;
      turn.refuse = false;
      turn.call = null;
      turn.refusal = null;
      turn.error = null;
      turn.function_calls = [];
      turn.execution = "respond";
      this.responded = true;
    } else {
      turn.kind = "refuse";
      turn.call = null;
      turn.refusal = { reason: turn.provenance?.detail || "model_refusal", detail: null };
    }
    turn.state = this.state();
    return turn;
  }

  submitToolResult(result) {
    if (this.closed) fail("session_closed", "session is closed");
    if (!result || typeof result !== "object" || Array.isArray(result)) fail("invalid_argument", "ToolResultV2 must be an object");
    validateSemanticJson(result);
    const unknown = Object.keys(result).find((key) => !["wire_version", "call_id", "status", "payload", "error", "provenance"].includes(key));
    if (unknown) fail("invalid_argument", `unknown ToolResultV2 field ${unknown}`);
    if (!Object.hasOwn(result, "payload")) fail("invalid_argument", "ToolResultV2.payload is required");
    if (!this.pendingCall) fail("stale_call_id", "no pending call");
    if (result?.wire_version !== "mei-runtime-wire-v2") fail("invalid_argument", "ToolResultV2 wire_version is required");
    if (result.call_id !== this.pendingCall.call_id) {
      fail("stale_call_id", `expected ${this.pendingCall.call_id}, got ${String(result.call_id || "")}`);
    }
    if (!["ok", "error", "cancelled"].includes(result.status)) fail("invalid_argument", "invalid tool result status");
    if (result.status === "error" && (!result.error || typeof result.error.code !== "string" || typeof result.error.message !== "string"
        || Object.keys(result.error).some((key) => !["code", "message"].includes(key)))) {
      fail("invalid_argument", "error status requires {code,message}");
    }
    if (result.status === "ok" && result.error != null) fail("invalid_argument", "ok status cannot carry error");
    if (!result.provenance?.verified || typeof result.provenance?.source !== "string" || !result.provenance.source) {
      fail("protocol_violation", "tool result provenance must be verified");
    }
    if (Object.keys(result.provenance).some((key) => !["source", "verified", "receipt_sha256"].includes(key))
        || (result.provenance.receipt_sha256 != null && !/^[0-9a-f]{64}$/.test(result.provenance.receipt_sha256))) {
      fail("protocol_violation", "invalid provenance fields");
    }
    const size = new TextEncoder().encode(JSON.stringify(result)).length;
    if (size > this.maxToolResultBytes) fail("tool_result_too_large", `${size} > ${this.maxToolResultBytes} bytes`);
    if (result.status === "cancelled") this.cancelled = true;
    this.toolResults.push(JSON.parse(dumpsCanonical(result)));
    const callId = this.pendingCall.call_id;
    this.completedCalls.set(callId, JSON.parse(dumpsCanonical(this.pendingCall)));
    this.pendingCall = null;
    return { wire_version: sdkVersions().wire_version, accepted: true, call_id: callId, state: this.state() };
  }

  async run(request, executors = {}) {
    const turns = [];
    const toolResults = [];
    let stoppedReason = "max_steps";
    while (this.step < this.maxSteps) {
      const turn = this.complete(request);
      turns.push(turn);
      if (turn.kind === "error") {
        stoppedReason = turn.error?.id === "cancelled" ? "cancelled" : "error";
        break;
      }
      if (turn.kind === "refuse") {
        stoppedReason = "refuse";
        break;
      }
      if (turn.kind === "respond") {
        stoppedReason = "respond";
        break;
      }
      const executor = executors instanceof Map ? executors.get(turn.call.name) : executors[turn.call.name];
      if (typeof executor !== "function") {
        const terminal = {
          wire_version: sdkVersions().wire_version,
          call_id: turn.call.call_id,
          status: "error",
          payload: null,
          error: { code: "executor_missing", message: `no executor for ${turn.call.name}` },
          provenance: { source: "host-runtime", verified: true },
        };
        this.submitToolResult(terminal);
        toolResults.push(terminal);
        turns.push(this.errorTurn("executor_missing", `no executor for ${turn.call.name}`));
        stoppedReason = "error";
        break;
      }
      let result;
      try {
        const payload = await executor(turn.call.arguments, turn.call);
        result = {
          wire_version: sdkVersions().wire_version,
          call_id: turn.call.call_id,
          status: "ok",
          payload,
          provenance: { source: `host-executor:${turn.call.name}`, verified: true },
        };
      } catch (error) {
        result = {
          wire_version: sdkVersions().wire_version,
          call_id: turn.call.call_id,
          status: "error",
          payload: null,
          error: { code: String(error?.code || "executor_error"), message: String(error?.message || error) },
          provenance: { source: `host-executor:${turn.call.name}`, verified: true },
        };
      }
      this.submitToolResult(result);
      toolResults.push(result);
      if (result.status !== "ok") {
        stoppedReason = result.status === "cancelled" ? "cancelled" : "error";
        break;
      }
    }
    return {
      wire_version: sdkVersions().wire_version,
      ok: stoppedReason === "respond" && turns.length > 0 && turns.every((turn) => turn.kind !== "error"),
      turns,
      tool_results: toolResults,
      stopped_reason: stoppedReason,
    };
  }

  narrate(options = {}) {
    const terminalToolResult = this.toolResults.some((row) => ["error", "cancelled"].includes(row.status));
    if (!this.responded && !terminalToolResult) fail("protocol_violation", "narration_requires_respond");
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      fail("invalid_argument", "narration options must be an object");
    }
    const unknown = Object.keys(options).find((key) => !["mode", "call_ids", "locale"].includes(key));
    if (unknown) fail("invalid_argument", `unknown narration option ${unknown}`);
    const requestedMode = String(options.mode || "deterministic");
    if (!["off", "deterministic", "adapter"].includes(requestedMode)) {
      fail("invalid_argument", "narration mode must be off|deterministic|adapter");
    }
    const locale = String(options.locale || "zh-CN");
    if (locale !== "zh-CN") fail("invalid_argument", "only zh-CN narration is frozen");
    if (options.call_ids != null && (!Array.isArray(options.call_ids)
        || options.call_ids.some((value) => typeof value !== "string"))) {
      fail("invalid_argument", "narration call_ids must be a string array");
    }
    const allowed = new Set(options.call_ids || this.toolResults.map((row) => row.call_id));
    const views = this.toolResults.filter((row) => allowed.has(row.call_id));
    if (requestedMode !== "off" && views.length === 0) {
      fail("protocol_violation", "narration_requires_terminal_result");
    }
    const fallbackUsed = requestedMode === "adapter";
    const mode = fallbackUsed ? "deterministic" : requestedMode;
    const text = mode === "off" ? null : views.map((row) => {
      const call = this.completedCalls.get(row.call_id);
      if (!call) fail("protocol_violation", "narration_requires_session_result");
      return deterministicNarration(call, row);
    }).join("\n");
    return {
      wire_version: sdkVersions().wire_version,
      mode,
      requested_mode: requestedMode,
      locale,
      text,
      grounded: true,
      fallback_used: fallbackUsed,
      adapter_verified: false,
      provider_id: "mei-zh-deterministic-narration-v1",
      call_ids: views.map((row) => row.call_id),
    };
  }

  cancel() {
    this.cancelled = true;
  }

  close() {
    this.closed = true;
    this.pendingCall = null;
    this.toolResults = [];
    this.completedCalls.clear();
    this.responded = false;
  }
}

function validateCompleteRequestV2(request) {
  const allowed = new Set([
    "wire_version", "query", "context", "evidence", "history", "tool_results", "permissions", "state",
    "mw", "mw_disposition", "confidence", "enforce_confidence", "oracle_tools", "candidate_text",
    "decode_mode", "max_new",
  ]);
  const unknown = Object.keys(request).find((key) => !allowed.has(key));
  if (unknown) fail("invalid_argument", `unknown CompleteRequestV2 field ${unknown}`);
  if (typeof request.query !== "string" || !request.query) {
    fail("invalid_argument", "CompleteRequestV2.query must be a non-empty string");
  }
  if (Object.hasOwn(request, "mw") && Object.hasOwn(request, "mw_disposition")) {
    fail("invalid_argument", "mw and mw_disposition are mutually exclusive");
  }
  if (request.max_new != null && (!Number.isInteger(request.max_new) || request.max_new < 1 || request.max_new > 128)) {
    fail("invalid_argument", "max_new must be between 1 and 128");
  }
  if (request.decode_mode != null && !["raw", "constrained"].includes(request.decode_mode)) {
    fail("invalid_argument", "invalid decode_mode");
  }
  if (request.candidate_text != null && typeof request.candidate_text !== "string") {
    fail("invalid_argument", "candidate_text must be string or null");
  }
  if (request.oracle_tools != null && (!Array.isArray(request.oracle_tools) || request.oracle_tools.length > 5)) {
    fail("too_many_tools", "oracle_tools accepts at most 5 entries");
  }
  for (const key of ["context", "permissions", "state", "mw", "mw_disposition", "confidence"]) {
    if (request[key] != null && (typeof request[key] !== "object" || Array.isArray(request[key]))) {
      fail("invalid_argument", `${key} must be an object`);
    }
  }
  for (const key of ["evidence", "history", "tool_results"]) {
    if (request[key] != null && !Array.isArray(request[key])) fail("invalid_argument", `${key} must be an array`);
  }
  if (request.enforce_confidence != null && typeof request.enforce_confidence !== "boolean") {
    fail("invalid_argument", "enforce_confidence must be boolean");
  }
}
