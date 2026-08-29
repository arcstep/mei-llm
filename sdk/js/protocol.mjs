import { createHash } from "node:crypto";
import { ERRORS, PROTOCOL, sdkVersions } from "./version.mjs";

export function dumpsCanonical(obj) {
  return JSON.stringify(obj);
}

export function compactTools(tools) {
  return tools.map((tool) => ({
    name: tool.name ?? null,
    description: tool.description || "",
    parameters: tool.parameters || { type: "object", properties: {} },
  }));
}

export function schemaFingerprint(tools) {
  return createHash("sha256").update(dumpsCanonical(compactTools(tools)), "utf8").digest("hex");
}

export function leakMarkers(text) {
  return PROTOCOL.forbidden_markers.filter((m) => (text || "").includes(m));
}

export function requestLeaks(request) {
  const blob = [
    String(request.query || ""),
    String(request.system_facts || ""),
    String(request.selected_entity || ""),
    String(request.candidate_text || ""),
    dumpsCanonical(request.history || []),
    dumpsCanonical(request.prior_tool_results || []),
    dumpsCanonical(request.catalog || []),
    dumpsCanonical(request.oracle_tools || []),
    dumpsCanonical(request.permissions || []),
  ].join("\n");
  return leakMarkers(blob);
}

export function parseV2Text(text) {
  const raw = (text || "").trim();
  if (!raw) return { ok: false, function_calls: [], error: "empty", refuse: true };
  let obj;
  try {
    obj = JSON.parse(raw);
  } catch {
    return { ok: false, function_calls: [], error: "json", refuse: true };
  }
  if (Array.isArray(obj) && obj.length === 0) {
    return { ok: true, function_calls: [], error: null, refuse: true };
  }
  if (!Array.isArray(obj) || obj.length > PROTOCOL.max_calls || obj.length !== 1 || typeof obj[0] !== "object") {
    return { ok: false, function_calls: [], error: "illegal_shape", refuse: true };
  }
  const name = obj[0].name;
  const args = obj[0].arguments;
  if (typeof name !== "string" || typeof args !== "object" || args === null || Array.isArray(args)) {
    return { ok: false, function_calls: [], error: "illegal_item", refuse: true };
  }
  return {
    ok: true,
    function_calls: [{ name, arguments: args }],
    error: null,
    refuse: false,
  };
}

export function renderRequest(request, tools) {
  if (tools.length > PROTOCOL.max_selected_tools) {
    throw new Error("complete() accepts at most 5 selected tools; retrieval must run first");
  }
  const facts = String(request.system_facts || "").trim();
  let sink = PROTOCOL.task_contract;
  if (facts) sink += `\n系统事实：${facts}`;
  if (request.permissions && request.permissions.length) {
    sink += `\n权限：${request.permissions.join("、")}`;
  }
  sink += `\n点选实体：${request.selected_entity ? String(request.selected_entity) : "无"}`;
  sink += `\n<tools>${dumpsCanonical(compactTools(tools))}</tools>`;
  const ordinary = [];
  for (const turn of request.history || []) {
    const role = turn.role || "user";
    const content = String(turn.content || turn.text || "").trim();
    if (content) ordinary.push(`${role}：${content}`);
  }
  for (const result of request.prior_tool_results || []) {
    ordinary.push(`tool：${result}`);
  }
  ordinary.push(`user：${String(request.query || "").trim()}`);
  return {
    prompt: `${sink}\n${ordinary.join("\n")}`,
    schema_fingerprint: schemaFingerprint(tools),
    serializer_id: PROTOCOL.serializer_id,
    protocol_id: PROTOCOL.protocol_id,
    selected_tools: tools.map((t) => String(t.name || "")),
  };
}

function errorFromId(id, message) {
  const row = ERRORS[id];
  return { code: row.code, id, message: message || row.message };
}

function turn(fields) {
  return {
    wire_version: sdkVersions().wire_version,
    ok: fields.ok,
    error: fields.error,
    refuse: fields.refuse,
    selected_tools: fields.selected_tools,
    schema_fingerprint: fields.schema_fingerprint,
    function_calls: fields.function_calls,
    raw_text: fields.raw_text,
    confidence: { available: false, value: null, source: null },
    provenance: fields.provenance,
    capabilities: fields.capabilities,
    stats: { backend: "protocol", wall_ms: fields.wall_ms ?? 0, decode_mode: fields.decode_mode || "constrained" },
  };
}

export function complete(request, capabilities) {
  const decodeMode = request.decode_mode || "constrained";
  const leaks = requestLeaks(request);
  if (leaks.length) {
    return turn({
      ok: false,
      refuse: true,
      selected_tools: [],
      schema_fingerprint: null,
      function_calls: [],
      raw_text: null,
      error: errorFromId("gold_leak", `forbidden markers: ${leaks.join(",")}`),
      provenance: { validated: true, ok: false, detail: "gold_leak" },
      capabilities,
      decode_mode: decodeMode,
    });
  }
  const tools = request.oracle_tools ?? request.catalog ?? [];
  if (!Array.isArray(tools)) {
    return turn({
      ok: false,
      refuse: true,
      selected_tools: [],
      schema_fingerprint: null,
      function_calls: [],
      raw_text: null,
      error: errorFromId("invalid_argument", "tools must be a list"),
      provenance: { validated: false, ok: false, detail: "invalid_tools" },
      capabilities,
      decode_mode: decodeMode,
    });
  }
  if (tools.length > PROTOCOL.max_selected_tools) {
    return turn({
      ok: false,
      refuse: true,
      selected_tools: [],
      schema_fingerprint: null,
      function_calls: [],
      raw_text: null,
      error: errorFromId("too_many_tools"),
      provenance: { validated: true, ok: false, detail: "too_many_tools" },
      capabilities,
      decode_mode: decodeMode,
    });
  }
  const rendered = renderRequest(request, tools);
  if (request.candidate_text == null) {
    return turn({
      ok: false,
      refuse: true,
      selected_tools: rendered.selected_tools,
      schema_fingerprint: rendered.schema_fingerprint,
      function_calls: [],
      raw_text: null,
      error: errorFromId(
        "engine_unavailable",
        "portable inference is not in this experimental SDK; pass candidate_text for protocol validation",
      ),
      provenance: { validated: false, ok: false, detail: "engine_unavailable" },
      capabilities,
      decode_mode: decodeMode,
    });
  }
  const parsed = parseV2Text(String(request.candidate_text));
  if (parsed.function_calls.length) {
    const name = parsed.function_calls[0].name;
    if (!rendered.selected_tools.includes(name)) {
      return turn({
        ok: false,
        refuse: true,
        selected_tools: rendered.selected_tools,
        schema_fingerprint: rendered.schema_fingerprint,
        function_calls: [],
        raw_text: String(request.candidate_text),
        error: errorFromId("protocol_violation", `tool ${name} not in selected_tools`),
        provenance: { validated: true, ok: false, detail: "unknown_tool" },
        capabilities,
        decode_mode: decodeMode,
      });
    }
  }
  return turn({
    ok: parsed.ok,
    refuse: parsed.refuse,
    selected_tools: rendered.selected_tools,
    schema_fingerprint: rendered.schema_fingerprint,
    function_calls: parsed.ok ? parsed.function_calls : [],
    raw_text: String(request.candidate_text),
    error: parsed.ok ? null : errorFromId("protocol_violation", parsed.error || "protocol"),
    provenance: { validated: true, ok: parsed.ok, detail: parsed.error },
    capabilities,
    decode_mode: decodeMode,
  });
}
