import { createHash } from "node:crypto";
import { isIP } from "node:net";
import { ERRORS, PROTOCOL, sdkVersions } from "./version.mjs";

export function dumpsCanonical(obj) {
  return JSON.stringify(canonicalValue(obj));
}

export function canonicalValue(value) {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort((left, right) => {
      const a = new TextEncoder().encode(left);
      const b = new TextEncoder().encode(right);
      const length = Math.min(a.length, b.length);
      for (let index = 0; index < length; index += 1) {
        if (a[index] !== b[index]) return a[index] - b[index];
      }
      return a.length - b.length;
    })) out[key] = canonicalValue(value[key]);
    return out;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) {
      throw Object.assign(new Error("canonical JSON rejects non-finite or unsafe integers"), { code: "invalid_json" });
    }
    return Object.is(value, -0) ? 0 : value;
  }
  if (value !== null && !["string", "boolean"].includes(typeof value)) {
    throw Object.assign(new Error(`canonical JSON rejects ${typeof value}`), { code: "invalid_json" });
  }
  return value;
}

export function validateSemanticJson(value) {
  canonicalValue(value);
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

export function catalogFingerprint(tools) {
  const identity = tools.map((tool) => ({
    name: tool.name ?? null,
    description: tool.description || "",
    parameters: tool.parameters || { type: "object", properties: {} },
    required_permissions: tool.required_permissions ?? null,
    required_state: tool.required_state ?? null,
    "x-mei-permissions": tool["x-mei-permissions"] ?? null,
    "x-mei-state": tool["x-mei-state"] ?? null,
  }));
  const utf8 = new TextEncoder();
  identity.sort((left, right) => {
    const a = utf8.encode(String(left.name || ""));
    const b = utf8.encode(String(right.name || ""));
    const length = Math.min(a.length, b.length);
    for (let index = 0; index < length; index += 1) {
      if (a[index] !== b[index]) return a[index] - b[index];
    }
    return a.length - b.length;
  });
  return createHash("sha256").update(dumpsCanonical(identity), "utf8").digest("hex");
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
    dumpsCanonical(request.context || {}),
    dumpsCanonical(request.evidence || []),
    dumpsCanonical(request.tool_results || []),
    dumpsCanonical(request.state || {}),
    dumpsCanonical(request.mw || request.mw_disposition || {}),
    dumpsCanonical(request.confidence ?? null),
  ].join("\n");
  return leakMarkers(blob);
}

const SCALARS = new Set(["string", "boolean", "integer", "number", "null"]);
const FORMATS = new Set(["date", "date-time", "time", "email", "uuid", "uri", "ipv4", "ipv6"]);
const COMBINATORS = [
  "$ref", "$dynamicRef", "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
  "dependentSchemas", "patternProperties", "propertyNames", "prefixItems", "contains",
  "unevaluatedItems", "unevaluatedProperties", "additionalItems",
];

function schemaError(message) {
  throw Object.assign(new Error(message), { code: "unsupported_schema" });
}

function onlySchemaKeys(schema, allowed, path) {
  const unknown = Object.keys(schema || {}).find((key) => !allowed.includes(key));
  if (unknown) schemaError(`${path}.${unknown} is outside the portable subset`);
}

function valueType(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value;
}

function schemaTypes(schema, path) {
  let types;
  if (typeof schema.type === "string") types = [schema.type];
  else if (Array.isArray(schema.type) && schema.type.length && schema.type.every((value) => typeof value === "string")) {
    types = [...new Set(schema.type)];
  } else if (Object.hasOwn(schema, "const") || Array.isArray(schema.enum)) {
    const sample = Object.hasOwn(schema, "const") ? schema.const : schema.enum[0];
    if (sample === undefined) schemaError(`${path}.enum must be non-empty`);
    types = [valueType(sample)];
  } else schemaError(`${path}.type is required`);
  if (types.length > 2 || (types.length === 2 && !types.includes("null"))
      || types.some((kind) => !SCALARS.has(kind) && kind !== "array")) {
    schemaError(`${path}.type is outside the portable subset`);
  }
  return types;
}

function typeMatches(value, kind) {
  if (kind === "number") return typeof value === "number" && Number.isFinite(value);
  if (kind === "integer") return Number.isSafeInteger(value);
  if (kind === "array") return Array.isArray(value);
  if (kind === "null") return value === null;
  return typeof value === kind;
}

function validFormat(value, format) {
  if (format === "email") return /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$/.test(value);
  if (format === "uuid") return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
  if (format === "uri") {
    if (!/^[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$/.test(value)) return false;
    for (let index = value.indexOf("%"); index !== -1; index = value.indexOf("%", index + 1)) {
      if (!/^[0-9A-Fa-f]{2}$/.test(value.slice(index + 1, index + 3))) return false;
    }
    return true;
  }
  if (format === "ipv4") return isIP(value) === 4;
  if (format === "ipv6") return isIP(value) === 6;
  const time = /^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]{1,9})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$/;
  if (format === "time") return time.test(value);
  const date = /^(\d{4})-(\d{2})-(\d{2})$/;
  const match = date.exec(format === "date-time" ? String(value).split("T")[0] : value);
  if (!match) return false;
  const [, yearText, monthText, dayText] = match;
  const year = Number(yearText); const month = Number(monthText); const day = Number(dayText);
  const leap = year > 0 && year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const maxDay = [0, 31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month] || 0;
  const dateOk = year > 0 && day >= 1 && day <= maxDay;
  if (format === "date") return dateOk;
  return dateOk && String(value).includes("T") && time.test(String(value).slice(String(value).indexOf("T") + 1));
}

function validatePortablePattern(pattern, path) {
  const chars = [...pattern];
  let escaped = false;
  let inClass = false;
  for (let index = 0; index < chars.length; index += 1) {
    const character = chars[index];
    if (escaped) {
      if (/^[A-Za-z0-9]$/.test(character)) {
        schemaError(`${path}.pattern uses non-portable escape \\${character}`);
      }
      escaped = false;
      continue;
    }
    if (character === "\\") { escaped = true; continue; }
    if (character === "(" && chars[index + 1] === "?") {
      schemaError(`${path}.pattern uses non-portable group extension`);
    }
    if (inClass && ((character === "&" && chars[index + 1] === "&")
        || (character === "-" && chars[index + 1] === "-") || character === "[")) {
      schemaError(`${path}.pattern uses non-portable character class syntax`);
    }
    if (character === "[") inClass = true;
    else if (character === "]") inClass = false;
  }
  if (escaped) schemaError(`${path}.pattern ends with an escape`);
  try { new RegExp(pattern, "u"); } catch (error) { schemaError(`${path}.pattern: ${error.message}`); }
}

export function validateJsonValue(value, schema) {
  const types = schemaTypes(schema, "$");
  if (!types.some((kind) => typeMatches(value, kind))) return `type mismatch: expected ${types.join("|")}`;
  if (Object.hasOwn(schema, "const") && !deepEqual(value, schema.const)) return "value does not equal const";
  if (Array.isArray(schema.enum) && !schema.enum.some((item) => JSON.stringify(item) === JSON.stringify(value))) return "value is not in enum";
  if (value === null) return null;
  if (typeof value === "string") {
    const length = [...value].length;
    if (schema.minLength != null && length < schema.minLength) return "string is shorter than minLength";
    if (schema.maxLength != null && length > schema.maxLength) return "string exceeds maxLength";
    if (schema.pattern != null) {
      validatePortablePattern(schema.pattern, "$");
      if (!new RegExp(schema.pattern, "u").test(value)) return "string does not match pattern";
    }
    if (schema.format != null && !validFormat(value, schema.format)) return `invalid ${schema.format}`;
  }
  if (typeof value === "number") {
    if (schema.minimum != null && value < schema.minimum) return "number violates minimum";
    if (schema.maximum != null && value > schema.maximum) return "number violates maximum";
    if (schema.exclusiveMinimum != null && value <= schema.exclusiveMinimum) return "number violates exclusiveMinimum";
    if (schema.exclusiveMaximum != null && value >= schema.exclusiveMaximum) return "number violates exclusiveMaximum";
    if (schema.multipleOf != null) {
      const quotient = value / schema.multipleOf;
      if (Math.abs(quotient - Math.round(quotient)) > 1e-9 * Math.max(1, Math.abs(quotient))) return "number is not a multiple";
    }
  }
  if (Array.isArray(value)) {
    if (schema.minItems != null && value.length < schema.minItems) return "array is shorter than minItems";
    if (schema.maxItems != null && value.length > schema.maxItems) return "array exceeds maxItems";
    for (const item of value) { const issue = validateJsonValue(item, schema.items); if (issue) return issue; }
  }
  return null;
}

function validateScalarSchema(schema, path) {
  onlySchemaKeys(schema, [
    "type", "enum", "const", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern", "format", "title", "description", "examples", "$comment",
  ], path);
  const types = schemaTypes(schema, path);
  if (types.includes("array")) schemaError(`${path} nested arrays are unsupported`);
  if (schema.pattern != null) {
    if (!types.includes("string") || typeof schema.pattern !== "string") schemaError(`${path}.pattern`);
    validatePortablePattern(schema.pattern, path);
  }
  if (schema.format != null && (!types.includes("string") || !FORMATS.has(schema.format))) schemaError(`${path}.format is unsupported`);
  for (const key of ["minLength", "maxLength"]) if (schema[key] != null && (!types.includes("string") || !Number.isSafeInteger(schema[key]) || schema[key] < 0)) schemaError(`${path}.${key}`);
  if ((schema.minLength ?? 0) > (schema.maxLength ?? Number.MAX_SAFE_INTEGER)) schemaError(`${path}.minLength exceeds maxLength`);
  for (const key of ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]) {
    if (schema[key] != null && (!types.some((kind) => kind === "integer" || kind === "number") || !Number.isFinite(schema[key]) || (key === "multipleOf" && schema[key] <= 0))) schemaError(`${path}.${key}`);
  }
  if (types.some((kind) => kind === "integer" || kind === "number")) {
    const lowers = [[schema.minimum, false], [schema.exclusiveMinimum, true]].filter(([value]) => value != null);
    const uppers = [[schema.maximum, false], [schema.exclusiveMaximum, true]].filter(([value]) => value != null);
    lowers.sort((a, b) => b[0] - a[0] || Number(b[1]) - Number(a[1]));
    uppers.sort((a, b) => a[0] - b[0] || Number(b[1]) - Number(a[1]));
    if (lowers.length && uppers.length) {
      const [low, lowExclusive] = lowers[0];
      const [high, highExclusive] = uppers[0];
      if (low > high || (low === high && (lowExclusive || highExclusive))) schemaError(`${path} numeric interval is empty`);
      if (types.includes("integer")) {
        const first = lowExclusive ? Math.floor(low) + 1 : Math.ceil(low);
        const last = highExclusive ? Math.ceil(high) - 1 : Math.floor(high);
        if (first > last) schemaError(`${path} integer interval is empty`);
      }
    }
  }
}

export function validateToolSchema(tool) {
  const name = typeof tool?.name === "string" && tool.name ? tool.name : null;
  if (!name) throw Object.assign(new Error("tool.name is required"), { code: "unsupported_schema" });
  const allowedToolKeys = [
    "name", "description", "parameters", "required_permissions", "required_state",
    "x-mei-permissions", "x-mei-state",
  ];
  const unknownToolKey = Object.keys(tool).find((key) => !allowedToolKeys.includes(key));
  if (unknownToolKey) schemaError(`tool.${unknownToolKey} is outside the portable subset`);
  const schema = tool.parameters;
  if (tool.description != null && typeof tool.description !== "string") schemaError(`${name}.description must be a string`);
  if (!schema || schema.type !== "object" || typeof schema.properties !== "object" || Array.isArray(schema.properties)) {
    throw Object.assign(new Error(`${name}.parameters must be a root object`), { code: "unsupported_schema" });
  }
  const reject = (value, path) => {
    for (const key of COMBINATORS) {
      if (Object.hasOwn(value || {}, key)) {
        throw Object.assign(new Error(`${path}.${key} is unsupported`), { code: "unsupported_schema" });
      }
    }
  };
  reject(schema, name);
  onlySchemaKeys(schema, ["type", "properties", "required", "additionalProperties", "$schema", "$id", "title", "description", "examples", "$comment"], name);
  if (schema.additionalProperties != null && schema.additionalProperties !== false) schemaError(`${name}.additionalProperties must be false`);
  const required = schema.required || [];
  if (!Array.isArray(required) || required.some((value) => typeof value !== "string") || new Set(required).size !== required.length || required.some((value) => !Object.hasOwn(schema.properties, value))) schemaError(`${name}.required is invalid`);
  for (const [property, spec] of Object.entries(schema.properties)) {
    reject(spec, `${name}.${property}`);
    const types = schemaTypes(spec, `${name}.${property}`);
    if (types.includes("array")) {
      if (types.length !== 1) schemaError(`${name}.${property} nullable arrays are unsupported`);
      onlySchemaKeys(spec, ["type", "items", "minItems", "maxItems", "enum", "const", "default", "title", "description", "examples", "$comment"], `${name}.${property}`);
      reject(spec.items || {}, `${name}.${property}.items`);
      validateScalarSchema(spec.items || {}, `${name}.${property}.items`);
      for (const key of ["minItems", "maxItems"]) if (spec[key] != null && (!Number.isSafeInteger(spec[key]) || spec[key] < 0)) schemaError(`${name}.${property}.${key}`);
      if ((spec.minItems || 0) > (spec.maxItems ?? Number.MAX_SAFE_INTEGER)) schemaError(`${name}.${property}.minItems exceeds maxItems`);
    } else {
      validateScalarSchema(spec, `${name}.${property}`);
    }
    for (const key of ["const", "default"]) {
      if (Object.hasOwn(spec, key)) { const issue = validateJsonValue(spec[key], spec); if (issue) schemaError(`${name}.${property}.${key}: ${issue}`); }
    }
    if (spec.enum != null) {
      if (!Array.isArray(spec.enum) || !spec.enum.length) schemaError(`${name}.${property}.enum must be non-empty`);
      for (const value of spec.enum) { const issue = validateJsonValue(value, spec); if (issue) schemaError(`${name}.${property}.enum: ${issue}`); }
    }
  }
  for (const key of ["required_permissions", "x-mei-permissions"]) {
    if (tool[key] != null && (!Array.isArray(tool[key]) || tool[key].some((value) => typeof value !== "string"))) schemaError(`${name}.${key} must be a string array`);
  }
  for (const key of ["required_state", "x-mei-state"]) {
    if (tool[key] != null && (typeof tool[key] !== "object" || Array.isArray(tool[key]))) schemaError(`${name}.${key} must be an object`);
  }
  return true;
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

function deepEqual(left, right) {
  return dumpsCanonical(left) === dumpsCanonical(right);
}

function gate(name, ok, detail = null, extra = {}) {
  return { gate: name, ok, detail, ...extra };
}

function blocked(gates, error, detail = null) {
  return {
    ok: false, refuse: true, execution: "refuse", function_calls: [], error, detail, gates,
    provenance: {}, unsupported_accepted: 0, unprovenanced_argument_accepted: 0,
  };
}

function stringSet(value) {
  return new Set(Array.isArray(value) ? value.map(String) : []);
}

export function validateGeneratedCall(
  text,
  tools,
  request,
  confidence = null,
  enforceConfidence = false,
  trustedToolResults = [],
) {
  const parsed = parseV2Text(text);
  const gates = [gate("grammar", parsed.ok, parsed.error)];
  if (!parsed.ok) return blocked(gates, "grammar_violation", parsed.error);
  if (parsed.refuse) {
    gates.push(gate("schema", true));
    return { ok: true, refuse: true, execution: "refuse", function_calls: [], error: null, gates, provenance: {} };
  }
  const call = parsed.function_calls[0];
  try {
    validateSemanticJson(call);
  } catch (error) {
    gates.push(gate("schema", false, "numeric_domain"));
    return blocked(gates, "schema_validation", String(error?.message || error));
  }
  const tool = tools.find((candidate) => candidate.name === call.name);
  if (!tool) { gates.push(gate("schema", false, "unsupported_tool")); return blocked(gates, "schema_validation", "unsupported_tool"); }
  const properties = tool.parameters?.properties || {};
  const required = new Set(tool.parameters?.required || []);
  let schemaIssue = null;
  for (const [argument, value] of Object.entries(call.arguments)) {
    if (!Object.hasOwn(properties, argument)) { schemaIssue = `unsupported argument ${argument}`; break; }
    const issue = validateJsonValue(value, properties[argument]);
    if (issue) { schemaIssue = `${argument}: ${issue}`; break; }
  }
  if (!schemaIssue) {
    const missing = [...required].find((argument) => !Object.hasOwn(call.arguments, argument));
    if (missing) schemaIssue = `missing required ${missing}`;
  }
  gates.push(gate("schema", !schemaIssue, schemaIssue));
  if (schemaIssue) return blocked(gates, "schema_validation", schemaIssue);

  const evidenceRows = [...(request.evidence || []), ...(request.context?.facts || [])];
  const provenance = {};
  const missing = [];
  for (const [argument, value] of Object.entries(call.arguments)) {
    const schema = properties[argument];
    if (Object.hasOwn(schema, "const") && deepEqual(schema.const, value)) {
      provenance[argument] = { source: "schema_const", verified: true, canonical_value: value };
      continue;
    }
    const evidence = evidenceRows.find((row) => row?.verified === true && typeof row.source === "string" && row.source
      && deepEqual(row.value, value)
      && ((row.tool === call.name && row.argument === argument) || (row.subject === call.name && row.predicate === argument)));
    if (evidence) provenance[argument] = evidence;
    else {
      const result = trustedToolResults.find((candidate) => candidate?.status === "ok"
        && candidate.provenance?.verified === true
        && typeof candidate.provenance?.source === "string" && candidate.provenance.source
        && candidate.payload && typeof candidate.payload === "object" && !Array.isArray(candidate.payload)
        && Object.hasOwn(candidate.payload, argument) && deepEqual(candidate.payload[argument], value));
      if (result) {
        provenance[argument] = {
          source: "verified_tool_result",
          locator: result.call_id || null,
          canonical_value: value,
          verified: true,
        };
      } else missing.push(argument);
    }
  }
  gates.push(gate("provenance", missing.length === 0, missing.length ? "missing" : null, { missing }));
  if (missing.length) return blocked(gates, "provenance_missing", missing.join(","));

  const rawPermissions = request.permissions ?? {};
  let scopes; let denies; let allowedTools = null; let deniedTools; let permissionError = null;
  if (Array.isArray(rawPermissions)) {
    scopes = stringSet(rawPermissions); denies = new Set(); deniedTools = new Set();
  } else if (rawPermissions && typeof rawPermissions === "object") {
    scopes = stringSet(rawPermissions.scopes || rawPermissions.grants);
    denies = stringSet(rawPermissions.denies);
    allowedTools = Array.isArray(rawPermissions.allowed_tools) ? stringSet(rawPermissions.allowed_tools) : null;
    deniedTools = stringSet(rawPermissions.denied_tools);
  } else permissionError = "permissions_invalid";
  const requiredPermissions = stringSet(tool.required_permissions || tool["x-mei-permissions"]);
  if (!permissionError && (deniedTools.has(call.name) || [...requiredPermissions].some((scope) => denies.has(scope)))) permissionError = "permission_denied";
  if (!permissionError && allowedTools && !allowedTools.has(call.name)) permissionError = "permission_not_granted";
  if (!permissionError && [...requiredPermissions].some((scope) => !scopes.has(scope))) permissionError = "permission_scope_missing";
  gates.push(gate("permission", !permissionError, permissionError));
  if (permissionError) return blocked(gates, permissionError);

  const state = request.state ?? {};
  const requiredState = tool.required_state || tool["x-mei-state"] || {};
  let stateError = null;
  if (!state || typeof state !== "object" || Array.isArray(state)) stateError = "state_invalid";
  else if (state.invalid) stateError = "state_invalid";
  else if (state.conflict) stateError = "state_conflict";
  else {
    const missingState = Object.entries(requiredState).find(([key, value]) => !deepEqual(state[key], value));
    if (missingState) stateError = `state_requirement_missing:${missingState[0]}`;
  }
  gates.push(gate("state", !stateError, stateError));
  if (stateError) return blocked(gates, stateError);

  const wireV2 = request.wire_version === "mei-runtime-wire-v2";
  const hasMwOverride = request.mw != null;
  const hasMwRuntime = request.mw_disposition != null;
  const mw = request.mw ?? request.mw_disposition ?? { decision: "continue", source: "deterministic-policy" };
  const decision = typeof mw === "string" ? mw : String(mw?.decision || mw?.disposition || "continue");
  const mwAllowed = typeof mw === "object" && Array.isArray(mw.allowed_tools) ? stringSet(mw.allowed_tools) : null;
  let mwError = null;
  if (wireV2 && hasMwOverride && hasMwRuntime) mwError = "mw_invalid";
  if (wireV2 && !mwError) {
    const allowedKeys = new Set(["decision", "allowed_tools", "source", "receipt_sha256"]);
    const expectedSources = hasMwOverride ? new Set(["protocol-test"]) : new Set(["mw-head", "deterministic-policy"]);
    if (!mw || typeof mw !== "object" || Array.isArray(mw)
        || Object.keys(mw).some((key) => !allowedKeys.has(key))
        || typeof mw.decision !== "string" || !expectedSources.has(mw.source)
        || (mw.source === "mw-head" && !/^[0-9a-f]{64}$/.test(String(mw.receipt_sha256 || "")))
        || (mw.allowed_tools != null && (!Array.isArray(mw.allowed_tools)
          || mw.allowed_tools.some((value) => typeof value !== "string")
          || new Set(mw.allowed_tools).size !== mw.allowed_tools.length))) mwError = "mw_invalid";
  }
  if (!mwError && decision === "constrain" && (!mwAllowed || mwAllowed.size === 0)) mwError = "mw_invalid";
  if (!mwError && ["stop", "block", "refuse"].includes(decision)) mwError = "mw_stop";
  if (!mwError && decision === "constrain" && mwAllowed && !mwAllowed.has(call.name)) mwError = "mw_constrained";
  if (!mwError && !["continue", "constrain"].includes(decision)) mwError = "mw_invalid";
  gates.push(gate("mw", !mwError, mwError));
  if (mwError) return blocked(gates, mwError);

  let confidenceShapeValid = true;
  if (wireV2 && confidence != null) {
    const allowedKeys = new Set(["value", "execute_high", "escalate_low", "source"]);
    confidenceShapeValid = confidence && typeof confidence === "object" && !Array.isArray(confidence)
      && Object.keys(confidence).every((key) => allowedKeys.has(key))
      && confidence.source === "protocol-test" && typeof confidence.value === "number";
  }
  if (wireV2 && request.enforce_confidence != null && typeof request.enforce_confidence !== "boolean") confidenceShapeValid = false;
  const rawConfidence = confidence && typeof confidence === "object" ? confidence.value : confidence;
  const executeHigh = confidence && typeof confidence === "object" ? Number(confidence.execute_high ?? 0.70) : 0.70;
  const escalateLow = confidence && typeof confidence === "object" ? Number(confidence.escalate_low ?? 0.35) : 0.35;
  if (rawConfidence == null && enforceConfidence) { gates.push(gate("confidence", false, "confidence_unavailable")); return blocked(gates, "confidence_unavailable"); }
  if (!confidenceShapeValid || (rawConfidence != null && (!Number.isFinite(rawConfidence) || rawConfidence < 0 || rawConfidence > 1
      || !Number.isFinite(executeHigh) || !Number.isFinite(escalateLow) || executeHigh < 0 || executeHigh > 1
      || escalateLow < 0 || escalateLow > 1 || escalateLow > executeHigh))) {
    gates.push(gate("confidence", false, "confidence_invalid")); return blocked(gates, "confidence_invalid");
  }
  const execution = rawConfidence == null || rawConfidence >= executeHigh ? "execute" : rawConfidence >= escalateLow ? "escalate" : "refuse";
  gates.push(gate("confidence", execution === "execute", execution === "execute" ? null : execution, { value: rawConfidence, execution }));
  return {
    ok: true, refuse: execution !== "execute", execution,
    function_calls: execution === "execute" ? parsed.function_calls : [], error: null, gates,
    provenance, confidence_value: rawConfidence,
    unsupported_accepted: 0, unprovenanced_argument_accepted: 0,
  };
}

export function renderRequest(request, tools) {
  if (tools.length > PROTOCOL.max_selected_tools) {
    throw new Error("complete() accepts at most 5 selected tools; retrieval must run first");
  }
  const wireV2 = request.wire_version === "mei-runtime-wire-v2";
  let sink = PROTOCOL.task_contract;
  if (!wireV2) {
    const facts = String(request.context?.system_facts || request.system_facts || "").trim();
    if (facts) sink += `\n系统事实：${facts}`;
    if (request.permissions && (Array.isArray(request.permissions) ? request.permissions.length : Object.keys(request.permissions).length)) {
      sink += `\n权限：${Array.isArray(request.permissions) ? request.permissions.join("、") : dumpsCanonical(request.permissions)}`;
    }
    sink += `\n点选实体：${request.selected_entity ? String(request.selected_entity) : "无"}`;
  }
  sink += `\n<tools>${dumpsCanonical(compactTools(tools))}</tools>`;
  const ordinary = [];
  for (const turn of request.history || []) {
    const role = turn.role || "user";
    const content = String(turn.content || turn.text || "").trim();
    if (content) ordinary.push(`${role}：${content}`);
  }
  const results = request.tool_results?.length ? request.tool_results : (request.prior_tool_results || []);
  for (const result of results) {
    ordinary.push(`tool：${typeof result === "object" && result !== null ? dumpsCanonical(result) : String(result)}`);
  }
  if (wireV2) {
    for (const [tag, value] of [
      ["context", request.context || {}],
      ["evidence", request.evidence || []],
      ["permissions", request.permissions || {}],
      ["state", request.state || {}],
      ["mw", request.mw || request.mw_disposition || {}],
    ]) ordinary.push(`<${tag}>${dumpsCanonical(value)}</${tag}>`);
  }
  ordinary.push(`user：${String(request.query || "").trim()}`);
  const ordinaryText = ordinary.join("\n");
  return {
    prompt: `${sink}\n${ordinaryText}`,
    sink,
    ordinary: ordinaryText,
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
  const result = {
    wire_version: sdkVersions().wire_version,
    ok: fields.ok,
    error: fields.error,
    refuse: fields.refuse,
    selected_tools: fields.selected_tools,
    schema_fingerprint: fields.schema_fingerprint,
    function_calls: fields.function_calls,
    raw_text: fields.raw_text,
    confidence: fields.confidence || { available: false, value: null, source: null },
    provenance: fields.provenance,
    capabilities: fields.capabilities,
    stats: { backend: "protocol", wall_ms: fields.wall_ms ?? 0, decode_mode: fields.decode_mode || "constrained" },
  };
  if (fields.execution) result.execution = fields.execution;
  return result;
}

export function complete(request, capabilities, trustedToolResults = []) {
  const decodeMode = request.decode_mode || "constrained";
  if (decodeMode === "raw" && ["candidate", "release"].includes(capabilities?.release_class)) {
    return turn({
      ok: false, refuse: true, selected_tools: [], schema_fingerprint: null,
      function_calls: [], raw_text: null, error: errorFromId("decode_mode_forbidden"),
      provenance: { validated: false, ok: false, detail: "decode_mode_forbidden" },
      capabilities, decode_mode: decodeMode,
    });
  }
  if (["candidate", "release"].includes(capabilities?.release_class)
      && (request.candidate_text != null || Object.hasOwn(request, "mw")
        || Object.hasOwn(request, "mw_disposition") || Object.hasOwn(request, "confidence")
        || request.enforce_confidence === false)) {
    return turn({
      ok: false, refuse: true, selected_tools: [], schema_fingerprint: null,
      function_calls: [], raw_text: null,
      error: errorFromId("decode_mode_forbidden", "candidate/release sessions forbid protocol-test and learned-head overrides"),
      provenance: { validated: false, ok: false, detail: "decode_mode_forbidden" },
      capabilities, decode_mode: decodeMode,
    });
  }
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
  const validated = validateGeneratedCall(
    String(request.candidate_text),
    tools,
    request,
    request.confidence ?? null,
    Boolean(request.enforce_confidence),
    trustedToolResults,
  );
  const validationError = validated.error;
  const deterministicRefusal = typeof validationError === "string" && (
    validationError === "provenance_missing" || validationError.startsWith("permission_")
    || validationError.startsWith("state_") || validationError.startsWith("mw_")
    || validationError.startsWith("confidence_")
  );
  return turn({
    ok: deterministicRefusal || validated.ok,
    refuse: deterministicRefusal || validated.refuse,
    selected_tools: rendered.selected_tools,
    schema_fingerprint: rendered.schema_fingerprint,
    function_calls: validated.ok && !deterministicRefusal ? validated.function_calls : [],
    raw_text: String(request.candidate_text),
    error: validationError && !deterministicRefusal ? errorFromId("protocol_violation", validationError) : null,
    provenance: {
      validated: true,
      ok: validated.ok,
      detail: validationError,
      arguments: validated.provenance || {},
      gates: validated.gates || [],
    },
    confidence: validated.confidence_value == null
      ? undefined
      : { available: true, value: validated.confidence_value, source: "protocol-test" },
    execution: validated.execution,
    capabilities,
    decode_mode: decodeMode,
  });
}
