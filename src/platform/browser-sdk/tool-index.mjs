import { createHash } from "node:crypto";
import {
  catalogFingerprint, compactTools, dumpsCanonical, validateToolSchema,
} from "./protocol.mjs";

const SHA256 = /^[0-9a-f]{64}$/;
const ROOT_KEYS = new Set([
  "format", "fingerprint", "catalog_sha256", "model_sha256", "head_sha256",
  "tokenizer_sha256", "serializer_id", "dtype", "normalized", "dimension", "records",
]);
const RECORD_KEYS = new Set(["tool_id", "schema", "schema_sha256", "embedding_f16_base64"]);

function fail(message) {
  throw Object.assign(new Error(message), { code: "package_invalid" });
}

function exactKeys(value, expected, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)
      || Object.keys(value).length !== expected.size
      || Object.keys(value).some((key) => !expected.has(key))) fail(`${path} fields do not match v2`);
}

function decodeF16(bits) {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let f32;
  if (exponent === 0) {
    if (mantissa === 0) f32 = sign;
    else {
      exponent = -14;
      while ((mantissa & 0x0400) === 0) { mantissa <<= 1; exponent -= 1; }
      f32 = sign | ((exponent + 127) << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent === 31) f32 = sign | 0x7f800000 | (mantissa << 13);
  else f32 = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setUint32(0, f32 >>> 0, false);
  return view.getFloat32(0, false);
}

export function parseToolIndex(input) {
  let raw;
  try { raw = typeof input === "string" ? JSON.parse(input) : JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(input)); }
  catch { fail("tool index is not valid UTF-8 JSON"); }
  exactKeys(raw, ROOT_KEYS, "tool index");
  if (raw.format !== "mei-tool-index-v2" || raw.serializer_id !== "mei-tool-call-serializer-v2"
      || raw.dtype !== "float16" || raw.normalized !== true) fail("unsupported tool index format");
  for (const field of ["fingerprint", "catalog_sha256", "model_sha256", "head_sha256", "tokenizer_sha256"]) {
    if (!SHA256.test(String(raw[field] || ""))) fail(`tool index ${field} must be lowercase sha256`);
  }
  if (!Number.isSafeInteger(raw.dimension) || raw.dimension < 0 || !Array.isArray(raw.records)
      || (raw.records.length && raw.dimension === 0)) fail("tool index dimension/records mismatch");
  const ids = new Set();
  const records = raw.records.map((record) => {
    exactKeys(record, RECORD_KEYS, "tool index record");
    if (typeof record.tool_id !== "string" || !record.tool_id || ids.has(record.tool_id)) fail("duplicate or empty tool ID");
    ids.add(record.tool_id);
    validateToolSchema(record.schema);
    if (record.schema.name !== record.tool_id) fail("tool index ID/schema mismatch");
    const schemaSha = createHash("sha256").update(dumpsCanonical(compactTools([record.schema]))).digest("hex");
    if (record.schema_sha256 !== schemaSha) fail("tool index schema hash mismatch");
    if (typeof record.embedding_f16_base64 !== "string"
        || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(record.embedding_f16_base64)) {
      fail("tool index embedding is invalid base64");
    }
    const bytes = Buffer.from(record.embedding_f16_base64, "base64");
    if (bytes.toString("base64") !== record.embedding_f16_base64 || bytes.length !== raw.dimension * 2) {
      fail("tool index embedding dimension mismatch");
    }
    const embedding = Array.from({ length: raw.dimension }, (_, index) => decodeF16(bytes.readUInt16LE(index * 2)));
    const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm < 0.99 || norm > 1.01) fail("tool index embedding is not normalized");
    return Object.freeze({ tool_id: record.tool_id, schema: structuredClone(record.schema), schema_sha256: schemaSha, embedding: Object.freeze(embedding) });
  });
  const utf8 = new TextEncoder();
  const compareId = (left, right) => {
    const a = utf8.encode(left.tool_id); const b = utf8.encode(right.tool_id);
    const length = Math.min(a.length, b.length);
    for (let index = 0; index < length; index += 1) if (a[index] !== b[index]) return a[index] - b[index];
    return a.length - b.length;
  };
  for (let index = 1; index < records.length; index += 1) {
    if (compareId(records[index - 1], records[index]) >= 0) fail("tool index records must be sorted by UTF-8 tool ID");
  }
  const catalogSha = catalogFingerprint(records.map((record) => record.schema));
  if (raw.catalog_sha256 !== catalogSha) fail("tool index catalog hash mismatch");
  const schemaMap = Object.fromEntries(records.map((record) => [record.tool_id, record.schema_sha256]));
  const schemaSha = createHash("sha256").update(dumpsCanonical(schemaMap)).digest("hex");
  const fingerprint = createHash("sha256").update(dumpsCanonical({
    catalog_sha256: catalogSha,
    head_sha256: raw.head_sha256,
    model_sha256: raw.model_sha256,
    schema_sha256: schemaSha,
    serializer_id: raw.serializer_id,
    tokenizer_sha256: raw.tokenizer_sha256,
  })).digest("hex");
  if (raw.fingerprint !== fingerprint) fail("tool index provenance fingerprint mismatch");
  return Object.freeze({
    fingerprint, catalog_sha256: catalogSha, model_sha256: raw.model_sha256,
    head_sha256: raw.head_sha256, tokenizer_sha256: raw.tokenizer_sha256,
    dimension: raw.dimension, records: Object.freeze(records),
    topK(query, k = 5) {
      if (!Array.isArray(query) && !ArrayBuffer.isView(query)) fail("query embedding must be an array");
      const values = Array.from(query, Number);
      const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
      if (values.length !== raw.dimension || !Number.isFinite(norm) || norm <= 0 || values.some((value) => !Number.isFinite(value))) {
        fail("query embedding dimension/norm mismatch");
      }
      return [...records].map((record) => ({
        record,
        score: record.embedding.reduce((sum, value, index) => sum + value * (values[index] / norm), 0),
      })).sort((left, right) => right.score - left.score || compareId(left.record, right.record))
        .slice(0, Math.min(Math.max(Number(k) || 0, 0), records.length)).map((row) => row.record);
    },
  });
}
