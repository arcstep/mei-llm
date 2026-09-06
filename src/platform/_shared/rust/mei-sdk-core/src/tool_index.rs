//! Portable normalized-f16 tool-index parser and stable top-k primitive.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::canonical::{catalog_fingerprint, compact_tools, dumps_canonical, sha256_bytes};
use crate::error::SdkError;
use crate::protocol::validate_tool_schema;

pub const TOOL_INDEX_FORMAT: &str = "mei-tool-index-v2";
pub const TOOL_INDEX_SERIALIZER: &str = "mei-tool-call-serializer-v2";

#[derive(Debug, Clone)]
pub struct ToolRecord {
    pub tool_id: String,
    pub schema: Value,
    pub schema_sha256: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ToolIndex {
    pub fingerprint: String,
    pub catalog_sha256: String,
    pub model_sha256: String,
    pub head_sha256: String,
    pub tokenizer_sha256: String,
    pub dimension: usize,
    pub records: Vec<ToolRecord>,
}

fn invalid(message: impl Into<String>) -> SdkError {
    SdkError::new("package_invalid", message)
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn only_keys(value: &Value, allowed: &[&str], path: &str) -> Result<(), SdkError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(format!("{path} must be an object")))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(format!("unknown {path}.{key}")));
    }
    Ok(())
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn decode_base64(value: &str) -> Result<Vec<u8>, SdkError> {
    let bytes = value.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(invalid("tool index embedding is invalid base64"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = index + 1 == bytes.len() / 4;
        let a = base64_value(chunk[0]).ok_or_else(|| invalid("invalid base64"))?;
        let b = base64_value(chunk[1]).ok_or_else(|| invalid("invalid base64"))?;
        let pad2 = chunk[2] == b'=';
        let pad3 = chunk[3] == b'=';
        if pad2 && (!pad3 || !last || b & 0x0f != 0) {
            return Err(invalid("tool index embedding is non-canonical base64"));
        }
        if pad3 && !last {
            return Err(invalid("tool index embedding is non-canonical base64"));
        }
        let c = if pad2 {
            0
        } else {
            base64_value(chunk[2]).ok_or_else(|| invalid("invalid base64"))?
        };
        let d = if pad3 {
            0
        } else {
            base64_value(chunk[3]).ok_or_else(|| invalid("invalid base64"))?
        };
        if pad3 && !pad2 && c & 0x03 != 0 {
            return Err(invalid("tool index embedding is non-canonical base64"));
        }
        out.push((a << 2) | (b >> 4));
        if !pad2 {
            out.push((b << 4) | (c >> 2));
        }
        if !pad3 {
            out.push((c << 6) | d);
        }
    }
    Ok(out)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let mantissa = (value & 0x03ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mantissa = mantissa;
            let mut exponent = -14i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

impl ToolIndex {
    pub fn parse(bytes: &[u8]) -> Result<Self, SdkError> {
        let raw: Value = serde_json::from_slice(bytes)
            .map_err(|error| SdkError::new("invalid_json", error.to_string()))?;
        only_keys(
            &raw,
            &[
                "format",
                "fingerprint",
                "catalog_sha256",
                "model_sha256",
                "head_sha256",
                "tokenizer_sha256",
                "serializer_id",
                "dtype",
                "normalized",
                "dimension",
                "records",
            ],
            "tool_index",
        )?;
        if raw.get("format").and_then(Value::as_str) != Some(TOOL_INDEX_FORMAT)
            || raw.get("serializer_id").and_then(Value::as_str) != Some(TOOL_INDEX_SERIALIZER)
            || raw.get("dtype").and_then(Value::as_str) != Some("float16")
            || raw.get("normalized").and_then(Value::as_bool) != Some(true)
        {
            return Err(invalid("unsupported tool index format"));
        }
        for field in [
            "fingerprint",
            "catalog_sha256",
            "model_sha256",
            "head_sha256",
            "tokenizer_sha256",
        ] {
            if !raw.get(field).and_then(Value::as_str).is_some_and(sha) {
                return Err(invalid(format!(
                    "tool index {field} must be lowercase sha256"
                )));
            }
        }
        let dimension = raw
            .get("dimension")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| invalid("tool index dimension"))?;
        let rows = raw
            .get("records")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("tool index records"))?;
        if !rows.is_empty() && dimension == 0 {
            return Err(invalid(
                "non-empty tool index requires a positive dimension",
            ));
        }
        let mut ids = HashSet::new();
        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            only_keys(
                row,
                &["tool_id", "schema", "schema_sha256", "embedding_f16_base64"],
                "tool_index.records[]",
            )?;
            let tool_id = row
                .get("tool_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| invalid("tool index tool_id"))?;
            if !ids.insert(tool_id.to_string()) {
                return Err(invalid("duplicate tool ID in index"));
            }
            let schema = row
                .get("schema")
                .filter(|value| value.is_object())
                .cloned()
                .ok_or_else(|| invalid("tool index schema"))?;
            validate_tool_schema(&schema).map_err(invalid)?;
            if schema.get("name").and_then(Value::as_str) != Some(tool_id) {
                return Err(invalid("tool index ID/schema mismatch"));
            }
            let expected_schema_sha = sha256_bytes(
                dumps_canonical(&compact_tools(std::slice::from_ref(&schema))).as_bytes(),
            );
            if row.get("schema_sha256").and_then(Value::as_str)
                != Some(expected_schema_sha.as_str())
            {
                return Err(invalid("tool index schema hash mismatch"));
            }
            let packed = decode_base64(
                row.get("embedding_f16_base64")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("tool index embedding must be base64 text"))?,
            )?;
            if packed.len() != dimension.saturating_mul(2) {
                return Err(invalid("tool index embedding dimension mismatch"));
            }
            let embedding = packed
                .chunks_exact(2)
                .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                .collect::<Vec<_>>();
            let norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if !norm.is_finite() || !(0.99..=1.01).contains(&norm) {
                return Err(invalid("tool index embedding is not normalized"));
            }
            records.push(ToolRecord {
                tool_id: tool_id.to_string(),
                schema,
                schema_sha256: expected_schema_sha,
                embedding,
            });
        }
        if records
            .windows(2)
            .any(|pair| pair[0].tool_id.as_bytes() >= pair[1].tool_id.as_bytes())
        {
            return Err(invalid(
                "tool index records must be sorted by UTF-8 tool ID",
            ));
        }
        let catalog = records
            .iter()
            .map(|record| record.schema.clone())
            .collect::<Vec<_>>();
        let catalog_sha256 = catalog_fingerprint(&catalog);
        if raw.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256.as_str()) {
            return Err(invalid("tool index catalog hash mismatch"));
        }
        let schema_map = Value::Object(
            records
                .iter()
                .map(|record| (record.tool_id.clone(), json!(record.schema_sha256)))
                .collect(),
        );
        let combined_schema = sha256_bytes(dumps_canonical(&schema_map).as_bytes());
        let identity = json!({
            "catalog_sha256":catalog_sha256,
            "head_sha256":raw["head_sha256"],
            "model_sha256":raw["model_sha256"],
            "schema_sha256":combined_schema,
            "serializer_id":TOOL_INDEX_SERIALIZER,
            "tokenizer_sha256":raw["tokenizer_sha256"]
        });
        let expected_fingerprint = sha256_bytes(dumps_canonical(&identity).as_bytes());
        if raw.get("fingerprint").and_then(Value::as_str) != Some(expected_fingerprint.as_str()) {
            return Err(invalid("tool index provenance fingerprint mismatch"));
        }
        Ok(Self {
            fingerprint: expected_fingerprint,
            catalog_sha256,
            model_sha256: raw["model_sha256"].as_str().unwrap().to_string(),
            head_sha256: raw["head_sha256"].as_str().unwrap().to_string(),
            tokenizer_sha256: raw["tokenizer_sha256"].as_str().unwrap().to_string(),
            dimension,
            records,
        })
    }

    pub fn ranked(&self, query: &[f32]) -> Result<Vec<(f32, &ToolRecord)>, SdkError> {
        if query.len() != self.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::new(
                "invalid_argument",
                "query embedding dimension",
            ));
        }
        let norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(SdkError::new("invalid_argument", "query embedding norm"));
        }
        let mut scored = self
            .records
            .iter()
            .map(|record| {
                let score = record
                    .embedding
                    .iter()
                    .zip(query)
                    .map(|(left, right)| left * (right / norm))
                    .sum::<f32>();
                (score, record)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left.tool_id.as_bytes().cmp(right.tool_id.as_bytes()))
        });
        Ok(scored)
    }

    pub fn topk(&self, query: &[f32], k: usize) -> Result<Vec<&ToolRecord>, SdkError> {
        Ok(self
            .ranked(query)?
            .into_iter()
            .take(k.min(self.records.len()))
            .map(|(_, record)| record)
            .collect())
    }
}
