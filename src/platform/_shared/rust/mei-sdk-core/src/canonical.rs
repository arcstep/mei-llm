use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn dumps_canonical(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

/// Canonical JSON for every semantic hash: object keys are recursively sorted
/// by their UTF-8 bytes and arrays retain order. `serde_json::Value` cannot
/// represent NaN/Inf; strict parsers reject duplicate keys at trust boundaries.
pub fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
            let mut canonical = serde_json::Map::new();
            for (key, child) in entries {
                canonical.insert(key.clone(), canonical_value(child));
            }
            Value::Object(canonical)
        }
        other => other.clone(),
    }
}

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Reject values that cannot survive the Python/Rust/JavaScript wire without
/// numeric coercion. JSON numbers are binary64 on the portable boundary;
/// integral values therefore stay within ±(2^53-1). `serde_json::Value`
/// already excludes NaN and infinity.
pub fn validate_semantic_json(value: &Value) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            for item in items {
                validate_semantic_json(item)?;
            }
        }
        Value::Object(object) => {
            for child in object.values() {
                validate_semantic_json(child)?;
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if value.unsigned_abs() > MAX_SAFE_INTEGER {
                    return Err("integer is outside the portable ±(2^53-1) domain".into());
                }
            } else if let Some(value) = number.as_u64() {
                if value > MAX_SAFE_INTEGER {
                    return Err("integer is outside the portable ±(2^53-1) domain".into());
                }
            } else if let Some(value) = number.as_f64() {
                if !value.is_finite()
                    || (value.fract() == 0.0 && value.abs() > MAX_SAFE_INTEGER as f64)
                {
                    return Err("number is outside the portable binary64 domain".into());
                }
            } else {
                return Err("number is outside the portable binary64 domain".into());
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => out.push_str(&js_number(value)),
        Value::String(value) => out
            .push_str(&serde_json::to_string(value).expect("a JSON string is always serializable")),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
            out.push('{');
            for (index, (key, child)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(
                    &serde_json::to_string(key).expect("a JSON object key is serializable"),
                );
                out.push(':');
                write_canonical(child, out);
            }
            out.push('}');
        }
    }
}

/// ECMAScript/JCS-style finite binary64 rendering. `serde_json` already uses
/// a shortest-round-trip representation; this normalizes the exponent and
/// decimal thresholds where Rust and `JSON.stringify` differ.
fn js_number(number: &serde_json::Number) -> String {
    let value = number.as_f64().unwrap_or(0.0);
    if value == 0.0 {
        return "0".into();
    }
    if value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER as f64 {
        return format!("{value:.0}");
    }
    let raw = number.to_string().to_ascii_lowercase();
    let abs = value.abs();
    if (1e-6..1e21).contains(&abs) {
        return expand_decimal(&raw);
    }
    normalize_exponent(&raw)
}

fn split_number(raw: &str) -> (&str, &str, i32) {
    let (sign, unsigned) = raw
        .strip_prefix('-')
        .map(|value| ("-", value))
        .unwrap_or(("", raw));
    let (mantissa, exponent) = unsigned
        .split_once('e')
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or(0)))
        .unwrap_or((unsigned, 0));
    (sign, mantissa, exponent)
}

fn expand_decimal(raw: &str) -> String {
    let (sign, mantissa, exponent) = split_number(raw);
    if !raw.contains('e') {
        return format!("{sign}{}", mantissa.strip_suffix(".0").unwrap_or(mantissa));
    }
    let before = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits = mantissa.replace('.', "");
    let point = before + exponent;
    let body = if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        format!("{}{}", digits, "0".repeat(point as usize - digits.len()))
    } else {
        let point = point as usize;
        format!("{}.{}", &digits[..point], &digits[point..])
    };
    format!("{sign}{body}")
}

fn normalize_exponent(raw: &str) -> String {
    let (sign, mantissa, exponent) = split_number(raw);
    if raw.contains('e') {
        let exponent = if exponent >= 0 {
            format!("+{exponent}")
        } else {
            exponent.to_string()
        };
        return format!("{sign}{mantissa}e{exponent}");
    }
    // Shortest serializers normally choose exponent notation in this range.
    // This fallback handles a plain decimal deterministically.
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{integer}{fraction}");
    let first = digits.find(|ch| ch != '0').unwrap_or(0);
    let exponent = integer.len() as i32 - first as i32 - 1;
    let significant = digits[first..].trim_end_matches('0');
    let rendered = if significant.len() <= 1 {
        significant.to_string()
    } else {
        format!("{}.{}", &significant[..1], &significant[1..])
    };
    let exponent = if exponent >= 0 {
        format!("+{exponent}")
    } else {
        exponent.to_string()
    };
    format!("{sign}{rendered}e{exponent}")
}

pub fn compact_tools(tools: &[Value]) -> Value {
    let items: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let name = tool.get("name").cloned().unwrap_or(Value::Null);
            let description = match tool.get("description") {
                Some(Value::String(s)) if !s.is_empty() => Value::String(s.clone()),
                Some(Value::Null) | None => Value::String(String::new()),
                Some(other) => Value::String(other.to_string().trim_matches('"').to_string()),
            };
            let parameters = tool
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object","properties":{}}));
            json!({
                "name": name,
                "description": description,
                "parameters": parameters,
            })
        })
        .collect();
    Value::Array(items)
}

pub fn schema_fingerprint(tools: &[Value]) -> String {
    let payload = dumps_canonical(&compact_tools(tools));
    let digest = Sha256::digest(payload.as_bytes());
    hex::encode(digest)
}

pub fn catalog_fingerprint(tools: &[Value]) -> String {
    let mut items: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            json!({
                "name": tool.get("name").cloned().unwrap_or(Value::Null),
                "description": description,
                "parameters": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                "required_permissions": tool.get("required_permissions").cloned().unwrap_or(Value::Null),
                "required_state": tool.get("required_state").cloned().unwrap_or(Value::Null),
                "x-mei-permissions": tool.get("x-mei-permissions").cloned().unwrap_or(Value::Null),
                "x-mei-state": tool.get("x-mei-state").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    // Tool ID is the cross-runtime stable tie-breaker. UTF-8 byte order
    // preserves Unicode scalar order and is reproducible in browsers.
    items.sort_by(|left, right| {
        left.get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .as_bytes()
            .cmp(
                right
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .as_bytes(),
            )
    });
    sha256_bytes(dumps_canonical(&Value::Array(items)).as_bytes())
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
