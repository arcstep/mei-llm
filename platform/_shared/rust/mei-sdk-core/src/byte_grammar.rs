//! UTF-8 byte-prefix grammar for the portable tool-call schema subset.
//!
//! The accepted wire is exactly `[]` or one compact call object with ordered
//! `name` and `arguments` fields. Argument objects may use any property order,
//! while values cover the registered scalar/nullable/scalar-array subset.

use std::collections::{HashMap, HashSet};

use serde_json::{Number, Value};

use crate::protocol::validate_json_value;

#[derive(Clone)]
struct ArgConstraint {
    name: String,
    schema: Value,
    required: bool,
}

#[derive(Clone)]
pub struct ByteGrammar {
    names: Vec<String>,
    args: HashMap<String, Vec<ArgConstraint>>,
}

#[derive(Debug)]
struct StringPrefix {
    value: String,
    used: usize,
    done: bool,
    valid: bool,
}

#[derive(Debug)]
struct ValuePrefix {
    used: usize,
    done: bool,
    value: Option<Value>,
    valid: bool,
}

impl ByteGrammar {
    pub fn compile(tools: &[Value]) -> Self {
        let mut names = Vec::with_capacity(tools.len());
        let mut args = HashMap::new();
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            names.push(name.to_string());
            let required = tool
                .pointer("/parameters/required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<HashSet<_>>();
            let constraints = tool
                .pointer("/parameters/properties")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|properties| properties.iter())
                .map(|(property, schema)| ArgConstraint {
                    name: property.clone(),
                    schema: schema.clone(),
                    required: required.contains(property.as_str()),
                })
                .collect::<Vec<_>>();
            args.insert(name.to_string(), constraints);
        }
        Self { names, args }
    }

    pub fn legal_prefix(&self, bytes: &[u8]) -> bool {
        match std::str::from_utf8(bytes) {
            Ok(text) => self.legal_text_prefix(text, false),
            Err(error) if error.error_len().is_none() => {
                let valid = &bytes[..error.valid_up_to()];
                std::str::from_utf8(valid)
                    .map(|text| self.legal_text_prefix(text, false))
                    .unwrap_or(false)
            }
            Err(_) => false,
        }
    }

    pub fn accepting(&self, bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes)
            .map(|text| self.legal_text_prefix(text, true))
            .unwrap_or(false)
    }

    fn legal_text_prefix(&self, text: &str, must_accept: bool) -> bool {
        if text.is_empty() {
            return !must_accept;
        }
        if !text.starts_with('[') {
            return "[".starts_with(text) && !must_accept;
        }
        if text == "[" {
            return !must_accept;
        }
        let rest = &text[1..];
        if rest.starts_with(']') {
            return rest == "]";
        }
        if !rest.starts_with('{') {
            return "{".starts_with(rest) && !must_accept;
        }
        let object = &rest[1..];
        const NAME_KEY: &str = "\"name\":";
        if !object.starts_with(NAME_KEY) {
            return NAME_KEY.starts_with(object) && !must_accept;
        }
        let after_key = &object[NAME_KEY.len()..];
        let parsed = parse_json_string_prefix(after_key);
        if !parsed.valid {
            return false;
        }
        if !parsed.done {
            return !must_accept
                && self
                    .names
                    .iter()
                    .any(|candidate| candidate.starts_with(&parsed.value));
        }
        if !self
            .names
            .iter()
            .any(|candidate| candidate == &parsed.value)
        {
            return false;
        }
        let after_name = &after_key[parsed.used..];
        const MIDDLE: &str = ",\"arguments\":";
        if !after_name.starts_with(MIDDLE) {
            return MIDDLE.starts_with(after_name) && !must_accept;
        }
        let constraints = self
            .args
            .get(&parsed.value)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        legal_args_prefix(&after_name[MIDDLE.len()..], constraints, must_accept)
    }
}

fn parse_json_string_prefix(text: &str) -> StringPrefix {
    if text.is_empty() {
        return StringPrefix {
            value: String::new(),
            used: 0,
            done: false,
            valid: true,
        };
    }
    if !text.starts_with('"') {
        return StringPrefix {
            value: String::new(),
            used: 0,
            done: false,
            valid: false,
        };
    }
    let bytes = text.as_bytes();
    let mut index = 1usize;
    while index < bytes.len() {
        match bytes[index] {
            0x00..=0x1f => {
                return StringPrefix {
                    value: String::new(),
                    used: 0,
                    done: false,
                    valid: false,
                };
            }
            b'"' => {
                let used = index + 1;
                let Ok(Value::String(value)) = serde_json::from_str::<Value>(&text[..used]) else {
                    return StringPrefix {
                        value: String::new(),
                        used: 0,
                        done: false,
                        valid: false,
                    };
                };
                return StringPrefix {
                    value,
                    used,
                    done: true,
                    valid: true,
                };
            }
            b'\\' => {
                index += 1;
                if index >= bytes.len() {
                    break;
                }
                match bytes[index] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => index += 1,
                    b'u' => {
                        let available = bytes.len().saturating_sub(index + 1).min(4);
                        if bytes[index + 1..index + 1 + available]
                            .iter()
                            .any(|byte| !byte.is_ascii_hexdigit())
                        {
                            return StringPrefix {
                                value: String::new(),
                                used: 0,
                                done: false,
                                valid: false,
                            };
                        }
                        if available < 4 {
                            break;
                        }
                        index += 5;
                    }
                    _ => {
                        return StringPrefix {
                            value: String::new(),
                            used: 0,
                            done: false,
                            valid: false,
                        };
                    }
                }
            }
            _ => index += 1,
        }
    }
    let body = &text[1..index.min(text.len())];
    let complete = format!("\"{}\"", body.trim_end_matches('\\'));
    let value = serde_json::from_str::<String>(&complete)
        .unwrap_or_else(|_| body.split('\\').next().unwrap_or("").to_string());
    StringPrefix {
        value,
        used: text.len(),
        done: false,
        valid: true,
    }
}

fn legal_args_prefix(text: &str, constraints: &[ArgConstraint], must_accept: bool) -> bool {
    if text.is_empty() {
        return !must_accept;
    }
    if !text.starts_with('{') {
        return "{".starts_with(text) && !must_accept;
    }
    let body = &text[1..];
    let by_name = constraints
        .iter()
        .map(|constraint| (constraint.name.as_str(), constraint))
        .collect::<HashMap<_, _>>();
    let required = constraints
        .iter()
        .filter(|constraint| constraint.required)
        .map(|constraint| constraint.name.as_str())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::<String>::new();
    let mut index = 0usize;
    if body.is_empty() {
        return !must_accept;
    }
    if body.starts_with('}') {
        return required.is_empty() && close_call(&body[1..], must_accept);
    }
    while index < body.len() {
        let parsed = parse_json_string_prefix(&body[index..]);
        if !parsed.valid {
            return false;
        }
        if !parsed.done {
            return !must_accept
                && by_name
                    .keys()
                    .any(|name| !seen.contains(*name) && name.starts_with(&parsed.value));
        }
        let Some(constraint) = by_name.get(parsed.value.as_str()) else {
            return false;
        };
        if !seen.insert(parsed.value) {
            return false;
        }
        index += parsed.used;
        if index >= body.len() {
            return !must_accept;
        }
        if body.as_bytes()[index] != b':' {
            return ":".starts_with(&body[index..]) && !must_accept;
        }
        index += 1;
        let parsed_value = parse_value_prefix(&body[index..], &constraint.schema);
        if !parsed_value.valid {
            return false;
        }
        index += parsed_value.used;
        if !parsed_value.done {
            return !must_accept;
        }
        if index >= body.len() {
            return !must_accept;
        }
        let Some(value) = parsed_value.value else {
            return false;
        };
        if validate_json_value(&value, &constraint.schema).is_err() {
            return false;
        }
        match body.as_bytes()[index] {
            b',' => {
                index += 1;
                if index >= body.len() {
                    return !must_accept;
                }
            }
            b'}' => {
                if required.iter().any(|name| !seen.contains(*name)) {
                    return false;
                }
                return close_call(&body[index + 1..], must_accept);
            }
            _ => return false,
        }
    }
    !must_accept
}

fn close_call(rest: &str, must_accept: bool) -> bool {
    if must_accept {
        rest == "}]"
    } else {
        "}]".starts_with(rest)
    }
}

fn parse_value_prefix(text: &str, schema: &Value) -> ValuePrefix {
    if let Some(value) = schema.get("const") {
        return literal_prefix(text, value);
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return choose_prefix(values.iter().map(|value| literal_prefix(text, value)));
    }
    let mut candidates = Vec::new();
    for kind in schema_types(schema) {
        candidates.push(match kind.as_str() {
            "null" => literal_prefix(text, &Value::Null),
            "boolean" => choose_prefix(
                [Value::Bool(true), Value::Bool(false)]
                    .iter()
                    .map(|value| literal_prefix(text, value)),
            ),
            "integer" => number_prefix(text, true),
            "number" => number_prefix(text, false),
            "string" => {
                let parsed = parse_json_string_prefix(text);
                ValuePrefix {
                    used: parsed.used,
                    done: parsed.done,
                    value: parsed.done.then_some(Value::String(parsed.value)),
                    valid: parsed.valid,
                }
            }
            "array" => array_prefix(text, schema),
            _ => ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            },
        });
    }
    choose_prefix(candidates)
}

fn schema_types(schema: &Value) -> Vec<String> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.clone()],
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => schema
            .get("const")
            .or_else(|| {
                schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
            })
            .map(value_type)
            .into_iter()
            .map(str::to_string)
            .collect(),
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn literal_prefix(text: &str, value: &Value) -> ValuePrefix {
    let literal = serde_json::to_string(value).unwrap_or_default();
    if text.starts_with(&literal) {
        ValuePrefix {
            used: literal.len(),
            done: true,
            value: Some(value.clone()),
            valid: true,
        }
    } else if literal.starts_with(text) {
        ValuePrefix {
            used: text.len(),
            done: false,
            value: None,
            valid: true,
        }
    } else {
        ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: false,
        }
    }
}

fn choose_prefix(rows: impl IntoIterator<Item = ValuePrefix>) -> ValuePrefix {
    let rows = rows.into_iter().filter(|row| row.valid).collect::<Vec<_>>();
    rows.iter()
        .filter(|row| row.done)
        .max_by_key(|row| row.used)
        .or_else(|| rows.iter().max_by_key(|row| row.used))
        .map(|row| ValuePrefix {
            used: row.used,
            done: row.done,
            value: row.value.clone(),
            valid: row.valid,
        })
        .unwrap_or(ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: false,
        })
}

fn number_prefix(text: &str, integer: bool) -> ValuePrefix {
    if text.is_empty() {
        return ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: true,
        };
    }
    let bytes = text.as_bytes();
    let mut index = 0usize;
    if bytes[index] == b'-' {
        index += 1;
        if index == bytes.len() {
            return ValuePrefix {
                used: index,
                done: false,
                value: None,
                valid: true,
            };
        }
    }
    if index >= bytes.len() || !bytes[index].is_ascii_digit() {
        return ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: false,
        };
    }
    if bytes[index] == b'0' {
        index += 1;
    } else {
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    }
    if !integer && index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        if index == bytes.len() {
            return ValuePrefix {
                used: index,
                done: false,
                value: None,
                valid: true,
            };
        }
        if !bytes[index].is_ascii_digit() {
            return ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            };
        }
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    }
    if !integer && index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
        index += 1;
        if index == bytes.len() {
            return ValuePrefix {
                used: index,
                done: false,
                value: None,
                valid: true,
            };
        }
        if matches!(bytes[index], b'+' | b'-') {
            index += 1;
            if index == bytes.len() {
                return ValuePrefix {
                    used: index,
                    done: false,
                    value: None,
                    valid: true,
                };
            }
        }
        if !bytes[index].is_ascii_digit() {
            return ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            };
        }
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    }
    let token = &text[..index];
    let value = if integer {
        token
            .parse::<i64>()
            .ok()
            .map(|number| Value::Number(Number::from(number)))
    } else {
        token
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .and_then(Number::from_f64)
            .map(Value::Number)
    };
    ValuePrefix {
        used: index,
        done: value.is_some(),
        value,
        valid: true,
    }
}

fn array_prefix(text: &str, schema: &Value) -> ValuePrefix {
    if text.is_empty() {
        return ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: true,
        };
    }
    if !text.starts_with('[') {
        return ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: false,
        };
    }
    let Some(item_schema) = schema.get("items") else {
        return ValuePrefix {
            used: 0,
            done: false,
            value: None,
            valid: false,
        };
    };
    let mut index = 1usize;
    let mut values = Vec::new();
    if index >= text.len() {
        return ValuePrefix {
            used: index,
            done: false,
            value: None,
            valid: true,
        };
    }
    if text.as_bytes()[index] == b']' {
        let value = Value::Array(values);
        return ValuePrefix {
            used: index + 1,
            done: true,
            valid: validate_json_value(&value, schema).is_ok(),
            value: Some(value),
        };
    }
    loop {
        let item = parse_value_prefix(&text[index..], item_schema);
        if !item.valid {
            return ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            };
        }
        index += item.used;
        if !item.done || index >= text.len() {
            return ValuePrefix {
                used: index,
                done: false,
                value: None,
                valid: true,
            };
        }
        let Some(value) = item.value else {
            return ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            };
        };
        if validate_json_value(&value, item_schema).is_err() {
            return ValuePrefix {
                used: 0,
                done: false,
                value: None,
                valid: false,
            };
        }
        values.push(value);
        match text.as_bytes()[index] {
            b',' => {
                index += 1;
                if index >= text.len() {
                    return ValuePrefix {
                        used: index,
                        done: false,
                        value: None,
                        valid: true,
                    };
                }
            }
            b']' => {
                let value = Value::Array(values);
                return ValuePrefix {
                    used: index + 1,
                    done: true,
                    valid: validate_json_value(&value, schema).is_ok(),
                    value: Some(value),
                };
            }
            _ => {
                return ValuePrefix {
                    used: 0,
                    done: false,
                    value: None,
                    valid: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ByteGrammar;

    fn grammar() -> ByteGrammar {
        ByteGrammar::compile(&[json!({
            "name":"calendar.add",
            "description":"add",
            "parameters":{
                "type":"object",
                "additionalProperties":false,
                "required":["title","count","enabled","ratio","tags"],
                "properties":{
                    "title":{"type":"string","minLength":1,"maxLength":8},
                    "count":{"type":"integer","minimum":1,"maximum":9},
                    "enabled":{"type":"boolean"},
                    "ratio":{"type":"number","multipleOf":0.5},
                    "tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":2},
                    "mode":{"type":"string","enum":["快","慢"]},
                    "note":{"type":["string","null"]}
                }
            }
        })])
    }

    #[test]
    fn full_subset_accepts_unicode_numbers_nullable_and_scalar_array() {
        let grammar = grammar();
        let text = r#"[{"name":"calendar.add","arguments":{"title":"会议","count":3,"enabled":true,"ratio":1.5,"tags":["甲","乙"],"mode":"快","note":null}}]"#.as_bytes();
        for end in 0..=text.len() {
            assert!(
                grammar.legal_prefix(&text[..end]),
                "illegal byte prefix at {end}"
            );
        }
        assert!(grammar.accepting(text));
    }

    #[test]
    fn subset_rejects_bounds_duplicate_keys_and_missing_required() {
        let grammar = grammar();
        assert!(!grammar.accepting(r#"[{"name":"calendar.add","arguments":{"title":"会议","count":30,"enabled":true,"ratio":1.5,"tags":["甲"]}}]"#.as_bytes()));
        assert!(!grammar.accepting(r#"[{"name":"calendar.add","arguments":{"title":"会议","title":"重复","count":3,"enabled":true,"ratio":1.5,"tags":["甲"]}}]"#.as_bytes()));
        assert!(!grammar
            .accepting(r#"[{"name":"calendar.add","arguments":{"title":"会议"}}]"#.as_bytes()));
        assert!(grammar.accepting(b"[]"));
    }
}
