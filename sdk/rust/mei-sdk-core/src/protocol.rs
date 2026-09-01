use regex::Regex;
use serde_json::{json, Value};

use crate::canonical::{
    compact_tools, dumps_canonical, schema_fingerprint, validate_semantic_json,
};
use crate::version::{
    forbidden_markers, max_calls, max_selected_tools, protocol_id, serializer_id, task_contract,
};

const SCALAR_TYPES: [&str; 5] = ["string", "boolean", "integer", "number", "null"];
const SUPPORTED_FORMATS: [&str; 8] = [
    "date",
    "date-time",
    "time",
    "email",
    "uuid",
    "uri",
    "ipv4",
    "ipv6",
];
const COMBINATORS: [&str; 17] = [
    "$ref",
    "$dynamicRef",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "patternProperties",
    "propertyNames",
    "prefixItems",
    "contains",
    "unevaluatedItems",
    "unevaluatedProperties",
    "additionalItems",
];

/// Validate the deterministic tool-schema subset before a catalog is registered.
/// Nested objects, object arrays, recursion, and JSON-Schema combinators fail closed.
pub fn validate_tool_schema(tool: &Value) -> Result<(), String> {
    ensure_keys(
        tool,
        &[
            "name",
            "description",
            "parameters",
            "required_permissions",
            "required_state",
            "x-mei-permissions",
            "x-mei-state",
        ],
        "tool",
    )?;
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "tool.name is required".to_string())?;
    let schema = tool
        .get("parameters")
        .ok_or_else(|| format!("{name}.parameters is required"))?;
    if tool
        .get("description")
        .is_some_and(|value| !value.is_string())
    {
        return Err(format!("{name}.description must be a string"));
    }
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(format!("{name}.parameters must be a root object"));
    }
    reject_combinators(schema, name)?;
    ensure_keys(
        schema,
        &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "$schema",
            "$id",
            "title",
            "description",
            "examples",
            "$comment",
        ],
        name,
    )?;
    if schema
        .get("additionalProperties")
        .map(|value| value != &json!(false))
        .unwrap_or(false)
    {
        return Err(format!("{name}.additionalProperties must be false"));
    }
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{name}.parameters.properties must be an object"))?;
    let required = schema
        .get("required")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| format!("{name}.required must be a string array"))
        })
        .transpose()?
        .cloned()
        .unwrap_or_default();
    let mut required_names = std::collections::HashSet::new();
    for value in required {
        let property = value
            .as_str()
            .ok_or_else(|| format!("{name}.required must be a string array"))?;
        if !properties.contains_key(property) || !required_names.insert(property.to_string()) {
            return Err(format!(
                "{name}.required contains unknown or duplicate {property}"
            ));
        }
    }
    for (property, spec) in properties {
        reject_combinators(spec, &format!("{name}.{property}"))?;
        let types = schema_types(spec, &format!("{name}.{property}"))?;
        if types.iter().any(|kind| kind == "array") {
            if types.len() != 1 {
                return Err(format!("{name}.{property} nullable arrays are unsupported"));
            }
            ensure_keys(
                spec,
                &[
                    "type",
                    "items",
                    "minItems",
                    "maxItems",
                    "enum",
                    "const",
                    "default",
                    "title",
                    "description",
                    "examples",
                    "$comment",
                ],
                &format!("{name}.{property}"),
            )?;
            let item = spec
                .get("items")
                .ok_or_else(|| format!("{name}.{property}.items is required"))?;
            reject_combinators(item, &format!("{name}.{property}.items"))?;
            validate_scalar_schema(item, &format!("{name}.{property}.items"))?;
            for key in ["minItems", "maxItems"] {
                if let Some(value) = spec.get(key) {
                    if value.as_u64().is_none() {
                        return Err(format!(
                            "{name}.{property}.{key} must be non-negative integer"
                        ));
                    }
                }
            }
            if spec.get("minItems").and_then(Value::as_u64).unwrap_or(0)
                > spec
                    .get("maxItems")
                    .and_then(Value::as_u64)
                    .unwrap_or(u64::MAX)
            {
                return Err(format!("{name}.{property}.minItems exceeds maxItems"));
            }
        } else {
            validate_scalar_schema(spec, &format!("{name}.{property}"))?;
        }
        for key in ["const", "default"] {
            if let Some(value) = spec.get(key) {
                validate_json_value(value, spec).map_err(|issue| {
                    format!("{name}.{property}.{key} violates its schema: {issue}")
                })?;
            }
        }
        if let Some(values) = spec.get("enum") {
            let list = values
                .as_array()
                .filter(|values| !values.is_empty())
                .ok_or_else(|| format!("{name}.{property}.enum must be non-empty"))?;
            for value in list {
                validate_json_value(value, spec).map_err(|issue| {
                    format!("{name}.{property}.enum violates its schema: {issue}")
                })?;
            }
        }
    }
    for key in ["required_permissions", "x-mei-permissions"] {
        if let Some(value) = tool.get(key) {
            if value
                .as_array()
                .map(|items| items.iter().all(Value::is_string))
                != Some(true)
            {
                return Err(format!("{name}.{key} must be a string array"));
            }
        }
    }
    for key in ["required_state", "x-mei-state"] {
        if tool.get(key).is_some_and(|value| !value.is_object()) {
            return Err(format!("{name}.{key} must be an object"));
        }
    }
    Ok(())
}

fn reject_combinators(value: &Value, path: &str) -> Result<(), String> {
    for key in COMBINATORS {
        if value.get(key).is_some() {
            return Err(format!("{path}.{key} is unsupported"));
        }
    }
    Ok(())
}

fn ensure_keys(value: &Value, allowed: &[&str], path: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{path} must be an object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("{path}.{key} is outside the portable subset"));
    }
    Ok(())
}

fn schema_types(schema: &Value, path: &str) -> Result<Vec<String>, String> {
    let mut types = match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.clone()],
        Some(Value::Array(values)) if !values.is_empty() => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("{path}.type must contain strings"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None if schema.get("const").is_some() || schema.get("enum").is_some() => {
            let sample = schema
                .get("const")
                .or_else(|| {
                    schema
                        .get("enum")
                        .and_then(Value::as_array)
                        .and_then(|items| items.first())
                })
                .ok_or_else(|| format!("{path}.enum must be non-empty"))?;
            vec![value_type(sample)
                .ok_or_else(|| format!("{path} literal type is unsupported"))?
                .to_string()]
        }
        _ => return Err(format!("{path}.type is required")),
    };
    let mut seen = std::collections::HashSet::new();
    types.retain(|kind| seen.insert(kind.clone()));
    if types.len() > 2
        || (types.len() == 2 && !types.iter().any(|kind| kind == "null"))
        || types
            .iter()
            .any(|kind| !SCALAR_TYPES.contains(&kind.as_str()) && kind != "array")
    {
        return Err(format!("{path}.type is outside the portable subset"));
    }
    Ok(types)
}

fn value_type(value: &Value) -> Option<&'static str> {
    match value {
        Value::Null => Some("null"),
        Value::Bool(_) => Some("boolean"),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some("integer"),
        Value::Number(_) => Some("number"),
        Value::String(_) => Some("string"),
        Value::Array(_) => Some("array"),
        Value::Object(_) => None,
    }
}

fn validate_scalar_schema(schema: &Value, path: &str) -> Result<(), String> {
    ensure_keys(
        schema,
        &[
            "type",
            "enum",
            "const",
            "default",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "title",
            "description",
            "examples",
            "$comment",
        ],
        path,
    )?;
    let types = schema_types(schema, path)?;
    if types.iter().any(|kind| kind == "array") {
        return Err(format!("{path} nested arrays are unsupported"));
    }
    if let Some(pattern) = schema.get("pattern") {
        if !types.iter().any(|kind| kind == "string") {
            return Err(format!("{path}.pattern requires string type"));
        }
        validate_portable_pattern(
            pattern
                .as_str()
                .ok_or_else(|| format!("{path}.pattern must be a string"))?,
            path,
        )?;
    }
    if let Some(format) = schema.get("format") {
        let format = format
            .as_str()
            .ok_or_else(|| format!("{path}.format must be a string"))?;
        if !types.iter().any(|kind| kind == "string") || !SUPPORTED_FORMATS.contains(&format) {
            return Err(format!("{path}.format {format} is unsupported"));
        }
    }
    for key in ["minLength", "maxLength"] {
        if let Some(value) = schema.get(key) {
            if !types.iter().any(|kind| kind == "string") || value.as_u64().is_none() {
                return Err(format!("{path}.{key} is invalid"));
            }
        }
    }
    if schema.get("minLength").and_then(Value::as_u64).unwrap_or(0)
        > schema
            .get("maxLength")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX)
    {
        return Err(format!("{path}.minLength exceeds maxLength"));
    }
    for key in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ] {
        if let Some(value) = schema.get(key) {
            let number = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| format!("{path}.{key} must be finite number"))?;
            if !types
                .iter()
                .any(|kind| matches!(kind.as_str(), "integer" | "number"))
                || (key == "multipleOf" && number <= 0.0)
            {
                return Err(format!("{path}.{key} is invalid"));
            }
        }
    }
    validate_numeric_interval(schema, &types, path)?;
    Ok(())
}

fn validate_numeric_interval(schema: &Value, types: &[String], path: &str) -> Result<(), String> {
    if !types
        .iter()
        .any(|kind| matches!(kind.as_str(), "integer" | "number"))
    {
        return Ok(());
    }
    let lower = [
        (schema.get("minimum").and_then(Value::as_f64), false),
        (schema.get("exclusiveMinimum").and_then(Value::as_f64), true),
    ]
    .into_iter()
    .filter_map(|(value, exclusive)| value.map(|value| (value, exclusive)))
    .max_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let upper = [
        (schema.get("maximum").and_then(Value::as_f64), false),
        (schema.get("exclusiveMaximum").and_then(Value::as_f64), true),
    ]
    .into_iter()
    .filter_map(|(value, exclusive)| value.map(|value| (value, exclusive)))
    .min_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| right.1.cmp(&left.1))
    });
    if let (Some((low, low_exclusive)), Some((high, high_exclusive))) = (lower, upper) {
        if low > high || (low == high && (low_exclusive || high_exclusive)) {
            return Err(format!("{path} numeric interval is empty"));
        }
        if types.iter().any(|kind| kind == "integer") {
            let first = if low_exclusive {
                low.floor() + 1.0
            } else {
                low.ceil()
            };
            let last = if high_exclusive {
                high.ceil() - 1.0
            } else {
                high.floor()
            };
            if first > last {
                return Err(format!("{path} integer interval is empty"));
            }
        }
    }
    Ok(())
}

fn type_matches(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "integer" => value
            .as_number()
            .map(|number| number.is_i64() || number.is_u64())
            .unwrap_or(false),
        "number" => value.as_f64().map(f64::is_finite).unwrap_or(false),
        "string" => value.is_string(),
        "array" => value.is_array(),
        _ => false,
    }
}

fn valid_format(value: &str, format: &str) -> bool {
    match format {
        "date" => {
            let parts = value
                .split('-')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>();
            let Ok(parts) = parts else { return false };
            if parts.len() != 3 || value.len() != 10 {
                return false;
            }
            let (year, month, day) = (parts[0], parts[1], parts[2]);
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            let max_day = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 if leap => 29,
                2 => 28,
                _ => return false,
            };
            year > 0 && day >= 1 && day <= max_day
        }
        "time" => Regex::new(
            r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]{1,9})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$",
        )
        .unwrap()
        .is_match(value),
        "date-time" => value
            .split_once('T')
            .map(|(date, time)| valid_format(date, "date") && valid_format(time, "time"))
            .unwrap_or(false),
        "email" => Regex::new(
            r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$",
        )
            .unwrap()
            .is_match(value),
        "uuid" => Regex::new(
            r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )
        .unwrap()
        .is_match(value),
        "uri" => {
            let syntax = Regex::new(
                r#"^[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$"#,
            )
            .unwrap()
            .is_match(value);
            syntax
                && value.char_indices().all(|(index, character)| {
                    character != '%'
                        || value
                            .get(index + 1..index + 3)
                            .is_some_and(|pair| pair.bytes().all(|byte| byte.is_ascii_hexdigit()))
                })
        }
        "ipv4" => value.parse::<std::net::Ipv4Addr>().is_ok(),
        "ipv6" => value.parse::<std::net::Ipv6Addr>().is_ok(),
        _ => false,
    }
}

fn validate_portable_pattern(pattern: &str, path: &str) -> Result<(), String> {
    let chars = pattern.chars().collect::<Vec<_>>();
    let mut escaped = false;
    let mut in_class = false;
    for (index, character) in chars.iter().copied().enumerate() {
        if escaped {
            // Alphanumeric escapes have different semantics across Python,
            // Rust regex, and ECMAScript (classes, properties, boundaries,
            // backreferences, and octal/Unicode escapes).
            if character.is_ascii_alphanumeric() {
                return Err(format!(
                    "{path}.pattern uses non-portable escape \\{character}"
                ));
            }
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == '(' && chars.get(index + 1) == Some(&'?') {
            return Err(format!("{path}.pattern uses non-portable group extension"));
        }
        if in_class
            && ((character == '&' && chars.get(index + 1) == Some(&'&'))
                || (character == '-' && chars.get(index + 1) == Some(&'-'))
                || character == '[')
        {
            return Err(format!(
                "{path}.pattern uses non-portable character class syntax"
            ));
        }
        if character == '[' {
            in_class = true;
        } else if character == ']' {
            in_class = false;
        }
    }
    if escaped {
        return Err(format!("{path}.pattern ends with an escape"));
    }
    Regex::new(pattern).map_err(|error| format!("{path}.pattern: {error}"))?;
    Ok(())
}

pub fn validate_json_value(value: &Value, schema: &Value) -> Result<(), String> {
    let types = schema_types(schema, "$")?;
    if !types.iter().any(|kind| type_matches(value, kind)) {
        return Err(format!("type mismatch: expected {types:?}"));
    }
    if schema
        .get("const")
        .map(|expected| expected != value)
        .unwrap_or(false)
    {
        return Err("value does not equal const".into());
    }
    if let Some(values) = schema.get("enum") {
        if !values
            .as_array()
            .map(|values| values.contains(value))
            .unwrap_or(false)
        {
            return Err("value is not in enum".into());
        }
    }
    if value.is_null() {
        return Ok(());
    }
    if let Some(text) = value.as_str() {
        let length = text.chars().count() as u64;
        if length < schema.get("minLength").and_then(Value::as_u64).unwrap_or(0) {
            return Err("string is shorter than minLength".into());
        }
        if length
            > schema
                .get("maxLength")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX)
        {
            return Err("string exceeds maxLength".into());
        }
        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            validate_portable_pattern(pattern, "$")?;
            if !Regex::new(pattern)
                .map_err(|error| error.to_string())?
                .is_match(text)
            {
                return Err("string does not match pattern".into());
            }
        }
        if let Some(format) = schema.get("format").and_then(Value::as_str) {
            if !valid_format(text, format) {
                return Err(format!("invalid {format}"));
            }
        }
    }
    if let Some(number) = value.as_f64() {
        for (key, valid) in [
            (
                "minimum",
                schema
                    .get("minimum")
                    .and_then(Value::as_f64)
                    .map(|bound| number >= bound),
            ),
            (
                "maximum",
                schema
                    .get("maximum")
                    .and_then(Value::as_f64)
                    .map(|bound| number <= bound),
            ),
            (
                "exclusiveMinimum",
                schema
                    .get("exclusiveMinimum")
                    .and_then(Value::as_f64)
                    .map(|bound| number > bound),
            ),
            (
                "exclusiveMaximum",
                schema
                    .get("exclusiveMaximum")
                    .and_then(Value::as_f64)
                    .map(|bound| number < bound),
            ),
        ] {
            if valid == Some(false) {
                return Err(format!("number violates {key}"));
            }
        }
        if let Some(divisor) = schema.get("multipleOf").and_then(Value::as_f64) {
            let quotient = number / divisor;
            if (quotient - quotient.round()).abs() > 1e-9 * quotient.abs().max(1.0) {
                return Err("number is not a multiple".into());
            }
        }
    }
    if let Some(values) = value.as_array() {
        if (values.len() as u64) < schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) {
            return Err("array is shorter than minItems".into());
        }
        if values.len() as u64
            > schema
                .get("maxItems")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX)
        {
            return Err("array exceeds maxItems".into());
        }
        let item_schema = schema
            .get("items")
            .ok_or_else(|| "array items schema is required".to_string())?;
        for item in values {
            validate_json_value(item, item_schema)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ParsedCall {
    pub ok: bool,
    pub function_calls: Vec<Value>,
    pub error: Option<String>,
    pub refuse: bool,
}

impl ParsedCall {
    pub fn to_value(&self) -> Value {
        json!({
            "ok": self.ok,
            "function_calls": self.function_calls,
            "error": self.error,
            "refuse": self.refuse,
        })
    }
}

pub fn leak_markers(text: &str) -> Vec<String> {
    forbidden_markers()
        .into_iter()
        .filter(|marker| text.contains(marker.as_str()))
        .collect()
}

pub fn request_leaks(request: &Value) -> Vec<String> {
    let chunks = [
        request.get("query").and_then(Value::as_str).unwrap_or(""),
        request
            .get("system_facts")
            .and_then(Value::as_str)
            .unwrap_or(""),
        request
            .get("selected_entity")
            .and_then(Value::as_str)
            .unwrap_or(""),
        request
            .get("candidate_text")
            .and_then(Value::as_str)
            .unwrap_or(""),
        &dumps_canonical(request.get("history").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("prior_tool_results").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("catalog").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("oracle_tools").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("permissions").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("context").unwrap_or(&json!({}))),
        &dumps_canonical(request.get("evidence").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("tool_results").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("state").unwrap_or(&json!({}))),
        &dumps_canonical(
            request
                .get("mw")
                .or_else(|| request.get("mw_disposition"))
                .unwrap_or(&json!({})),
        ),
        &dumps_canonical(request.get("confidence").unwrap_or(&Value::Null)),
    ];
    let blob = chunks.join("\n");
    leak_markers(&blob)
}

pub fn parse_v2_text(text: &str) -> ParsedCall {
    let raw = text.trim();
    if raw.is_empty() {
        return ParsedCall {
            ok: false,
            function_calls: vec![],
            error: Some("empty".into()),
            refuse: true,
        };
    }
    let obj: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => {
            return ParsedCall {
                ok: false,
                function_calls: vec![],
                error: Some("json".into()),
                refuse: true,
            };
        }
    };
    if obj == json!([]) {
        return ParsedCall {
            ok: true,
            function_calls: vec![],
            error: None,
            refuse: true,
        };
    }
    let Some(list) = obj.as_array() else {
        return ParsedCall {
            ok: false,
            function_calls: vec![],
            error: Some("illegal_shape".into()),
            refuse: true,
        };
    };
    if list.len() > max_calls() || list.len() != 1 || !list[0].is_object() {
        return ParsedCall {
            ok: false,
            function_calls: vec![],
            error: Some("illegal_shape".into()),
            refuse: true,
        };
    }
    let item = &list[0];
    let name_ok = item.get("name").and_then(Value::as_str).is_some();
    let args_ok = item.get("arguments").map(Value::is_object).unwrap_or(false);
    if !name_ok || !args_ok {
        return ParsedCall {
            ok: false,
            function_calls: vec![],
            error: Some("illegal_item".into()),
            refuse: true,
        };
    }
    ParsedCall {
        ok: true,
        function_calls: vec![json!({
            "name": item.get("name").and_then(Value::as_str).unwrap(),
            "arguments": item.get("arguments").cloned().unwrap(),
        })],
        error: None,
        refuse: false,
    }
}

fn validation_gate(name: &str, ok: bool, detail: Option<&str>) -> Value {
    json!({"gate": name, "ok": ok, "detail": detail})
}

fn blocked(gates: Vec<Value>, error: &str, detail: Option<&str>) -> Value {
    json!({
        "ok": false,
        "refuse": true,
        "execution": "refuse",
        "function_calls": [],
        "error": error,
        "detail": detail,
        "gates": gates,
        "provenance": {},
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
    })
}

fn string_set(value: Option<&Value>) -> std::collections::HashSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn explicit_evidence<'a>(request: &'a Value) -> impl Iterator<Item = &'a Value> {
    request
        .get("evidence")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(
            request
                .pointer("/context/facts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
}

/// Deterministic validation gates before confidence. Learned heads may only
/// further restrict this result; they cannot restore a rejected call.
pub fn validate_generated_call(text: &str, tools: &[Value], request: &Value) -> Value {
    validate_generated_call_with_trusted_results(text, tools, request, &[])
}

pub(crate) fn validate_generated_call_with_trusted_results(
    text: &str,
    tools: &[Value],
    request: &Value,
    trusted_tool_results: &[Value],
) -> Value {
    let parsed = parse_v2_text(text);
    let grammar_ok = parsed.ok;
    let mut gates = vec![validation_gate(
        "grammar",
        grammar_ok,
        parsed.error.as_deref(),
    )];
    if !grammar_ok {
        return blocked(gates, "grammar_violation", parsed.error.as_deref());
    }
    if parsed.refuse {
        gates.push(validation_gate("schema", true, None));
        return json!({
            "ok": true, "refuse": true, "execution": "refuse", "function_calls": [],
            "error": null, "gates": gates, "provenance": {},
            "unsupported_accepted": 0, "unprovenanced_argument_accepted": 0,
        });
    }
    let call = &parsed.function_calls[0];
    if let Err(issue) = validate_semantic_json(call) {
        gates.push(validation_gate("schema", false, Some("numeric_domain")));
        return blocked(gates, "schema_validation", Some(&issue));
    }
    let name = call.get("name").and_then(Value::as_str).unwrap_or("");
    let Some(tool) = tools
        .iter()
        .find(|tool| tool.get("name").and_then(Value::as_str) == Some(name))
    else {
        gates.push(validation_gate("schema", false, Some("unsupported_tool")));
        return blocked(gates, "schema_validation", Some("unsupported_tool"));
    };
    let Some(arguments) = call.get("arguments").and_then(Value::as_object) else {
        gates.push(validation_gate("schema", false, Some("arguments")));
        return blocked(gates, "schema_validation", Some("arguments must be object"));
    };
    let properties = tool
        .pointer("/parameters/properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required = string_set(tool.pointer("/parameters/required"));
    let schema_issue = arguments
        .iter()
        .find_map(|(argument, value)| match properties.get(argument) {
            None => Some(format!("unsupported argument {argument}")),
            Some(schema) => validate_json_value(value, schema)
                .err()
                .map(|issue| format!("{argument}: {issue}")),
        })
        .or_else(|| {
            required
                .iter()
                .find(|argument| !arguments.contains_key(argument.as_str()))
                .map(|argument| format!("missing required {argument}"))
        });
    gates.push(validation_gate(
        "schema",
        schema_issue.is_none(),
        schema_issue.as_deref(),
    ));
    if let Some(issue) = schema_issue {
        return blocked(gates, "schema_validation", Some(&issue));
    }

    let mut provenance = serde_json::Map::new();
    let mut missing = Vec::new();
    for (argument, value) in arguments {
        let schema = &properties[argument];
        if schema.get("const") == Some(value) {
            provenance.insert(
                argument.clone(),
                json!({"source":"schema_const","verified":true,"canonical_value":value}),
            );
            continue;
        }
        let match_evidence = explicit_evidence(request).find(|evidence| {
            evidence.get("verified").and_then(Value::as_bool) == Some(true)
                && evidence
                    .get("source")
                    .and_then(Value::as_str)
                    .map(|source| !source.is_empty())
                    .unwrap_or(false)
                && evidence.get("value") == Some(value)
                && ((evidence.get("tool").and_then(Value::as_str) == Some(name)
                    && evidence.get("argument").and_then(Value::as_str) == Some(argument.as_str()))
                    || (evidence.get("subject").and_then(Value::as_str) == Some(name)
                        && evidence.get("predicate").and_then(Value::as_str)
                            == Some(argument.as_str())))
        });
        if let Some(evidence) = match_evidence {
            provenance.insert(argument.clone(), evidence.clone());
        } else if let Some(result) = trusted_tool_results.iter().find(|result| {
            result.get("status").and_then(Value::as_str) == Some("ok")
                && result
                    .pointer("/provenance/verified")
                    .and_then(Value::as_bool)
                    == Some(true)
                && result
                    .pointer("/provenance/source")
                    .and_then(Value::as_str)
                    .is_some_and(|source| !source.is_empty())
                && result
                    .get("payload")
                    .and_then(Value::as_object)
                    .and_then(|payload| payload.get(argument))
                    == Some(value)
        }) {
            provenance.insert(
                argument.clone(),
                json!({
                    "source":"verified_tool_result",
                    "locator":result.get("call_id").cloned().unwrap_or(Value::Null),
                    "canonical_value":value,
                    "verified":true,
                }),
            );
        } else {
            missing.push(argument.clone());
        }
    }
    gates.push(json!({
        "gate":"provenance", "ok":missing.is_empty(),
        "detail":if missing.is_empty() { Value::Null } else { json!("missing") },
        "missing":missing,
    }));
    if !missing.is_empty() {
        return blocked(gates, "provenance_missing", Some(&missing.join(",")));
    }

    let permissions = request.get("permissions").unwrap_or(&Value::Null);
    let (scopes, denies, allowed_tools, denied_tools, permissions_valid) = if permissions.is_array()
    {
        (
            string_set(Some(permissions)),
            std::collections::HashSet::new(),
            None,
            std::collections::HashSet::new(),
            true,
        )
    } else if let Some(object) = permissions.as_object() {
        (
            string_set(object.get("scopes").or_else(|| object.get("grants"))),
            string_set(object.get("denies")),
            object
                .get("allowed_tools")
                .map(|value| string_set(Some(value))),
            string_set(object.get("denied_tools")),
            true,
        )
    } else if permissions.is_null() {
        (
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
            None,
            std::collections::HashSet::new(),
            true,
        )
    } else {
        (
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
            None,
            std::collections::HashSet::new(),
            false,
        )
    };
    let required_permissions = string_set(
        tool.get("required_permissions")
            .or_else(|| tool.get("x-mei-permissions")),
    );
    let permission_error = if !permissions_valid {
        Some("permissions_invalid")
    } else if denied_tools.contains(name) || !required_permissions.is_disjoint(&denies) {
        Some("permission_denied")
    } else if allowed_tools
        .as_ref()
        .map(|allowed| !allowed.contains(name))
        .unwrap_or(false)
    {
        Some("permission_not_granted")
    } else if !required_permissions.is_subset(&scopes) {
        Some("permission_scope_missing")
    } else {
        None
    };
    gates.push(validation_gate(
        "permission",
        permission_error.is_none(),
        permission_error,
    ));
    if let Some(error) = permission_error {
        return blocked(gates, error, None);
    }

    let state = request.get("state").and_then(Value::as_object);
    let required_state = tool
        .get("required_state")
        .or_else(|| tool.get("x-mei-state"))
        .and_then(Value::as_object);
    let state_error = if request.get("state").is_some() && state.is_none() {
        Some("state_invalid".to_string())
    } else if state
        .and_then(|value| value.get("invalid"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        Some("state_invalid".to_string())
    } else if state
        .and_then(|value| value.get("conflict"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        Some("state_conflict".to_string())
    } else {
        required_state.and_then(|required| {
            required.iter().find_map(|(key, expected)| {
                if state.and_then(|actual| actual.get(key)) == Some(expected) {
                    None
                } else {
                    Some(format!("state_requirement_missing:{key}"))
                }
            })
        })
    };
    gates.push(validation_gate(
        "state",
        state_error.is_none(),
        state_error.as_deref(),
    ));
    if let Some(error) = state_error {
        return blocked(gates, &error, None);
    }

    let wire_v2 =
        request.get("wire_version").and_then(Value::as_str) == Some("mei-runtime-wire-v2");
    let mw_override = request.get("mw");
    let mw_runtime = request.get("mw_disposition");
    let mw = mw_override.or(mw_runtime);
    let mut mw_shape_valid = true;
    if wire_v2 {
        if mw_override.is_some() && mw_runtime.is_some() {
            mw_shape_valid = false;
        }
        if let Some(value) = mw {
            let expected_sources: &[&str] = if mw_override.is_some() {
                &["protocol-test"]
            } else {
                &["mw-head", "deterministic-policy"]
            };
            mw_shape_valid &= value.as_object().is_some_and(|object| {
                object.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "decision" | "allowed_tools" | "source" | "receipt_sha256"
                    )
                }) && object
                    .get("source")
                    .and_then(Value::as_str)
                    .is_some_and(|source| expected_sources.contains(&source))
                    && object.get("decision").and_then(Value::as_str).is_some()
                    && (object.get("source").and_then(Value::as_str) != Some("mw-head")
                        || object
                            .get("receipt_sha256")
                            .and_then(Value::as_str)
                            .is_some_and(|receipt| {
                                receipt.len() == 64
                                    && receipt.bytes().all(|byte| {
                                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                                    })
                            }))
                    && object.get("allowed_tools").map_or(true, |tools| {
                        tools.as_array().is_some_and(|items| {
                            items.iter().all(Value::is_string)
                                && items
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .collect::<std::collections::HashSet<_>>()
                                    .len()
                                    == items.len()
                        })
                    })
            });
        }
    }
    let (decision, allowed) = match mw {
        None => ("continue", None),
        Some(Value::String(value)) => (value.as_str(), None),
        Some(Value::Object(value)) => (
            value
                .get("decision")
                .or_else(|| value.get("disposition"))
                .and_then(Value::as_str)
                .unwrap_or("continue"),
            value
                .get("allowed_tools")
                .map(|tools| string_set(Some(tools))),
        ),
        _ => ("invalid", None),
    };
    let mw_error = if !mw_shape_valid {
        Some("mw_invalid")
    } else if decision == "constrain"
        && allowed
            .as_ref()
            .map(|tools| tools.is_empty())
            .unwrap_or(true)
    {
        Some("mw_invalid")
    } else if matches!(decision, "stop" | "block" | "refuse") {
        Some("mw_stop")
    } else if decision == "constrain"
        && allowed
            .as_ref()
            .map(|tools| !tools.contains(name))
            .unwrap_or(false)
    {
        Some("mw_constrained")
    } else if !matches!(decision, "continue" | "constrain") {
        Some("mw_invalid")
    } else {
        None
    };
    gates.push(validation_gate("mw", mw_error.is_none(), mw_error));
    if let Some(error) = mw_error {
        return blocked(gates, error, None);
    }
    json!({
        "ok": true, "refuse": false, "execution": "pending_confidence",
        "function_calls": parsed.function_calls, "error": null, "gates": gates,
        "provenance": provenance, "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
    })
}

pub fn apply_confidence(validated: Value, confidence: Option<f64>, enforce: bool) -> Value {
    apply_confidence_with_thresholds(validated, confidence, enforce, 0.70, 0.35)
}

pub fn apply_confidence_with_thresholds(
    mut validated: Value,
    confidence: Option<f64>,
    enforce: bool,
    execute_high: f64,
    escalate_low: f64,
) -> Value {
    if !validated
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || validated
            .get("refuse")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    {
        return validated;
    }
    let mut gates = validated
        .get("gates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let thresholds_valid = execute_high.is_finite()
        && escalate_low.is_finite()
        && (0.0..=1.0).contains(&execute_high)
        && (0.0..=1.0).contains(&escalate_low)
        && escalate_low <= execute_high;
    let execution = match confidence {
        _ if !thresholds_valid => {
            gates.push(validation_gate(
                "confidence",
                false,
                Some("confidence_invalid"),
            ));
            if let Some(object) = validated.as_object_mut() {
                object.insert("ok".into(), json!(false));
                object.insert("refuse".into(), json!(true));
                object.insert("error".into(), json!("confidence_invalid"));
                object.insert("function_calls".into(), json!([]));
            }
            "refuse"
        }
        Some(value) if !value.is_finite() || !(0.0..=1.0).contains(&value) => {
            gates.push(validation_gate(
                "confidence",
                false,
                Some("confidence_invalid"),
            ));
            if let Some(object) = validated.as_object_mut() {
                object.insert("ok".into(), json!(false));
                object.insert("refuse".into(), json!(true));
                object.insert("error".into(), json!("confidence_invalid"));
                object.insert("function_calls".into(), json!([]));
            }
            "refuse"
        }
        None if enforce => {
            gates.push(validation_gate(
                "confidence",
                false,
                Some("confidence_unavailable"),
            ));
            if let Some(object) = validated.as_object_mut() {
                object.insert("ok".into(), json!(false));
                object.insert("refuse".into(), json!(true));
                object.insert("error".into(), json!("confidence_unavailable"));
                object.insert("function_calls".into(), json!([]));
            }
            "refuse"
        }
        None => {
            gates.push(validation_gate("confidence", true, None));
            "execute"
        }
        Some(value) if value >= execute_high => {
            gates.push(json!({"gate":"confidence","ok":true,"detail":null,"value":value,"execution":"execute"}));
            "execute"
        }
        Some(value) if value >= escalate_low => {
            gates.push(json!({"gate":"confidence","ok":false,"detail":"escalate","value":value,"execution":"escalate"}));
            if let Some(object) = validated.as_object_mut() {
                object.insert("refuse".into(), json!(true));
                object.insert("function_calls".into(), json!([]));
            }
            "escalate"
        }
        Some(value) => {
            gates.push(json!({"gate":"confidence","ok":false,"detail":"refuse","value":value,"execution":"refuse"}));
            if let Some(object) = validated.as_object_mut() {
                object.insert("refuse".into(), json!(true));
                object.insert("function_calls".into(), json!([]));
            }
            "refuse"
        }
    };
    if let Some(object) = validated.as_object_mut() {
        object.insert("execution".into(), json!(execution));
        object.insert("confidence_value".into(), json!(confidence));
        object.insert("gates".into(), Value::Array(gates));
    }
    validated
}

pub fn render_tools_block(tools: &[Value]) -> String {
    format!("<tools>{}</tools>", dumps_canonical(&compact_tools(tools)))
}

pub fn selected_tools(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| {
            tool.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

pub fn render_request(request: &Value, tools: &[Value]) -> Result<Value, String> {
    if tools.len() > max_selected_tools() {
        return Err("complete() accepts at most 5 selected tools; retrieval must run first".into());
    }
    let wire_v2 =
        request.get("wire_version").and_then(Value::as_str) == Some("mei-runtime-wire-v2");
    let mut sink = task_contract();
    if !wire_v2 {
        let facts = request
            .get("context")
            .and_then(|context| context.get("system_facts"))
            .or_else(|| request.get("system_facts"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if !facts.is_empty() {
            sink.push_str("\n系统事实：");
            sink.push_str(facts);
        }
        if let Some(permissions) = request.get("permissions").filter(|value| {
            value
                .as_array()
                .map(|items| !items.is_empty())
                .unwrap_or(false)
                || value
                    .as_object()
                    .map(|items| !items.is_empty())
                    .unwrap_or(false)
        }) {
            let rendered = permissions
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .map(|value| value.as_str().unwrap_or("").to_string())
                        .collect::<Vec<_>>()
                        .join("、")
                })
                .unwrap_or_else(|| dumps_canonical(permissions));
            sink.push_str("\n权限：");
            sink.push_str(&rendered);
        }
        sink.push_str("\n点选实体：");
        match request.get("selected_entity").and_then(Value::as_str) {
            Some(entity) if !entity.is_empty() => sink.push_str(entity),
            _ => sink.push('无'),
        }
    }
    sink.push('\n');
    sink.push_str(&render_tools_block(tools));
    let mut ordinary: Vec<String> = Vec::new();
    if let Some(history) = request.get("history").and_then(Value::as_array) {
        for turn in history {
            let role = turn.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = turn
                .get("content")
                .and_then(Value::as_str)
                .or_else(|| turn.get("text").and_then(Value::as_str))
                .unwrap_or("")
                .trim();
            if !content.is_empty() {
                ordinary.push(format!("{role}：{content}"));
            }
        }
    }
    let results = request
        .get("tool_results")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .or_else(|| request.get("prior_tool_results").and_then(Value::as_array));
    if let Some(results) = results {
        for result in results {
            let rendered = result
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| dumps_canonical(result));
            ordinary.push(format!("tool：{rendered}"));
        }
    }
    if wire_v2 {
        let empty_object = json!({});
        let empty_array = json!([]);
        for (tag, value) in [
            ("context", request.get("context").unwrap_or(&empty_object)),
            ("evidence", request.get("evidence").unwrap_or(&empty_array)),
            (
                "permissions",
                request.get("permissions").unwrap_or(&empty_object),
            ),
            ("state", request.get("state").unwrap_or(&empty_object)),
            (
                "mw",
                request
                    .get("mw")
                    .or_else(|| request.get("mw_disposition"))
                    .unwrap_or(&empty_object),
            ),
        ] {
            ordinary.push(format!("<{tag}>{}</{tag}>", dumps_canonical(value)));
        }
    }
    let query = request
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    ordinary.push(format!("user：{query}"));
    let ordinary = ordinary.join("\n");
    let prompt = format!("{sink}\n{ordinary}");
    Ok(json!({
        "prompt": prompt,
        "sink": sink,
        "ordinary": ordinary,
        "schema_fingerprint": schema_fingerprint(tools),
        "serializer_id": serializer_id(),
        "protocol_id": protocol_id(),
        "selected_tools": selected_tools(tools),
    }))
}
