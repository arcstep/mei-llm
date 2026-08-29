use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn dumps_canonical(value: &Value) -> String {
    serde_json::to_string(value).expect("json value is serializable")
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

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
