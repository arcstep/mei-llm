use serde_json::{json, Value};

use crate::canonical::{compact_tools, dumps_canonical, schema_fingerprint};
use crate::version::{forbidden_markers, max_calls, max_selected_tools, protocol_id, serializer_id, task_contract};

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
        request.get("system_facts").and_then(Value::as_str).unwrap_or(""),
        request.get("selected_entity").and_then(Value::as_str).unwrap_or(""),
        request.get("candidate_text").and_then(Value::as_str).unwrap_or(""),
        &dumps_canonical(request.get("history").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("prior_tool_results").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("catalog").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("oracle_tools").unwrap_or(&json!([]))),
        &dumps_canonical(request.get("permissions").unwrap_or(&json!([]))),
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

pub fn render_tools_block(tools: &[Value]) -> String {
    format!("<tools>{}</tools>", dumps_canonical(&compact_tools(tools)))
}

pub fn selected_tools(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| tool.get("name").and_then(Value::as_str).unwrap_or("").to_string())
        .collect()
}

pub fn render_request(request: &Value, tools: &[Value]) -> Result<Value, String> {
    if tools.len() > max_selected_tools() {
        return Err("complete() accepts at most 5 selected tools; retrieval must run first".into());
    }
    let facts = request
        .get("system_facts")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let mut sink = task_contract();
    if !facts.is_empty() {
        sink.push_str("\n系统事实：");
        sink.push_str(facts);
    }
    if let Some(permissions) = request.get("permissions").and_then(Value::as_array) {
        if !permissions.is_empty() {
            let joined = permissions
                .iter()
                .map(|p| p.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
                .join("、");
            sink.push_str("\n权限：");
            sink.push_str(&joined);
        }
    }
    sink.push_str("\n点选实体：");
    match request.get("selected_entity").and_then(Value::as_str) {
        Some(entity) if !entity.is_empty() => sink.push_str(entity),
        _ => sink.push('无'),
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
    if let Some(results) = request.get("prior_tool_results").and_then(Value::as_array) {
        for result in results {
            let text = result
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| result.to_string());
            ordinary.push(format!("tool：{text}"));
        }
    }
    let query = request.get("query").and_then(Value::as_str).unwrap_or("").trim();
    ordinary.push(format!("user：{query}"));
    let prompt = format!("{sink}\n{}", ordinary.join("\n"));
    Ok(json!({
        "prompt": prompt,
        "schema_fingerprint": schema_fingerprint(tools),
        "serializer_id": serializer_id(),
        "protocol_id": protocol_id(),
        "selected_tools": selected_tools(tools),
    }))
}
