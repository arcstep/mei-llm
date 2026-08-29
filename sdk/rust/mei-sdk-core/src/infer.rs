//! Constrained decode, retrieval, validator, and confidence execution gate
//! over a packed 51M model. WASM loads only the quantized package.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::canonical::schema_fingerprint;
use crate::error::{ErrorInfo, SdkError};
use crate::model::{ForwardOut, NeedleModel};
use crate::protocol::{parse_v2_text, render_request};
use crate::version::{max_selected_tools, wire_version};
use crate::vocab::Vocab;

pub const EXECUTE_HIGH: f32 = 0.70;
pub const ESCALATE_LOW: f32 = 0.35;

#[derive(Clone)]
pub struct InferRuntime {
    pub model: Arc<NeedleModel>,
    pub vocab: Arc<Vocab>,
}

impl InferRuntime {
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, SdkError> {
        let out = self.model.forward(tokens, &mut None, false)?;
        let d = self.model.arch.d_model;
        let t = out.t.max(1);
        let last = out.hidden[(t - 1) * d..t * d].to_vec();
        Ok(l2_normalize(&last))
    }

    pub fn search_top_k(&self, query: &str, catalog: &[Value], k: usize) -> Result<Vec<Value>, SdkError> {
        if catalog.len() <= k {
            return Ok(catalog.to_vec());
        }
        let q_ids = self.vocab.encode(query, true);
        let q = self.embed(&q_ids)?;
        let mut scored: Vec<(f32, Value)> = Vec::new();
        for tool in catalog {
            let blob = serde_json::to_string(tool).unwrap_or_default();
            let ids = self.vocab.encode(&blob, true);
            let v = self.embed(&ids)?;
            let mut sim = 0f32;
            for (a, b) in q.iter().zip(v.iter()) {
                sim += a * b;
            }
            scored.push((sim, tool.clone()));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).map(|(_, t)| t).collect())
    }

    pub fn greedy(
        &self,
        prompt_ids: &[u32],
        tools: &[Value],
        max_new: usize,
        constrained: bool,
    ) -> Result<(String, Vec<u32>, f32, Option<f32>), SdkError> {
        let mut cache = None;
        let mut prefix = Vec::<u8>::new();
        let mut pieces = Vec::<u32>::new();
        let mut logprob_sum = 0f32;
        let mut out: ForwardOut = self.model.forward(prompt_ids, &mut cache, true)?;
        let conf = out.confidence_logit;
        cache = Some(out.cache);
        let vocab = out.vocab;
        let mut logits = out.logits[(out.t - 1) * vocab..out.t * vocab].to_vec();
        for _ in 0..max_new {
            if constrained {
                mask_grammar(&mut logits, &self.vocab, &prefix, tools);
            }
            let (tok, logp) = argmax_logp(&logits);
            if tok == self.vocab.eos_id {
                break;
            }
            let extra = self.vocab.token_bytes(tok);
            prefix.extend_from_slice(&extra);
            pieces.push(tok);
            logprob_sum += logp;
            let text = String::from_utf8_lossy(&prefix).to_string();
            if constrained && is_accept(&text) {
                break;
            }
            out = self.model.forward(&[tok], &mut cache, false)?;
            logits = out.logits[(out.t - 1) * vocab..out.t * vocab].to_vec();
            cache = Some(out.cache);
        }
        let text = String::from_utf8_lossy(&prefix).to_string();
        Ok((text, pieces, logprob_sum, conf))
    }
}

pub fn complete_infer(
    runtime: &InferRuntime,
    request: &Value,
    capabilities: &Value,
    wall_ms: f64,
) -> Result<Value, SdkError> {
    let decode_mode = request
        .get("decode_mode")
        .and_then(Value::as_str)
        .unwrap_or("constrained");
    let query = request.get("query").and_then(Value::as_str).unwrap_or("");
    let selected_tools = select_tools(runtime, request)?;
    let rendered = render_request(request, &selected_tools)
        .map_err(|e| SdkError::new("invalid_argument", e))?;
    let prompt = rendered.get("prompt").and_then(Value::as_str).unwrap_or("");
    let ids = request
        .get("token_ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| runtime.vocab.encode(prompt, true));
    let max_new = request.get("max_new").and_then(Value::as_u64).unwrap_or(96) as usize;
    let (text, pieces, logprob, conf_logit) =
        runtime.greedy(&ids, &selected_tools, max_new, decode_mode != "raw")?;
    let validated = validate_call(&text, &selected_tools, query);
    let conf = match conf_logit {
        Some(logit) => combine_confidence(logit, logprob),
        None => sigmoid_from_logit(logprob),
    };
    let gated = apply_confidence_gate(validated, Some(conf));
    let selected_names: Vec<String> = selected_tools
        .iter()
        .map(|t| t.get("name").and_then(Value::as_str).unwrap_or("").to_string())
        .collect();
    let fp = rendered
        .get("schema_fingerprint")
        .cloned()
        .unwrap_or_else(|| json!(schema_fingerprint(&selected_tools)));
    let execution = gated
        .get("execution")
        .and_then(Value::as_str)
        .unwrap_or("refuse");
    let refuse = gated.get("refuse").and_then(Value::as_bool).unwrap_or(true);
    let ok = gated.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let calls = gated
        .get("function_calls")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let err = gated.get("error").cloned();
    Ok(json!({
        "wire_version": wire_version(),
        "ok": ok,
        "error": err.and_then(|e| {
            if e.is_null() { None } else {
                Some(ErrorInfo::new("protocol_violation", e.as_str()).to_value())
            }
        }),
        "refuse": refuse,
        "execution": execution,
        "selected_tools": selected_names,
        "schema_fingerprint": fp,
        "function_calls": if ok && !refuse { calls } else { json!([]) },
        "raw_text": text,
        "confidence": {
            "available": true,
            "value": conf,
            "source": "mei-1.0-51m-q4",
            "version": "conf-head-v1"
        },
        "provenance": gated,
        "capabilities": capabilities,
        "stats": {
            "backend": "portable-q4",
            "wall_ms": wall_ms,
            "decode_mode": decode_mode,
            "prompt_tokens": ids.len(),
            "output_tokens": pieces.len(),
            "quantized_only": true
        }
    }))
}

fn select_tools(runtime: &InferRuntime, request: &Value) -> Result<Vec<Value>, SdkError> {
    if let Some(oracle) = request.get("oracle_tools").and_then(Value::as_array) {
        if oracle.len() > max_selected_tools() {
            return Err(SdkError::from_id("too_many_tools"));
        }
        return Ok(oracle.clone());
    }
    let catalog = request
        .get("catalog")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| SdkError::new("invalid_argument", "tools must be a list"))?;
    let query = request.get("query").and_then(Value::as_str).unwrap_or("");
    runtime.search_top_k(query, &catalog, max_selected_tools())
}

fn l2_normalize(x: &[f32]) -> Vec<f32> {
    let mut n = 1e-8f32;
    for v in x {
        n += *v * *v;
    }
    let inv = n.sqrt().recip();
    x.iter().map(|v| *v * inv).collect()
}

fn argmax_logp(logits: &[f32]) -> (u32, f32) {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0f32;
    for v in logits {
        z += (*v - m).exp();
    }
    let logp = best - m - z.ln();
    (best_i as u32, logp)
}

fn is_accept(text: &str) -> bool {
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(Value::Array(a)) if a.is_empty() => true,
        Ok(Value::Array(a)) if a.len() == 1 && a[0].is_object() => {
            a[0].get("name").is_some() && a[0].get("arguments").is_some()
        }
        _ => false,
    }
}

fn call_templates(tools: &[Value]) -> Vec<String> {
    let mut out = vec!["[]".to_string()];
    for tool in tools {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let props = tool
            .pointer("/parameters/properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let bool_keys: Vec<String> = props
            .iter()
            .filter(|(_, spec)| spec.get("type").and_then(Value::as_str) == Some("boolean"))
            .map(|(k, _)| k.clone())
            .take(3)
            .collect();
        if bool_keys.is_empty() {
            out.push(format!(r#"[{{"name":"{name}","arguments":{{}}}}]"#));
            continue;
        }
        let n = 1usize << bool_keys.len();
        for mask in 0..n {
            let args: Vec<String> = bool_keys
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    format!(
                        r#""{k}":{}"#,
                        if (mask >> i) & 1 == 1 { "true" } else { "false" }
                    )
                })
                .collect();
            out.push(format!(
                r#"[{{"name":"{name}","arguments":{{{}}}}}]"#,
                args.join(",")
            ));
        }
    }
    out
}

fn legal_prefix(text: &str, templates: &[String]) -> bool {
    if text.is_empty() {
        return true;
    }
    templates.iter().any(|tmpl| tmpl.starts_with(text))
}

fn mask_grammar(logits: &mut [f32], vocab: &Vocab, prefix: &[u8], tools: &[Value]) {
    let prefix_text = String::from_utf8_lossy(prefix).to_string();
    let templates = call_templates(tools);
    if is_accept(&prefix_text) || templates.iter().any(|t| t == &prefix_text) {
        for (i, v) in logits.iter_mut().enumerate() {
            if i as u32 != vocab.eos_id {
                *v = f32::NEG_INFINITY;
            }
        }
        return;
    }
    let mut allowed = Vec::new();
    let mut ranked: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let consider = |tok: u32, allowed: &mut Vec<usize>| {
        if tok == vocab.pad_id || tok == vocab.bos_id {
            return;
        }
        let extra = vocab.token_bytes(tok);
        if extra.is_empty() {
            return;
        }
        let mut trial = prefix.to_vec();
        trial.extend_from_slice(&extra);
        if let Ok(text) = String::from_utf8(trial) {
            if legal_prefix(&text, &templates) {
                allowed.push(tok as usize);
            }
        }
    };
    for (_, tok) in ranked.iter().take(1024) {
        consider(*tok as u32, &mut allowed);
        if allowed.len() >= 16 {
            break;
        }
    }
    if allowed.is_empty() {
        for tok in 0..logits.len() as u32 {
            consider(tok, &mut allowed);
            if allowed.len() >= 16 {
                break;
            }
        }
    }
    if allowed.is_empty() {
        return;
    }
    let mut keep = vec![false; logits.len()];
    for i in allowed {
        if i < keep.len() {
            keep[i] = true;
        }
    }
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn validate_call(text: &str, tools: &[Value], query: &str) -> Value {
    let parsed = parse_v2_text(text);
    if !parsed.ok {
        return json!({
            "ok": false,
            "function_calls": [],
            "error": parsed.error,
            "refuse": true,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0
        });
    }
    if parsed.refuse {
        return json!({
            "ok": true,
            "function_calls": [],
            "error": null,
            "refuse": true,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0
        });
    }
    let call = &parsed.function_calls[0];
    let name = call.get("name").and_then(Value::as_str).unwrap_or("");
    let tool = tools.iter().find(|t| t.get("name").and_then(Value::as_str) == Some(name));
    let Some(tool) = tool else {
        return json!({
            "ok": false,
            "function_calls": [],
            "error": "unsupported_tool",
            "refuse": true,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0
        });
    };
    let props = tool
        .pointer("/parameters/properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required: Vec<String> = tool
        .pointer("/parameters/required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let args = call.get("arguments").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut unsupported = 0u32;
    let mut unprov = 0u32;
    for (key, value) in &args {
        if !props.contains_key(key) {
            unsupported += 1;
            continue;
        }
        let needle = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(_) => String::new(),
            _ => value.to_string(),
        };
        if !needle.is_empty() && !query.contains(&needle) {
            unprov += 1;
        }
    }
    let missing: Vec<String> = required.into_iter().filter(|k| !args.contains_key(k)).collect();
    if unsupported > 0 || unprov > 0 || !missing.is_empty() {
        return json!({
            "ok": false,
            "function_calls": [],
            "error": "validator",
            "refuse": true,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
            "would_have_accepted_unsupported": unsupported,
            "would_have_accepted_unprovenanced": unprov,
            "missing_required": missing
        });
    }
    json!({
        "ok": true,
        "function_calls": parsed.function_calls,
        "error": null,
        "refuse": false,
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0
    })
}

fn combine_confidence(head_logit: f32, decode_logprob: f32) -> f32 {
    let p_head = 1.0 / (1.0 + (-head_logit).exp());
    let p_dec = decode_logprob.min(0.0).exp();
    p_head.min(p_dec)
}

fn sigmoid_from_logit(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_confidence_gate(mut validated: Value, confidence: Option<f32>) -> Value {
    let value = confidence;
    if let Some(obj) = validated.as_object_mut() {
        obj.insert("confidence_value".into(), json!(value));
    }
    let ok = validated.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let refuse = validated.get("refuse").and_then(Value::as_bool).unwrap_or(true);
    if !ok || refuse {
        if let Some(obj) = validated.as_object_mut() {
            obj.insert("execution".into(), json!("refuse"));
        }
        return validated;
    }
    let Some(v) = value else {
        if let Some(obj) = validated.as_object_mut() {
            obj.insert("ok".into(), json!(false));
            obj.insert("refuse".into(), json!(true));
            obj.insert("function_calls".into(), json!([]));
            obj.insert("error".into(), json!("confidence_unavailable"));
            obj.insert("execution".into(), json!("refuse"));
        }
        return validated;
    };
    if v >= EXECUTE_HIGH {
        if let Some(obj) = validated.as_object_mut() {
            obj.insert("execution".into(), json!("execute"));
        }
    } else if v >= ESCALATE_LOW {
        if let Some(obj) = validated.as_object_mut() {
            obj.insert("execution".into(), json!("escalate"));
            obj.insert("ok".into(), json!(true));
            obj.insert("refuse".into(), json!(true));
            obj.insert("function_calls".into(), json!([]));
            obj.insert("escalate".into(), json!(true));
        }
    } else if let Some(obj) = validated.as_object_mut() {
        obj.insert("execution".into(), json!("refuse"));
        obj.insert("ok".into(), json!(true));
        obj.insert("refuse".into(), json!(true));
        obj.insert("function_calls".into(), json!([]));
    }
    validated
}
