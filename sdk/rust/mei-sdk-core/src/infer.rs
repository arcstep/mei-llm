//! Constrained decode, retrieval, validator, and confidence execution gate
//! over a packed 51M model. WASM loads only the quantized package.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::byte_grammar::ByteGrammar;
use crate::canonical::schema_fingerprint;
use crate::error::{ErrorInfo, SdkError};
use crate::model::{ForwardOut, NeedleModel};
use crate::protocol::{
    apply_confidence, render_request, render_tools_block,
    validate_generated_call_with_trusted_results,
};
use crate::tool_index::ToolIndex;
use crate::version::{max_selected_tools, wire_version};
use crate::vocab::Vocab;

#[derive(Clone)]
pub struct InferRuntime {
    pub model: Arc<NeedleModel>,
    pub vocab: Arc<Vocab>,
    pub tool_index: Option<Arc<ToolIndex>>,
    pub mw_receipt_sha256: Option<String>,
}

impl InferRuntime {
    pub fn generate_narration(&self, prompt: &str, max_new: usize) -> Result<String, SdkError> {
        if !self.model.has_tensor("heads.narration_adapter.down.weight")
            || !self.model.has_tensor("heads.narration_adapter.up.weight")
        {
            return Err(SdkError::new(
                "capability_missing",
                "narration adapter is unavailable",
            ));
        }
        let encoded = self.vocab.encode(prompt, true);
        if encoded.is_empty() {
            return Err(SdkError::new(
                "invalid_argument",
                "narration prompt produced no tokens",
            ));
        }
        let stable_len = encoded.len().min(crate::model::STABLE_PREFIX_TOKENS_MAX);
        let stable_history = encoded[..stable_len].to_vec();
        let mut rolling_history = encoded[stable_len..].to_vec();
        if rolling_history.len() > crate::model::ROLLING_WINDOW_TOKENS {
            let drop = rolling_history.len() - crate::model::ROLLING_WINDOW_TOKENS;
            rolling_history.drain(..drop);
        }
        let visible = stable_history
            .iter()
            .chain(rolling_history.iter())
            .copied()
            .collect::<Vec<_>>();
        let mut cache = None;
        let mut out = self.model.forward_bounded_last_logits(
            &visible,
            &mut cache,
            false,
            None,
            stable_history.len(),
        )?;
        cache = Some(std::mem::take(&mut out.cache));
        let mut bytes = Vec::new();
        let mut pieces = Vec::new();
        for _ in 0..max_new.clamp(1, 48) {
            let hidden =
                &out.hidden[(out.t - 1) * self.model.arch.d_model..out.t * self.model.arch.d_model];
            let residual = self.model.narration_residual(hidden)?;
            let mut logits = out.last_logits()?.to_vec();
            for (value, add) in logits.iter_mut().zip(residual.iter()) {
                *value += *add;
            }
            let (token, _) = argmax_logp(&logits);
            if token == self.vocab.eos_id {
                break;
            }
            bytes.extend_from_slice(&self.vocab.generated_token_bytes(token, pieces.is_empty()));
            pieces.push(token);
            let prefix_history = stable_history
                .iter()
                .chain(rolling_history.iter())
                .copied()
                .collect::<Vec<_>>();
            out = self.model.forward_bounded_last_logits(
                &[token],
                &mut cache,
                false,
                Some(&prefix_history),
                0,
            )?;
            rolling_history.push(token);
            if rolling_history.len() > crate::model::ROLLING_WINDOW_TOKENS {
                rolling_history.remove(0);
            }
            cache = Some(std::mem::take(&mut out.cache));
        }
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, SdkError> {
        let out = self.model.forward_features(tokens, &mut None, false)?;
        if self.model.has_tensor("heads.contrastive.proj.weight") {
            return self.model.contrastive_embedding(&out);
        }
        let d = self.model.arch.d_model;
        let t = out.t.max(1);
        let last = out.hidden[(t - 1) * d..t * d].to_vec();
        Ok(l2_normalize(&last))
    }

    pub fn search_top_k(
        &self,
        query: &str,
        catalog: &[Value],
        k: usize,
    ) -> Result<Vec<Value>, SdkError> {
        let q_ids = self.vocab.encode(query, true);
        let q = self.embed(&q_ids)?;
        if let Some(index) = &self.tool_index {
            let catalog_sha = crate::canonical::catalog_fingerprint(catalog);
            if catalog_sha != index.catalog_sha256 {
                return Err(SdkError::new(
                    "package_hash_mismatch",
                    "request catalog does not match the frozen tool index",
                ));
            }
            return Ok(index
                .topk(&q, k)?
                .into_iter()
                .map(|record| record.schema.clone())
                .collect());
        }
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
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let left = a.1.get("name").and_then(Value::as_str).unwrap_or("");
                    let right = b.1.get("name").and_then(Value::as_str).unwrap_or("");
                    left.as_bytes().cmp(right.as_bytes())
                })
        });
        Ok(scored.into_iter().take(k).map(|(_, t)| t).collect())
    }

    pub fn head_diagnostics(&self, text: &str) -> Result<Value, SdkError> {
        let ids = self.vocab.encode(text, true);
        if ids.is_empty() {
            return Err(SdkError::new(
                "invalid_argument",
                "head diagnostic text produced no tokens",
            ));
        }
        let out = self.model.forward_features(&ids, &mut None, true)?;
        let embedding = self.model.contrastive_embedding(&out)?;
        let mw_probabilities = self.model.mw_disposition_probabilities(&out)?;
        let d = self.model.arch.d_model;
        let last_hidden = &out.hidden[(out.t - 1) * d..out.t * d];
        let narration_residual = self.model.narration_residual(last_hidden)?;
        let top5 = if let Some(index) = &self.tool_index {
            index
                .topk(&embedding, 5)?
                .into_iter()
                .map(|record| record.tool_id.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        Ok(json!({
            "token_ids": ids,
            "contrastive_embedding": embedding,
            "retrieval_top5": top5,
            "mw_probabilities": mw_probabilities,
            "mw_reason_code": mw_probabilities
                .iter()
                .enumerate()
                .max_by(|(left_id, left), (right_id, right)| {
                    left.total_cmp(right).then_with(|| right_id.cmp(left_id))
                })
                .map(|(index, _)| index),
            "confidence_logit": out.confidence_logit,
            "narration_residual": narration_residual,
            "semantic_boundaries": {
                "retrieval": "independent-contrastive-head",
                "mw_disposition": "independent-20class-sidecar",
                "confidence": "independent-calibrated-sidecar",
                "narration": "independent-frozen-backbone-rank16-generation-sidecar",
            }
        }))
    }

    pub fn greedy(
        &self,
        prompt_ids: &[u32],
        stable_prefix_tokens: usize,
        tools: &[Value],
        max_new: usize,
        constrained: bool,
    ) -> Result<
        (
            String,
            Vec<u32>,
            Vec<u32>,
            f32,
            Option<f32>,
            Value,
            Value,
            Value,
        ),
        SdkError,
    > {
        let mut cache = None;
        let mut prefix = Vec::<u8>::new();
        let mut pieces = Vec::<u32>::new();
        let stable_prefix_tokens = stable_prefix_tokens.min(prompt_ids.len());
        let stable_history = prompt_ids[..stable_prefix_tokens].to_vec();
        let mut rolling_history = prompt_ids[stable_prefix_tokens..].to_vec();
        if rolling_history.len() > crate::model::ROLLING_WINDOW_TOKENS {
            let keep = rolling_history.len() - crate::model::ROLLING_WINDOW_TOKENS;
            rolling_history.drain(..keep);
        }
        let mut logprob_sum = 0f32;
        let grammar = constrained.then(|| ByteGrammar::compile(tools));
        let mut out: ForwardOut = self.model.forward_bounded_last_logits(
            prompt_ids,
            &mut cache,
            true,
            None,
            stable_prefix_tokens,
        )?;
        let conf = out.confidence_logit;
        let (mw_disposition, mw_audit) = self.mw_disposition(&out)?;
        cache = Some(std::mem::take(&mut out.cache));
        let mut logits = out.last_logits()?.to_vec();
        let mut ranked: Vec<usize> = (0..logits.len()).collect();
        ranked.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let prefill_topk_ids = ranked.into_iter().take(5).map(|id| id as u32).collect();
        for _ in 0..max_new {
            if constrained {
                mask_grammar(
                    &mut logits,
                    &self.vocab,
                    &prefix,
                    grammar.as_ref().expect("compiled constrained grammar"),
                    pieces.len(),
                )?;
            }
            let (tok, logp) = argmax_logp(&logits);
            if tok == self.vocab.eos_id {
                break;
            }
            let extra = self.vocab.generated_token_bytes(tok, pieces.is_empty());
            prefix.extend_from_slice(&extra);
            pieces.push(tok);
            logprob_sum += logp;
            if constrained
                && grammar
                    .as_ref()
                    .expect("compiled constrained grammar")
                    .accepting(&prefix)
            {
                break;
            }
            let prefix_history = stable_history
                .iter()
                .chain(rolling_history.iter())
                .copied()
                .collect::<Vec<_>>();
            out = self.model.forward_bounded_last_logits(
                &[tok],
                &mut cache,
                false,
                Some(&prefix_history),
                0,
            )?;
            rolling_history.push(tok);
            if rolling_history.len() > crate::model::ROLLING_WINDOW_TOKENS {
                rolling_history.remove(0);
            }
            logits = out.last_logits()?.to_vec();
            cache = Some(std::mem::take(&mut out.cache));
        }
        let text = String::from_utf8_lossy(&prefix).to_string();
        let cache_report = cache
            .as_ref()
            .map(|layers| crate::model::cache_evidence(layers))
            .unwrap_or_else(|| json!({"cache_growth_bounded": false}));
        Ok((
            text,
            pieces,
            prefill_topk_ids,
            logprob_sum,
            conf,
            cache_report,
            mw_disposition,
            mw_audit,
        ))
    }

    fn mw_disposition(&self, out: &ForwardOut) -> Result<(Value, Value), SdkError> {
        let Some(receipt) = self.mw_receipt_sha256.as_deref() else {
            return Ok((Value::Null, json!({"available":false})));
        };
        let probabilities = self.model.mw_disposition_probabilities(out)?;
        let (decision, reason_code, top_probability, margin) =
            mw_policy_from_probabilities(&probabilities)?;
        let wire = json!({
            "decision": decision,
            "source": "mw-head",
            "receipt_sha256": receipt,
        });
        let audit = json!({
            "decision": decision,
            "source": "mw-head",
            "receipt_sha256": receipt,
            "reason_code": reason_code,
            "top_probability": top_probability,
            "margin": margin,
            "continue_min_probability": 0.70,
            "continue_min_margin": 0.15,
        });
        Ok((wire, audit))
    }
}

fn mw_policy_from_probabilities(
    probabilities: &[f32],
) -> Result<(&'static str, usize, f32, f32), SdkError> {
    if probabilities.len() != 20
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        || (probabilities.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err(SdkError::new(
            "package_invalid",
            "MW disposition head returned invalid probabilities",
        ));
    }
    let mut ranked = probabilities
        .iter()
        .copied()
        .enumerate()
        .map(|(class, probability)| (probability, class))
        .collect::<Vec<_>>();
    ranked.sort_by(
        |(left_probability, left_class), (right_probability, right_class)| {
            right_probability
                .total_cmp(left_probability)
                .then_with(|| left_class.cmp(right_class))
        },
    );
    let (top_probability, reason_code) = ranked[0];
    let margin = top_probability - ranked[1].0;
    let decision = if reason_code == 0 && top_probability >= 0.70 && margin >= 0.15 {
        "continue"
    } else {
        "stop"
    };
    Ok((decision, reason_code, top_probability, margin))
}

pub fn complete_infer(
    runtime: &InferRuntime,
    request: &Value,
    capabilities: &Value,
    trusted_tool_results: &[Value],
    wall_ms: f64,
) -> Result<Value, SdkError> {
    let decode_mode = request
        .get("decode_mode")
        .and_then(Value::as_str)
        .unwrap_or("constrained");
    let selected_tools = select_tools(runtime, request)?;
    let schema_tokens = runtime
        .vocab
        .encode(&render_tools_block(&selected_tools), false)
        .len();
    if schema_tokens > 1_024 {
        return Err(SdkError::new(
            "tool_schema_budget_exceeded",
            format!("selected tool schema uses {schema_tokens} tokens; maximum is 1024"),
        ));
    }
    let rendered = render_request(request, &selected_tools)
        .map_err(|e| SdkError::new("invalid_argument", e))?;
    let (ids, stable_prefix_tokens) = if let Some(raw) = request.get("token_ids") {
        let values = raw
            .as_array()
            .ok_or_else(|| SdkError::new("invalid_argument", "token_ids must be an array"))?;
        let all = values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .filter(|id| *id < runtime.vocab.vocab_size as u64)
                    .map(|id| id as u32)
                    .ok_or_else(|| {
                        SdkError::new("invalid_argument", "token_ids contains an out-of-vocab ID")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stable = all.len().min(crate::model::STABLE_PREFIX_TOKENS_MAX);
        let mut bounded = all[..stable].to_vec();
        let ordinary = &all[stable..];
        let start = ordinary
            .len()
            .saturating_sub(crate::model::ROLLING_WINDOW_TOKENS);
        bounded.extend_from_slice(&ordinary[start..]);
        (bounded, stable)
    } else {
        let sink = rendered.get("sink").and_then(Value::as_str).unwrap_or("");
        let ordinary = rendered
            .get("ordinary")
            .and_then(Value::as_str)
            .unwrap_or("");
        let sink_ids = runtime.vocab.encode(sink, true);
        if sink_ids.len() > crate::model::STABLE_PREFIX_TOKENS_MAX {
            return Err(SdkError::new(
                "tool_schema_budget_exceeded",
                format!(
                    "stable tool prefix uses {} tokens; maximum is {}",
                    sink_ids.len(),
                    crate::model::STABLE_PREFIX_TOKENS_MAX
                ),
            ));
        }
        let ordinary_ids = runtime.vocab.encode(ordinary, false);
        let start = ordinary_ids
            .len()
            .saturating_sub(crate::model::ROLLING_WINDOW_TOKENS);
        let mut bounded = sink_ids.clone();
        bounded.extend_from_slice(&ordinary_ids[start..]);
        (bounded, sink_ids.len())
    };
    if ids.is_empty()
        || ids.len() > crate::model::STABLE_PREFIX_TOKENS_MAX + crate::model::ROLLING_WINDOW_TOKENS
    {
        return Err(SdkError::new(
            "invalid_argument",
            format!(
                "bounded prompt token length {} is outside 1..=1280",
                ids.len()
            ),
        ));
    }
    let max_new = request
        .get("max_new")
        .and_then(Value::as_u64)
        .unwrap_or(128);
    if !(1..=128).contains(&max_new) {
        return Err(SdkError::new(
            "invalid_argument",
            "max_new must be between 1 and 128",
        ));
    }
    let (
        text,
        pieces,
        prefill_topk_ids,
        logprob,
        conf_logit,
        cache_report,
        mw_disposition,
        mw_audit,
    ) = runtime.greedy(
        &ids,
        stable_prefix_tokens,
        &selected_tools,
        max_new as usize,
        decode_mode != "raw",
    )?;
    let mut gate_request = request.clone();
    if !mw_disposition.is_null() {
        gate_request["mw_disposition"] = mw_disposition.clone();
    }
    let validated = validate_generated_call_with_trusted_results(
        &text,
        &selected_tools,
        &gate_request,
        trusted_tool_results,
    );
    let conf = match conf_logit {
        Some(logit) => combine_confidence(logit, logprob),
        None => sigmoid_from_logit(logprob),
    };
    let gated = apply_confidence(validated, Some(conf as f64), true);
    let selected_names: Vec<String> = selected_tools
        .iter()
        .map(|t| {
            t.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        })
        .collect();
    let fp = rendered
        .get("schema_fingerprint")
        .cloned()
        .unwrap_or_else(|| json!(schema_fingerprint(&selected_tools)));
    let execution = gated
        .get("execution")
        .and_then(Value::as_str)
        .unwrap_or("refuse");
    let validation_error = gated.get("error").and_then(Value::as_str);
    let deterministic_refusal = validation_error
        .map(|error| {
            error == "provenance_missing"
                || error.starts_with("permission_")
                || error.starts_with("state_")
                || error.starts_with("mw_")
                || error.starts_with("confidence_")
        })
        .unwrap_or(false);
    let refuse =
        gated.get("refuse").and_then(Value::as_bool).unwrap_or(true) || deterministic_refusal;
    let ok = gated.get("ok").and_then(Value::as_bool).unwrap_or(false) || deterministic_refusal;
    let calls = if deterministic_refusal {
        json!([])
    } else {
        gated
            .get("function_calls")
            .cloned()
            .unwrap_or_else(|| json!([]))
    };
    let err = if deterministic_refusal {
        None
    } else {
        gated.get("error").cloned()
    };
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
        "generated_token_ids": pieces,
        "prefill_topk_ids": prefill_topk_ids,
        "runtime_cache": cache_report,
        "mw_disposition": mw_audit,
        "confidence": {
            "available": true,
            "value": conf,
            "source": "mei-1.0-51m-q4",
            "version": "conf-head-v1"
        },
        "provenance": {
            "validated": true,
            "ok": gated.get("ok").and_then(Value::as_bool).unwrap_or(false),
            "detail": validation_error,
            "arguments": gated.get("provenance").cloned().unwrap_or_else(|| json!({})),
            "gates": gated.get("gates").cloned().unwrap_or_else(|| json!([])),
        },
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

#[cfg(test)]
mod independent_head_policy_tests {
    use super::mw_policy_from_probabilities;

    #[test]
    fn mw_disposition_is_a_20_class_fail_closed_policy_only() {
        let mut ready = vec![0.2 / 19.0; 20];
        ready[0] = 0.8;
        let (decision, reason, _, _) = mw_policy_from_probabilities(&ready).unwrap();
        assert_eq!((decision, reason), ("continue", 0));

        let mut non_ready = vec![0.1 / 19.0; 20];
        non_ready[7] = 0.9;
        let (decision, reason, _, _) = mw_policy_from_probabilities(&non_ready).unwrap();
        assert_eq!((decision, reason), ("stop", 7));

        let mut low_margin = vec![0.0; 20];
        low_margin[0] = 0.55;
        low_margin[1] = 0.45;
        assert_eq!(mw_policy_from_probabilities(&low_margin).unwrap().0, "stop");

        assert!(mw_policy_from_probabilities(&[1.0; 19]).is_err());
        assert!(mw_policy_from_probabilities(&[f32::NAN; 20]).is_err());
    }
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

fn mask_grammar(
    logits: &mut [f32],
    vocab: &Vocab,
    prefix: &[u8],
    grammar: &ByteGrammar,
    generated_token_count: usize,
) -> Result<(), SdkError> {
    if grammar.accepting(prefix) {
        for (i, v) in logits.iter_mut().enumerate() {
            if i as u32 != vocab.eos_id {
                *v = f32::NEG_INFINITY;
            }
        }
        return Ok(());
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
        let extra = vocab.generated_token_bytes(tok, generated_token_count == 0);
        if extra.is_empty() {
            return;
        }
        let mut trial = prefix.to_vec();
        trial.extend_from_slice(&extra);
        if grammar.legal_prefix(&trial) {
            allowed.push(tok as usize);
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
        return Err(SdkError::new(
            "protocol_violation",
            "constrained decoder has no legal UTF-8 grammar transition",
        ));
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
    Ok(())
}

fn combine_confidence(head_logit: f32, decode_logprob: f32) -> f32 {
    let p_head = 1.0 / (1.0 + (-head_logit).exp());
    let p_dec = decode_logprob.min(0.0).exp();
    p_head.min(p_dec)
}

fn sigmoid_from_logit(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
