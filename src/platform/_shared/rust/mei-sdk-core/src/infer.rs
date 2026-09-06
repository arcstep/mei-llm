//! Constrained decode, retrieval, validator, and confidence execution gate
//! over a packed 51M model. WASM loads only the quantized package.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::byte_grammar::ByteGrammar;
use crate::canonical::schema_fingerprint;
use crate::context_budget::{
    plan_candidate_batches, platt_relevance, render_budgeted_request, RankedCandidate,
    RETRIEVAL_BATCH_POLICY_ID, RETRIEVAL_CALIBRATION_ID,
};
use crate::error::{ErrorInfo, SdkError};
use crate::model::{ForwardOut, NeedleModel};
use crate::protocol::{apply_confidence, validate_generated_call_with_trusted_results};
use crate::tool_index::ToolIndex;
use crate::version::wire_version;
use crate::vocab::Vocab;

#[derive(Clone)]
pub struct InferRuntime {
    pub model: Arc<NeedleModel>,
    pub vocab: Arc<Vocab>,
    pub tool_index: Option<Arc<ToolIndex>>,
    pub mw_receipt_sha256: Option<String>,
    pub retrieval_scale: f64,
    pub retrieval_bias: f64,
    pub retrieval_discard_threshold: f64,
    pub retrieval_expand_threshold: f64,
    pub retrieval_calibration_validated: bool,
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
        let ordinary_capacity = crate::model::MAX_CONTEXT_TOKENS.saturating_sub(stable_len);
        if rolling_history.len() > ordinary_capacity {
            let drop = rolling_history.len() - ordinary_capacity;
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
            if rolling_history.len() > ordinary_capacity {
                rolling_history.remove(0);
            }
            cache = Some(std::mem::take(&mut out.cache));
        }
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, SdkError> {
        #[cfg(feature = "wasm-prefix-cache")]
        if let Some(embedding) = self.model.cached_retrieval_embedding(tokens) {
            return Ok(embedding);
        }
        let out = self.model.forward_features(tokens, &mut None, false)?;
        let embedding = if self.model.has_tensor("heads.contrastive.proj.weight") {
            self.model.contrastive_embedding(&out)?
        } else {
            let d = self.model.arch.d_model;
            let t = out.t.max(1);
            l2_normalize(&out.hidden[(t - 1) * d..t * d])
        };
        #[cfg(feature = "wasm-prefix-cache")]
        self.model.store_retrieval_embedding(tokens, &embedding);
        Ok(embedding)
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

    pub fn search_ranked(
        &self,
        query: &str,
        catalog: &[Value],
        scale: f64,
        bias: f64,
    ) -> Result<Vec<RankedCandidate>, SdkError> {
        let q_ids = self.vocab.encode(query, true);
        let q = self.embed(&q_ids)?;
        let mut scored = if let Some(index) = &self.tool_index {
            let catalog_sha = crate::canonical::catalog_fingerprint(catalog);
            if catalog_sha != index.catalog_sha256 {
                return Err(SdkError::new(
                    "package_hash_mismatch",
                    "request catalog does not match the frozen tool index",
                ));
            }
            index
                .ranked(&q)?
                .into_iter()
                .map(|(score, record)| (score, record.schema.clone()))
                .collect::<Vec<_>>()
        } else {
            let mut rows = Vec::with_capacity(catalog.len());
            for tool in catalog {
                let blob = serde_json::to_string(tool).unwrap_or_default();
                let ids = self.vocab.encode(&blob, true);
                let embedding = self.embed(&ids)?;
                let score = q
                    .iter()
                    .zip(embedding.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                rows.push((score, tool.clone()));
            }
            rows
        };
        scored.sort_by(|(left_score, left), (right_score, right)| {
            right_score.total_cmp(left_score).then_with(|| {
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
            })
        });
        scored
            .into_iter()
            .enumerate()
            .map(|(index, (raw_score, schema))| {
                Ok(RankedCandidate {
                    tool_id: schema
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    schema,
                    raw_score,
                    relevance: platt_relevance(raw_score, scale, bias)?,
                    rank: index + 1,
                })
            })
            .collect()
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
        let ordinary_capacity =
            crate::model::MAX_CONTEXT_TOKENS.saturating_sub(stable_prefix_tokens);
        if rolling_history.len() > ordinary_capacity {
            let keep = rolling_history.len() - ordinary_capacity;
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
        for decode_index in 0..max_new {
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
            if decode_index + 1 == max_new {
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
            if rolling_history.len() > ordinary_capacity {
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
        #[cfg(feature = "wasm-prefix-cache")]
        let cache_report = {
            let mut report = cache_report;
            report["prefix_cache"] = self.model.prefix_cache_stats();
            report
        };
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

fn complete_infer_batch(
    runtime: &InferRuntime,
    request: &Value,
    capabilities: &Value,
    trusted_tool_results: &[Value],
    wall_ms: f64,
    selected_tools: &[Value],
    rendered: &Value,
) -> Result<Value, SdkError> {
    let decode_mode = request
        .get("decode_mode")
        .and_then(Value::as_str)
        .unwrap_or("constrained");
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
        (all, stable)
    } else {
        let sink = rendered.get("sink").and_then(Value::as_str).unwrap_or("");
        let ordinary = rendered
            .get("ordinary")
            .and_then(Value::as_str)
            .unwrap_or("");
        let sink_ids = runtime.vocab.encode(sink, true);
        let ordinary_ids = runtime.vocab.encode(ordinary, false);
        let mut bounded = sink_ids.clone();
        bounded.extend_from_slice(&ordinary_ids);
        (bounded, sink_ids.len())
    };
    if ids.is_empty() || ids.len() > crate::model::MAX_CONTEXT_TOKENS {
        return Err(SdkError::new(
            "invalid_argument",
            format!(
                "bounded prompt token length {} is outside 1..=2048",
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
        selected_tools,
        max_new as usize,
        decode_mode != "raw",
    )?;
    let mut gate_request = request.clone();
    if !mw_disposition.is_null() {
        gate_request["mw_disposition"] = mw_disposition.clone();
    }
    let validated = validate_generated_call_with_trusted_results(
        &text,
        selected_tools,
        &gate_request,
        trusted_tool_results,
    );
    let conf = match conf_logit {
        Some(logit) => combine_confidence(logit, logprob, pieces.len()),
        None => decode_probability(logprob, pieces.len()),
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
        .unwrap_or_else(|| json!(schema_fingerprint(selected_tools)));
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
        "schema_budget": rendered.get("schema_budget").cloned().unwrap_or(Value::Null),
        "input_budget": rendered.get("input_budget").cloned().unwrap_or(Value::Null),
        "schema_projection_sha256": rendered.get("schema_projection_sha256").cloned().unwrap_or(Value::Null),
        "function_calls": if ok && !refuse { calls } else { json!([]) },
        "raw_text": text,
        "generated_token_ids": pieces,
        "prefill_topk_ids": prefill_topk_ids,
        "runtime_cache": cache_report,
        "compute_profile": crate::version::compute_profile(),
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

fn terminal_scan_result(
    reason: &str,
    capabilities: &Value,
    retrieval: Value,
    last: Option<&Value>,
    total_prompt_tokens: usize,
    total_output_tokens: usize,
) -> Value {
    json!({
        "wire_version": wire_version(),
        "ok": true,
        "error": Value::Null,
        "refuse": true,
        "execution": "refuse",
        "selected_tools": last.and_then(|value| value.get("selected_tools")).cloned().unwrap_or_else(|| json!([])),
        "schema_fingerprint": last.and_then(|value| value.get("schema_fingerprint")).cloned().unwrap_or(Value::Null),
        "schema_budget": last.and_then(|value| value.get("schema_budget")).cloned().unwrap_or(Value::Null),
        "input_budget": last.and_then(|value| value.get("input_budget")).cloned().unwrap_or(Value::Null),
        "schema_projection_sha256": last.and_then(|value| value.get("schema_projection_sha256")).cloned().unwrap_or(Value::Null),
        "function_calls": [],
        "raw_text": last.and_then(|value| value.get("raw_text")).cloned().unwrap_or(Value::Null),
        "generated_token_ids": last.and_then(|value| value.get("generated_token_ids")).cloned().unwrap_or_else(|| json!([])),
        "prefill_topk_ids": last.and_then(|value| value.get("prefill_topk_ids")).cloned().unwrap_or_else(|| json!([])),
        "runtime_cache": last.and_then(|value| value.get("runtime_cache")).cloned().unwrap_or(Value::Null),
        "mw_disposition": last.and_then(|value| value.get("mw_disposition")).cloned().unwrap_or_else(|| json!({"available":false})),
        "confidence": {"available":false,"value":null,"source":null},
        "provenance": {
            "validated": true,
            "ok": true,
            "detail": reason,
            "gates": [{"gate":"candidate_scan","ok":true,"terminal":reason}],
        },
        "capabilities": capabilities,
        "retrieval": retrieval,
        "stats": {
            "backend":"portable-cq2",
            "wall_ms":0.0,
            "decode_mode":"constrained",
            "prompt_tokens":total_prompt_tokens,
            "output_tokens":total_output_tokens,
        },
        "scan_terminal": reason,
    })
}

pub fn complete_infer(
    runtime: &InferRuntime,
    request: &Value,
    capabilities: &Value,
    trusted_tool_results: &[Value],
    wall_ms: f64,
) -> Result<Value, SdkError> {
    // Raw/token-id decoding is a diagnostic numerical-oracle path.  It has no
    // retrieval decision to make, so an empty catalog must not be converted
    // into the normal `retrieval_no_match` terminal.
    if request.get("decode_mode").and_then(Value::as_str) == Some("raw")
        || request.get("token_ids").is_some()
    {
        let selected_tools = request
            .get("oracle_tools")
            .or_else(|| request.get("catalog"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let rendered = json!({
            "schema_fingerprint": schema_fingerprint(&selected_tools),
            "schema_budget": Value::Null,
            "input_budget": Value::Null,
            "schema_projection_sha256": Value::Null,
        });
        return complete_infer_batch(
            runtime,
            request,
            capabilities,
            trusted_tool_results,
            wall_ms,
            &selected_tools,
            &rendered,
        );
    }
    let options = request.get("_runtime_options").unwrap_or(&Value::Null);
    let profile = options
        .get("runtime_profile")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    let discard = options
        .get("retrieval_discard_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(runtime.retrieval_discard_threshold);
    let requested_expand = options
        .get("retrieval_expand_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(runtime.retrieval_expand_threshold);
    let expand = if runtime.retrieval_calibration_validated {
        requested_expand
    } else {
        discard
    };
    let max_batches = options
        .get("max_candidate_batches")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let max_new = request
        .get("max_new")
        .and_then(Value::as_u64)
        .unwrap_or(128) as usize;
    let ranked = if let Some(oracle) = request.get("oracle_tools").and_then(Value::as_array) {
        oracle
            .iter()
            .enumerate()
            .map(|(index, schema)| RankedCandidate {
                tool_id: schema
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                schema: schema.clone(),
                raw_score: 1.0,
                relevance: 1.0,
                rank: index + 1,
            })
            .collect::<Vec<_>>()
    } else {
        let catalog = request
            .get("catalog")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| SdkError::new("invalid_argument", "tools must be a list"))?;
        let query = request.get("query").and_then(Value::as_str).unwrap_or("");
        runtime.search_ranked(
            query,
            &catalog,
            runtime.retrieval_scale,
            runtime.retrieval_bias,
        )?
    };
    let plan = plan_candidate_batches(ranked.clone(), discard, expand, None)?;
    let mut queue = plan
        .batches
        .iter()
        .flat_map(|batch| batch.iter().cloned())
        .collect::<Vec<_>>();
    let mut retrieval = json!({
        "policy_id": RETRIEVAL_BATCH_POLICY_ID,
        "calibration_id": RETRIEVAL_CALIBRATION_ID,
        "calibration_validated": runtime.retrieval_calibration_validated,
        "candidates": ranked.iter().map(RankedCandidate::to_value).collect::<Vec<_>>(),
        "scanned_batches": [],
        "scanned_tools": [],
        "remaining_candidates": [],
        "thresholds": {"discard":discard,"expand":expand},
        "non_expandable": plan.non_expandable.iter().map(RankedCandidate::to_value).collect::<Vec<_>>(),
        "discarded": plan.discarded.iter().map(RankedCandidate::to_value).collect::<Vec<_>>(),
    });
    if queue.is_empty() {
        return Ok(terminal_scan_result(
            "retrieval_no_match",
            capabilities,
            retrieval,
            None,
            0,
            0,
        ));
    }
    let mut scanned = 0usize;
    let mut total_prompt_tokens = 0usize;
    let mut total_output_tokens = 0usize;
    let mut last = None::<Value>;
    while !queue.is_empty() {
        if max_batches.is_some_and(|limit| scanned >= limit) {
            retrieval["remaining_candidates"] =
                Value::Array(queue.iter().map(RankedCandidate::to_value).collect());
            return Ok(terminal_scan_result(
                "candidate_scan_limit_reached",
                capabilities,
                retrieval,
                last.as_ref(),
                total_prompt_tokens,
                total_output_tokens,
            ));
        }
        let take = queue.len().min(5);
        let requested = queue.drain(..take).collect::<Vec<_>>();
        let tools = requested
            .iter()
            .map(|candidate| candidate.schema.clone())
            .collect::<Vec<_>>();
        let relevances = requested
            .iter()
            .map(|candidate| candidate.relevance)
            .collect::<Vec<_>>();
        let rendered = render_budgeted_request(
            &runtime.vocab,
            request,
            &tools,
            &relevances,
            profile,
            max_new,
        )?;
        let selected_names = rendered
            .get("selected_tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let selected_count = selected_names.len();
        if selected_count < requested.len() {
            let mut deferred = requested[selected_count..].to_vec();
            deferred.append(&mut queue);
            queue = deferred;
        }
        if selected_count == 0 || rendered.get("error").and_then(Value::as_str).is_some() {
            last = Some(json!({
                "selected_tools":[],
                "schema_budget":rendered.get("schema_budget"),
                "input_budget":rendered.get("input_budget"),
                "schema_projection_sha256":rendered.get("schema_projection_sha256"),
            }));
            retrieval["remaining_candidates"] =
                Value::Array(queue.iter().map(RankedCandidate::to_value).collect());
            return Ok(terminal_scan_result(
                "context_unrepresentable",
                capabilities,
                retrieval,
                last.as_ref(),
                total_prompt_tokens,
                total_output_tokens,
            ));
        }
        let validation_tools = rendered
            .get("_validation_tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut batch = complete_infer_batch(
            runtime,
            request,
            capabilities,
            trusted_tool_results,
            wall_ms,
            &validation_tools,
            &rendered,
        )?;
        scanned += 1;
        total_prompt_tokens += rendered
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        total_output_tokens += batch
            .get("generated_token_ids")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let reason_code = batch
            .pointer("/mw_disposition/reason_code")
            .and_then(Value::as_u64);
        let raw_empty = batch
            .get("raw_text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.trim() == "[]");
        let has_call = batch
            .get("function_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        let outcome = if has_call {
            "call"
        } else if raw_empty && reason_code == Some(10) {
            "capability_insufficient"
        } else if raw_empty && reason_code == Some(0) {
            "model_no_call"
        } else {
            "terminal_disposition"
        };
        retrieval["scanned_tools"]
            .as_array_mut()
            .expect("retrieval array")
            .extend(
                requested[..selected_count]
                    .iter()
                    .map(RankedCandidate::to_value),
            );
        retrieval["scanned_batches"]
            .as_array_mut()
            .expect("retrieval array")
            .push(json!({
                "batch":scanned,
                "tools":selected_names,
                "schema_budget":rendered.get("schema_budget"),
                "input_budget":rendered.get("input_budget"),
                "schema_projection_sha256":rendered.get("schema_projection_sha256"),
                "outcome":outcome,
            }));
        retrieval["remaining_candidates"] =
            Value::Array(queue.iter().map(RankedCandidate::to_value).collect());
        batch["retrieval"] = retrieval.clone();
        if let Some(stats) = batch.get_mut("stats").and_then(Value::as_object_mut) {
            stats.insert("prompt_tokens".into(), json!(total_prompt_tokens));
            stats.insert("output_tokens".into(), json!(total_output_tokens));
        }
        if has_call || outcome == "terminal_disposition" {
            return Ok(batch);
        }
        last = Some(batch);
        if outcome == "model_no_call" {
            return Ok(terminal_scan_result(
                "model_no_call",
                capabilities,
                retrieval,
                last.as_ref(),
                total_prompt_tokens,
                total_output_tokens,
            ));
        }
        if queue.is_empty() {
            return Ok(terminal_scan_result(
                "candidate_exhausted",
                capabilities,
                retrieval,
                last.as_ref(),
                total_prompt_tokens,
                total_output_tokens,
            ));
        }
    }
    Ok(terminal_scan_result(
        "candidate_exhausted",
        capabilities,
        retrieval,
        last.as_ref(),
        total_prompt_tokens,
        total_output_tokens,
    ))
}

#[cfg(test)]
mod independent_head_policy_tests {
    use super::{combine_confidence, decode_probability, mw_policy_from_probabilities};

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

    #[test]
    fn confidence_uses_mean_output_token_logprob() {
        let expected = (-1.0f32).exp();
        assert!((decode_probability(-100.0, 100) - expected).abs() < 1e-6);
        assert!((combine_confidence(10.0, -100.0, 100) - expected).abs() < 1e-4);
        assert_eq!(decode_probability(0.0, 0), 1.0);
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
        let extra = vocab.generated_token_bytes_ref(tok, generated_token_count == 0);
        if extra.is_empty() {
            return;
        }
        let mut trial = prefix.to_vec();
        trial.extend_from_slice(extra);
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
        // 死胡同回退：无合法 UTF-8 转移时终止生成（强制 EOS）。已生成的
        // 前缀交给后续 gate 校验走确定性拒绝路径，产出正常 refuse turn
        // （保留 selected_tools/confidence/mw），而不是抛协议违规丢掉整轮。
        for (i, v) in logits.iter_mut().enumerate() {
            if i as u32 != vocab.eos_id {
                *v = f32::NEG_INFINITY;
            }
        }
        return Ok(());
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

fn combine_confidence(head_logit: f32, decode_logprob: f32, output_tokens: usize) -> f32 {
    let p_head = 1.0 / (1.0 + (-head_logit).exp());
    let p_dec = decode_probability(decode_logprob, output_tokens);
    p_head.min(p_dec)
}

fn decode_probability(logprob_sum: f32, output_tokens: usize) -> f32 {
    (logprob_sum / output_tokens.max(1) as f32).min(0.0).exp()
}
