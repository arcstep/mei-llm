#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{ErrorInfo, SdkError};
use crate::infer::{complete_infer, InferRuntime};
use crate::model::{Arch, NeedleModel};
use crate::package::{package_from_manifest, ModelPackage};
use crate::packed::PackedWeights;
use crate::protocol::{leak_markers, parse_v2_text, render_request, request_leaks};
use crate::version::{max_selected_tools, wire_version};
use crate::vocab::Vocab;

pub type TurnResult = Value;
pub type LoopResult = Value;

#[derive(Clone)]
pub struct Engine {
    pub package: ModelPackage,
    pub closed: bool,
    infer: Option<InferRuntime>,
}

impl Engine {
    pub fn load(package_dir: &std::path::Path, verify_hashes: bool) -> Result<Self, SdkError> {
        let package = crate::package::load_package(package_dir, verify_hashes)?;
        let infer = if package.packed_inference_ready() {
            Some(load_infer_from_package(&package)?)
        } else {
            None
        };
        Ok(Self {
            package,
            closed: false,
            infer,
        })
    }

    /// WASM / in-memory path: quantized package only. Float weights are refused.
    pub fn from_quantized_bytes(
        manifest: Value,
        weights: Vec<u8>,
        vocab: Vec<u8>,
    ) -> Result<Self, SdkError> {
        let package = package_from_manifest(manifest, true)?;
        if !package.packed_inference_ready() {
            return Err(SdkError::new(
                "engine_unavailable",
                "WASM/portable inference loads only mei-q4-packed-v1 quantized weights",
            ));
        }
        let expected = package
            .manifest
            .get("weights")
            .and_then(|w| w.get("sha256"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        if !expected.is_empty() && crate::canonical::sha256_bytes(&weights) != expected {
            return Err(SdkError::from_id("package_hash_mismatch"));
        }
        let packed = PackedWeights::parse(weights)?;
        let arch = Arch::from_manifest(&package.manifest);
        let model = Arc::new(NeedleModel::new(arch, packed));
        let vocab = Arc::new(Vocab::from_json(&vocab)?);
        Ok(Self {
            package,
            closed: false,
            infer: Some(InferRuntime { model, vocab }),
        })
    }

    pub fn capabilities(&self) -> Value {
        let mut caps = self.package.capabilities();
        if let Some(obj) = caps.as_object_mut() {
            obj.insert("inference".into(), Value::Bool(self.infer.is_some()));
            obj.insert("protocol".into(), Value::Bool(true));
            if self.infer.is_some() {
                obj.insert("quantized_only".into(), Value::Bool(true));
                obj.insert("wasm_tier".into(), json!(1));
            }
        }
        caps
    }

    pub fn create_session(&self) -> Result<Session, SdkError> {
        if self.closed {
            return Err(SdkError::new("session_closed", "engine is closed"));
        }
        Ok(Session {
            capabilities: self.capabilities(),
            infer: self.infer.clone(),
            closed: false,
            cancelled: false,
        })
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

fn load_infer_from_package(package: &ModelPackage) -> Result<InferRuntime, SdkError> {
    if !package.packed_inference_ready() {
        return Err(SdkError::from_id("engine_unavailable"));
    }
    let weights_rel = package
        .manifest
        .get("weights")
        .and_then(|w| w.get("file"))
        .and_then(Value::as_str)
        .ok_or_else(|| SdkError::new("package_invalid", "weights.file"))?;
    let vocab_rel = package
        .manifest
        .get("tokenizer")
        .and_then(|t| t.get("vocab_file"))
        .and_then(Value::as_str)
        .unwrap_or("tokenizer.vocab.json");
    let weights = std::fs::read(package.path.join(weights_rel))
        .map_err(|_| SdkError::new("file_not_found", "weights.q4"))?;
    let vocab_bytes = std::fs::read(package.path.join(vocab_rel))
        .map_err(|_| SdkError::new("file_not_found", "tokenizer.vocab.json"))?;
    let packed = PackedWeights::parse(weights)?;
    let arch = Arch::from_manifest(&package.manifest);
    Ok(InferRuntime {
        model: Arc::new(NeedleModel::new(arch, packed)),
        vocab: Arc::new(Vocab::from_json(&vocab_bytes)?),
    })
}

#[derive(Clone)]
pub struct Session {
    capabilities: Value,
    infer: Option<InferRuntime>,
    pub closed: bool,
    pub cancelled: bool,
}

impl Session {
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn complete(&self, request: &Value) -> Result<TurnResult, SdkError> {
        if self.closed {
            return Err(SdkError::from_id("session_closed"));
        }
        if self.infer.is_some() && request.get("candidate_text").is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            let started = Instant::now();
            #[cfg(target_arch = "wasm32")]
            let wall_ms = 0.0;
            #[cfg(not(target_arch = "wasm32"))]
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            if self.cancelled {
                return Ok(complete_request(request, &self.capabilities, true));
            }
            let runtime = self.infer.as_ref().unwrap();
            return complete_infer(runtime, request, &self.capabilities, wall_ms);
        }
        Ok(complete_request(request, &self.capabilities, self.cancelled))
    }

    pub fn run(&self, request: &Value, max_turns: usize) -> Result<LoopResult, crate::SdkError> {
        let mut turns = Vec::new();
        let mut stopped = "max_turns";
        let n = max_turns.max(1);
        for _ in 0..n {
            let turn = self.complete(request)?;
            let ok = turn.get("ok").and_then(Value::as_bool).unwrap_or(false);
            let refuse = turn.get("refuse").and_then(Value::as_bool).unwrap_or(false);
            let err_id = turn
                .get("error")
                .and_then(|e| e.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let has_calls = turn
                .get("function_calls")
                .and_then(Value::as_array)
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            turns.push(turn);
            if let Some(id) = err_id {
                stopped = if id == "cancelled" { "cancelled" } else { "error" };
                break;
            }
            if refuse {
                stopped = "refuse";
                break;
            }
            if has_calls {
                stopped = "call";
                break;
            }
            let _ = ok;
        }
        let all_ok = !turns.is_empty()
            && turns
                .iter()
                .all(|t| t.get("ok").and_then(Value::as_bool).unwrap_or(false));
        Ok(json!({
            "wire_version": wire_version(),
            "ok": all_ok,
            "turns": turns,
            "stopped_reason": stopped,
        }))
    }
}

fn empty_confidence() -> Value {
    json!({"available": false, "value": null, "source": null})
}

fn turn(
    ok: bool,
    refuse: bool,
    selected: Vec<String>,
    fingerprint: Option<String>,
    calls: Vec<Value>,
    raw_text: Option<String>,
    error: Option<ErrorInfo>,
    provenance: Value,
    capabilities: &Value,
    wall_ms: f64,
    decode_mode: &str,
) -> TurnResult {
    json!({
        "wire_version": wire_version(),
        "ok": ok,
        "error": error.map(|e| e.to_value()),
        "refuse": refuse,
        "selected_tools": selected,
        "schema_fingerprint": fingerprint,
        "function_calls": calls,
        "raw_text": raw_text,
        "confidence": empty_confidence(),
        "provenance": provenance,
        "capabilities": capabilities,
        "stats": {
            "backend": "protocol",
            "wall_ms": wall_ms,
            "decode_mode": decode_mode,
        }
    })
}

pub fn complete_request(request: &Value, capabilities: &Value, cancelled: bool) -> TurnResult {
    #[cfg(not(target_arch = "wasm32"))]
    let started = Instant::now();
    let decode_mode = request
        .get("decode_mode")
        .and_then(Value::as_str)
        .unwrap_or("constrained");
    let wall = || {
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            started.elapsed().as_secs_f64() * 1000.0
        }
    };
    if cancelled {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new("cancelled", None)),
            json!({"validated": false, "ok": false, "detail": "cancelled"}),
            capabilities,
            wall(),
            decode_mode,
        );
    }
    let leaks = request_leaks(request);
    if !leaks.is_empty() {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new(
                "gold_leak",
                Some(&format!("forbidden markers: {}", leaks.join(","))),
            )),
            json!({"validated": true, "ok": false, "detail": "gold_leak"}),
            capabilities,
            wall(),
            decode_mode,
        );
    }
    let tools = request
        .get("oracle_tools")
        .and_then(Value::as_array)
        .or_else(|| request.get("catalog").and_then(Value::as_array));
    let Some(tools) = tools else {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new("invalid_argument", Some("tools must be a list"))),
            json!({"validated": false, "ok": false, "detail": "invalid_tools"}),
            capabilities,
            wall(),
            decode_mode,
        );
    };
    if tools.len() > max_selected_tools() {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new("too_many_tools", None)),
            json!({"validated": true, "ok": false, "detail": "too_many_tools"}),
            capabilities,
            wall(),
            decode_mode,
        );
    }
    let rendered = render_request(request, tools).expect("tool count already checked");
    let selected = rendered
        .get("selected_tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect::<Vec<_>>();
    let fingerprint = rendered
        .get("schema_fingerprint")
        .and_then(Value::as_str)
        .map(str::to_string);
    match request.get("candidate_text") {
        None | Some(Value::Null) => turn(
            false,
            true,
            selected,
            fingerprint,
            vec![],
            None,
            Some(ErrorInfo::new(
                "engine_unavailable",
                Some("portable inference is not in this experimental SDK; pass candidate_text for protocol validation"),
            )),
            json!({"validated": false, "ok": false, "detail": "engine_unavailable"}),
            capabilities,
            wall(),
            decode_mode,
        ),
        Some(Value::String(candidate)) => {
            if !leak_markers(candidate).is_empty() {
                return turn(
                    false,
                    true,
                    selected,
                    fingerprint,
                    vec![],
                    Some(candidate.clone()),
                    Some(ErrorInfo::new(
                        "gold_leak",
                        Some("generation contains forbidden markers"),
                    )),
                    json!({"validated": true, "ok": false, "detail": "gold_leak"}),
                    capabilities,
                    wall(),
                    decode_mode,
                );
            }
            let parsed = parse_v2_text(candidate);
            if let Some(call) = parsed.function_calls.first() {
                let name = call.get("name").and_then(Value::as_str).unwrap_or("");
                if !selected.iter().any(|s| s == name) {
                    return turn(
                        false,
                        true,
                        selected,
                        fingerprint,
                        vec![],
                        Some(candidate.clone()),
                        Some(ErrorInfo::new(
                            "protocol_violation",
                            Some(&format!("tool {name} not in selected_tools")),
                        )),
                        json!({"validated": true, "ok": false, "detail": "unknown_tool"}),
                        capabilities,
                        wall(),
                        decode_mode,
                    );
                }
            }
            let error = if parsed.ok {
                None
            } else {
                Some(ErrorInfo::new(
                    "protocol_violation",
                    Some(parsed.error.as_deref().unwrap_or("protocol")),
                ))
            };
            turn(
                parsed.ok,
                parsed.refuse,
                selected,
                fingerprint,
                if parsed.ok {
                    parsed.function_calls
                } else {
                    vec![]
                },
                Some(candidate.clone()),
                error,
                json!({
                    "validated": true,
                    "ok": parsed.ok,
                    "detail": parsed.error,
                }),
                capabilities,
                wall(),
                decode_mode,
            )
        }
        Some(_) => turn(
            false,
            true,
            selected,
            fingerprint,
            vec![],
            None,
            Some(ErrorInfo::new("invalid_json", Some("candidate_text must be a string"))),
            json!({"validated": false, "ok": false, "detail": "invalid_candidate"}),
            capabilities,
            wall(),
            decode_mode,
        ),
    }
}
