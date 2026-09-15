use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use serde_json::{json, Value};

use crate::canonical::validate_semantic_json;
use crate::context_budget::minimum_tool_projection_tokens;
use crate::error::{ErrorInfo, SdkError};
use crate::infer::{complete_infer, InferRuntime};
use crate::model::{Arch, NeedleModel};
use crate::package::{
    package_from_manifest, validate_container_directory, validate_in_memory_payload,
    validate_in_memory_tool_index, ModelPackage, PackageGeneration,
};
use crate::packed::PackedWeights;
use crate::protocol::{
    apply_confidence_with_thresholds, leak_markers, render_request, request_leaks,
    validate_generated_call_with_trusted_results, validate_tool_schema,
};
use crate::version::{max_selected_tools, wire_version};
use crate::vocab::Vocab;

pub type TurnResult = Value;
pub type LoopResult = Value;

fn is_successful_respond(turn: &Value, tool_results: &[Value]) -> bool {
    if tool_results.is_empty()
        || tool_results
            .iter()
            .any(|row| row.get("status").and_then(Value::as_str) != Some("ok"))
    {
        return false;
    }
    if turn.get("error").is_some_and(|value| !value.is_null())
        || turn
            .get("function_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
    {
        return false;
    }
    if turn
        .pointer("/provenance/gates")
        .and_then(Value::as_array)
        .is_some_and(|gates| {
            gates
                .iter()
                .any(|gate| gate.get("ok").and_then(Value::as_bool) == Some(false))
        })
    {
        return false;
    }
    turn.get("raw_text")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|value| value.as_array().cloned())
        .is_some_and(|items| items.is_empty())
}

fn trusted_call_history(
    tool_results: &[Value],
    completed_calls: &HashMap<String, Value>,
) -> Vec<Value> {
    let mut history = Vec::new();
    for result in tool_results {
        let call_id = result.get("call_id").and_then(Value::as_str).unwrap_or("");
        let Some(call) = completed_calls.get(call_id) else {
            continue;
        };
        history.push(json!({
            "role": "assistant",
            "call_id": call_id,
            "content": crate::canonical::dumps_canonical(&json!({
                "call_id": call_id,
                "name": call.get("name").and_then(Value::as_str).unwrap_or(""),
                "arguments": call.get("arguments").cloned().unwrap_or_else(|| json!({})),
            })),
        }));
    }
    history
}

fn truncate_chars(value: &str, limit: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= limit {
        return value.to_string();
    }
    chars[..limit.saturating_sub(1)]
        .iter()
        .chain(std::iter::once(&'…'))
        .collect()
}

fn compact_narration_value(value: &Value, limit: usize) -> String {
    let text = value
        .as_str()
        .map(str::trim)
        .map(str::to_string)
        .unwrap_or_else(|| crate::canonical::dumps_canonical(value));
    truncate_chars(&text, limit)
}

fn deterministic_narration(call: &Value, result: &Value) -> String {
    let tool_name = call.get("name").and_then(Value::as_str).unwrap_or("该工具");
    let arguments = call
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let payload = result.get("payload").unwrap_or(&Value::Null);
    let payload_object = payload.as_object();
    let device = arguments
        .get("device")
        .and_then(Value::as_str)
        .unwrap_or("设备");
    let zone = arguments.get("zone").and_then(Value::as_str).unwrap_or("");
    let door = arguments
        .get("door")
        .and_then(Value::as_str)
        .unwrap_or("门");
    match result.get("status").and_then(Value::as_str).unwrap_or("") {
        "error" => {
            let detail = result
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    compact_narration_value(result.get("error").unwrap_or(payload), 240)
                });
            return match tool_name {
                "start_device" => format!("{device}启动失败：{detail}。"),
                "stop_device" => format!("{device}关闭失败：{detail}。"),
                "unlock_door" => format!("{door}解锁失败：{detail}。"),
                _ => format!("{tool_name}执行未成功：{detail}。"),
            };
        }
        "cancelled" => {
            return if tool_name == "start_device" {
                format!("{device}的启动操作已取消。")
            } else {
                format!("{tool_name}已取消，未继续执行。")
            };
        }
        "ok" => {}
        _ => return format!("{tool_name}执行未成功：结果状态无效。"),
    }
    if let Some(object) = payload_object {
        let compact =
            |key: &str| compact_narration_value(object.get(key).unwrap_or(&Value::Null), 96);
        match tool_name {
            "get_temperature" if object.contains_key("temperature_c") => {
                return format!(
                    "{}温度为{}℃。",
                    if zone.is_empty() {
                        "当前".to_string()
                    } else {
                        format!("{zone}当前")
                    },
                    compact("temperature_c")
                );
            }
            "set_temperature" if object.contains_key("temperature_c") => {
                let area = if zone.is_empty() {
                    "目标区域"
                } else {
                    zone
                };
                if object.get("applied").and_then(Value::as_bool) == Some(false) {
                    return format!("{area}已经是{}℃，无需调整。", compact("temperature_c"));
                }
                return format!("已将{area}温度设为{}℃。", compact("temperature_c"));
            }
            "adjust_temperature" if object.contains_key("temperature_c") => {
                let area = if zone.is_empty() {
                    "目标区域"
                } else {
                    zone
                };
                let lower = arguments
                    .get("delta_c")
                    .and_then(Value::as_f64)
                    .is_some_and(|v| v < 0.0);
                return format!(
                    "已将{area}温度{}到{}℃。",
                    if lower { "调低" } else { "调高" },
                    compact("temperature_c")
                );
            }
            "get_humidity" if object.contains_key("humidity_percent") => {
                return format!(
                    "{}湿度为{}%。",
                    if zone.is_empty() {
                        "当前".to_string()
                    } else {
                        format!("{zone}当前")
                    },
                    compact("humidity_percent")
                );
            }
            "start_device" if object.get("state").and_then(Value::as_str) == Some("on") => {
                return format!("{device}已启动。")
            }
            "stop_device" if object.get("state").and_then(Value::as_str) == Some("off") => {
                return format!("{device}已关闭。")
            }
            "set_brightness" if object.contains_key("brightness_percent") => {
                let light = arguments
                    .get("light")
                    .and_then(Value::as_str)
                    .unwrap_or("灯光");
                return format!("已将{light}亮度调到{}%。", compact("brightness_percent"));
            }
            "lock_door" if object.get("locked").and_then(Value::as_bool) == Some(true) => {
                return format!("{door}已锁定。")
            }
            "set_fan_speed" if object.contains_key("level") => {
                return format!("已将{device}调到{}档。", compact("level"))
            }
            "create_timer" if object.contains_key("timer_id") => {
                let minutes = object
                    .get("minutes")
                    .or_else(|| arguments.get("minutes"))
                    .unwrap_or(&Value::Null);
                return format!(
                    "已设置{}分钟计时器，编号为{}。",
                    compact_narration_value(minutes, 96),
                    compact("timer_id")
                );
            }
            "cancel_timer" if object.get("cancelled").and_then(Value::as_bool) == Some(true) => {
                let timer = object
                    .get("timer_id")
                    .or_else(|| arguments.get("timer_id"))
                    .unwrap_or(&Value::Null);
                return format!("计时器{}已取消。", compact_narration_value(timer, 96));
            }
            "activate_scene" => {
                let scene = object
                    .get("scene")
                    .and_then(Value::as_str)
                    .or_else(|| arguments.get("scene").and_then(Value::as_str))
                    .unwrap_or("场景");
                if object.get("status").and_then(Value::as_str) == Some("partial") {
                    return format!(
                        "{scene}模式已部分启动：{}项完成，{}项失败。",
                        compact("completed"),
                        compact("failed")
                    );
                }
                if object.get("active").and_then(Value::as_bool) == Some(true) {
                    return format!("{scene}模式已启动。");
                }
            }
            "play_music" if object.get("playing").and_then(Value::as_bool) == Some(true) => {
                let playlist = object
                    .get("playlist")
                    .or_else(|| arguments.get("playlist"))
                    .unwrap_or(&Value::Null);
                return format!("已开始播放{}。", compact_narration_value(playlist, 96));
            }
            _ => {}
        }
    }
    if payload.is_null()
        || payload.as_array().is_some_and(Vec::is_empty)
        || payload.as_object().is_some_and(serde_json::Map::is_empty)
    {
        return format!("{tool_name}已执行完成。");
    }
    if let Some(object) = payload.as_object() {
        let mut keys = object.keys().collect::<Vec<_>>();
        keys.sort();
        let pairs = keys
            .into_iter()
            .map(|key| {
                format!(
                    "{}为{}",
                    key,
                    compact_narration_value(object.get(key).unwrap_or(&Value::Null), 96)
                )
            })
            .collect::<Vec<_>>()
            .join("；");
        return format!("{tool_name}已执行完成，{pairs}。");
    }
    format!(
        "{tool_name}已执行完成，结果为{}。",
        compact_narration_value(payload, 240)
    )
}

fn narration_prompt_v2(user_request: &str, views: &[(&Value, &Value)]) -> String {
    let calls = views
        .iter()
        .map(|(call, result)| {
            json!({
                "call_id": result.get("call_id").and_then(Value::as_str).unwrap_or(""),
                "name": call.get("name").and_then(Value::as_str).unwrap_or("该工具"),
                "arguments": call.get("arguments").cloned().unwrap_or_else(|| json!({})),
            })
        })
        .collect::<Vec<_>>();
    let results = views
        .iter()
        .map(|(_, result)| {
            let mut value = json!({
                "call_id": result.get("call_id").and_then(Value::as_str).unwrap_or(""),
                "status": result.get("status").and_then(Value::as_str).unwrap_or(""),
                "payload": result.get("payload").cloned().unwrap_or(Value::Null),
            });
            if let Some(error) = result.get("error") {
                value
                    .as_object_mut()
                    .expect("result object")
                    .insert("error".into(), error.clone());
            }
            value
        })
        .collect::<Vec<_>>();
    let public = json!({"user_request": user_request, "calls": calls, "verified_results": results});
    format!(
        "任务：只依据用户请求、已执行调用和 <verified_results> 中的已验证结果，生成不超过48个token的简洁中文结果说明。必须保留数值、单位、标识符和成功/失败极性；不得补充结果中不存在的事实，不得调用工具。\n<verified_results>{}</verified_results>\n解说：",
        crate::canonical::dumps_canonical(&public)
    )
}

// A ToolResult is meaningful only inside the Session that emitted its call.
// Process-local monotonic allocation is sufficient because Session state
// cannot be submitted across processes. It also works on wasm32 targets that
// do not provide 64-bit atomics.
static NEXT_SESSION_NONCE: AtomicU32 = AtomicU32::new(1);

fn allocate_session_nonce() -> String {
    format!(
        "s{:08x}",
        NEXT_SESSION_NONCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn retrieval_calibration(manifest: &Value) -> (f64, f64, f64, f64, bool) {
    let calibration = manifest
        .get("retrieval_calibration")
        .filter(|value| value.is_object());
    (
        calibration
            .and_then(|value| value.get("scale"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        calibration
            .and_then(|value| value.get("bias"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        calibration
            .and_then(|value| value.get("discard_threshold"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        calibration
            .and_then(|value| value.get("expand_threshold"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        calibration
            .and_then(|value| value.get("validated"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )
}

#[derive(Clone)]
pub struct Engine {
    pub package: ModelPackage,
    pub closed: bool,
    infer: Option<InferRuntime>,
    registered_tools: Vec<Value>,
}

impl Engine {
    /// Experimental numerical benchmark access; never an Agent execution path.
    #[cfg(feature = "numeric-bench")]
    pub fn diagnostic_model(&self) -> Option<&NeedleModel> {
        self.infer.as_ref().map(|r| r.model.as_ref())
    }

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
            registered_tools: vec![],
        })
    }

    /// WASM / in-memory path: quantized package only. Float weights are refused.
    pub fn from_quantized_bytes(
        manifest: Value,
        weights: Vec<u8>,
        vocab: Vec<u8>,
    ) -> Result<Self, SdkError> {
        Self::from_quantized_package_bytes(manifest, weights, vocab, None)
    }

    /// Complete in-memory package load used by WASM. Native v2 requires its
    /// immutable tool-index payload; legacy v1 remains read-only/degraded.
    pub fn from_quantized_package_bytes(
        manifest: Value,
        weights: Vec<u8>,
        vocab: Vec<u8>,
        tool_index: Option<Vec<u8>>,
    ) -> Result<Self, SdkError> {
        // The memory API receives the tensor payload and runtime vocab, but not
        // every package file. It must not claim full-package hash verification.
        let mut package = package_from_manifest(manifest, false)?;
        if !package.packed_structure_ready() {
            return Err(SdkError::new(
                "engine_unavailable",
                "portable inference requires a CQ2 v2 container or legacy q4 package",
            ));
        }
        let expected = package
            .tensor_spec()
            .and_then(|weights| weights.get("sha256"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        if expected.is_empty() || crate::canonical::sha256_bytes(&weights) != expected {
            return Err(SdkError::from_id("package_hash_mismatch"));
        }
        let expected_vocab = package
            .manifest
            .pointer("/tokenizer/vocab_sha256")
            .and_then(Value::as_str)
            .or_else(|| {
                package
                    .manifest
                    .pointer("/tokenizer/sha256")
                    .and_then(Value::as_str)
            })
            .unwrap_or("");
        if expected_vocab.is_empty() || crate::canonical::sha256_bytes(&vocab) != expected_vocab {
            return Err(SdkError::from_id("package_hash_mismatch"));
        }
        validate_in_memory_payload(&package.manifest, weights.len())?;
        let packed = PackedWeights::parse(weights)?;
        if package.generation == crate::package::PackageGeneration::V2 {
            validate_container_directory(&package.manifest, &packed)?;
            let index_bytes = tool_index.as_deref().ok_or_else(|| {
                SdkError::new(
                    "capability_missing",
                    "native v2 in-memory load requires tool-index.json",
                )
            })?;
            package.tool_index = Some(validate_in_memory_tool_index(
                &package.manifest,
                &packed,
                index_bytes,
            )?);
        } else if tool_index.is_some() {
            return Err(SdkError::new(
                "invalid_argument",
                "legacy v1 load does not accept a v2 tool index",
            ));
        }
        package.inference_payload_verified = true;
        let arch = Arch::from_manifest(&package.manifest);
        let model = Arc::new(NeedleModel::new(arch, packed));
        let vocab = Arc::new(Vocab::from_package_payload(&vocab)?);
        let native_v2 = package.generation == PackageGeneration::V2;
        let (
            retrieval_scale,
            retrieval_bias,
            retrieval_discard_threshold,
            retrieval_expand_threshold,
            retrieval_calibration_validated,
        ) = retrieval_calibration(&package.manifest);
        let runtime_index = package.tool_index.clone().map(Arc::new);
        let mw_receipt_sha256 = if native_v2 {
            package.heads.mw_disposition.training_receipt_sha256.clone()
        } else {
            None
        };
        Ok(Self {
            package,
            closed: false,
            infer: Some(InferRuntime {
                model,
                vocab,
                tool_index: runtime_index,
                mw_receipt_sha256,
                retrieval_scale,
                retrieval_bias,
                retrieval_discard_threshold,
                retrieval_expand_threshold,
                retrieval_calibration_validated,
            }),
            registered_tools: vec![],
        })
    }

    pub fn capabilities(&self) -> Value {
        let mut caps = self.package.capabilities();
        if let Some(obj) = caps.as_object_mut() {
            obj.insert("protocol".into(), Value::Bool(true));
            let loaded = self.infer.is_some();
            obj.insert(
                "inference".into(),
                Value::Bool(loaded && self.package.packed_inference_ready()),
            );
            obj.insert("backend_loaded".into(), Value::Bool(loaded));
            obj.insert("compute_profile".into(), crate::version::compute_profile());
            if cfg!(feature = "wasm-fast-kernels") || cfg!(feature = "wasm-prefix-cache") {
                obj.insert("release_eligible".into(), Value::Bool(false));
                obj.insert("inference".into(), Value::Bool(false));
                obj.insert("experimental_runtime".into(), Value::Bool(true));
            }
            obj.insert(
                "backend".into(),
                json!(if loaded { "rust-portable" } else { "protocol" }),
            );
            obj.insert("stateful_tool_loop".into(), Value::Bool(true));
            obj.insert("max_steps_default".into(), json!(4));
            obj.insert("max_steps_hard".into(), json!(8));
            obj.insert(
                "runtime_profiles".into(),
                json!({"compact":1024,"standard":1536}),
            );
            obj.insert("candidate_batch_size".into(), json!(5));
            obj.insert("candidate_batch_scan".into(), json!(true));
            if self.infer.is_some() {
                obj.insert("diagnostic_inference".into(), Value::Bool(true));
                obj.insert("quantized_only".into(), Value::Bool(true));
            }
        }
        caps
    }

    pub fn diagnose_heads(&self, text: &str) -> Result<Value, SdkError> {
        self.infer
            .as_ref()
            .ok_or_else(|| SdkError::from_id("engine_unavailable"))?
            .head_diagnostics(text)
    }

    pub fn register_tools(&mut self, tools: &[Value]) -> Result<Value, SdkError> {
        if self.closed {
            return Err(SdkError::from_id("session_closed"));
        }
        let mut names = std::collections::HashSet::new();
        for tool in tools {
            validate_semantic_json(tool)
                .map_err(|message| SdkError::new("invalid_json", message))?;
            validate_tool_schema(tool)
                .map_err(|message| SdkError::new("unsupported_schema", message))?;
            let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
            if !names.insert(name.to_string()) {
                return Err(SdkError::new(
                    "invalid_argument",
                    format!("duplicate tool {name}"),
                ));
            }
        }
        let fingerprint = crate::canonical::catalog_fingerprint(tools);
        if self
            .package
            .tool_index
            .as_ref()
            .is_some_and(|index| index.catalog_sha256 != fingerprint)
        {
            return Err(SdkError::new(
                "package_hash_mismatch",
                "registered catalog does not match the package tool index",
            ));
        }
        self.registered_tools = tools.to_vec();
        let profile_rows = tools
            .iter()
            .map(|tool| {
                let tool_id = tool.get("name").and_then(Value::as_str).unwrap_or("");
                if let Some(runtime) = self.infer.as_ref() {
                    let minimum = minimum_tool_projection_tokens(
                        &runtime.vocab,
                        tool,
                        &format!("{}\n", crate::version::task_contract()),
                    );
                    json!({
                        "tool_id":tool_id,
                        "profiles":{
                            "compact":{"eligible":minimum <= 1024,"minimum_structure_tokens":minimum},
                            "standard":{"eligible":minimum <= 1536,"minimum_structure_tokens":minimum},
                        }
                    })
                } else {
                    json!({
                        "tool_id":tool_id,
                        "profiles":{
                            "compact":{"eligible":null,"minimum_structure_tokens":null},
                            "standard":{"eligible":null,"minimum_structure_tokens":null},
                        }
                    })
                }
            })
            .collect::<Vec<_>>();
        let compact_ineligible = profile_rows
            .iter()
            .filter(|row| row.pointer("/profiles/compact/eligible") == Some(&json!(false)))
            .filter_map(|row| row.get("tool_id").and_then(Value::as_str))
            .collect::<Vec<_>>();
        let standard_ineligible = profile_rows
            .iter()
            .filter(|row| row.pointer("/profiles/standard/eligible") == Some(&json!(false)))
            .filter_map(|row| row.get("tool_id").and_then(Value::as_str))
            .collect::<Vec<_>>();
        Ok(json!({
            "registered": tools.len(),
            "catalog_fingerprint": fingerprint,
            "measured": self.infer.is_some(),
            "tools": profile_rows,
            "profile_ineligible": {
                "compact":compact_ineligible,
                "standard":standard_ineligible,
            },
        }))
    }

    pub fn create_session(&self) -> Result<Session, SdkError> {
        self.create_session_with_options(&json!({}))
    }

    pub fn create_session_with_options(&self, options: &Value) -> Result<Session, SdkError> {
        if self.closed {
            return Err(SdkError::new("session_closed", "engine is closed"));
        }
        let options = options.as_object().ok_or_else(|| {
            SdkError::new("invalid_argument", "session options must be an object")
        })?;
        if let Some(key) = options.keys().find(|key| {
            !matches!(
                key.as_str(),
                "max_steps"
                    | "max_tool_result_bytes"
                    | "runtime_profile"
                    | "retrieval_discard_threshold"
                    | "retrieval_expand_threshold"
                    | "max_candidate_batches"
            )
        }) {
            return Err(SdkError::new(
                "invalid_argument",
                format!("unknown session option {key}"),
            ));
        }
        let requested_steps = options
            .get("max_steps")
            .and_then(Value::as_u64)
            .unwrap_or(4);
        if !(1..=8).contains(&requested_steps) {
            return Err(SdkError::new(
                "invalid_argument",
                "max_steps must be between 1 and 8",
            ));
        }
        let max_steps = requested_steps as usize;
        let requested_result_bytes = options
            .get("max_tool_result_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(65_536);
        if !(1..=1_048_576).contains(&requested_result_bytes) {
            return Err(SdkError::new(
                "invalid_argument",
                "max_tool_result_bytes must be between 1 and 1048576",
            ));
        }
        let max_result_bytes = requested_result_bytes as usize;
        let runtime_profile = options
            .get("runtime_profile")
            .and_then(Value::as_str)
            .unwrap_or("standard");
        if !matches!(runtime_profile, "compact" | "standard") {
            return Err(SdkError::new(
                "invalid_argument",
                "runtime_profile must be compact or standard",
            ));
        }
        let default_discard = self
            .infer
            .as_ref()
            .map(|runtime| runtime.retrieval_discard_threshold)
            .unwrap_or(0.0);
        let default_expand = self
            .infer
            .as_ref()
            .map(|runtime| runtime.retrieval_expand_threshold)
            .unwrap_or(0.0);
        let discard_threshold = options
            .get("retrieval_discard_threshold")
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    SdkError::new(
                        "invalid_argument",
                        "retrieval_discard_threshold must be a finite number in [0,1]",
                    )
                })
            })
            .transpose()?
            .unwrap_or(default_discard);
        let expand_threshold = options
            .get("retrieval_expand_threshold")
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    SdkError::new(
                        "invalid_argument",
                        "retrieval_expand_threshold must be a finite number in [0,1]",
                    )
                })
            })
            .transpose()?
            .unwrap_or(default_expand);
        if !discard_threshold.is_finite()
            || !expand_threshold.is_finite()
            || !(0.0..=1.0).contains(&discard_threshold)
            || !(0.0..=1.0).contains(&expand_threshold)
            || expand_threshold < discard_threshold
        {
            return Err(SdkError::new(
                "invalid_argument",
                "retrieval thresholds require 0 <= discard <= expand <= 1",
            ));
        }
        let max_candidate_batches = match options.get("max_candidate_batches") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .filter(|number| *number >= 1)
                    .ok_or_else(|| {
                        SdkError::new(
                            "invalid_argument",
                            "max_candidate_batches must be >= 1 or null",
                        )
                    })?,
            ),
        };
        Ok(Session {
            session_nonce: allocate_session_nonce(),
            capabilities: self.capabilities(),
            infer: self.infer.clone(),
            registered_tools: self.registered_tools.clone(),
            closed: false,
            cancelled: false,
            pending_call: None,
            tool_results: vec![],
            completed_calls: HashMap::new(),
            responded: false,
            narration_query: String::new(),
            step: 0,
            max_steps,
            max_result_bytes,
            runtime_profile: runtime_profile.to_string(),
            retrieval_discard_threshold: discard_threshold,
            retrieval_expand_threshold: expand_threshold,
            max_candidate_batches,
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
        .tensor_file()
        .ok_or_else(|| SdkError::new("package_invalid", "tensor payload file"))?;
    let vocab_rel = package
        .manifest
        .get("tokenizer")
        .and_then(|t| t.get("vocab_file"))
        .and_then(Value::as_str)
        .or_else(|| {
            package
                .manifest
                .get("tokenizer")
                .and_then(|t| t.get("file"))
                .and_then(Value::as_str)
        })
        .unwrap_or("tokenizer.model");
    let weights = std::fs::read(package.path.join(weights_rel))
        .map_err(|_| SdkError::new("file_not_found", "weights.q4"))?;
    let vocab_bytes = std::fs::read(package.path.join(vocab_rel))
        .map_err(|_| SdkError::new("file_not_found", vocab_rel))?;
    let packed = PackedWeights::parse(weights)?;
    if package.generation == crate::package::PackageGeneration::V2 {
        validate_container_directory(&package.manifest, &packed)?;
    }
    let arch = Arch::from_manifest(&package.manifest);
    let (
        retrieval_scale,
        retrieval_bias,
        retrieval_discard_threshold,
        retrieval_expand_threshold,
        retrieval_calibration_validated,
    ) = retrieval_calibration(&package.manifest);
    Ok(InferRuntime {
        model: Arc::new(NeedleModel::new(arch, packed)),
        vocab: Arc::new(Vocab::from_package_payload(&vocab_bytes)?),
        tool_index: package.tool_index.clone().map(Arc::new),
        mw_receipt_sha256: package.heads.mw_disposition.training_receipt_sha256.clone(),
        retrieval_scale,
        retrieval_bias,
        retrieval_discard_threshold,
        retrieval_expand_threshold,
        retrieval_calibration_validated,
    })
}

#[derive(Clone)]
pub struct Session {
    session_nonce: String,
    capabilities: Value,
    infer: Option<InferRuntime>,
    registered_tools: Vec<Value>,
    pub closed: bool,
    pub cancelled: bool,
    pending_call: Option<Value>,
    tool_results: Vec<Value>,
    completed_calls: HashMap<String, Value>,
    responded: bool,
    narration_query: String,
    step: usize,
    max_steps: usize,
    max_result_bytes: usize,
    runtime_profile: String,
    retrieval_discard_threshold: f64,
    retrieval_expand_threshold: f64,
    max_candidate_batches: Option<usize>,
}

impl Session {
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn complete(&mut self, request: &Value) -> Result<TurnResult, SdkError> {
        if self.closed {
            return Err(SdkError::from_id("session_closed"));
        }
        if self.pending_call.is_some() {
            return Ok(self.error_turn("tool_result_required", None));
        }
        if self.step >= self.max_steps {
            return Ok(self.error_turn("max_steps", None));
        }
        let effective = self.effective_request(request)?;
        if self.narration_query.is_empty() {
            self.narration_query = effective
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if (effective.get("decode_mode").and_then(Value::as_str) == Some("raw")
            || effective.get("token_ids").is_some())
            && matches!(
                self.capabilities
                    .get("release_class")
                    .and_then(Value::as_str),
                Some("candidate" | "release")
            )
        {
            let mut turn = self.error_turn(
                "decode_mode_forbidden",
                Some("candidate/release sessions require constrained request rendering"),
            );
            self.attach_state(&mut turn);
            return Ok(turn);
        }
        if matches!(
            self.capabilities
                .get("release_class")
                .and_then(Value::as_str),
            Some("candidate" | "release")
        ) && (effective
            .get("candidate_text")
            .is_some_and(|value| !value.is_null())
            || effective.get("mw").is_some()
            || effective.get("mw_disposition").is_some()
            || effective.get("confidence").is_some()
            || effective.get("enforce_confidence").and_then(Value::as_bool) == Some(false))
        {
            let mut turn = self.error_turn(
                "decode_mode_forbidden",
                Some("candidate/release sessions forbid protocol-test and learned-head overrides"),
            );
            self.attach_state(&mut turn);
            return Ok(turn);
        }
        let mut turn = if self.infer.is_some() && effective.get("candidate_text").is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            let started = Instant::now();
            #[cfg(target_arch = "wasm32")]
            let wall_ms = 0.0;
            #[cfg(not(target_arch = "wasm32"))]
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            if self.cancelled {
                complete_request_with_trusted_results(
                    &effective,
                    &self.capabilities,
                    true,
                    &self.tool_results,
                )
            } else {
                let runtime = self.infer.as_ref().unwrap();
                complete_infer(
                    runtime,
                    &effective,
                    &self.capabilities,
                    &self.tool_results,
                    wall_ms,
                )?
            }
        } else {
            complete_request_with_trusted_results(
                &effective,
                &self.capabilities,
                self.cancelled,
                &self.tool_results,
            )
        };
        self.step += 1;
        let call = turn
            .get("function_calls")
            .and_then(Value::as_array)
            .and_then(|calls| calls.first())
            .cloned();
        let error = turn.get("error").filter(|value| !value.is_null()).cloned();
        let refuse = turn.get("refuse").and_then(Value::as_bool).unwrap_or(false);
        if let Some(mut call) = call {
            let seed = format!(
                "{}:{}:{}",
                self.session_nonce,
                self.step,
                crate::canonical::dumps_canonical(&call)
            );
            let digest = crate::canonical::sha256_bytes(seed.as_bytes());
            let call_id = format!(
                "call-{}-{}-{}",
                self.session_nonce,
                self.step,
                &digest[..12]
            );
            if let Some(object) = call.as_object_mut() {
                object.insert("call_id".into(), json!(call_id));
            }
            self.pending_call = Some(call.clone());
            if let Some(object) = turn.as_object_mut() {
                object.insert("kind".into(), json!("call"));
                object.insert("call".into(), call.clone());
                object.insert("function_calls".into(), json!([call]));
                object.insert("refusal".into(), Value::Null);
            }
        } else if error.is_some() {
            if let Some(object) = turn.as_object_mut() {
                object.insert("kind".into(), json!("error"));
                object.insert("call".into(), Value::Null);
                object.insert("refusal".into(), Value::Null);
            }
        } else if refuse && is_successful_respond(&turn, &self.tool_results) {
            if let Some(object) = turn.as_object_mut() {
                object.insert("kind".into(), json!("respond"));
                object.insert("ok".into(), json!(true));
                object.insert("refuse".into(), json!(false));
                object.insert("execution".into(), json!("respond"));
                object.insert("call".into(), Value::Null);
                object.insert("refusal".into(), Value::Null);
                object.insert("error".into(), Value::Null);
                object.insert("function_calls".into(), json!([]));
                self.responded = true;
            }
        } else if refuse {
            let scan_terminal = turn
                .get("scan_terminal")
                .and_then(Value::as_str)
                .map(str::to_string);
            if let Some(object) = turn.as_object_mut() {
                let reason = object
                    .get("provenance")
                    .and_then(|value| value.get("gates"))
                    .and_then(Value::as_array)
                    .and_then(|gates| {
                        gates.iter().find_map(|gate| {
                            if gate.get("ok").and_then(Value::as_bool) != Some(false) {
                                return None;
                            }
                            let name = gate.get("gate").and_then(Value::as_str).unwrap_or("");
                            let detail = gate.get("detail").and_then(Value::as_str)?;
                            Some(match (name, detail) {
                                ("provenance", "missing") => "provenance_missing",
                                _ => detail,
                            })
                        })
                    })
                    .or(scan_terminal.as_deref())
                    .unwrap_or("model_refusal")
                    .to_string();
                object.insert("kind".into(), json!("refuse"));
                object.insert("call".into(), Value::Null);
                object.insert("refusal".into(), json!({"reason": reason, "detail": null}));
            }
        } else if let Some(object) = turn.as_object_mut() {
            object.insert("kind".into(), json!("error"));
            object.insert(
                "error".into(),
                ErrorInfo::new("protocol_violation", Some("turn has no terminal variant"))
                    .to_value(),
            );
            object.insert("call".into(), Value::Null);
            object.insert("refusal".into(), Value::Null);
        }
        self.attach_state(&mut turn);
        Ok(turn)
    }

    pub fn submit_tool_result(&mut self, result: &Value) -> Result<Value, SdkError> {
        if self.closed {
            return Err(SdkError::from_id("session_closed"));
        }
        validate_semantic_json(result).map_err(|message| SdkError::new("invalid_json", message))?;
        let object = result
            .as_object()
            .ok_or_else(|| SdkError::new("invalid_argument", "ToolResultV2 must be an object"))?;
        if let Some(key) = object.keys().find(|key| {
            ![
                "wire_version",
                "call_id",
                "status",
                "payload",
                "error",
                "provenance",
            ]
            .contains(&key.as_str())
        }) {
            return Err(SdkError::new(
                "invalid_argument",
                format!("unknown ToolResultV2 field {key}"),
            ));
        }
        if !object.contains_key("payload") {
            return Err(SdkError::new(
                "invalid_argument",
                "ToolResultV2.payload is required",
            ));
        }
        let Some(pending) = self.pending_call.as_ref() else {
            return Err(SdkError::new("stale_call_id", "no pending call"));
        };
        let expected = pending.get("call_id").and_then(Value::as_str).unwrap_or("");
        let actual = result.get("call_id").and_then(Value::as_str).unwrap_or("");
        if actual.is_empty() || actual != expected {
            return Err(SdkError::new(
                "stale_call_id",
                format!("expected {expected}, got {actual}"),
            ));
        }
        if result.get("wire_version").and_then(Value::as_str) != Some("mei-runtime-wire-v2") {
            return Err(SdkError::new(
                "invalid_argument",
                "ToolResultV2 wire_version is required",
            ));
        }
        if !matches!(
            result.get("status").and_then(Value::as_str),
            Some("ok" | "error" | "cancelled")
        ) {
            return Err(SdkError::new(
                "invalid_argument",
                "invalid tool result status",
            ));
        }
        let status = result.get("status").and_then(Value::as_str).unwrap_or("");
        match (status, result.get("error")) {
            ("error", Some(Value::Object(error)))
                if error.get("code").and_then(Value::as_str).is_some()
                    && error.get("message").and_then(Value::as_str).is_some()
                    && error
                        .keys()
                        .all(|key| matches!(key.as_str(), "code" | "message")) => {}
            ("error", _) => {
                return Err(SdkError::new(
                    "invalid_argument",
                    "error status requires {code,message}",
                ))
            }
            ("ok", Some(value)) if !value.is_null() => {
                return Err(SdkError::new(
                    "invalid_argument",
                    "ok status cannot carry error",
                ))
            }
            _ => {}
        }
        let provenance = result
            .get("provenance")
            .and_then(Value::as_object)
            .ok_or_else(|| SdkError::new("protocol_violation", "provenance object is required"))?;
        if provenance
            .keys()
            .any(|key| !matches!(key.as_str(), "source" | "verified" | "receipt_sha256"))
            || provenance
                .get("receipt_sha256")
                .and_then(Value::as_str)
                .is_some_and(|value| {
                    value.len() != 64
                        || value
                            .bytes()
                            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
                })
        {
            return Err(SdkError::new(
                "protocol_violation",
                "invalid provenance fields",
            ));
        }
        let verified = result
            .pointer("/provenance/verified")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let source = result
            .pointer("/provenance/source")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !verified || source.is_empty() {
            return Err(SdkError::new(
                "protocol_violation",
                "tool result provenance must be verified",
            ));
        }
        let bytes = serde_json::to_vec(result)
            .map_err(|error| SdkError::new("invalid_json", error.to_string()))?;
        if bytes.len() > self.max_result_bytes {
            return Err(SdkError::new(
                "tool_result_too_large",
                format!("{} > {} bytes", bytes.len(), self.max_result_bytes),
            ));
        }
        if result.get("status").and_then(Value::as_str) == Some("cancelled") {
            self.cancelled = true;
        }
        self.tool_results.push(result.clone());
        self.completed_calls
            .insert(actual.to_string(), pending.clone());
        self.pending_call = None;
        Ok(json!({
            "wire_version": wire_version(),
            "accepted": true,
            "call_id": actual,
            "state": self.state_value(),
        }))
    }

    pub fn narrate(&self, options: &Value) -> Result<Value, SdkError> {
        if self.closed {
            return Err(SdkError::from_id("session_closed"));
        }
        let terminal_tool_result = !self.tool_results.is_empty()
            && self.tool_results.iter().any(|row| {
                matches!(
                    row.get("status").and_then(Value::as_str),
                    Some("error" | "cancelled")
                )
            });
        if !self.responded && !terminal_tool_result {
            return Err(SdkError::new(
                "protocol_violation",
                "narration_requires_respond",
            ));
        }
        validate_semantic_json(options)
            .map_err(|message| SdkError::new("invalid_json", message))?;
        let object = options.as_object().ok_or_else(|| {
            SdkError::new("invalid_argument", "narration options must be an object")
        })?;
        if let Some(key) = object
            .keys()
            .find(|key| !matches!(key.as_str(), "mode" | "call_ids" | "locale"))
        {
            return Err(SdkError::new(
                "invalid_argument",
                format!("unknown narration option {key}"),
            ));
        }
        let requested_mode = object
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("deterministic");
        if !matches!(requested_mode, "off" | "deterministic" | "adapter") {
            return Err(SdkError::new(
                "invalid_argument",
                "narration mode must be off|deterministic|adapter",
            ));
        }
        let locale = object
            .get("locale")
            .and_then(Value::as_str)
            .unwrap_or("zh-CN");
        if locale != "zh-CN" {
            return Err(SdkError::new(
                "invalid_argument",
                "only zh-CN narration is frozen",
            ));
        }
        let selected = match object.get("call_ids") {
            None => None,
            Some(Value::Array(values)) if values.iter().all(Value::is_string) => Some(
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<HashSet<_>>(),
            ),
            Some(_) => {
                return Err(SdkError::new(
                    "invalid_argument",
                    "narration call_ids must be a string array",
                ))
            }
        };
        let mut texts = Vec::new();
        let mut call_ids = Vec::new();
        let mut views = Vec::<(&Value, &Value)>::new();
        for result in &self.tool_results {
            let call_id = result.get("call_id").and_then(Value::as_str).unwrap_or("");
            if selected.as_ref().is_some_and(|ids| !ids.contains(call_id)) {
                continue;
            }
            let call = self.completed_calls.get(call_id).ok_or_else(|| {
                SdkError::new("protocol_violation", "narration_requires_session_result")
            })?;
            texts.push(deterministic_narration(call, result));
            views.push((call, result));
            call_ids.push(call_id.to_string());
        }
        if requested_mode != "off" && texts.is_empty() {
            return Err(SdkError::new(
                "protocol_violation",
                "narration_requires_terminal_result",
            ));
        }
        let deterministic_text = texts.join("\n");
        let adapter_ready = self.infer.as_ref().is_some_and(|runtime| {
            runtime
                .model
                .has_tensor("heads.narration_adapter.down.weight")
                && runtime
                    .model
                    .has_tensor("heads.narration_adapter.up.weight")
        });
        let adapter_text = if requested_mode == "adapter" && adapter_ready {
            let prompt = narration_prompt_v2(&self.narration_query, &views);
            Some(
                self.infer
                    .as_ref()
                    .expect("adapter runtime")
                    .generate_narration(&prompt, 48)?,
            )
        } else {
            None
        };
        let adapter_verified = adapter_text.as_deref() == Some(deterministic_text.as_str());
        let fallback_used = requested_mode == "adapter" && !adapter_verified;
        let mode = if adapter_verified {
            "adapter"
        } else if fallback_used {
            "deterministic"
        } else {
            requested_mode
        };
        let text = if mode == "off" {
            Value::Null
        } else if adapter_verified {
            json!(adapter_text.unwrap_or_default())
        } else {
            json!(deterministic_text)
        };
        Ok(json!({
            "wire_version": wire_version(),
            "mode": mode,
            "requested_mode": requested_mode,
            "locale": locale,
            "text": text,
            "grounded": true,
            "fallback_used": fallback_used,
            "adapter_verified": adapter_verified,
            "provider_id": if adapter_verified { "mei-zh-narration-adapter-r16-v2" } else { "mei-zh-deterministic-narration-v1" },
            "call_ids": call_ids,
        }))
    }

    pub fn run<F>(&mut self, request: &Value, mut executor: F) -> Result<LoopResult, SdkError>
    where
        F: FnMut(&str, &Value) -> Result<Value, SdkError>,
    {
        let mut turns = Vec::new();
        let mut executed = Vec::new();
        let mut stopped = "max_steps";
        while self.step < self.max_steps {
            let turn = self.complete(request)?;
            let kind = turn.get("kind").and_then(Value::as_str).unwrap_or("error");
            turns.push(turn.clone());
            if kind == "error" {
                let id = turn
                    .pointer("/error/id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                stopped = if id == "cancelled" {
                    "cancelled"
                } else {
                    "error"
                };
                break;
            }
            if kind == "refuse" {
                stopped = "refuse";
                break;
            }
            if kind == "respond" {
                stopped = "respond";
                break;
            }
            let call = turn.get("call").cloned().unwrap_or(Value::Null);
            let name = call.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = call.get("arguments").unwrap_or(&Value::Null);
            let call_id = call.get("call_id").and_then(Value::as_str).unwrap_or("");
            let payload = match executor(name, arguments) {
                Ok(payload) => json!({
                    "wire_version": wire_version(), "call_id": call_id, "status": "ok",
                    "payload": payload,
                    "provenance": {"source": format!("host-executor:{name}"), "verified": true}
                }),
                Err(error) => json!({
                    "wire_version": wire_version(), "call_id": call_id, "status": "error",
                    "payload": null, "error": {"code": error.info().id, "message": error.info().message},
                    "provenance": {"source": format!("host-executor:{name}"), "verified": true}
                }),
            };
            let executor_failed = payload.get("status").and_then(Value::as_str) != Some("ok");
            self.submit_tool_result(&payload)?;
            executed.push(payload);
            if executor_failed {
                stopped = if self.cancelled { "cancelled" } else { "error" };
                break;
            }
            if self.cancelled {
                stopped = "cancelled";
                break;
            }
        }
        let all_ok = stopped == "respond"
            && !turns.is_empty()
            && turns
                .iter()
                .all(|turn| turn.get("kind") != Some(&json!("error")));
        Ok(json!({
            "wire_version": wire_version(),
            "ok": all_ok,
            "turns": turns,
            "tool_results": executed,
            "stopped_reason": stopped,
        }))
    }

    fn effective_request(&self, request: &Value) -> Result<Value, SdkError> {
        validate_semantic_json(request)
            .map_err(|message| SdkError::new("invalid_json", message))?;
        let mut effective = request.clone();
        let object = effective
            .as_object_mut()
            .ok_or_else(|| SdkError::new("invalid_json", "request must be an object"))?;
        let native_v2 = self
            .capabilities
            .get("compatibility_mode")
            .and_then(Value::as_str)
            == Some("v2-native");
        if native_v2 && !object.contains_key("wire_version") {
            return Err(SdkError::new(
                "invalid_argument",
                "CompleteRequestV2.wire_version is required",
            ));
        }
        if !native_v2 && !object.contains_key("wire_version") {
            object.insert("wire_version".into(), json!("mei-runtime-wire-v1"));
        }
        let wire = object
            .get("wire_version")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !matches!(wire, "mei-runtime-wire-v1" | "mei-runtime-wire-v2") {
            return Err(SdkError::new(
                "invalid_argument",
                "unsupported wire_version",
            ));
        }
        if native_v2 && wire != "mei-runtime-wire-v2" {
            return Err(SdkError::new(
                "abi_version_mismatch",
                "native v2 sessions cannot be downgraded to wire v1",
            ));
        }
        if wire == "mei-runtime-wire-v2" {
            validate_complete_request_v2(object)?;
        }
        if !object.contains_key("oracle_tools")
            && !object.contains_key("catalog")
            && !self.registered_tools.is_empty()
        {
            object.insert(
                "catalog".into(),
                Value::Array(self.registered_tools.clone()),
            );
        }
        if !self.tool_results.is_empty() {
            let mut history = object
                .get("history")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            history.extend(trusted_call_history(
                &self.tool_results,
                &self.completed_calls,
            ));
            object.insert("history".into(), Value::Array(history));
            object.insert(
                "tool_results".into(),
                Value::Array(self.tool_results.clone()),
            );
        }
        object.insert(
            "_runtime_options".into(),
            json!({
                "runtime_profile": self.runtime_profile,
                "retrieval_discard_threshold": self.retrieval_discard_threshold,
                "retrieval_expand_threshold": self.retrieval_expand_threshold,
                "max_candidate_batches": self.max_candidate_batches,
            }),
        );
        Ok(effective)
    }

    fn state_value(&self) -> Value {
        json!({
            "step": self.step,
            "pending_call_id": self.pending_call.as_ref().and_then(|call| call.get("call_id")).cloned(),
            "cancelled": self.cancelled,
        })
    }

    fn attach_state(&self, turn: &mut Value) {
        if let Some(object) = turn.as_object_mut() {
            object.insert("state".into(), self.state_value());
        }
    }

    fn error_turn(&self, id: &str, message: Option<&str>) -> Value {
        let mut result = json!({
            "wire_version": wire_version(), "kind": "error", "ok": false, "refuse": true,
            "call": null, "refusal": null, "function_calls": [], "selected_tools": [],
            "schema_fingerprint": null, "raw_text": null,
            "error": ErrorInfo::new(id, message).to_value(),
            "confidence": empty_confidence(),
            "provenance": {"validated": false, "ok": false, "detail": id},
            "capabilities": self.capabilities,
            "stats": {"backend": "state-machine", "wall_ms": 0.0, "decode_mode": null}
        });
        self.attach_state(&mut result);
        result
    }
}

fn validate_complete_request_v2(object: &serde_json::Map<String, Value>) -> Result<(), SdkError> {
    const ALLOWED: &[&str] = &[
        "wire_version",
        "query",
        "context",
        "evidence",
        "history",
        "tool_results",
        "permissions",
        "state",
        "mw",
        "mw_disposition",
        "confidence",
        "enforce_confidence",
        "oracle_tools",
        "candidate_text",
        "decode_mode",
        "max_new",
        "token_ids",
    ];
    if let Some(key) = object.keys().find(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(SdkError::new(
            "invalid_argument",
            format!("unknown CompleteRequestV2 field {key}"),
        ));
    }
    if object
        .get("query")
        .and_then(Value::as_str)
        .filter(|query| !query.is_empty())
        .is_none()
    {
        return Err(SdkError::new(
            "invalid_argument",
            "CompleteRequestV2.query must be a non-empty string",
        ));
    }
    if object.contains_key("mw") && object.contains_key("mw_disposition") {
        return Err(SdkError::new(
            "invalid_argument",
            "mw and mw_disposition are mutually exclusive",
        ));
    }
    if let Some(max_new) = object.get("max_new") {
        if !max_new
            .as_u64()
            .is_some_and(|value| (1..=128).contains(&value))
        {
            return Err(SdkError::new(
                "invalid_argument",
                "max_new must be between 1 and 128",
            ));
        }
    }
    if let Some(mode) = object.get("decode_mode") {
        if !matches!(mode.as_str(), Some("raw" | "constrained")) {
            return Err(SdkError::new("invalid_argument", "invalid decode_mode"));
        }
    }
    if let Some(token_ids) = object.get("token_ids") {
        let valid = token_ids.as_array().is_some_and(|values| {
            !values.is_empty()
                && values.len() <= 2_048
                && values.iter().all(|value| value.as_u64().is_some())
        });
        if !valid {
            return Err(SdkError::new(
                "invalid_argument",
                "token_ids must be a non-empty integer array of at most 2048 entries",
            ));
        }
    }
    if let Some(candidate) = object.get("candidate_text") {
        if !candidate.is_null() && !candidate.is_string() {
            return Err(SdkError::new(
                "invalid_argument",
                "candidate_text must be string or null",
            ));
        }
    }
    if let Some(tools) = object.get("oracle_tools") {
        if !tools.is_null()
            && !tools
                .as_array()
                .is_some_and(|items| items.len() <= max_selected_tools())
        {
            return Err(SdkError::new(
                "too_many_tools",
                "oracle_tools accepts at most 5 entries",
            ));
        }
    }
    for key in [
        "context",
        "permissions",
        "state",
        "mw",
        "mw_disposition",
        "confidence",
    ] {
        if object.get(key).is_some_and(|value| !value.is_object()) {
            return Err(SdkError::new(
                "invalid_argument",
                format!("{key} must be an object"),
            ));
        }
    }
    for key in ["evidence", "history", "tool_results"] {
        if object.get(key).is_some_and(|value| !value.is_array()) {
            return Err(SdkError::new(
                "invalid_argument",
                format!("{key} must be an array"),
            ));
        }
    }
    if object
        .get("enforce_confidence")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err(SdkError::new(
            "invalid_argument",
            "enforce_confidence must be boolean",
        ));
    }
    Ok(())
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
    complete_request_with_trusted_results(request, capabilities, cancelled, &[])
}

fn complete_request_with_trusted_results(
    request: &Value,
    capabilities: &Value,
    cancelled: bool,
    trusted_tool_results: &[Value],
) -> TurnResult {
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
    if (decode_mode == "raw" || request.get("token_ids").is_some())
        && matches!(
            capabilities.get("release_class").and_then(Value::as_str),
            Some("candidate" | "release")
        )
    {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new(
                "decode_mode_forbidden",
                Some("candidate/release sessions require constrained request rendering"),
            )),
            json!({"validated": false, "ok": false, "detail": "decode_mode_forbidden"}),
            capabilities,
            wall(),
            decode_mode,
        );
    }
    if matches!(
        capabilities.get("release_class").and_then(Value::as_str),
        Some("candidate" | "release")
    ) && (request
        .get("candidate_text")
        .is_some_and(|value| !value.is_null())
        || request.get("mw").is_some()
        || request.get("mw_disposition").is_some()
        || request.get("confidence").is_some()
        || request.get("enforce_confidence").and_then(Value::as_bool) == Some(false))
    {
        return turn(
            false,
            true,
            vec![],
            None,
            vec![],
            None,
            Some(ErrorInfo::new(
                "decode_mode_forbidden",
                Some("candidate/release sessions forbid protocol-test and learned-head overrides"),
            )),
            json!({"validated": false, "ok": false, "detail": "decode_mode_forbidden"}),
            capabilities,
            wall(),
            decode_mode,
        );
    }
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
            Some(ErrorInfo::new(
                "invalid_argument",
                Some("tools must be a list"),
            )),
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
            let (confidence, enforce_confidence, execute_high, escalate_low) =
                protocol_confidence(request);
            let validated = apply_confidence_with_thresholds(
                validate_generated_call_with_trusted_results(
                    candidate,
                    tools,
                    request,
                    trusted_tool_results,
                ),
                confidence,
                enforce_confidence,
                execute_high,
                escalate_low,
            );
            let validation_error = validated.get("error").and_then(Value::as_str);
            let deterministic_refusal = validation_error
                .map(|error| {
                    error == "provenance_missing"
                        || error.starts_with("permission_")
                        || error.starts_with("state_")
                        || error.starts_with("mw_")
                        || error.starts_with("confidence_")
                })
                .unwrap_or(false);
            let error = validation_error
                .filter(|_| !deterministic_refusal)
                .map(|detail| ErrorInfo::new("protocol_violation", Some(detail)));
            let calls = if error.is_none() && !deterministic_refusal {
                validated
                    .get("function_calls")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            } else {
                vec![]
            };
            let mut gates = vec![json!({
                "gate": "retrieval",
                "ok": true,
                "selected_tools": selected,
            })];
            gates.extend(
                validated
                    .get("gates")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            );
            let mut provenance = json!({
                "validated": true,
                "ok": validated.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "arguments": validated.get("provenance").cloned().unwrap_or_else(|| json!({})),
                "gates": gates,
            });
            if request.get("wire_version").and_then(Value::as_str) != Some("mei-runtime-wire-v2") {
                provenance
                    .as_object_mut()
                    .expect("provenance object")
                    .insert(
                        "request_compatibility".into(),
                        json!({
                            "source_wire": "mei-runtime-wire-v1",
                            "mode": "read_only_degraded_adapter",
                            "capability_complete": false,
                        }),
                    );
            }
            let mut result = turn(
                error.is_none(),
                validated.get("refuse").and_then(Value::as_bool).unwrap_or(true)
                    || deterministic_refusal,
                selected.clone(),
                fingerprint,
                calls,
                Some(candidate.clone()),
                error,
                provenance,
                capabilities,
                wall(),
                decode_mode,
            );
            if let Some(object) = result.as_object_mut() {
                if let Some(value) = validated.get("confidence_value").and_then(Value::as_f64) {
                    object.insert(
                        "confidence".into(),
                        json!({"available":true,"value":value,"source":"protocol-test"}),
                    );
                }
            }
            result
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

fn protocol_confidence(request: &Value) -> (Option<f64>, bool, f64, f64) {
    let wire_v2 =
        request.get("wire_version").and_then(Value::as_str) == Some("mei-runtime-wire-v2");
    let mut valid = true;
    let enforce = match request.get("enforce_confidence") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            valid = false;
            true
        }
    };
    let mut confidence = None;
    let mut execute_high = 0.70;
    let mut escalate_low = 0.35;
    if let Some(value) = request.get("confidence") {
        if wire_v2 {
            if let Some(object) = value.as_object() {
                valid &= object.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "value" | "execute_high" | "escalate_low" | "source"
                    )
                });
                valid &= object.get("source").and_then(Value::as_str) == Some("protocol-test");
                confidence = object.get("value").and_then(Value::as_f64);
                valid &= confidence.is_some();
                if let Some(value) = object.get("execute_high") {
                    execute_high = value.as_f64().unwrap_or(f64::NAN);
                }
                if let Some(value) = object.get("escalate_low") {
                    escalate_low = value.as_f64().unwrap_or(f64::NAN);
                }
            } else {
                valid = false;
            }
        } else {
            confidence = value
                .as_f64()
                .or_else(|| value.get("value").and_then(Value::as_f64));
            valid &= confidence.is_some();
            if let Some(value) = value.get("execute_high") {
                execute_high = value.as_f64().unwrap_or(f64::NAN);
            }
            if let Some(value) = value.get("escalate_low") {
                escalate_low = value.as_f64().unwrap_or(f64::NAN);
            }
        }
    }
    if !valid {
        execute_high = f64::NAN;
    }
    (confidence, enforce, execute_high, escalate_low)
}

#[cfg(test)]
mod tests {
    use super::{deterministic_narration, narration_prompt_v2, trusted_call_history};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn trusted_result_is_paired_with_its_session_call() {
        let call_id = "call-s00000001-1-0123456789ab";
        let mut calls = HashMap::new();
        calls.insert(
            call_id.to_string(),
            json!({
                "call_id": call_id,
                "name": "book_flight",
                "arguments": {"date":"2026-09-01","from_city":"北京","to_city":"上海"}
            }),
        );
        let history = trusted_call_history(
            &[json!({"call_id":call_id,"status":"ok","payload":{"pnr":"PNR-7"}})],
            &calls,
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["role"], "assistant");
        assert_eq!(history[0]["call_id"], call_id);
        assert_eq!(
            history[0]["content"],
            concat!(
                "{\"arguments\":{\"date\":\"2026-09-01\",\"from_city\":\"北京\",",
                "\"to_city\":\"上海\"},\"call_id\":\"call-s00000001-1-0123456789ab\",",
                "\"name\":\"book_flight\"}"
            )
        );
    }

    #[test]
    fn chinese_narration_covers_query_action_and_failure() {
        let temperature = json!({
            "call_id":"call-s00000001-1-0123456789ab",
            "name":"get_temperature", "arguments":{"zone":"客厅"}
        });
        let temperature_result = json!({
            "call_id":"call-s00000001-1-0123456789ab", "status":"ok",
            "payload":{"temperature_c":26}
        });
        assert_eq!(
            deterministic_narration(&temperature, &temperature_result),
            "客厅当前温度为26℃。"
        );
        let start = json!({
            "call_id":"call-s00000001-2-0123456789ab",
            "name":"start_device", "arguments":{"device":"空调"}
        });
        let failed = json!({
            "call_id":"call-s00000001-2-0123456789ab", "status":"error",
            "payload":null, "error":{"code":"offline","message":"设备离线"}
        });
        assert_eq!(
            deterministic_narration(&start, &failed),
            "空调启动失败：设备离线。"
        );
        let prompt =
            narration_prompt_v2("客厅现在多少度？", &[(&temperature, &temperature_result)]);
        assert!(prompt.contains("客厅现在多少度？"));
        assert!(prompt.contains("temperature_c"));
        assert!(prompt.ends_with("解说："));
    }
}
