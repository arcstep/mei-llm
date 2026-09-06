use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::canonical::{dumps_canonical, sha256_bytes};
use crate::error::SdkError;
use crate::packed::PackedWeights;
use crate::tool_index::ToolIndex;
use crate::version::sdk_versions;

const REQUIRED_HEADS: [&str; 4] = ["lm", "contrastive", "mw_disposition", "confidence"];
const ALL_HEADS: [&str; 5] = [
    "lm",
    "contrastive",
    "mw_disposition",
    "confidence",
    "narration_adapter",
];
const V1: &str = "mei-model-package-v1";
const V2: &str = "mei-model-package-v2";
const CQ2_FORMAT: &str = "mei-cq-tensor-v2";
const CQ2_MATH: &str = "mei-cq-v2-g128-wht-codebook";
const WEIGHT_CONTRACT_SHA256: &str =
    "c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b";
const LEGACY_RUNTIME_PROFILE_SHA256: &str =
    "74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43";
const PREVIOUS_ADAPTIVE_RUNTIME_PROFILE_SHA256: &str =
    "7d2d97b2f4fabc5e638f29a64789298ac8b2ebb3484b03d94925a9f08d3e2611";
const RUNTIME_PROFILE_SHA256: &str =
    "f3a4ab1151e82299fee0214c5f78a18c20d83367d24a9dc073bbcd56312fa512";
const TRAINING_AUX_SHA256: &str =
    "83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac";
// Rust/WASM implements the v2 container, frozen-index retrieval, all three
// independent sidecars, UTF-8 schema-constrained decode and bounded int8
// activation/KV execution. Resource and release eligibility remain separate,
// receipt-backed fail-closed gates.
const PORTABLE_V2_RUNTIME_COMPLETE: bool = true;
const MW_LABEL_CODEBOOK_SHA256: &str =
    "913d2c4bca8a796c9baddb9a79539cc703aa6420af4260be2c5ca0e0d5a68d40";
const CONTRASTIVE_TENSORS: [(&str, &[u64]); 3] = [
    ("heads.contrastive.tok_probes", &[4, 512]),
    ("heads.contrastive.lay_probes", &[4, 512]),
    ("heads.contrastive.proj.weight", &[128, 2048]),
];
const MW_DISPOSITION_TENSORS: [(&str, &[u64]); 2] = [
    ("heads.mw_disposition.proj.weight", &[20, 512]),
    ("heads.mw_disposition.proj.bias", &[20]),
];
const CONFIDENCE_TENSORS: [(&str, &[u64]); 3] = [
    ("heads.confidence.cell_probes", &[8, 512]),
    ("heads.confidence.proj.weight", &[1, 4096]),
    ("heads.confidence.proj.bias", &[1]),
];
const NARRATION_ADAPTER_TENSORS: [(&str, &[u64]); 2] = [
    ("heads.narration_adapter.down.weight", &[16, 512]),
    ("heads.narration_adapter.up.weight", &[24_000, 16]),
];

fn canonical_runtime_quantization() -> Value {
    json!({
        "weight_math_id": CQ2_MATH,
        "activation_semantics_id": "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2",
        "activation_sites": ["engram_input", "attention_input", "attention_output", "lm_head_input"],
        "activation_q_quantized": false,
        "activation_reconstruction_dtype": "f32",
        "kv_semantics_id": "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1",
        "kv_sites": ["attention_key_after_rope", "attention_value"],
        "kv_storage_dtype": "int8",
        "kv_scale_granularity": "per-head-vector",
        "int8_code_min": -128,
        "int8_code_max": 127,
        "scale_denominator": 127
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageGeneration {
    V1ReadOnly,
    V2,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeadStatus {
    pub present: bool,
    pub trained: bool,
    pub status: String,
    #[serde(default)]
    pub tensor_prefixes: Vec<String>,
    #[serde(default)]
    pub training_receipt_sha256: Option<String>,
    #[serde(default)]
    pub label_codebook_sha256: Option<String>,
}

impl HeadStatus {
    pub fn to_value(&self) -> Value {
        json!({
            "present": self.present,
            "trained": self.trained,
            "status": self.status,
            "tensor_prefixes": self.tensor_prefixes,
            "training_receipt_sha256": self.training_receipt_sha256,
            "label_codebook_sha256": self.label_codebook_sha256,
        })
    }

    pub fn is_missing_or_untrained(&self) -> bool {
        self.status != "ready" || !self.present || !self.trained
    }
}

#[derive(Debug, Clone)]
pub struct HeadReport {
    pub lm: HeadStatus,
    pub contrastive: HeadStatus,
    pub mw_disposition: HeadStatus,
    pub confidence: HeadStatus,
    pub narration_adapter: HeadStatus,
}

impl HeadReport {
    pub fn to_value(&self) -> Value {
        json!({
            "lm": self.lm.to_value(),
            "contrastive": self.contrastive.to_value(),
            "mw_disposition": self.mw_disposition.to_value(),
            "confidence": self.confidence.to_value(),
            "narration_adapter": self.narration_adapter.to_value(),
        })
    }

    pub fn missing(&self) -> Vec<String> {
        [
            ("lm", &self.lm),
            ("contrastive", &self.contrastive),
            ("mw_disposition", &self.mw_disposition),
            ("confidence", &self.confidence),
        ]
        .into_iter()
        .filter(|(_, head)| head.is_missing_or_untrained())
        .map(|(name, _)| name.to_string())
        .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ModelPackage {
    pub path: PathBuf,
    pub manifest: Value,
    pub heads: HeadReport,
    pub verified_hashes: bool,
    pub inference_payload_verified: bool,
    pub training_receipts_verified: bool,
    pub resource_measurement_verified: bool,
    pub tool_index: Option<ToolIndex>,
    pub generation: PackageGeneration,
}

impl ModelPackage {
    pub fn package_id(&self) -> String {
        self.manifest
            .get("package_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    pub fn tensor_spec(&self) -> Option<&Value> {
        match self.generation {
            PackageGeneration::V1ReadOnly => self.manifest.get("weights"),
            PackageGeneration::V2 => self.manifest.get("tensor_container"),
        }
    }

    pub fn tensor_file(&self) -> Option<&str> {
        self.tensor_spec()?.get("file")?.as_str()
    }

    pub fn packed_structure_ready(&self) -> bool {
        let Some(spec) = self.tensor_spec() else {
            return false;
        };
        let format = spec.get("format").and_then(Value::as_str);
        match self.generation {
            PackageGeneration::V1ReadOnly => {
                let scheme = spec
                    .get("quantization")
                    .and_then(|q| q.get("scheme"))
                    .and_then(Value::as_str);
                format == Some("mei-q4-packed-v1")
                    && matches!(
                        scheme,
                        Some("q4") | Some("cq2") | Some("qat-q4") | Some("qat-cq2")
                    )
            }
            PackageGeneration::V2 => {
                format == Some(CQ2_FORMAT)
                    && spec.get("quant_math_id").and_then(Value::as_str) == Some(CQ2_MATH)
                    && self.tensor_identity_complete()
                    && self.portable_quantization_policy_complete()
                    && self.heads.missing().is_empty()
                    && [
                        "retrieval",
                        "full_call",
                        "mw_disposition",
                        "confidence",
                        "multi_step",
                    ]
                    .into_iter()
                    .all(|key| {
                        self.manifest
                            .pointer(&format!("/capabilities/{key}"))
                            .and_then(Value::as_bool)
                            == Some(true)
                    })
            }
        }
    }

    pub fn packed_inference_ready(&self) -> bool {
        self.inference_payload_verified
            && (self.generation == PackageGeneration::V1ReadOnly || self.training_receipts_verified)
            && (self.generation == PackageGeneration::V1ReadOnly || self.tool_index.is_some())
            && self.packed_structure_ready()
    }

    pub fn tensor_identity_complete(&self) -> bool {
        if self.generation != PackageGeneration::V2 {
            return false;
        }
        let Some(directory) = self
            .manifest
            .pointer("/tensor_container/directory")
            .and_then(Value::as_array)
        else {
            return false;
        };
        if self
            .manifest
            .pointer("/contracts/weight_contract_sha256")
            .and_then(Value::as_str)
            != Some(WEIGHT_CONTRACT_SHA256)
        {
            return false;
        }
        let mut actual = Vec::<(String, Vec<u64>)>::new();
        for entry in directory {
            let name = entry.get("name").and_then(Value::as_str).unwrap_or("");
            if name
                .split('.')
                .any(|component| component == "mtp" || component.starts_with("mtp_"))
            {
                return false;
            }
            if entry.get("role").and_then(Value::as_str) == Some("lm") {
                let Some(name) = entry.get("name").and_then(Value::as_str) else {
                    return false;
                };
                let Some(shape) = entry.get("shape").and_then(Value::as_array) else {
                    return false;
                };
                let Some(shape) = shape.iter().map(Value::as_u64).collect::<Option<Vec<_>>>()
                else {
                    return false;
                };
                actual.push((name.to_string(), shape));
            }
        }
        actual == canonical_lm_tensor_geometry()
    }

    pub fn portable_quantization_policy_complete(&self) -> bool {
        if !self.tensor_identity_complete() {
            return false;
        }
        self.manifest
            .pointer("/tensor_container/directory")
            .and_then(Value::as_array)
            .is_some_and(|directory| {
                directory
                    .iter()
                    .filter(|entry| entry.get("role").and_then(Value::as_str) == Some("lm"))
                    .all(|entry| {
                        let name = entry.get("name").and_then(Value::as_str).unwrap_or("");
                        entry.get("dtype").and_then(Value::as_str) == Some(portable_lm_dtype(name))
                    })
            })
    }

    pub fn runtime_quantization_complete(&self) -> bool {
        self.generation == PackageGeneration::V2
            && self.manifest.get("runtime_quantization") == Some(&canonical_runtime_quantization())
    }

    pub fn product_ready(&self) -> bool {
        PORTABLE_V2_RUNTIME_COMPLETE
            && self.generation == PackageGeneration::V2
            && self.verified_hashes
            && self.packed_inference_ready()
            && self.runtime_quantization_complete()
            && self.heads.missing().is_empty()
    }

    pub fn capabilities(&self) -> Value {
        let legacy = self.generation == PackageGeneration::V1ReadOnly;
        let declared = self
            .manifest
            .get("capabilities")
            .cloned()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let v2_trusted = !legacy
            && self.verified_hashes
            && self.tensor_identity_complete()
            && self.runtime_quantization_complete();
        let ready = |head: &HeadStatus| !head.is_missing_or_untrained();
        let effective = json!({
            "retrieval": declared.get("retrieval").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.contrastive),
            "full_call": declared.get("full_call").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.lm),
            "mw_disposition": declared.get("mw_disposition").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.mw_disposition),
            "confidence": declared.get("confidence").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.confidence),
            "multi_step": declared.get("multi_step").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.lm),
            "narration": declared.get("narration").and_then(Value::as_bool) == Some(true)
                && v2_trusted && ready(&self.heads.narration_adapter),
        });
        let resource_limits_reported = self.generation == PackageGeneration::V2
            && self.verified_hashes
            && self.resource_measurement_verified
            && self
                .manifest
                .pointer("/resources/package_bytes")
                .and_then(Value::as_u64)
                .is_some_and(|value| value <= 18 * 1024 * 1024)
            && self
                .manifest
                .pointer("/resources/rust_session_peak_bytes")
                .and_then(Value::as_u64)
                .is_some_and(|value| value <= 64 * 1024 * 1024)
            && self
                .manifest
                .pointer("/resources/wasm_heap_peak_bytes")
                .and_then(Value::as_u64)
                .is_some_and(|value| value <= 96 * 1024 * 1024);
        // The receipt currently authenticates package/runner/command identity
        // and binds the three numbers, but no trusted runner allow-list or
        // reproducible peak-memory harness is frozen yet. Keep eligibility
        // fail-closed instead of trusting producer-authored measurements.
        let resource_eligible = false;
        let diagnostic_inference = self.packed_inference_ready();
        let capability_complete = !legacy
            && self.heads.missing().is_empty()
            && self.tensor_identity_complete()
            && self.runtime_quantization_complete()
            && self.verified_hashes
            && declared.as_object().is_some_and(|items| {
                !items.is_empty() && items.values().all(|value| value == &json!(true))
            });
        let mut capabilities = json!({
            "package_id": self.package_id(),
            "package_format": if legacy { V1 } else { V2 },
            "release_class": self.manifest.get("release_class"),
            "inference": legacy && diagnostic_inference,
            "diagnostic_inference": diagnostic_inference,
            "protocol": true,
            "heads": self.heads.to_value(),
            "missing_or_untrained_heads": self.heads.missing(),
            "hash_verified": self.verified_hashes,
            "inference_payload_verified": self.inference_payload_verified,
            "resource_measurement_verified": self.resource_measurement_verified,
            "resource_limits_reported": resource_limits_reported,
            "tensor_identity_complete": self.tensor_identity_complete(),
            "portable_quantization_policy_complete": self.portable_quantization_policy_complete(),
            "runtime_quantization_complete": self.runtime_quantization_complete(),
            "tool_index_payload_complete": !legacy && self.tool_index.is_some(),
            "compatibility_mode": if legacy { "v1-read-only-degraded" } else { "v2-native" },
            "read_only": legacy,
            "degraded": legacy || !self.product_ready(),
            "implementation_complete": !legacy
                && PORTABLE_V2_RUNTIME_COMPLETE
                && self.runtime_quantization_complete(),
            "open_capabilities": if legacy { json!(["v2-native"]) } else { json!([
                "int8-bounded-kv", "trusted-resource-measurement"
            ]) },
            "product_ready": self.product_ready(),
            "resource_eligible": resource_eligible,
            "release_eligible": false,
            "quantized_only": diagnostic_inference,
            "compatibility": {
                "mode": if legacy { "read_only_degraded_adapter" } else { "native_v2" },
                "source_format": if legacy { V1 } else { V2 },
                "capability_complete": capability_complete,
            },
            "declared_capabilities": declared,
            "versions": sdk_versions(),
        });
        capabilities
            .as_object_mut()
            .expect("capability object")
            .extend(
                effective
                    .as_object()
                    .expect("effective capability object")
                    .clone(),
            );
        capabilities
    }
}

fn head_from(
    raw: &Value,
    name: &str,
    generation: PackageGeneration,
) -> Result<HeadStatus, SdkError> {
    let item = raw.get(name).ok_or_else(|| {
        SdkError::new(
            "package_invalid",
            format!("heads.{name} must be listed explicitly"),
        )
    })?;
    let present = item
        .get("present")
        .and_then(Value::as_bool)
        .ok_or_else(|| SdkError::new("package_invalid", format!("heads.{name}.present")))?;
    let trained = item
        .get("trained")
        .and_then(Value::as_bool)
        .ok_or_else(|| SdkError::new("package_invalid", format!("heads.{name}.trained")))?;
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("missing");
    if !matches!(status, "ready" | "untrained" | "missing" | "disabled") {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name}.status"),
        ));
    }
    let tensor_prefixes: Vec<String> = item
        .get("tensor_prefixes")
        .and_then(Value::as_array)
        .map(|xs| {
            xs.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let training_receipt_sha256 = item
        .get("training_receipt_sha256")
        .and_then(Value::as_str)
        .map(str::to_string);
    let label_codebook_sha256 = item
        .get("label_codebook_sha256")
        .and_then(Value::as_str)
        .map(str::to_string);
    if training_receipt_sha256
        .as_deref()
        .is_some_and(|value| !is_sha256(value))
    {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name}.training_receipt_sha256"),
        ));
    }
    if label_codebook_sha256
        .as_deref()
        .is_some_and(|value| !is_sha256(value))
    {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name}.label_codebook_sha256"),
        ));
    }
    if tensor_prefixes
        .iter()
        .any(|prefix| !is_component_path(prefix))
    {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name}.tensor_prefixes"),
        ));
    }
    if generation == PackageGeneration::V2 && present && tensor_prefixes.is_empty() {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name}.tensor_prefixes must identify tensors"),
        ));
    }
    if status == "ready" && (!present || !trained) {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name} ready status requires present and trained"),
        ));
    }
    Ok(HeadStatus {
        present,
        trained,
        status: status.to_string(),
        tensor_prefixes,
        training_receipt_sha256,
        label_codebook_sha256,
    })
}

fn parse_manifest(manifest: &Value) -> Result<(PackageGeneration, HeadReport), SdkError> {
    let generation = match manifest.get("package_format").and_then(Value::as_str) {
        Some(V1) => PackageGeneration::V1ReadOnly,
        Some(V2) => PackageGeneration::V2,
        _ => {
            return Err(SdkError::new(
                "package_invalid",
                "unsupported package_format",
            ))
        }
    };
    match manifest.get("product").and_then(Value::as_str) {
        Some("mei-1.0-51m") | Some("mei-1.2-51m") => {}
        _ => {
            return Err(SdkError::new(
                "package_invalid",
                "product must be mei-1.0-51m or mei-1.2-51m",
            ))
        }
    }
    if generation == PackageGeneration::V2 {
        if manifest.get("runtime_min").and_then(Value::as_str) != Some("mei-runtime-abi-2") {
            return Err(SdkError::new(
                "abi_version_mismatch",
                "v2 packages require mei-runtime-abi-2",
            ));
        }
        validate_sha_contracts(manifest)?;
        validate_v2_metadata(manifest)?;
        validate_v2_directory(manifest, None)?;
    }
    let raw = manifest
        .get("heads")
        .ok_or_else(|| SdkError::new("package_invalid", "heads object is required"))?;
    for name in REQUIRED_HEADS {
        let _ = head_from(raw, name, generation)?;
    }
    let narration_adapter = if raw.get("narration_adapter").is_some() {
        head_from(raw, "narration_adapter", generation)?
    } else {
        HeadStatus {
            present: false,
            trained: false,
            status: "missing".to_string(),
            tensor_prefixes: Vec::new(),
            training_receipt_sha256: None,
            label_codebook_sha256: None,
        }
    };
    Ok((
        generation,
        HeadReport {
            lm: head_from(raw, "lm", generation)?,
            contrastive: head_from(raw, "contrastive", generation)?,
            mw_disposition: head_from(raw, "mw_disposition", generation)?,
            confidence: head_from(raw, "confidence", generation)?,
            narration_adapter,
        },
    ))
}

fn validate_v2_metadata(manifest: &Value) -> Result<(), SdkError> {
    only_keys(
        manifest,
        &[
            "package_format",
            "product",
            "package_id",
            "runtime_min",
            "parent_package_id",
            "contracts",
            "architecture",
            "runtime_profile",
            "retrieval_calibration",
            "runtime_quantization",
            "tokenizer",
            "tensor_container",
            "files",
            "heads",
            "capabilities",
            "training_receipts",
            "resources",
            "release_class",
        ],
        "manifest",
    )?;
    if manifest
        .get("package_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .is_none()
    {
        return Err(SdkError::new("package_invalid", "package_id is required"));
    }
    if !matches!(
        manifest.get("release_class").and_then(Value::as_str),
        Some("experimental" | "candidate" | "release")
    ) {
        return Err(SdkError::new(
            "package_invalid",
            "release_class is required",
        ));
    }
    if let Some(parent) = manifest.get("parent_package_id") {
        if !parent.is_null() && parent.as_str().filter(|value| !value.is_empty()).is_none() {
            return Err(SdkError::new("package_invalid", "parent_package_id"));
        }
    }
    let architecture = manifest
        .get("architecture")
        .ok_or_else(|| SdkError::new("package_invalid", "architecture is required"))?;
    only_keys(
        architecture,
        &[
            "id",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "vocab_size",
            "max_seq_len",
            "parameter_count",
            "rope_theta",
            "engram_layers",
            "engram_orders",
            "engram_slots",
            "engram_conv_taps",
            "mhc_lanes",
            "sinkhorn_iters",
            "tie_embeddings",
            "rms_eps",
            "conf_probes",
            "mlp",
            "confidence_head",
        ],
        "architecture",
    )?;
    let expected = [
        ("d_model", 512),
        ("n_layers", 27),
        ("n_heads", 8),
        ("n_kv_heads", 4),
        ("head_dim", 64),
        ("vocab_size", 24_000),
        ("max_seq_len", 2_048),
        ("parameter_count", 51_463_797),
    ];
    if expected
        .into_iter()
        .any(|(key, value)| architecture.get(key).and_then(Value::as_u64) != Some(value))
    {
        return Err(SdkError::new(
            "package_invalid",
            "51M architecture identity mismatch",
        ));
    }
    let exact_architecture = [
        ("id", json!("mei-1.0-51m-arch-v1")),
        ("engram_layers", json!([2, 15])),
        ("engram_orders", json!([2, 3])),
        ("engram_slots", json!(8_192)),
        ("engram_conv_taps", json!(4)),
        ("mhc_lanes", json!(4)),
        ("sinkhorn_iters", json!(20)),
        ("tie_embeddings", json!(true)),
        ("conf_probes", json!(8)),
        ("mlp", json!("FixedWalshHadamardMLP")),
        ("confidence_head", json!(true)),
    ];
    if exact_architecture
        .into_iter()
        .any(|(key, expected)| architecture.get(key) != Some(&expected))
    {
        return Err(SdkError::new(
            "package_invalid",
            "51M architecture topology mismatch",
        ));
    }
    // JSON.parse/stringify in JavaScript is permitted to serialize an
    // integer-valued float such as 100000.0 as 100000. Compare numeric
    // semantics so the browser cannot change architecture identity merely by
    // transporting an otherwise identical manifest.
    if architecture.get("rope_theta").and_then(Value::as_f64) != Some(100_000.0)
        || architecture.get("rms_eps").and_then(Value::as_f64) != Some(1e-6)
    {
        return Err(SdkError::new(
            "package_invalid",
            "51M architecture numeric topology mismatch",
        ));
    }
    let profile = manifest
        .get("runtime_profile")
        .ok_or_else(|| SdkError::new("package_invalid", "runtime_profile is required"))?;
    let runtime_hash = manifest
        .pointer("/contracts/runtime_profile_sha256")
        .and_then(Value::as_str)
        .unwrap_or("");
    if runtime_hash == LEGACY_RUNTIME_PROFILE_SHA256 {
        only_keys(
            profile,
            &[
                "max_context_tokens",
                "stable_prefix_tokens",
                "rolling_window_tokens",
                "default_output_tokens",
                "kv_dtype",
                "activation_dtype",
            ],
            "runtime_profile",
        )?;
        for (key, value) in [
            ("max_context_tokens", 2048),
            ("stable_prefix_tokens", 1024),
            ("rolling_window_tokens", 256),
            ("default_output_tokens", 128),
        ] {
            if profile.get(key).and_then(Value::as_u64) != Some(value) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("runtime_profile.{key}"),
                ));
            }
        }
        if manifest.get("retrieval_calibration").is_some() {
            return Err(SdkError::new(
                "package_invalid",
                "legacy runtime profile cannot claim retrieval calibration",
            ));
        }
    } else {
        let previous_adaptive = runtime_hash == PREVIOUS_ADAPTIVE_RUNTIME_PROFILE_SHA256;
        only_keys(
            profile,
            if previous_adaptive {
                &[
                    "max_context_tokens",
                    "default_profile",
                    "stable_prefix_profiles",
                    "ordinary_window_policy",
                    "default_output_tokens",
                    "candidate_batch_size",
                    "context_packer_id",
                    "retrieval_batch_policy_id",
                    "kv_dtype",
                    "activation_dtype",
                ]
            } else {
                &[
                    "max_context_tokens",
                    "default_profile",
                    "stable_prefix_profiles",
                    "ordinary_window_policy",
                    "default_output_tokens",
                    "candidate_batch_size",
                    "context_packer_id",
                    "retrieval_batch_policy_id",
                    "prompt_framing_id",
                    "assistant_suffix",
                    "kv_dtype",
                    "activation_dtype",
                ]
            },
            "runtime_profile",
        )?;
        if profile.get("max_context_tokens").and_then(Value::as_u64) != Some(2048)
            || profile.get("default_profile").and_then(Value::as_str) != Some("standard")
            || profile.get("stable_prefix_profiles")
                != Some(&json!({"compact":1024,"standard":1536}))
            || profile
                .get("ordinary_window_policy")
                .and_then(Value::as_str)
                != Some("dynamic_remainder")
            || profile.get("default_output_tokens").and_then(Value::as_u64) != Some(128)
            || profile.get("candidate_batch_size").and_then(Value::as_u64) != Some(5)
            || profile.get("context_packer_id").and_then(Value::as_str)
                != Some("mei-tool-context-packer-v1")
            || profile
                .get("retrieval_batch_policy_id")
                .and_then(Value::as_str)
                != Some("mei-retrieval-fixed-five-batches-v1")
            || (!previous_adaptive
                && (profile.get("prompt_framing_id").and_then(Value::as_str)
                    != Some("mei-tool-prompt-framing-v1")
                    || profile.get("assistant_suffix").and_then(Value::as_str)
                        != Some("<|im_end|>\n<|im_start|>assistant\n")))
        {
            return Err(SdkError::new(
                "package_invalid",
                "runtime_profile does not match the adaptive v2 contract",
            ));
        }
        validate_retrieval_calibration(manifest)?;
    }
    if profile.get("kv_dtype").and_then(Value::as_str) != Some("i8")
        || profile.get("activation_dtype").and_then(Value::as_str) != Some("i8")
    {
        return Err(SdkError::new(
            "package_invalid",
            "runtime profile requires int8 KV/activations",
        ));
    }
    if let Some(runtime_quantization) = manifest.get("runtime_quantization") {
        only_keys(
            runtime_quantization,
            &[
                "weight_math_id",
                "activation_semantics_id",
                "activation_sites",
                "activation_q_quantized",
                "activation_reconstruction_dtype",
                "kv_semantics_id",
                "kv_sites",
                "kv_storage_dtype",
                "kv_scale_granularity",
                "int8_code_min",
                "int8_code_max",
                "scale_denominator",
            ],
            "runtime_quantization",
        )?;
        if runtime_quantization != &canonical_runtime_quantization() {
            return Err(SdkError::new(
                "package_invalid",
                "runtime_quantization does not match native v2 semantics",
            ));
        }
    }
    let tokenizer = manifest
        .get("tokenizer")
        .ok_or_else(|| SdkError::new("package_invalid", "tokenizer is required"))?;
    only_keys(
        tokenizer,
        &[
            "id",
            "file",
            "sha256",
            "vocab_file",
            "vocab_sha256",
            "pad_id",
            "eos_id",
            "bos_id",
            "unk_id",
        ],
        "tokenizer",
    )?;
    if tokenizer.get("id").and_then(Value::as_str) != Some("zh-24k-v1") {
        return Err(SdkError::new(
            "package_invalid",
            "tokenizer.id must be zh-24k-v1",
        ));
    }
    for (key, value) in [("pad_id", 0), ("eos_id", 1), ("bos_id", 2), ("unk_id", 3)] {
        if tokenizer.get(key).and_then(Value::as_u64) != Some(value) {
            return Err(SdkError::new("package_invalid", format!("tokenizer.{key}")));
        }
    }
    if !is_sha256(
        tokenizer
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or(""),
    ) {
        return Err(SdkError::new("package_invalid", "tokenizer.sha256"));
    }
    match (
        tokenizer.get("vocab_file").and_then(Value::as_str),
        tokenizer.get("vocab_sha256").and_then(Value::as_str),
    ) {
        (Some(_), Some(hash)) if is_sha256(hash) => {}
        (None, None) => {}
        _ => {
            return Err(SdkError::new(
                "package_invalid",
                "tokenizer vocab file/hash pair",
            ))
        }
    }
    let _ = safe_relative(tokenizer.get("file").and_then(Value::as_str).unwrap_or(""))?;
    if let Some(vocab_file) = tokenizer.get("vocab_file").and_then(Value::as_str) {
        let _ = safe_relative(vocab_file)?;
    }
    let _ = safe_relative(
        manifest
            .pointer("/tensor_container/file")
            .and_then(Value::as_str)
            .unwrap_or(""),
    )?;
    validate_v2_files(manifest)?;
    let capabilities = manifest
        .get("capabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| SdkError::new("package_invalid", "capabilities is required"))?;
    only_keys(
        &Value::Object(capabilities.clone()),
        &[
            "retrieval",
            "full_call",
            "mw_disposition",
            "confidence",
            "multi_step",
            "narration",
        ],
        "capabilities",
    )?;
    for key in [
        "retrieval",
        "full_call",
        "mw_disposition",
        "confidence",
        "multi_step",
    ] {
        if !capabilities
            .get(key)
            .map(Value::is_boolean)
            .unwrap_or(false)
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("capabilities.{key}"),
            ));
        }
    }
    let receipts = manifest
        .get("training_receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "training_receipts is required"))?;
    if receipts
        .iter()
        .any(|receipt| !is_sha256(receipt.as_str().unwrap_or("")))
    {
        return Err(SdkError::new(
            "package_invalid",
            "invalid training receipt sha256",
        ));
    }
    let mut unique_receipts = HashSet::new();
    let files = manifest
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "files is required"))?;
    for receipt in receipts.iter().filter_map(Value::as_str) {
        if !unique_receipts.insert(receipt)
            || files
                .iter()
                .filter(|row| {
                    row.get("role").and_then(Value::as_str) == Some("training_receipt")
                        && row.get("sha256").and_then(Value::as_str) == Some(receipt)
                })
                .count()
                != 1
        {
            return Err(SdkError::new(
                "package_invalid",
                "every training_receipts entry must have one inventoried payload",
            ));
        }
    }
    let inventoried_receipts = files
        .iter()
        .filter(|row| row.get("role").and_then(Value::as_str) == Some("training_receipt"))
        .collect::<Vec<_>>();
    if inventoried_receipts.len() != receipts.len()
        || inventoried_receipts.iter().any(|row| {
            !receipts
                .iter()
                .any(|receipt| receipt.as_str() == row.get("sha256").and_then(Value::as_str))
        })
    {
        return Err(SdkError::new(
            "package_invalid",
            "training_receipts and inventoried payloads must be one-to-one",
        ));
    }
    let resources = manifest
        .get("resources")
        .and_then(Value::as_object)
        .ok_or_else(|| SdkError::new("package_invalid", "resources is required"))?;
    only_keys(
        &Value::Object(resources.clone()),
        &[
            "package_bytes",
            "rust_session_peak_bytes",
            "wasm_heap_peak_bytes",
            "measurement_receipt",
        ],
        "resources",
    )?;
    for key in [
        "package_bytes",
        "rust_session_peak_bytes",
        "wasm_heap_peak_bytes",
    ] {
        if resources.get(key).and_then(Value::as_u64).is_none() {
            return Err(SdkError::new("package_invalid", format!("resources.{key}")));
        }
    }
    if let Some(receipt) = resources.get("measurement_receipt") {
        only_keys(
            receipt,
            &["file", "sha256"],
            "resources.measurement_receipt",
        )?;
        let path = receipt.get("file").and_then(Value::as_str).unwrap_or("");
        let hash = receipt.get("sha256").and_then(Value::as_str).unwrap_or("");
        let _ = safe_relative(path)?;
        if !is_sha256(hash)
            || files
                .iter()
                .filter(|row| {
                    row.get("path").and_then(Value::as_str) == Some(path)
                        && row.get("sha256").and_then(Value::as_str) == Some(hash)
                        && row.get("role").and_then(Value::as_str) == Some("resource_receipt")
                })
                .count()
                != 1
        {
            return Err(SdkError::new(
                "package_invalid",
                "resource measurement receipt is not uniquely inventoried",
            ));
        }
    }
    Ok(())
}

fn only_keys(value: &Value, allowed: &[&str], path: &str) -> Result<(), SdkError> {
    let object = value
        .as_object()
        .ok_or_else(|| SdkError::new("package_invalid", format!("{path} must be an object")))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(SdkError::new(
            "package_invalid",
            format!("unknown {path}.{key}"),
        ));
    }
    Ok(())
}

fn validate_sha_contracts(manifest: &Value) -> Result<(), SdkError> {
    let contracts = manifest
        .get("contracts")
        .ok_or_else(|| SdkError::new("package_invalid", "contracts object is required"))?;
    only_keys(
        contracts,
        &[
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        ],
        "contracts",
    )?;
    for (key, expected) in [
        ("weight_contract_sha256", WEIGHT_CONTRACT_SHA256),
        ("training_aux_sha256", TRAINING_AUX_SHA256),
    ] {
        let value = contracts.get(key).and_then(Value::as_str).unwrap_or("");
        if value != expected {
            return Err(SdkError::new(
                "package_invalid",
                format!("contracts.{key} does not match the frozen 51M contract"),
            ));
        }
    }
    let runtime = contracts
        .get("runtime_profile_sha256")
        .and_then(Value::as_str)
        .unwrap_or("");
    if runtime != RUNTIME_PROFILE_SHA256
        && runtime != PREVIOUS_ADAPTIVE_RUNTIME_PROFILE_SHA256
        && runtime != LEGACY_RUNTIME_PROFILE_SHA256
    {
        return Err(SdkError::new(
            "package_invalid",
            "contracts.runtime_profile_sha256 does not match a supported 51M runtime contract",
        ));
    }
    Ok(())
}

fn validate_retrieval_calibration(manifest: &Value) -> Result<(), SdkError> {
    let calibration = manifest
        .get("retrieval_calibration")
        .ok_or_else(|| SdkError::new("package_invalid", "retrieval_calibration is required"))?;
    only_keys(
        calibration,
        &[
            "calibration_id",
            "scale",
            "bias",
            "discard_threshold",
            "expand_threshold",
            "validated",
            "fallback_scan_all_eligible",
            "validation_manifest_sha256",
            "catalog_sizes",
            "metrics",
        ],
        "retrieval_calibration",
    )?;
    let scale = calibration.get("scale").and_then(Value::as_f64);
    let bias = calibration.get("bias").and_then(Value::as_f64);
    let discard = calibration.get("discard_threshold").and_then(Value::as_f64);
    let expand = calibration.get("expand_threshold").and_then(Value::as_f64);
    if calibration.get("calibration_id").and_then(Value::as_str) != Some("mei-retrieval-platt-v1")
        || !scale.is_some_and(f64::is_finite)
        || !bias.is_some_and(f64::is_finite)
        || !discard.is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        || !expand.is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        || expand.unwrap_or(-1.0) < discard.unwrap_or(2.0)
        || !calibration.get("validated").is_some_and(Value::is_boolean)
        || !calibration
            .get("fallback_scan_all_eligible")
            .is_some_and(Value::is_boolean)
        || calibration.get("catalog_sizes") != Some(&json!([10, 20, 50]))
        || !calibration.get("metrics").is_some_and(Value::is_object)
        || !calibration
            .get("validation_manifest_sha256")
            .and_then(Value::as_str)
            .is_some_and(is_sha256)
    {
        return Err(SdkError::new(
            "package_invalid",
            "retrieval_calibration does not match the adaptive v2 contract",
        ));
    }
    Ok(())
}

/// Canonical ordered geometry for the deployed mei-1.0-51m weight contract.
///
/// Keep this expansion byte-for-byte equivalent to
/// `models/mei-1.2-51m/architecture/architecture_contract.py`.  Runtime
/// profiles and training-only auxiliaries deliberately do not appear here.
fn canonical_lm_tensor_geometry() -> Vec<(String, Vec<u64>)> {
    let mut tensors = Vec::<(String, Vec<u64>)>::with_capacity(400);
    let mut add = |name: String, shape: &[u64]| tensors.push((name, shape.to_vec()));
    add("embed.weight".into(), &[24_000, 512]);
    for layer in 0..27 {
        let prefix = format!("blocks.{layer}");
        add(format!("{prefix}.attn_norm.scale"), &[512]);
        add(format!("{prefix}.attn.q_proj.weight"), &[512, 512]);
        add(format!("{prefix}.attn.k_proj.weight"), &[256, 512]);
        add(format!("{prefix}.attn.v_proj.weight"), &[256, 512]);
        add(format!("{prefix}.attn.gate_proj.weight"), &[512, 512]);
        add(format!("{prefix}.attn.o_proj.weight"), &[512, 512]);
        add(format!("{prefix}.attn.q_norm.scale"), &[64]);
        add(format!("{prefix}.attn.k_norm.scale"), &[64]);
        add(format!("{prefix}.post_attn_norm.scale"), &[512]);
        add(format!("{prefix}.attn_gate"), &[]);
        add(format!("{prefix}.mlp_norm.scale"), &[512]);
        add(format!("{prefix}.mlp.d1"), &[512]);
        add(format!("{prefix}.mlp.d2"), &[512]);
        add(format!("{prefix}.mlp.d3"), &[512]);
    }
    add("final_norm.scale".into(), &[512]);
    for site in 0..2 {
        let prefix = format!("engrams.{site}");
        add(format!("{prefix}.tables"), &[4, 8_192, 128]);
        add(format!("{prefix}.key_proj.weight"), &[512, 512]);
        add(format!("{prefix}.value_proj.weight"), &[512, 512]);
        add(format!("{prefix}.taps"), &[4, 512]);
    }
    add("mhc_phi_pre".into(), &[27, 2_048, 4]);
    add("mhc_phi_post".into(), &[27, 2_048, 4]);
    add("mhc_phi_res".into(), &[27, 2_048, 16]);
    add("mhc_b_pre".into(), &[27, 4]);
    add("mhc_b_post".into(), &[27, 4]);
    add("mhc_b_res".into(), &[27, 4, 4]);
    add("mhc_a_pre".into(), &[27]);
    add("mhc_a_post".into(), &[27]);
    add("mhc_a_res".into(), &[27]);
    add("conf_probes".into(), &[8, 512]);
    add("conf_proj.weight".into(), &[1, 4_096]);
    add("conf_proj.bias".into(), &[1]);
    debug_assert_eq!(tensors.len(), 400);
    tensors
}

fn portable_lm_dtype(name: &str) -> &'static str {
    if name == "embed.weight" || name.starts_with("mhc_") {
        "cq4"
    } else if (name.starts_with("engrams.") && !name.ends_with(".taps"))
        || (name.contains(".attn.") && name.ends_with("_proj.weight"))
    {
        "cq2"
    } else {
        // Norms, scalar gates, fixed-WHT diagonal MLP factors, engram taps,
        // and the deployed backbone confidence tensors remain explicit f16.
        "f16"
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_component_path(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|component| {
            !component.is_empty()
                && component
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
}

fn prefix_matches(name: &str, prefix: &str) -> bool {
    name == prefix
        || name
            .strip_prefix(prefix)
            .map(|suffix| suffix.starts_with('.'))
            .unwrap_or(false)
}

fn ready_head_contract(
    name: &str,
) -> Option<(&'static str, &'static [(&'static str, &'static [u64])])> {
    match name {
        "contrastive" => Some(("heads.contrastive", &CONTRASTIVE_TENSORS)),
        "mw_disposition" => Some(("heads.mw_disposition", &MW_DISPOSITION_TENSORS)),
        "confidence" => Some(("heads.confidence", &CONFIDENCE_TENSORS)),
        "narration_adapter" => Some(("heads.narration_adapter", &NARRATION_ADAPTER_TENSORS)),
        _ => None,
    }
}

fn validate_ready_head_contract(
    manifest: &Value,
    name: &str,
    head: &Value,
    entries: &[Value],
) -> Result<(), SdkError> {
    if head.get("status").and_then(Value::as_str) != Some("ready") {
        return Ok(());
    }
    let receipt = head
        .get("training_receipt_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_sha256(value))
        .ok_or_else(|| {
            SdkError::new(
                "package_invalid",
                format!("heads.{name} ready status requires training_receipt_sha256"),
            )
        })?;
    let receipts = manifest
        .get("training_receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "training_receipts is required"))?;
    if !receipts.iter().any(|value| value.as_str() == Some(receipt)) {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name} training receipt is absent from training_receipts"),
        ));
    }
    let files = manifest
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "files is required"))?;
    if !files.iter().any(|row| {
        row.get("role").and_then(Value::as_str) == Some("training_receipt")
            && row.get("sha256").and_then(Value::as_str) == Some(receipt)
    }) {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name} training receipt has no inventoried payload"),
        ));
    }
    if name == "mw_disposition" {
        if head.get("label_codebook_sha256").and_then(Value::as_str)
            != Some(MW_LABEL_CODEBOOK_SHA256)
        {
            return Err(SdkError::new(
                "package_invalid",
                "heads.mw_disposition ready status requires the frozen 20-class label codebook",
            ));
        }
        if files
            .iter()
            .filter(|row| {
                row.get("role").and_then(Value::as_str) == Some("head_codebook")
                    && row.get("sha256").and_then(Value::as_str) == Some(MW_LABEL_CODEBOOK_SHA256)
            })
            .count()
            != 1
        {
            return Err(SdkError::new(
                "package_invalid",
                "heads.mw_disposition label codebook has no inventoried payload",
            ));
        }
    }
    let Some((required_prefix, required_tensors)) = ready_head_contract(name) else {
        return Ok(());
    };
    let prefixes = head
        .get("tensor_prefixes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if prefixes != [required_prefix] {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name} must use canonical prefix {required_prefix}"),
        ));
    }
    let role_entries = entries
        .iter()
        .filter(|entry| entry.get("role").and_then(Value::as_str) == Some(name))
        .collect::<Vec<_>>();
    if role_entries.len() != required_tensors.len() {
        return Err(SdkError::new(
            "package_invalid",
            format!("heads.{name} ready tensor set is not canonical"),
        ));
    }
    for &(tensor_name, expected_shape) in required_tensors {
        let entry = role_entries
            .iter()
            .find(|entry| entry.get("name").and_then(Value::as_str) == Some(tensor_name))
            .ok_or_else(|| {
                SdkError::new(
                    "package_invalid",
                    format!("heads.{name} is missing {tensor_name}"),
                )
            })?;
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .collect::<Vec<_>>();
        let dtype = entry.get("dtype").and_then(Value::as_str).unwrap_or("");
        let narration_quantized = name == "narration_adapter"
            && matches!(dtype, "cq2" | "cq4")
            && entry.get("transform").and_then(Value::as_str) == Some("wht")
            && entry.get("codebook").and_then(Value::as_str)
                == Some(if dtype == "cq2" {
                    "gaussian-lloyd-q2-v1"
                } else {
                    "gaussian-lloyd-q4-v1"
                });
        let ordinary_f16 = name != "narration_adapter"
            && dtype == "f16"
            && entry.get("transform").and_then(Value::as_str) == Some("none")
            && entry.get("codebook").and_then(Value::as_str) == Some("none");
        if shape.as_slice() != expected_shape || !(narration_quantized || ordinary_f16) {
            return Err(SdkError::new(
                "package_invalid",
                format!("heads.{name} tensor {tensor_name} violates the canonical contract"),
            ));
        }
    }
    Ok(())
}

fn safe_relative(value: &str) -> Result<&Path, SdkError> {
    if value.is_empty() || value.contains('\\') {
        return Err(SdkError::from_id("package_path_unsafe"));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(SdkError::new(
            "package_path_unsafe",
            format!("unsafe package path: {value}"),
        ));
    }
    Ok(path)
}

fn package_path(root: &Path, value: &str) -> Result<PathBuf, SdkError> {
    let joined = root.join(safe_relative(value)?);
    if !joined.is_file() {
        return Err(SdkError::new(
            "file_not_found",
            format!("missing package file: {value}"),
        ));
    }
    let canonical = joined
        .canonicalize()
        .map_err(|_| SdkError::new("file_not_found", value))?;
    if !canonical.starts_with(root) {
        return Err(SdkError::from_id("package_path_unsafe"));
    }
    Ok(canonical)
}

fn file_sha256(path: &Path) -> Result<String, SdkError> {
    let bytes = fs::read(path).map_err(|_| {
        SdkError::new(
            "file_not_found",
            format!("missing file: {}", path.display()),
        )
    })?;
    Ok(sha256_bytes(&bytes))
}

fn verify_file(root: &Path, spec: &Value) -> Result<PathBuf, SdkError> {
    let rel = spec
        .get("file")
        .and_then(Value::as_str)
        .ok_or_else(|| SdkError::new("package_invalid", "file is required"))?;
    let expected = spec.get("sha256").and_then(Value::as_str).unwrap_or("");
    if !is_sha256(expected) {
        return Err(SdkError::new(
            "package_invalid",
            format!("invalid sha256 for {rel}"),
        ));
    }
    let path = package_path(root, rel)?;
    let digest = file_sha256(&path)?;
    if digest != expected {
        return Err(SdkError::new(
            "package_hash_mismatch",
            format!("{rel} sha256 mismatch: expected {expected}, got {digest}"),
        ));
    }
    Ok(path)
}

fn verify_resource_measurement_receipt(root: &Path, manifest: &Value) -> Result<bool, SdkError> {
    let Some(spec) = manifest.pointer("/resources/measurement_receipt") else {
        return Ok(false);
    };
    let path = verify_file(root, spec)?;
    let bytes = fs::read(&path).map_err(|_| SdkError::from_id("file_not_found"))?;
    let receipt: Value = serde_json::from_slice(&bytes)
        .map_err(|error| SdkError::new("invalid_json", error.to_string()))?;
    only_keys(
        &receipt,
        &[
            "schema",
            "package_id",
            "tensor_container_sha256",
            "runtime_abi",
            "measurements",
            "rust",
            "wasm",
        ],
        "resource_receipt",
    )?;
    if receipt.get("schema").and_then(Value::as_str) != Some("mei-resource-measurement-receipt-v1")
        || receipt.get("package_id") != manifest.get("package_id")
        || receipt.get("tensor_container_sha256") != manifest.pointer("/tensor_container/sha256")
        || receipt.get("runtime_abi").and_then(Value::as_str) != Some("mei-runtime-abi-2")
    {
        return Err(SdkError::new(
            "package_invalid",
            "resource measurement receipt identity mismatch",
        ));
    }
    let measurements = receipt
        .get("measurements")
        .ok_or_else(|| SdkError::new("package_invalid", "resource receipt measurements"))?;
    only_keys(
        measurements,
        &[
            "package_bytes",
            "rust_session_peak_bytes",
            "wasm_heap_peak_bytes",
        ],
        "resource_receipt.measurements",
    )?;
    for key in [
        "package_bytes",
        "rust_session_peak_bytes",
        "wasm_heap_peak_bytes",
    ] {
        let measured = measurements.get(key).and_then(Value::as_u64);
        if measured.is_none()
            || measured
                != manifest
                    .pointer(&format!("/resources/{key}"))
                    .and_then(Value::as_u64)
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("resource measurement receipt disagrees on {key}"),
            ));
        }
    }
    for runtime in ["rust", "wasm"] {
        let runner = receipt.get(runtime).ok_or_else(|| {
            SdkError::new("package_invalid", format!("resource receipt {runtime}"))
        })?;
        only_keys(
            runner,
            &[
                "target",
                "build_profile",
                "runner_id",
                "runner_sha256",
                "command_sha256",
            ],
            &format!("resource_receipt.{runtime}"),
        )?;
        if runner
            .get("target")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .is_none()
            || runner.get("build_profile").and_then(Value::as_str) != Some("release")
            || runner
                .get("runner_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .is_none()
            || !is_sha256(
                runner
                    .get("runner_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            )
            || !is_sha256(
                runner
                    .get("command_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            )
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("resource receipt {runtime} runner evidence"),
            ));
        }
    }
    Ok(true)
}

fn role_directory_sha256(manifest: &Value, component: &str) -> Result<String, SdkError> {
    let entries = manifest
        .pointer("/tensor_container/directory")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "tensor directory is required"))?
        .iter()
        .filter(|entry| entry.get("role").and_then(Value::as_str) == Some(component))
        .cloned()
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err(SdkError::new(
            "package_invalid",
            format!("training receipt component {component} has no tensors"),
        ));
    }
    Ok(sha256_bytes(
        dumps_canonical(&Value::Array(entries)).as_bytes(),
    ))
}

fn verify_training_receipts(root: &Path, manifest: &Value) -> Result<bool, SdkError> {
    let receipts = manifest
        .get("training_receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "training_receipts is required"))?;
    let files = manifest
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "files is required"))?;
    let mut components = HashMap::<String, String>::new();
    for digest in receipts.iter().filter_map(Value::as_str) {
        let row = files
            .iter()
            .find(|row| {
                row.get("role").and_then(Value::as_str) == Some("training_receipt")
                    && row.get("sha256").and_then(Value::as_str) == Some(digest)
            })
            .ok_or_else(|| SdkError::new("package_invalid", "training receipt payload"))?;
        let path = package_path(root, row.get("path").and_then(Value::as_str).unwrap_or(""))?;
        let bytes = fs::read(path).map_err(|_| SdkError::from_id("file_not_found"))?;
        let receipt: Value = serde_json::from_slice(&bytes)
            .map_err(|error| SdkError::new("invalid_json", error.to_string()))?;
        only_keys(
            &receipt,
            &[
                "schema",
                "product",
                "package_id",
                "component",
                "stage_id",
                "stage_fingerprint_sha256",
                "status",
                "contracts",
                "tensor_container_sha256",
                "tensor_directory_sha256",
            ],
            "training_receipt",
        )?;
        let component = receipt
            .get("component")
            .and_then(Value::as_str)
            .filter(|value| ALL_HEADS.contains(value))
            .ok_or_else(|| SdkError::new("package_invalid", "training receipt component"))?;
        let stage_id = receipt
            .get("stage_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let stage_fingerprint = receipt
            .get("stage_fingerprint_sha256")
            .and_then(Value::as_str)
            .filter(|value| is_sha256(value));
        if receipt.get("schema").and_then(Value::as_str) != Some("mei-training-receipt-v2")
            || receipt.get("product") != manifest.get("product")
            || receipt.get("package_id") != manifest.get("package_id")
            || receipt.get("status").and_then(Value::as_str) != Some("passed")
            || receipt.get("contracts") != manifest.get("contracts")
            || receipt.get("tensor_container_sha256")
                != manifest.pointer("/tensor_container/sha256")
            || stage_id.is_none()
            || stage_fingerprint.is_none()
            || receipt
                .get("tensor_directory_sha256")
                .and_then(Value::as_str)
                != Some(role_directory_sha256(manifest, component)?.as_str())
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("training receipt {digest} identity or terminal status mismatch"),
            ));
        }
        components.insert(digest.to_string(), component.to_string());
    }
    for name in ALL_HEADS {
        let head = manifest.pointer(&format!("/heads/{name}"));
        if head
            .and_then(|value| value.get("status"))
            .and_then(Value::as_str)
            == Some("ready")
        {
            let digest = head
                .and_then(|value| value.get("training_receipt_sha256"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if components.get(digest).map(String::as_str) != Some(name) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("heads.{name} receipt is not bound to that component"),
                ));
            }
        }
    }
    Ok(true)
}

fn contrastive_head_sha256(packed: &PackedWeights) -> Result<String, SdkError> {
    let mut entries = packed
        .tensors
        .values()
        .filter(|entry| entry.role == "contrastive")
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    if entries.is_empty() {
        return Err(SdkError::new(
            "package_invalid",
            "tool index requires a contrastive head payload",
        ));
    }
    let mut identity = Vec::new();
    for entry in entries {
        identity.extend_from_slice(entry.name.as_bytes());
        for (offset, nbytes) in [
            (entry.packed_offset, entry.packed_nbytes),
            (entry.scale_offset, entry.scale_nbytes),
            (entry.bit_map_offset, entry.bit_map_nbytes),
        ] {
            if nbytes > 0 {
                let end = offset.checked_add(nbytes).ok_or_else(|| {
                    SdkError::new("package_range_invalid", "tool-index head range overflow")
                })?;
                identity.extend_from_slice(packed.bytes.get(offset..end).ok_or_else(|| {
                    SdkError::new("package_range_invalid", "tool-index head range")
                })?);
            }
        }
    }
    Ok(sha256_bytes(&identity))
}

fn verify_tool_index(
    root: &Path,
    manifest: &Value,
    packed: &PackedWeights,
) -> Result<Option<ToolIndex>, SdkError> {
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| row.get("role").and_then(Value::as_str) == Some("tool_index"))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != 1 {
        return Err(SdkError::new(
            "package_invalid",
            "package supports exactly one canonical tool index",
        ));
    }
    let path = package_path(
        root,
        rows[0].get("path").and_then(Value::as_str).unwrap_or(""),
    )?;
    let bytes = fs::read(path).map_err(|_| SdkError::from_id("file_not_found"))?;
    let index = ToolIndex::parse(&bytes)?;
    if index.model_sha256
        != manifest
            .pointer("/tensor_container/sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
        || index.head_sha256 != contrastive_head_sha256(packed)?
        || index.tokenizer_sha256
            != manifest
                .pointer("/tokenizer/sha256")
                .and_then(Value::as_str)
                .unwrap_or("")
    {
        return Err(SdkError::new(
            "package_hash_mismatch",
            "tool index model/head/tokenizer fingerprint mismatch",
        ));
    }
    Ok(Some(index))
}

fn validate_v2_files(manifest: &Value) -> Result<HashMap<String, (String, u64, String)>, SdkError> {
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| SdkError::new("package_invalid", "files must inventory every payload"))?;
    let mut files = HashMap::new();
    for row in rows {
        only_keys(row, &["path", "sha256", "nbytes", "role"], "files[]")?;
        let path = row.get("path").and_then(Value::as_str).unwrap_or("");
        let _ = safe_relative(path)?;
        if path == "mei-model.json" {
            return Err(SdkError::new(
                "package_invalid",
                "mei-model.json is the sole self-hash exception and must not appear in files",
            ));
        }
        let hash = row.get("sha256").and_then(Value::as_str).unwrap_or("");
        let nbytes = row
            .get("nbytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| SdkError::new("package_invalid", format!("files[{path}].nbytes")))?;
        let role = row.get("role").and_then(Value::as_str).unwrap_or("");
        if !is_sha256(hash)
            || !matches!(
                role,
                "tokenizer"
                    | "tensor_container"
                    | "tool_index"
                    | "training_receipt"
                    | "head_codebook"
                    | "resource_receipt"
                    | "auxiliary"
            )
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("invalid files entry for {path}"),
            ));
        }
        if files
            .insert(
                path.to_string(),
                (hash.to_string(), nbytes, role.to_string()),
            )
            .is_some()
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("duplicate files path {path}"),
            ));
        }
    }
    if files
        .values()
        .filter(|(_, _, role)| role == "tool_index")
        .count()
        > 1
    {
        return Err(SdkError::new(
            "package_invalid",
            "package supports at most one canonical tool index",
        ));
    }
    let tokenizer = manifest.get("tokenizer").unwrap_or(&Value::Null);
    let container = manifest.get("tensor_container").unwrap_or(&Value::Null);
    let required = [
        (
            tokenizer.get("file").and_then(Value::as_str),
            tokenizer.get("sha256").and_then(Value::as_str),
            None,
            "tokenizer",
        ),
        (
            tokenizer.get("vocab_file").and_then(Value::as_str),
            tokenizer.get("vocab_sha256").and_then(Value::as_str),
            None,
            "tokenizer",
        ),
        (
            container.get("file").and_then(Value::as_str),
            container.get("sha256").and_then(Value::as_str),
            container.get("payload_bytes").and_then(Value::as_u64),
            "tensor_container",
        ),
    ];
    for (path, hash, nbytes, role) in required {
        let (Some(path), Some(hash)) = (path, hash) else {
            if path.is_some() || hash.is_some() {
                return Err(SdkError::new("package_invalid", "file/hash reference pair"));
            }
            continue;
        };
        let Some((listed_hash, listed_nbytes, listed_role)) = files.get(path) else {
            return Err(SdkError::new(
                "package_invalid",
                format!("referenced payload {path} is missing from files"),
            ));
        };
        if listed_hash != hash
            || listed_role != role
            || nbytes.is_some_and(|value| value != *listed_nbytes)
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("files entry disagrees with {path} reference"),
            ));
        }
    }
    Ok(files)
}

fn collect_payload_files(
    root: &Path,
    directory: &Path,
    out: &mut HashSet<String>,
) -> Result<(), SdkError> {
    for entry in fs::read_dir(directory)
        .map_err(|error| SdkError::new("file_not_found", error.to_string()))?
    {
        let entry = entry.map_err(|error| SdkError::new("file_not_found", error.to_string()))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| SdkError::new("file_not_found", error.to_string()))?;
        if metadata.file_type().is_symlink() {
            return Err(SdkError::new(
                "package_path_unsafe",
                format!("package symlink is forbidden: {}", path.display()),
            ));
        }
        if metadata.is_dir() {
            collect_payload_files(root, &path, out)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| SdkError::from_id("package_path_unsafe"))?;
            let components = relative
                .components()
                .map(|component| match component {
                    Component::Normal(value) => value
                        .to_str()
                        .map(str::to_string)
                        .ok_or_else(|| SdkError::from_id("package_path_unsafe")),
                    _ => Err(SdkError::from_id("package_path_unsafe")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            out.insert(components.join("/"));
        } else {
            return Err(SdkError::new(
                "package_path_unsafe",
                format!("special package file is forbidden: {}", path.display()),
            ));
        }
    }
    Ok(())
}

fn verify_v2_files(root: &Path, manifest: &Value, manifest_path: &Path) -> Result<(), SdkError> {
    let declared = validate_v2_files(manifest)?;
    let mut actual = HashSet::new();
    collect_payload_files(root, root, &mut actual)?;
    if !actual.remove("mei-model.json") {
        return Err(SdkError::new("file_not_found", "mei-model.json"));
    }
    let declared_names = declared.keys().cloned().collect::<HashSet<_>>();
    if actual != declared_names {
        let undeclared = actual
            .difference(&declared_names)
            .cloned()
            .collect::<Vec<_>>();
        let missing = declared_names
            .difference(&actual)
            .cloned()
            .collect::<Vec<_>>();
        return Err(SdkError::new(
            "package_invalid",
            format!("payload inventory mismatch undeclared={undeclared:?} missing={missing:?}"),
        ));
    }
    let mut payload_bytes = 0u64;
    for (path, (expected_hash, expected_nbytes, _role)) in declared {
        let full = package_path(root, &path)?;
        let metadata = fs::metadata(&full).map_err(|_| SdkError::from_id("file_not_found"))?;
        if metadata.len() != expected_nbytes {
            return Err(SdkError::new(
                "package_range_invalid",
                format!(
                    "{path} declares {expected_nbytes} bytes, file has {}",
                    metadata.len()
                ),
            ));
        }
        let digest = file_sha256(&full)?;
        if digest != expected_hash {
            return Err(SdkError::new(
                "package_hash_mismatch",
                format!("{path} sha256 mismatch"),
            ));
        }
        payload_bytes = payload_bytes
            .checked_add(metadata.len())
            .ok_or_else(|| SdkError::from_id("package_range_invalid"))?;
    }
    let actual_package_bytes = payload_bytes
        .checked_add(
            fs::metadata(manifest_path)
                .map_err(|_| SdkError::from_id("file_not_found"))?
                .len(),
        )
        .ok_or_else(|| SdkError::from_id("package_range_invalid"))?;
    let declared_package_bytes = manifest
        .pointer("/resources/package_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    if actual_package_bytes != declared_package_bytes {
        return Err(SdkError::new(
            "package_range_invalid",
            format!(
                "resources.package_bytes declares {declared_package_bytes}, measured {actual_package_bytes}"
            ),
        ));
    }
    Ok(())
}

fn range(entry: &Value, field: &str) -> Option<(u64, u64)> {
    let value = entry.get(field)?;
    Some((
        value.get("offset")?.as_u64()?,
        value.get("nbytes")?.as_u64()?,
    ))
}

fn validate_v2_directory(manifest: &Value, actual_bytes: Option<u64>) -> Result<(), SdkError> {
    let container = manifest
        .get("tensor_container")
        .ok_or_else(|| SdkError::new("package_invalid", "tensor_container is required"))?;
    only_keys(
        container,
        &[
            "file",
            "format",
            "sha256",
            "payload_bytes",
            "quant_math_id",
            "directory",
        ],
        "tensor_container",
    )?;
    if container.get("format").and_then(Value::as_str) != Some(CQ2_FORMAT)
        || container.get("quant_math_id").and_then(Value::as_str) != Some(CQ2_MATH)
        || !is_sha256(
            container
                .get("sha256")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )
    {
        return Err(SdkError::new(
            "package_invalid",
            "unsupported v2 tensor container",
        ));
    }
    let declared = container
        .get("payload_bytes")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| SdkError::new("package_invalid", "tensor_container.payload_bytes"))?;
    if let Some(actual) = actual_bytes {
        if declared != actual {
            return Err(SdkError::new(
                "package_range_invalid",
                format!("payload_bytes declares {declared}, file has {actual}"),
            ));
        }
    }
    let entries = container
        .get("directory")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "tensor_container.directory"))?;
    if entries.is_empty() {
        return Err(SdkError::new(
            "package_invalid",
            "tensor directory is empty",
        ));
    }
    let mut names = HashSet::new();
    let mut tensor_roles = Vec::<(String, String)>::new();
    let mut ranges = Vec::<(u64, u64, String)>::new();
    for entry in entries {
        let name = entry.get("name").and_then(Value::as_str).unwrap_or("");
        if !is_component_path(name) {
            return Err(SdkError::new(
                "package_invalid",
                format!("invalid tensor name: {name}"),
            ));
        }
        only_keys(
            entry,
            &[
                "name",
                "role",
                "shape",
                "n_params",
                "dtype",
                "data",
                "scales",
                "bit_map",
                "group_size",
                "transform",
                "codebook",
            ],
            &format!("tensor.{name}"),
        )?;
        for field in ["data", "scales", "bit_map"] {
            if let Some(value) = entry.get(field) {
                only_keys(
                    value,
                    &["offset", "nbytes"],
                    &format!("tensor.{name}.{field}"),
                )?;
            }
        }
        if !names.insert(name.to_string()) {
            return Err(SdkError::new("duplicate_tensor", name));
        }
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| SdkError::new("package_invalid", format!("{name}.shape")))?;
        let product = shape.iter().try_fold(1u64, |acc, dim| {
            let value = dim.as_u64().filter(|x| *x > 0)?;
            acc.checked_mul(value)
        });
        if product != entry.get("n_params").and_then(Value::as_u64) {
            return Err(SdkError::new(
                "package_invalid",
                format!("{name}.n_params != product(shape)"),
            ));
        }
        let role = entry.get("role").and_then(Value::as_str).unwrap_or("");
        if !matches!(
            role,
            "lm" | "contrastive" | "mw_disposition" | "confidence" | "narration_adapter"
        ) {
            return Err(SdkError::new("package_invalid", format!("{name}.role")));
        }
        tensor_roles.push((name.to_string(), role.to_string()));
        let dtype = entry.get("dtype").and_then(Value::as_str).unwrap_or("");
        if !matches!(dtype, "cq2" | "cq4" | "f16" | "f32" | "i8") {
            return Err(SdkError::new("package_invalid", format!("{name}.dtype")));
        }
        if matches!(dtype, "cq2" | "cq4") {
            let groups = entry
                .get("n_params")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .div_ceil(128);
            let data_nbytes = range(entry, "data").map(|(_, n)| n).unwrap_or(0);
            let expected_codebook = if dtype == "cq2" {
                "gaussian-lloyd-q2-v1"
            } else {
                "gaussian-lloyd-q4-v1"
            };
            if entry.get("group_size").and_then(Value::as_u64) != Some(128)
                || entry.get("transform").and_then(Value::as_str) != Some("wht")
                || entry.get("codebook").and_then(Value::as_str) != Some(expected_codebook)
                || range(entry, "scales").map(|(_, n)| n) != Some(groups * 2)
                || range(entry, "bit_map").map(|(_, n)| n) != Some(groups.div_ceil(8))
                || data_nbytes < groups * 32
                || data_nbytes > groups * 64
                || data_nbytes % 32 != 0
                || (dtype == "cq2" && data_nbytes == groups * 64)
                || (dtype == "cq4" && data_nbytes != groups * 64)
            {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{name} CQ2 metadata"),
                ));
            }
        } else if let Some((_, data_nbytes)) = range(entry, "data") {
            let expected = entry.get("n_params").and_then(Value::as_u64).unwrap_or(0)
                * match dtype {
                    "f32" => 4,
                    "f16" => 2,
                    "i8" => 1,
                    _ => 0,
                };
            if data_nbytes != expected
                || entry.get("transform").and_then(Value::as_str) != Some("none")
                || entry.get("codebook").and_then(Value::as_str) != Some("none")
                || entry.get("group_size").is_some()
                || entry.get("scales").is_some()
                || entry.get("bit_map").is_some()
            {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{name} payload byte length"),
                ));
            }
        }
        for field in ["data", "scales", "bit_map"] {
            if let Some((offset, nbytes)) = range(entry, field) {
                let end = offset
                    .checked_add(nbytes)
                    .ok_or_else(|| SdkError::from_id("package_range_invalid"))?;
                if end > declared {
                    return Err(SdkError::new(
                        "package_range_invalid",
                        format!("{name}.{field} exceeds payload"),
                    ));
                }
                if nbytes > 0 {
                    ranges.push((offset, end, format!("{name}.{field}")));
                }
            } else if field == "data" {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{name}.data is required"),
                ));
            }
        }
    }
    ranges.sort_by_key(|r| r.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(SdkError::new(
                "package_range_invalid",
                format!("{} overlaps {}", pair[0].2, pair[1].2),
            ));
        }
    }
    let heads = manifest
        .get("heads")
        .and_then(Value::as_object)
        .ok_or_else(|| SdkError::new("package_invalid", "heads object is required"))?;
    only_keys(
        &Value::Object(heads.clone()),
        &[
            "lm",
            "contrastive",
            "mw_disposition",
            "confidence",
            "narration_adapter",
        ],
        "heads",
    )?;
    for (head_name, head) in heads {
        only_keys(
            head,
            &[
                "present",
                "trained",
                "status",
                "tensor_prefixes",
                "training_receipt_sha256",
                "label_codebook_sha256",
            ],
            &format!("heads.{head_name}"),
        )?;
        if let Some(receipt) = head.get("training_receipt_sha256").and_then(Value::as_str) {
            if !is_sha256(receipt) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("heads.{head_name}.training_receipt_sha256"),
                ));
            }
        }
        if let Some(codebook) = head.get("label_codebook_sha256").and_then(Value::as_str) {
            if !is_sha256(codebook) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("heads.{head_name}.label_codebook_sha256"),
                ));
            }
        }
        if !head
            .get("present")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        for prefix in head
            .get("tensor_prefixes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !is_component_path(prefix)
                || !tensor_roles
                    .iter()
                    .any(|(name, role)| role == head_name && prefix_matches(name, prefix))
            {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("heads.{head_name} prefix {prefix} matches no same-role tensor"),
                ));
            }
        }
        validate_ready_head_contract(manifest, head_name, head, entries)?;
    }
    Ok(())
}

pub fn load_package(package_dir: &Path, verify_hashes: bool) -> Result<ModelPackage, SdkError> {
    let root = package_dir.canonicalize().map_err(|_| {
        SdkError::new(
            "file_not_found",
            format!("missing package dir: {}", package_dir.display()),
        )
    })?;
    if !root.is_dir() {
        return Err(SdkError::new(
            "file_not_found",
            "package root is not a directory",
        ));
    }
    let manifest_path = root.join("mei-model.json");
    let manifest_metadata = fs::symlink_metadata(&manifest_path).map_err(|_| {
        SdkError::new(
            "file_not_found",
            format!("missing mei-model.json: {}", manifest_path.display()),
        )
    })?;
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.is_file() {
        return Err(SdkError::new(
            "package_path_unsafe",
            "mei-model.json must be a regular non-symlink file",
        ));
    }
    let text = fs::read_to_string(&manifest_path).map_err(|_| {
        SdkError::new(
            "file_not_found",
            format!("missing mei-model.json: {}", manifest_path.display()),
        )
    })?;
    let manifest: Value = serde_json::from_str(&text)
        .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
    let (generation, heads) = parse_manifest(&manifest)?;
    let mut training_receipts_verified = false;
    let mut resource_measurement_verified = false;
    let mut tool_index = None;
    if verify_hashes {
        if generation == PackageGeneration::V2 {
            verify_v2_files(&root, &manifest, &manifest_path)?;
            training_receipts_verified = verify_training_receipts(&root, &manifest)?;
            resource_measurement_verified = verify_resource_measurement_receipt(&root, &manifest)?;
        }
        let tokenizer = manifest
            .get("tokenizer")
            .ok_or_else(|| SdkError::new("package_invalid", "tokenizer is required"))?;
        verify_file(&root, tokenizer)?;
        if let (Some(rel), Some(expected)) = (
            tokenizer.get("vocab_file").and_then(Value::as_str),
            tokenizer.get("vocab_sha256").and_then(Value::as_str),
        ) {
            let path = package_path(&root, rel)?;
            let digest = file_sha256(&path)?;
            if digest != expected {
                return Err(SdkError::new(
                    "package_hash_mismatch",
                    format!("{rel} sha256 mismatch"),
                ));
            }
        }
        let tensor_spec = match generation {
            PackageGeneration::V1ReadOnly => manifest.get("weights"),
            PackageGeneration::V2 => manifest.get("tensor_container"),
        }
        .ok_or_else(|| SdkError::new("package_invalid", "tensor payload is required"))?;
        let tensor_path = verify_file(&root, tensor_spec)?;
        if let Some(head_artifact) = manifest.pointer("/heads/artifact") {
            let _ = verify_file(&root, head_artifact)?;
        }
        if generation == PackageGeneration::V2 {
            let actual = fs::metadata(&tensor_path)
                .map_err(|_| SdkError::from_id("file_not_found"))?
                .len();
            validate_v2_directory(&manifest, Some(actual))?;
            let bytes = fs::read(&tensor_path).map_err(|_| SdkError::from_id("file_not_found"))?;
            let packed = PackedWeights::parse(bytes)?;
            validate_container_directory(&manifest, &packed)?;
            tool_index = verify_tool_index(&root, &manifest, &packed)?;
        }
    }
    Ok(ModelPackage {
        path: root,
        manifest,
        heads,
        verified_hashes: verify_hashes,
        inference_payload_verified: verify_hashes,
        training_receipts_verified,
        resource_measurement_verified,
        tool_index,
        generation,
    })
}

pub fn package_from_manifest(
    manifest: Value,
    _verified_hashes: bool,
) -> Result<ModelPackage, SdkError> {
    let (generation, heads) = parse_manifest(&manifest)?;
    Ok(ModelPackage {
        path: PathBuf::new(),
        manifest,
        heads,
        // A manifest without its filesystem payload can never establish hash
        // or complete-file verification, regardless of the caller's claim.
        verified_hashes: false,
        inference_payload_verified: false,
        training_receipts_verified: false,
        resource_measurement_verified: false,
        tool_index: None,
        generation,
    })
}

pub fn validate_in_memory_payload(manifest: &Value, actual_bytes: usize) -> Result<(), SdkError> {
    if manifest.get("package_format").and_then(Value::as_str) == Some(V2) {
        validate_v2_directory(manifest, Some(actual_bytes as u64))?;
    }
    Ok(())
}

/// Validate the separately inventoried frozen tool index supplied to an
/// in-memory/WASM load.  A v2 runtime cannot reconstruct this artifact from a
/// request catalog because it is bound to the final LM, contrastive R1 head,
/// tokenizer, serializer and schemas.
pub fn validate_in_memory_tool_index(
    manifest: &Value,
    packed: &PackedWeights,
    bytes: &[u8],
) -> Result<ToolIndex, SdkError> {
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| row.get("role").and_then(Value::as_str) == Some("tool_index"))
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err(SdkError::new(
            "package_invalid",
            "v2 in-memory runtime requires exactly one inventoried tool index",
        ));
    }
    let expected_sha = rows[0].get("sha256").and_then(Value::as_str).unwrap_or("");
    let expected_nbytes = rows[0].get("nbytes").and_then(Value::as_u64).unwrap_or(0);
    if sha256_bytes(bytes) != expected_sha || bytes.len() as u64 != expected_nbytes {
        return Err(SdkError::new(
            "package_hash_mismatch",
            "in-memory tool index hash/size mismatch",
        ));
    }
    let index = ToolIndex::parse(bytes)?;
    if index.model_sha256
        != manifest
            .pointer("/tensor_container/sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
        || index.head_sha256 != contrastive_head_sha256(packed)?
        || index.tokenizer_sha256
            != manifest
                .pointer("/tokenizer/sha256")
                .and_then(Value::as_str)
                .unwrap_or("")
    {
        return Err(SdkError::new(
            "package_hash_mismatch",
            "in-memory tool index model/head/tokenizer fingerprint mismatch",
        ));
    }
    Ok(index)
}

pub fn validate_container_directory(
    manifest: &Value,
    packed: &PackedWeights,
) -> Result<(), SdkError> {
    let directory = manifest
        .pointer("/tensor_container/directory")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("package_invalid", "tensor_container.directory"))?;
    if directory.len() != packed.tensors.len() {
        return Err(SdkError::new(
            "package_invalid",
            "manifest/container tensor count mismatch",
        ));
    }
    for item in directory {
        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
        let actual = packed.entry(name)?;
        let shape: Vec<usize> = item
            .get("shape")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .map(|value| value as usize)
            .collect();
        let manifest_range = |field: &str| {
            (
                item.pointer(&format!("/{field}/offset"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                item.pointer(&format!("/{field}/nbytes"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
            )
        };
        if actual.shape != shape
            || actual.n_params != item.get("n_params").and_then(Value::as_u64).unwrap_or(0) as usize
            || actual.dtype != item.get("dtype").and_then(Value::as_str).unwrap_or("")
            || actual.role != item.get("role").and_then(Value::as_str).unwrap_or("")
            || actual.transform != item.get("transform").and_then(Value::as_str).unwrap_or("")
            || actual.codebook != item.get("codebook").and_then(Value::as_str).unwrap_or("")
            || (actual.packed_offset, actual.packed_nbytes) != manifest_range("data")
            || (actual.scale_offset, actual.scale_nbytes) != manifest_range("scales")
            || (actual.bit_map_offset, actual.bit_map_nbytes) != manifest_range("bit_map")
            || (matches!(actual.dtype.as_str(), "cq2" | "cq4")
                && actual.block_size
                    != item.get("group_size").and_then(Value::as_u64).unwrap_or(0) as usize)
        {
            return Err(SdkError::new(
                "package_invalid",
                format!("manifest/container directory mismatch for {name}"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn canonical_51m_geometry_is_exact() {
        let geometry = canonical_lm_tensor_geometry();
        assert_eq!(geometry.len(), 400);
        assert_eq!(
            geometry
                .iter()
                .map(|(_, shape)| shape.iter().product::<u64>())
                .sum::<u64>(),
            51_463_797
        );
        assert_eq!(
            geometry
                .iter()
                .filter(|(name, shape)| name.ends_with(".attn_gate") && shape.is_empty())
                .count(),
            27
        );
        assert_eq!(portable_lm_dtype("embed.weight"), "cq4");
        assert_eq!(portable_lm_dtype("blocks.0.attn.q_proj.weight"), "cq2");
        assert_eq!(portable_lm_dtype("engrams.0.tables"), "cq2");
        assert_eq!(portable_lm_dtype("engrams.0.taps"), "f16");
        assert_eq!(portable_lm_dtype("blocks.0.attn_gate"), "f16");
    }
}
