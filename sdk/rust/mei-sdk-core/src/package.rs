use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::canonical::sha256_bytes;
use crate::error::SdkError;
use crate::version::sdk_versions;

const REQUIRED_HEADS: [&str; 4] = ["lm", "contrastive", "mw_disposition", "confidence"];

#[derive(Debug, Clone, Deserialize)]
pub struct HeadStatus {
    pub present: bool,
    pub trained: bool,
    pub status: String,
}

impl HeadStatus {
    pub fn to_value(&self) -> Value {
        json!({
            "present": self.present,
            "trained": self.trained,
            "status": self.status,
        })
    }

    pub fn is_missing_or_untrained(&self) -> bool {
        self.status == "missing"
            || self.status == "untrained"
            || !self.present
            || !self.trained
    }
}

#[derive(Debug, Clone)]
pub struct HeadReport {
    pub lm: HeadStatus,
    pub contrastive: HeadStatus,
    pub mw_disposition: HeadStatus,
    pub confidence: HeadStatus,
}

impl HeadReport {
    pub fn to_value(&self) -> Value {
        json!({
            "lm": self.lm.to_value(),
            "contrastive": self.contrastive.to_value(),
            "mw_disposition": self.mw_disposition.to_value(),
            "confidence": self.confidence.to_value(),
        })
    }

    pub fn missing(&self) -> Vec<String> {
        let pairs = [
            ("lm", &self.lm),
            ("contrastive", &self.contrastive),
            ("mw_disposition", &self.mw_disposition),
            ("confidence", &self.confidence),
        ];
        pairs
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
}

impl ModelPackage {
    pub fn package_id(&self) -> String {
        self.manifest
            .get("package_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    pub fn packed_inference_ready(&self) -> bool {
        let weights = self.manifest.get("weights");
        let fmt = weights.and_then(|w| w.get("format")).and_then(Value::as_str);
        let scheme = weights
            .and_then(|w| w.get("quantization"))
            .and_then(|q| q.get("scheme"))
            .and_then(Value::as_str);
        fmt == Some("mei-q4-packed-v1")
            && matches!(
                scheme,
                Some("q4") | Some("cq2") | Some("qat-q4") | Some("qat-cq2")
            )
    }

    pub fn capabilities(&self) -> Value {
        json!({
            "package_id": self.package_id(),
            "release_class": self.manifest.get("release_class"),
            "inference": self.packed_inference_ready(),
            "protocol": true,
            "heads": self.heads.to_value(),
            "missing_or_untrained_heads": self.heads.missing(),
            "hash_verified": self.verified_hashes,
            "versions": sdk_versions(),
        })
    }
}

fn head_from(raw: &Value, name: &str) -> Result<HeadStatus, SdkError> {
    let item = raw.get(name).ok_or_else(|| {
        SdkError::new("package_invalid", format!("heads.{name} must be listed explicitly"))
    })?;
    Ok(HeadStatus {
        present: item.get("present").and_then(Value::as_bool).unwrap_or(false),
        trained: item.get("trained").and_then(Value::as_bool).unwrap_or(false),
        status: item
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("missing")
            .to_string(),
    })
}

fn file_sha256(path: &Path) -> Result<String, SdkError> {
    let bytes = fs::read(path).map_err(|_| {
        SdkError::new("file_not_found", format!("missing file: {}", path.display()))
    })?;
    Ok(sha256_bytes(&bytes))
}

pub fn load_package(package_dir: &Path, verify_hashes: bool) -> Result<ModelPackage, SdkError> {
    let root = package_dir
        .canonicalize()
        .map_err(|_| SdkError::new("file_not_found", format!("missing package dir: {}", package_dir.display())))?;
    let manifest_path = root.join("mei-model.json");
    let text = fs::read_to_string(&manifest_path).map_err(|_| {
        SdkError::new(
            "file_not_found",
            format!("missing mei-model.json: {}", manifest_path.display()),
        )
    })?;
    let manifest: Value = serde_json::from_str(&text)
        .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
    if manifest.get("package_format").and_then(Value::as_str) != Some("mei-model-package-v1") {
        return Err(SdkError::new(
            "package_invalid",
            "package_format must be mei-model-package-v1",
        ));
    }
    let product = manifest.get("product").and_then(Value::as_str).unwrap_or("");
    if product != "mei-1.0-58m" && product != "mei-1.0-51m" {
        return Err(SdkError::new(
            "package_invalid",
            "product must be mei-1.0-58m or mei-1.0-51m",
        ));
    }
    let heads_raw = manifest
        .get("heads")
        .ok_or_else(|| SdkError::new("package_invalid", "heads object is required"))?;
    for name in REQUIRED_HEADS {
        let _ = head_from(heads_raw, name)?;
    }
    let heads = HeadReport {
        lm: head_from(heads_raw, "lm")?,
        contrastive: head_from(heads_raw, "contrastive")?,
        mw_disposition: head_from(heads_raw, "mw_disposition")?,
        confidence: head_from(heads_raw, "confidence")?,
    };
    let mut verified = false;
    if verify_hashes {
        for key in ["tokenizer", "weights"] {
            let spec = manifest
                .get(key)
                .ok_or_else(|| SdkError::new("package_invalid", format!("missing {key}")))?;
            let rel = spec
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| SdkError::new("package_invalid", format!("{key}.file")))?;
            let expected = spec
                .get("sha256")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            let path = root.join(rel);
            if !path.is_file() {
                return Err(SdkError::new("file_not_found", format!("missing {key} file: {rel}")));
            }
            let digest = file_sha256(&path)?;
            if digest != expected {
                return Err(SdkError::new(
                    "package_hash_mismatch",
                    format!("{key} sha256 mismatch: expected {expected}, got {digest}"),
                ));
            }
        }
        verified = true
    }
    if verify_hashes {
        if let Some(rel) = manifest
            .get("tokenizer")
            .and_then(|t| t.get("vocab_file"))
            .and_then(Value::as_str)
        {
            if let Some(expected) = manifest
                .get("tokenizer")
                .and_then(|t| t.get("vocab_sha256"))
                .and_then(Value::as_str)
            {
                let path = root.join(rel);
                if path.is_file() {
                    let digest = file_sha256(&path)?;
                    if digest != expected.to_ascii_lowercase() {
                        return Err(SdkError::new(
                            "package_hash_mismatch",
                            format!("tokenizer vocab sha256 mismatch: expected {expected}, got {digest}"),
                        ));
                    }
                }
            }
        }
    }
    Ok(ModelPackage {
        path: root,
        manifest,
        heads,
        verified_hashes: verified,
    })
}

pub fn package_from_manifest(manifest: Value, verified_hashes: bool) -> Result<ModelPackage, SdkError> {
    if manifest.get("package_format").and_then(Value::as_str) != Some("mei-model-package-v1") {
        return Err(SdkError::new(
            "package_invalid",
            "package_format must be mei-model-package-v1",
        ));
    }
    let product = manifest.get("product").and_then(Value::as_str).unwrap_or("");
    if product != "mei-1.0-58m" && product != "mei-1.0-51m" {
        return Err(SdkError::new(
            "package_invalid",
            "product must be mei-1.0-58m or mei-1.0-51m",
        ));
    }
    let heads_raw = manifest
        .get("heads")
        .ok_or_else(|| SdkError::new("package_invalid", "heads object is required"))?;
    let heads = HeadReport {
        lm: head_from(heads_raw, "lm")?,
        contrastive: head_from(heads_raw, "contrastive")?,
        mw_disposition: head_from(heads_raw, "mw_disposition")?,
        confidence: head_from(heads_raw, "confidence")?,
    };
    Ok(ModelPackage {
        path: PathBuf::new(),
        manifest,
        heads,
        verified_hashes,
    })
}
