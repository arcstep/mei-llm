//! Tokenizer identity for a model package. SentencePiece encode is not in this
//! experimental core; Python+MLX remains the golden oracle for token ids.

use std::path::PathBuf;

use crate::error::SdkError;
use crate::package::ModelPackage;

#[derive(Debug, Clone)]
pub struct TokenizerRef {
    pub id: String,
    pub path: PathBuf,
    pub sha256: String,
}

impl TokenizerRef {
    pub fn from_package(package: &ModelPackage) -> Result<Self, SdkError> {
        let tok = package
            .manifest
            .get("tokenizer")
            .ok_or_else(|| SdkError::new("package_invalid", "tokenizer missing"))?;
        let id = tok
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let rel = tok
            .get("file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SdkError::new("package_invalid", "tokenizer.file"))?;
        Ok(Self {
            id,
            path: package.path.join(rel),
            sha256: tok
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
        })
    }

    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, SdkError> {
        Err(SdkError::new(
            "not_implemented",
            "SentencePiece encode is not in the portable core yet; use the Python+MLX oracle",
        ))
    }
}
