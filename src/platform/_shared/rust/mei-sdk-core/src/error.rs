use serde::Serialize;
use serde_json::json;
use thiserror::Error;

use crate::version::{error_id_from_code, error_row};

#[derive(Debug, Clone, Serialize)]
pub struct ErrorInfo {
    pub code: i32,
    pub id: String,
    pub message: String,
}

impl std::fmt::Display for ErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.id)
    }
}

impl ErrorInfo {
    pub fn new(id: &str, message: Option<&str>) -> Self {
        let row = error_row(id);
        Self {
            code: row.code,
            id: row.id,
            message: message.unwrap_or(&row.message).to_string(),
        }
    }

    pub fn to_value(&self) -> serde_json::Value {
        json!({
            "code": self.code,
            "id": self.id,
            "message": self.message,
        })
    }
}

#[derive(Debug, Error)]
pub enum SdkError {
    #[error("{0}")]
    Coded(ErrorInfo),
}

impl SdkError {
    pub fn new(id: &str, message: impl Into<String>) -> Self {
        let msg = message.into();
        Self::Coded(ErrorInfo::new(id, Some(&msg)))
    }

    pub fn from_id(id: &str) -> Self {
        Self::Coded(ErrorInfo::new(id, None))
    }

    pub fn info(&self) -> &ErrorInfo {
        match self {
            Self::Coded(info) => info,
        }
    }

    pub fn code(&self) -> i32 {
        self.info().code
    }

    pub fn from_code(code: i32, message: impl Into<String>) -> Self {
        Self::new(&error_id_from_code(code), message)
    }
}
