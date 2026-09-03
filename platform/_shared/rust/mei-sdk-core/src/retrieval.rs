//! Retrieval contract: complete() never silently truncates a catalog.
//! Dense retrieval requires a trained contrastive head; this core reports missing.

use serde_json::Value;

use crate::error::SdkError;
use crate::version::max_selected_tools;

pub fn select_tools<'a>(request: &'a Value) -> Result<&'a [Value], SdkError> {
    if let Some(oracle) = request.get("oracle_tools") {
        if oracle.is_null() {
            // fall through
        } else if let Some(list) = oracle.as_array() {
            if list.len() > max_selected_tools() {
                return Err(SdkError::from_id("too_many_tools"));
            }
            return Ok(list);
        } else {
            return Err(SdkError::new(
                "invalid_argument",
                "oracle_tools must be a list",
            ));
        }
    }
    let catalog = request
        .get("catalog")
        .and_then(Value::as_array)
        .ok_or_else(|| SdkError::new("invalid_argument", "tools must be a list"))?;
    if catalog.len() > max_selected_tools() {
        return Err(SdkError::new(
            "engine_unavailable",
            "catalog exceeds 5 tools and contrastive retrieval is not in this experimental core",
        ));
    }
    Ok(catalog)
}
