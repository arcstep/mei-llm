//! MEI Runtime core. Product name is `mei-1.0-51m Runtime`. This crate does not
//! expose Needle 2 / libneedle identifiers in its public API.

use std::path::Path;

use serde_json::Value;

pub mod byte_grammar;
pub mod canonical;
pub mod cq2;
pub mod engine;
pub mod error;
pub mod infer;
pub mod model;
pub mod package;
pub mod packed;
pub mod protocol;
pub mod retrieval;
pub mod tokenizer;
pub mod tool_index;
pub mod version;
pub mod vocab;

pub use canonical::{catalog_fingerprint, schema_fingerprint};
pub use engine::{complete_request, Engine, LoopResult, Session, TurnResult};
pub use error::{ErrorInfo, SdkError};
pub use package::{load_package, ModelPackage};
pub use protocol::{
    apply_confidence, parse_v2_text, render_request, validate_generated_call, validate_json_value,
    validate_tool_schema, ParsedCall,
};
pub use version::sdk_versions;

pub const SDK_ROOT_FROM_CORE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Language-neutral v2 API façade. The methods on [`Engine`] and [`Session`]
/// remain available for idiomatic Rust use; these functions freeze the same
/// operation names used by the other bindings.
pub fn version() -> Value {
    sdk_versions()
}

pub fn load_model(package_dir: &Path) -> Result<Engine, SdkError> {
    Engine::load(package_dir, true)
}

pub fn register_tools(engine: &mut Engine, tools: &[Value]) -> Result<Value, SdkError> {
    engine.register_tools(tools)
}

pub fn create_session(engine: &Engine, options: &Value) -> Result<Session, SdkError> {
    engine.create_session_with_options(options)
}

pub fn complete(session: &mut Session, request: &Value) -> Result<TurnResult, SdkError> {
    session.complete(request)
}

pub fn submit_tool_result(session: &mut Session, result: &Value) -> Result<Value, SdkError> {
    session.submit_tool_result(result)
}

pub fn run<F>(session: &mut Session, request: &Value, executor: F) -> Result<LoopResult, SdkError>
where
    F: FnMut(&str, &Value) -> Result<Value, SdkError>,
{
    session.run(request, executor)
}

pub fn narrate(session: &Session, options: &Value) -> Result<Value, SdkError> {
    session.narrate(options)
}

pub fn cancel(session: &mut Session) {
    session.cancel();
}

pub fn close_session(session: &mut Session) {
    session.close();
}

pub fn close_engine(engine: &mut Engine) {
    engine.close();
}
