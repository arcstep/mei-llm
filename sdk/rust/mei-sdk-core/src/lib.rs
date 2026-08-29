//! MEI Runtime core. Product name is `mei-1.0-58m Runtime`. This crate does not
//! expose Needle 2 / libneedle identifiers in its public API.

pub mod canonical;
pub mod engine;
pub mod error;
pub mod infer;
pub mod model;
pub mod package;
pub mod packed;
pub mod protocol;
pub mod retrieval;
pub mod tokenizer;
pub mod version;
pub mod vocab;

pub use engine::{complete_request, Engine, LoopResult, Session, TurnResult};
pub use error::{ErrorInfo, SdkError};
pub use package::{load_package, ModelPackage};
pub use protocol::{parse_v2_text, render_request, ParsedCall};
pub use canonical::schema_fingerprint;
pub use version::sdk_versions;

pub const SDK_ROOT_FROM_CORE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
