use serde::Deserialize;
use serde_json::{json, Value};

const VERSIONS_JSON: &str = include_str!("../../../spec/versions.json");
const ERRORS_JSON: &str = include_str!("../../../spec/errors.json");
const PROTOCOL_JSON: &str = include_str!("../../../spec/protocol.json");

#[derive(Debug, Deserialize)]
struct VersionsFile {
    product: String,
    sdk_semver: String,
    release_class: String,
    wire_version: String,
    model_package_version: String,
    runtime_abi_version: String,
    protocol_id: String,
    serializer_id: String,
}

#[derive(Debug, Deserialize)]
struct ErrorsFile {
    codes: Vec<ErrorRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorRow {
    pub code: i32,
    pub id: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct ProtocolFile {
    protocol_id: String,
    serializer_id: String,
    max_selected_tools: usize,
    max_calls: usize,
    task_contract: String,
    forbidden_markers: Vec<String>,
}

fn versions_file() -> VersionsFile {
    serde_json::from_str(VERSIONS_JSON).expect("versions.json")
}

fn errors_file() -> ErrorsFile {
    serde_json::from_str(ERRORS_JSON).expect("errors.json")
}

fn protocol_file() -> ProtocolFile {
    serde_json::from_str(PROTOCOL_JSON).expect("protocol.json")
}

pub fn sdk_versions() -> Value {
    let v = versions_file();
    json!({
        "sdk_semver": v.sdk_semver,
        "wire_version": v.wire_version,
        "model_package_version": v.model_package_version,
        "runtime_abi_version": v.runtime_abi_version,
        "protocol_id": v.protocol_id,
        "serializer_id": v.serializer_id,
        "release_class": v.release_class,
        "product": v.product,
        "compute_profile": compute_profile(),
    })
}

pub fn compute_profile() -> Value {
    json!({
        "fast_kernels": cfg!(feature = "wasm-fast-kernels"),
        "parallel_kernels": cfg!(feature = "wasm-parallel"),
        "tiled_prefill": cfg!(feature = "wasm-tiled-prefill"),
        "prefix_cache": cfg!(feature = "wasm-prefix-cache"),
        "approximate_integer_codebooks_and_exp": cfg!(feature = "wasm-approx-kernels"),
        "requires_relaxed_simd": false,
        "canonical_numeric_parity_claimed": false,
        "experimental": cfg!(feature = "wasm-fast-kernels") || cfg!(feature = "wasm-prefix-cache"),
    })
}

pub fn wire_version() -> String {
    versions_file().wire_version
}

pub fn error_row(id: &str) -> ErrorRow {
    errors_file()
        .codes
        .into_iter()
        .find(|row| row.id == id)
        .unwrap_or_else(|| panic!("unknown error id {id}"))
}

pub fn error_id_from_code(code: i32) -> String {
    errors_file()
        .codes
        .into_iter()
        .find(|row| row.code == code)
        .map(|row| row.id)
        .unwrap_or_else(|| "invalid_argument".to_string())
}

pub fn protocol_id() -> String {
    protocol_file().protocol_id
}

pub fn serializer_id() -> String {
    protocol_file().serializer_id
}

pub fn max_selected_tools() -> usize {
    protocol_file().max_selected_tools
}

pub fn max_calls() -> usize {
    protocol_file().max_calls
}

pub fn task_contract() -> String {
    protocol_file().task_contract
}

pub fn forbidden_markers() -> Vec<String> {
    protocol_file().forbidden_markers
}
