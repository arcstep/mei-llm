use mei_sdk_core::Engine;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let package = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var("MEI_51M_PACKAGE_DIR").ok().map(PathBuf::from))
        .expect("package directory argument or MEI_51M_PACKAGE_DIR");
    let load_started = Instant::now();
    let mut engine = Engine::load(&package, true).expect("verified quantized engine");
    let catalog = engine
        .package
        .tool_index
        .as_ref()
        .expect("native v2 tool index")
        .records
        .iter()
        .map(|record| record.schema.clone())
        .collect::<Vec<_>>();
    engine
        .register_tools(&catalog)
        .expect("register frozen tool catalog");
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut session = engine.create_session().expect("session");
    let infer_started = Instant::now();
    let result = session
        .complete(&json!({
            "wire_version": "mei-runtime-wire-v2",
            "query": "厨房灯打开",
            "context": {},
            "evidence": [],
            "history": [],
            "tool_results": [],
            "permissions": {},
            "state": {},
            "decode_mode": "constrained",
            "max_new": 1
        }))
        .expect("complete");
    let greedy_ms = infer_started.elapsed().as_secs_f64() * 1000.0;
    let n_new = result["generated_token_ids"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    let prefill_topk = result["prefill_topk_ids"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    let runtime_cache = result["runtime_cache"].clone();
    let bounded_int8_cache = runtime_cache["kv_storage_dtype"] == "int8"
        && runtime_cache["cache_growth_bounded"] == true;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "ok": prefill_topk == 5 && bounded_int8_cache,
            "load_ms": load_ms,
            "greedy_ms": greedy_ms,
            "n_new": n_new,
            "prefill_topk_count": prefill_topk,
            "numeric_forward_ran": prefill_topk == 5,
            "bounded_int8_cache_ran": bounded_int8_cache,
            "measurement_profile": "prefill-plus-one-bounded-decode-v1",
            "runtime_cache": runtime_cache,
            "quantized_only": true,
            "package_dir": package
        }))
        .unwrap()
    );
}
