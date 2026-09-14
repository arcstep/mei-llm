use mei_sdk_core::{
    canonical, load_package, parse_v2_text, render_request, schema_fingerprint, sdk_versions,
    Engine,
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn sdk_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden(name: &str) -> Value {
    let text = std::fs::read_to_string(sdk_root().join("spec/golden").join(name)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn zero_wall(mut turn: Value) -> Value {
    if let Some(stats) = turn.get_mut("stats") {
        stats["wall_ms"] = json!(0);
    }
    turn
}

#[test]
fn versions_are_experimental_and_unrelated_to_needle() {
    let v = sdk_versions();
    let blob = v.to_string();
    assert!(blob.contains("experimental"));
    assert!(!blob.to_lowercase().contains("needle"));
    assert_eq!(v["product"], "mei-1.2-51m Runtime");
}

#[test]
fn schema_fingerprint_matches_python_golden() {
    let gold = golden("schema_fingerprint.json");
    let tools = gold["tools"].as_array().unwrap();
    assert_eq!(schema_fingerprint(tools), gold["sha256"].as_str().unwrap());
    assert_eq!(
        canonical::dumps_canonical(&canonical::compact_tools(tools)),
        gold["canonical"].as_str().unwrap()
    );
}

#[test]
fn parse_cases_match_python_golden() {
    let gold = golden("parse_cases.json");
    for case in gold["cases"].as_array().unwrap() {
        let parsed = parse_v2_text(case["text"].as_str().unwrap()).to_value();
        assert_eq!(parsed, case["parsed"]);
    }
}

#[test]
fn render_request_matches_python_golden() {
    let gold = golden("render_request.json");
    let request = &gold["request"];
    let tools = request["oracle_tools"].as_array().unwrap();
    let got = render_request(request, tools).unwrap();
    assert_eq!(got["prompt"], gold["prompt"]);
    assert_eq!(got["schema_fingerprint"], gold["schema_fingerprint"]);
}

#[test]
fn tiny_package_reports_missing_heads() {
    let pkg = load_package(&sdk_root().join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    let missing = pkg.heads.missing();
    assert!(missing.contains(&"contrastive".into()));
    assert!(missing.contains(&"mw_disposition".into()));
    assert!(missing.contains(&"confidence".into()));
    let caps = pkg.capabilities();
    assert_eq!(caps["compatibility_mode"], "v1-read-only-degraded");
    assert_eq!(caps["product_ready"], false);
}

#[test]
fn turn_results_match_python_golden() {
    let engine =
        Engine::load(&sdk_root().join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    let light = json!({"name":"light.set","parameters":{"type":"object","properties":{}}});
    let cases = [
        (
            "refuse",
            json!({"query":"开灯","oracle_tools":[light.clone()],"candidate_text":"[]"}),
        ),
        (
            "call",
            json!({"query":"开灯","oracle_tools":[light.clone()],"candidate_text":"[{\"name\":\"light.set\",\"arguments\":{}}]"}),
        ),
        (
            "too_many",
            json!({
                "query":"x",
                "oracle_tools":[
                    {"name":"t0","parameters":{"type":"object","properties":{}}},
                    {"name":"t1","parameters":{"type":"object","properties":{}}},
                    {"name":"t2","parameters":{"type":"object","properties":{}}},
                    {"name":"t3","parameters":{"type":"object","properties":{}}},
                    {"name":"t4","parameters":{"type":"object","properties":{}}},
                    {"name":"t5","parameters":{"type":"object","properties":{}}}
                ],
                "candidate_text":"[]"
            }),
        ),
        (
            "leak",
            json!({"query":"gold_route_id=1","oracle_tools":[light.clone()],"candidate_text":"[]"}),
        ),
        (
            "unavailable",
            json!({"query":"开灯","oracle_tools":[light]}),
        ),
    ];
    for (name, request) in cases {
        let mut session = engine.create_session().unwrap();
        let got = zero_wall(session.complete(&request).unwrap());
        assert_eq!(
            got["wire_version"], "mei-runtime-wire-v2",
            "mismatch at {name}"
        );
        assert!(matches!(
            got["kind"].as_str(),
            Some("call" | "refuse" | "error")
        ));
        assert_eq!(
            got["capabilities"]["compatibility_mode"],
            "v1-read-only-degraded"
        );
    }
}

#[test]
fn cancel_returns_cancelled() {
    let engine =
        Engine::load(&sdk_root().join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    let mut session = engine.create_session().unwrap();
    session.cancel();
    let turn = session
        .complete(
            &json!({"query":"开灯","oracle_tools":[{"name":"light.set"}],"candidate_text":"[]"}),
        )
        .unwrap();
    assert_eq!(turn["error"]["id"], "cancelled");
}
