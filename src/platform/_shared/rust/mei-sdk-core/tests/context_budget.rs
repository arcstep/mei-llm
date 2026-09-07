use std::path::PathBuf;

use mei_sdk_core::context_budget::{
    plan_candidate_batches, render_budgeted_request, RankedCandidate,
};
use mei_sdk_core::vocab::Vocab;
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../..")
}

fn load_fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(repo_root().join("platform/_shared/spec/golden/context_budget_v2.json"))
            .expect("context budget golden"),
    )
    .expect("context budget JSON")
}

fn fixture_inputs(fixture: &Value) -> (Vec<Value>, Value) {
    let recipe = &fixture["recipe"];
    let tool_unit = recipe["tool_description_unit"].as_str().unwrap();
    let tool_repeat = recipe["tool_description_repeat"].as_u64().unwrap() as usize;
    let parameter_unit = recipe["parameter_description_unit"].as_str().unwrap();
    let parameter_repeat = recipe["parameter_description_repeat"].as_u64().unwrap() as usize;
    let history = recipe["history_unit"]
        .as_str()
        .unwrap()
        .repeat(recipe["history_repeat"].as_u64().unwrap() as usize);
    let query = recipe["query_unit"]
        .as_str()
        .unwrap()
        .repeat(recipe["query_repeat"].as_u64().unwrap() as usize);
    let tools = (0..recipe["tool_count"].as_u64().unwrap())
        .map(|index| {
            json!({
                "name":format!("device.action_{index}"),
                "description":tool_unit.repeat(tool_repeat),
                "parameters":{
                    "type":"object",
                    "title":{"description":"注释内部描述不得参与配额".repeat(30)},
                    "examples":[{"level":3,"description":"示例内部描述同样整体删除".repeat(30)}],
                    "properties":{
                        "level":{
                            "type":"integer","minimum":0,"maximum":10,"multipleOf":1,
                            "description":parameter_unit.repeat(parameter_repeat),
                        },
                        "mode":{
                            "type":["string","null"],"enum":["auto","eco",null],
                            "description":parameter_unit.repeat(parameter_repeat),
                        }
                    },
                    "required":["level"]
                }
            })
        })
        .collect::<Vec<_>>();
    let request = json!({
        "wire_version":"mei-runtime-wire-v2",
        "query":query,
        "context":{
            "low":{"priority":0,"text":"低优先上下文".repeat(200)},
            "high":{"priority":10,"text":"高优先上下文".repeat(200)}
        },
        "evidence":[
            {"priority":0,"text":"低优先证据".repeat(200)},
            {"priority":10,"text":"高优先证据".repeat(200)}
        ],
        "permissions":{"scopes":["device:write"]},
        "state":{"device":"online"},
        "history":[
            {"role":"user","content":history},
            {"role":"assistant","content":history},
            {"role":"user","content":history}
        ],
        "tool_results":[
            {"wire_version":"mei-runtime-wire-v2","call_id":"old-call","status":"ok","payload":{"text":"旧结果".repeat(500)},"provenance":{"source":"fixture","verified":true}},
            {"wire_version":"mei-runtime-wire-v2","call_id":"latest-call","status":"ok","payload":{"text":"最新结果".repeat(500)},"provenance":{"source":"fixture","verified":true}}
        ]
    });
    (tools, request)
}

#[test]
fn fixed_five_batching_keeps_tail_order_and_applies_expand_threshold() {
    let rows = (0..12)
        .map(|index| RankedCandidate {
            tool_id: format!("tool-{index:02}"),
            schema: json!({"name":format!("tool-{index:02}")}),
            raw_score: 1.0 - index as f32 * 0.01,
            relevance: if index < 9 { 0.8 } else { 0.4 },
            rank: index + 1,
        })
        .collect();
    let plan = plan_candidate_batches(rows, 0.3, 0.7, None).unwrap();
    assert_eq!(
        plan.batches.iter().map(Vec::len).collect::<Vec<_>>(),
        [5, 4]
    );
    assert_eq!(plan.non_expandable.len(), 3);
    assert_eq!(plan.batches[1][0].tool_id, "tool-05");
}

#[test]
fn discard_never_cuts_first_batch_below_five() {
    // 回归（与 Python 侧同语义）：discard 砍进 rank 前 5（仅 4 个过闸）时首批仍必须为 5
    let relevances = [0.90, 0.88, 0.86, 0.84, 0.70, 0.30, 0.28];
    let rows = relevances
        .iter()
        .enumerate()
        .map(|(index, relevance)| RankedCandidate {
            tool_id: format!("tool.{index:02}"),
            schema: json!({"name":format!("tool.{index:02}")}),
            raw_score: 1.0 - index as f32 / 100.0,
            relevance: *relevance,
            rank: index + 1,
        })
        .collect();
    let plan = plan_candidate_batches(rows, 0.80, 0.80, None).unwrap();
    assert_eq!(plan.batches.iter().map(Vec::len).collect::<Vec<_>>(), [5]);
    assert_eq!(
        plan.batches[0]
            .iter()
            .map(|row| row.tool_id.as_str())
            .collect::<Vec<_>>(),
        ["tool.00", "tool.01", "tool.02", "tool.03", "tool.04"]
    );
    assert_eq!(
        plan.discarded
            .iter()
            .map(|row| row.tool_id.as_str())
            .collect::<Vec<_>>(),
        ["tool.04", "tool.05", "tool.06"]
    );
    assert_eq!(plan.candidates.len(), 5);
}

#[test]
fn all_below_discard_yields_no_batches() {
    let relevances = [0.2, 0.1, 0.05];
    let rows = relevances
        .iter()
        .enumerate()
        .map(|(index, relevance)| RankedCandidate {
            tool_id: format!("tool.{index:02}"),
            schema: json!({"name":format!("tool.{index:02}")}),
            raw_score: 1.0 - index as f32 / 100.0,
            relevance: *relevance,
            rank: index + 1,
        })
        .collect();
    let plan = plan_candidate_batches(rows, 0.50, 0.60, None).unwrap();
    assert!(plan.batches.is_empty());
    assert_eq!(plan.discarded.len(), 3);
}

#[test]
fn rust_budget_projection_matches_the_python_oracle_golden() {
    let fixture = load_fixture();
    let (tools, request) = fixture_inputs(&fixture);
    let vocab = Vocab::from_package_payload(
        &std::fs::read(repo_root().join("models/mei-1.2-51m/tokenizer/zh-24k-v1.model"))
            .expect("zh tokenizer"),
    )
    .expect("portable tokenizer");
    let out = render_budgeted_request(
        &vocab,
        &request,
        &tools,
        &[0.95, 0.85, 0.75, 0.65, 0.55],
        fixture["profile"].as_str().unwrap(),
        fixture["output_reserve"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let expected = &fixture["expected"];
    assert_eq!(out["selected_tools"], expected["selected_tools"]);
    assert_eq!(
        out["schema_projection_sha256"],
        expected["schema_projection_sha256"]
    );
    assert_eq!(out["prompt_tokens"], expected["prompt_tokens"]);
    assert_eq!(out["schema_budget"]["used"], expected["schema_tokens"]);
    assert_eq!(
        out["schema_budget"]["compression_level"],
        expected["schema_compression_level"]
    );
    assert_eq!(
        out["schema_budget"]["omitted_description_tokens"],
        expected["omitted_description_tokens"]
    );
    for key in [
        "compression_level",
        "dropped_history",
        "compacted_tool_results",
        "dropped_evidence",
        "dropped_context_keys",
        "query_clipped",
        "visible_query_tokens",
    ] {
        let expected_key = if key == "compression_level" {
            "input_compression_level"
        } else {
            key
        };
        assert_eq!(out["input_budget"][key], expected[expected_key], "{key}");
    }
    assert_eq!(out["prompt_tokens"].as_u64().unwrap() + 128, 2048);
    assert_eq!(
        out["prompt"]
            .as_str()
            .is_some_and(|value| value.ends_with("<|im_end|>\n<|im_start|>assistant\n")),
        expected["assistant_suffix"].as_bool().unwrap()
    );
    for tool in out["_projected_tools"].as_array().unwrap() {
        let level = &tool["parameters"]["properties"]["level"];
        assert_eq!(level["minimum"], 0);
        assert_eq!(level["maximum"], 10);
        assert_eq!(level["multipleOf"], 1);
        assert_eq!(tool["parameters"]["required"], json!(["level"]));
    }
}

#[test]
fn annotation_names_used_as_parameter_names_remain_structural() {
    let vocab = Vocab::from_package_payload(
        &std::fs::read(repo_root().join("models/mei-1.2-51m/tokenizer/zh-24k-v1.model"))
            .expect("zh tokenizer"),
    )
    .expect("portable tokenizer");
    let tools = (0..5)
        .map(|index| {
            json!({
                "name": format!("device.named_annotation_{index}"),
                "description": "很长的工具描述".repeat(100),
                "parameters": {
                    "type": "object",
                    "title": {"description": "这里是可删除的 schema 注释".repeat(20)},
                    "properties": {
                        "title": {
                            "type": "string",
                            "minLength": 1,
                            "description": "名为 title 的真实参数".repeat(100)
                        }
                    },
                    "required": ["title"]
                }
            })
        })
        .collect::<Vec<_>>();
    let request = json!({
        "wire_version": "mei-runtime-wire-v2",
        "query": "设置标题",
        "context": {},
        "evidence": [],
        "permissions": {},
        "state": {}
    });
    let out = render_budgeted_request(
        &vocab,
        &request,
        &tools,
        &[0.95, 0.85, 0.75, 0.65, 0.55],
        "compact",
        128,
    )
    .unwrap();
    for tool in out["_projected_tools"].as_array().unwrap() {
        let parameters = &tool["parameters"];
        assert!(parameters.get("title").is_none());
        assert_eq!(parameters["properties"]["title"]["type"], "string");
        assert_eq!(parameters["properties"]["title"]["minLength"], 1);
        assert_eq!(parameters["required"], json!(["title"]));
    }
}
