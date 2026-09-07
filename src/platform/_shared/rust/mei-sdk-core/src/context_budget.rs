//! Deterministic fixed-five batching and joint 2048-token context budgeting.
//!
//! This is the portable counterpart of `platform/_shared/runtime/context_budget.py`.
//! Only the model-visible schema projection is shortened; callers retain the
//! complete schemas for grammar, JSON Schema and execution validation.

use serde_json::{json, Map, Value};

use crate::canonical::{compact_tools, dumps_canonical, sha256_bytes};
use crate::error::SdkError;
use crate::version::task_contract;
use crate::vocab::Vocab;

pub const CONTEXT_PACKER_ID: &str = "mei-tool-context-packer-v1";
pub const PROJECTION_SERIALIZER_ID: &str = "mei-schema-projection-v1";
pub const RETRIEVAL_BATCH_POLICY_ID: &str = "mei-retrieval-fixed-five-batches-v1";
pub const RETRIEVAL_CALIBRATION_ID: &str = "mei-retrieval-platt-v1";
const ASSISTANT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
pub const MAX_CONTEXT_TOKENS: usize = 2048;
pub const DEFAULT_OUTPUT_RESERVE: usize = 128;
pub const TOOL_BATCH_SIZE: usize = 5;

#[derive(Clone, Debug)]
pub struct RankedCandidate {
    pub tool_id: String,
    pub schema: Value,
    pub raw_score: f32,
    pub relevance: f64,
    pub rank: usize,
}

impl RankedCandidate {
    pub fn to_value(&self) -> Value {
        json!({
            "tool_id": self.tool_id,
            "rank": self.rank,
            "raw_score": self.raw_score,
            "retrieval_relevance": self.relevance,
        })
    }
}

#[derive(Clone, Debug)]
pub struct CandidateBatchPlan {
    pub candidates: Vec<RankedCandidate>,
    pub batches: Vec<Vec<RankedCandidate>>,
    pub discarded: Vec<RankedCandidate>,
    pub non_expandable: Vec<RankedCandidate>,
    pub unscanned: Vec<RankedCandidate>,
    pub limited: bool,
}

pub fn platt_relevance(score: f32, scale: f64, bias: f64) -> Result<f64, SdkError> {
    let z = scale * f64::from(score) + bias;
    if !z.is_finite() {
        return Err(SdkError::new(
            "package_invalid",
            "retrieval calibration input is not finite",
        ));
    }
    Ok(if z >= 0.0 {
        let exp = (-z).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = z.exp();
        exp / (1.0 + exp)
    })
}

pub fn plan_candidate_batches(
    mut ranked: Vec<RankedCandidate>,
    discard_threshold: f64,
    expand_threshold: f64,
    max_candidate_batches: Option<usize>,
) -> Result<CandidateBatchPlan, SdkError> {
    if !discard_threshold.is_finite()
        || !expand_threshold.is_finite()
        || !(0.0..=1.0).contains(&discard_threshold)
        || !(0.0..=1.0).contains(&expand_threshold)
        || expand_threshold < discard_threshold
        || max_candidate_batches == Some(0)
    {
        return Err(SdkError::new(
            "invalid_argument",
            "retrieval thresholds or max_candidate_batches are invalid",
        ));
    }
    ranked.sort_by(|left, right| {
        right
            .raw_score
            .total_cmp(&left.raw_score)
            .then_with(|| left.tool_id.as_bytes().cmp(right.tool_id.as_bytes()))
    });
    let discarded = ranked
        .iter()
        .filter(|row| row.relevance < discard_threshold)
        .cloned()
        .collect::<Vec<_>>();
    // discard 只做 all-or-nothing 可用性闸门（AGENTS.md 候选扫描不变量）：
    // 全灭 → 空批次（no_match 终端）；有任一过闸 → 首批恒为 rank 前 5
    // （不得因阈值把首批砍到 1-2；目录不足 5 个时为实际数量）。
    // 与 Python `_shared/runtime/context_budget.py` 语义逐条一致。
    let eligible = ranked
        .iter()
        .filter(|row| row.relevance >= discard_threshold)
        .cloned()
        .collect::<Vec<_>>();
    let (first, expandable, non_expandable) = if eligible.is_empty() {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let first = ranked
            .iter()
            .take(TOOL_BATCH_SIZE)
            .cloned()
            .collect::<Vec<_>>();
        let tail_eligible = ranked
            .iter()
            .skip(TOOL_BATCH_SIZE)
            .filter(|row| row.relevance >= discard_threshold)
            .cloned()
            .collect::<Vec<_>>();
        let expandable = tail_eligible
            .iter()
            .filter(|row| row.relevance >= expand_threshold)
            .cloned()
            .collect::<Vec<_>>();
        let non_expandable = tail_eligible
            .iter()
            .filter(|row| row.relevance < expand_threshold)
            .cloned()
            .collect::<Vec<_>>();
        (first, expandable, non_expandable)
    };
    let mut all_batches = Vec::<Vec<RankedCandidate>>::new();
    if !first.is_empty() {
        all_batches.push(first.clone());
    }
    all_batches.extend(expandable.chunks(TOOL_BATCH_SIZE).map(<[_]>::to_vec));
    let limit = max_candidate_batches.unwrap_or(all_batches.len());
    let unscanned = all_batches
        .iter()
        .skip(limit)
        .flat_map(|batch| batch.iter().cloned())
        .collect::<Vec<_>>();
    let batches = all_batches.into_iter().take(limit).collect::<Vec<_>>();
    Ok(CandidateBatchPlan {
        // candidates = 模型可见的主候选集 = 恒定的首批（rank 前 5；与 Python 同语义）
        candidates: first,
        batches,
        discarded,
        non_expandable,
        limited: !unscanned.is_empty(),
        unscanned,
    })
}

fn stable_cap(profile: &str) -> Result<usize, SdkError> {
    match profile {
        "compact" => Ok(1024),
        "standard" => Ok(1536),
        _ => Err(SdkError::new(
            "invalid_argument",
            "runtime_profile must be compact or standard",
        )),
    }
}

fn token_count(vocab: &Vocab, text: &str, add_bos: bool) -> usize {
    vocab.encode(text, add_bos).len()
}

fn annotation_key(key: &str) -> bool {
    matches!(
        key,
        "$comment"
            | "comment"
            | "default"
            | "deprecated"
            | "examples"
            | "example"
            | "readOnly"
            | "title"
            | "writeOnly"
            | "description"
    )
}

fn strip_annotations(value: &Value, property_map: bool) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| property_map || !annotation_key(key))
                .map(|(key, child)| {
                    let child_is_property_map =
                        !property_map && key == "properties" && child.is_object();
                    (
                        key.clone(),
                        strip_annotations(child, child_is_property_map),
                    )
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|child| strip_annotations(child, false))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub fn structural_tool_projection(tool: &Value) -> Value {
    json!({
        "name": tool.get("name").and_then(Value::as_str).unwrap_or(""),
        "description": "",
        "parameters": strip_annotations(
            tool.get("parameters").unwrap_or(&json!({"type":"object","properties":{}})),
            false,
        ),
    })
}

#[derive(Clone, Debug)]
enum PathPart {
    Key(String),
    Index(usize),
}

#[derive(Clone, Debug)]
struct DescriptionField {
    tool_index: usize,
    path: Vec<PathPart>,
    text: String,
    weight: f64,
}

fn description_fields(
    value: &Value,
    tool_index: usize,
    path: &[PathPart],
    rank_weight: f64,
    required: bool,
    output: &mut Vec<DescriptionField>,
) {
    match value {
        Value::Object(object) => {
            if let Some(text) = object.get("description").and_then(Value::as_str) {
                if !text.is_empty() {
                    let mut field_path = path.to_vec();
                    field_path.push(PathPart::Key("description".into()));
                    output.push(DescriptionField {
                        tool_index,
                        path: field_path,
                        text: text.to_string(),
                        weight: rank_weight * if required { 2.5 } else { 1.0 },
                    });
                }
            }
            let required_names = object
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<std::collections::HashSet<_>>();
            for (key, child) in object {
                // Annotation subtrees are stripped from the structural
                // projection.  Never allocate a description nested below one
                // or materialization would target a path that cannot exist.
                if annotation_key(key) {
                    continue;
                }
                if key == "properties" {
                    if let Some(properties) = child.as_object() {
                        for (property_name, property) in properties {
                            let mut child_path = path.to_vec();
                            child_path.push(PathPart::Key(key.clone()));
                            child_path.push(PathPart::Key(property_name.clone()));
                            description_fields(
                                property,
                                tool_index,
                                &child_path,
                                rank_weight,
                                required_names.contains(property_name.as_str()),
                                output,
                            );
                        }
                    }
                    continue;
                }
                let mut child_path = path.to_vec();
                child_path.push(PathPart::Key(key.clone()));
                description_fields(
                    child,
                    tool_index,
                    &child_path,
                    rank_weight,
                    required,
                    output,
                );
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                let mut child_path = path.to_vec();
                child_path.push(PathPart::Index(index));
                description_fields(
                    child,
                    tool_index,
                    &child_path,
                    rank_weight,
                    required,
                    output,
                );
            }
        }
        _ => {}
    }
}

fn set_path(target: &mut Value, path: &[PathPart], value: Value) {
    let Some((first, rest)) = path.split_first() else {
        *target = value;
        return;
    };
    match first {
        PathPart::Key(key) => {
            if let Some(child) = target
                .as_object_mut()
                .and_then(|object| object.get_mut(key))
            {
                set_path(child, rest, value);
            } else if rest.is_empty() {
                if let Some(object) = target.as_object_mut() {
                    object.insert(key.clone(), value);
                }
            }
        }
        PathPart::Index(index) => {
            if let Some(child) = target
                .as_array_mut()
                .and_then(|items| items.get_mut(*index))
            {
                set_path(child, rest, value);
            }
        }
    }
}

fn clip_prefix(vocab: &Vocab, text: &str, tokens: usize) -> String {
    let ids = vocab.encode(text, false);
    if ids.len() <= tokens {
        return text.to_string();
    }
    if tokens == 0 {
        return String::new();
    }
    let marker = "…";
    let marker_ids = vocab.encode(marker, false);
    if marker_ids.len() >= tokens {
        return vocab.decode(&ids[..tokens]);
    }
    format!(
        "{}{}",
        vocab.decode(&ids[..tokens - marker_ids.len()]),
        marker
    )
}

fn clip_head_tail(vocab: &Vocab, text: &str, tokens: usize) -> String {
    let ids = vocab.encode(text, false);
    if ids.len() <= tokens {
        return text.to_string();
    }
    if tokens == 0 {
        return String::new();
    }
    let marker = "…已裁剪…";
    let marker_ids = vocab.encode(marker, false);
    if marker_ids.len() >= tokens {
        return vocab.decode(&ids[..tokens]);
    }
    let remaining = tokens - marker_ids.len();
    let left = (remaining + 1) / 2;
    let right = remaining - left;
    format!(
        "{}{}{}",
        vocab.decode(&ids[..left]),
        marker,
        if right == 0 {
            String::new()
        } else {
            vocab.decode(&ids[ids.len() - right..])
        }
    )
}

fn render_projected_tools(tools: &[Value]) -> String {
    format!(
        "<tools>{}</tools>",
        dumps_canonical(&Value::Array(tools.to_vec()))
    )
}

pub fn minimum_tool_projection_tokens(vocab: &Vocab, tool: &Value, stable_overhead: &str) -> usize {
    token_count(
        vocab,
        &format!(
            "{stable_overhead}{}",
            render_projected_tools(&[structural_tool_projection(tool)])
        ),
        true,
    )
}

#[derive(Clone, Debug)]
struct ToolProjection {
    projected: Vec<Value>,
    full_tools: Vec<Value>,
    dropped_tool_ids: Vec<String>,
    rendered: String,
    tokens: usize,
    cap: usize,
    profile: String,
    compression_level: String,
    omitted_description_tokens: usize,
    projection_sha256: String,
    context_unrepresentable: bool,
}

fn project_tool_batch(
    vocab: &Vocab,
    tools: &[Value],
    relevances: &[f64],
    profile: &str,
    stable_overhead: &str,
) -> Result<ToolProjection, SdkError> {
    if tools.len() > TOOL_BATCH_SIZE || tools.len() != relevances.len() {
        return Err(SdkError::new(
            "invalid_argument",
            "one relevance is required for each of at most five tools",
        ));
    }
    let cap = stable_cap(profile)?;
    let full_value = compact_tools(tools);
    let full = full_value.as_array().cloned().unwrap_or_default();
    let render = |rows: &[Value]| format!("{stable_overhead}{}", render_projected_tools(rows));
    let full_rendered = render(&full);
    let full_tokens = token_count(vocab, &full_rendered, true);
    if full_tokens <= cap {
        return Ok(ToolProjection {
            projected: full.clone(),
            full_tools: full.clone(),
            dropped_tool_ids: vec![],
            rendered: full_rendered,
            tokens: full_tokens,
            cap,
            profile: profile.to_string(),
            compression_level: "none".into(),
            omitted_description_tokens: 0,
            projection_sha256: sha256_bytes(dumps_canonical(&Value::Array(full)).as_bytes()),
            context_unrepresentable: false,
        });
    }
    let mut kept_full = full;
    let mut skeletons = kept_full
        .iter()
        .map(structural_tool_projection)
        .collect::<Vec<_>>();
    let mut dropped = Vec::new();
    while !skeletons.is_empty() && token_count(vocab, &render(&skeletons), true) > cap {
        dropped.push(
            kept_full
                .last()
                .and_then(|tool| tool.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        );
        kept_full.pop();
        skeletons.pop();
    }
    if skeletons.is_empty() {
        let rendered = render(&[]);
        return Ok(ToolProjection {
            projected: vec![],
            full_tools: vec![],
            dropped_tool_ids: dropped,
            tokens: token_count(vocab, &rendered, true),
            rendered,
            cap,
            profile: profile.to_string(),
            compression_level: "unrepresentable".into(),
            omitted_description_tokens: kept_full
                .iter()
                .map(|tool| {
                    token_count(
                        vocab,
                        tool.get("description")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        false,
                    )
                })
                .sum(),
            projection_sha256: sha256_bytes(b"[]"),
            context_unrepresentable: true,
        });
    }
    let mut fields = Vec::<DescriptionField>::new();
    for (index, tool) in kept_full.iter().enumerate() {
        let rank_weight = (kept_full.len() - index) as f64 * relevances[index].max(0.05);
        if let Some(text) = tool.get("description").and_then(Value::as_str) {
            if !text.is_empty() {
                fields.push(DescriptionField {
                    tool_index: index,
                    path: vec![PathPart::Key("description".into())],
                    text: text.to_string(),
                    weight: rank_weight * 2.0,
                });
            }
        }
        description_fields(
            tool.get("parameters").unwrap_or(&Value::Null),
            index,
            &[PathPart::Key("parameters".into())],
            rank_weight,
            false,
            &mut fields,
        );
    }
    let full_lengths = fields
        .iter()
        .map(|field| token_count(vocab, &field.text, false))
        .collect::<Vec<_>>();
    let mut allocations = vec![0usize; fields.len()];
    let base_tokens = token_count(vocab, &render(&skeletons), true);
    let allocatable = cap
        .saturating_sub(base_tokens)
        .saturating_sub(2 * fields.len());
    for _ in 0..allocatable {
        let winner = fields
            .iter()
            .enumerate()
            .filter(|(index, _)| allocations[*index] < full_lengths[*index])
            .max_by(|(left_index, left), (right_index, right)| {
                (left.weight / (allocations[*left_index] + 1) as f64)
                    .total_cmp(&(right.weight / (allocations[*right_index] + 1) as f64))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index);
        let Some(winner) = winner else { break };
        allocations[winner] += 1;
    }
    let materialize = |allocations: &[usize]| {
        let mut rows = skeletons.clone();
        for (index, field) in fields.iter().enumerate() {
            if allocations[index] > 0 {
                let clipped = clip_prefix(vocab, &field.text, allocations[index]);
                set_path(
                    &mut rows[field.tool_index],
                    &field.path,
                    Value::String(clipped),
                );
            }
        }
        let rendered = render(&rows);
        let tokens = token_count(vocab, &rendered, true);
        (rows, rendered, tokens)
    };
    let (mut projected, mut rendered, mut tokens) = materialize(&allocations);
    while tokens > cap && allocations.iter().any(|amount| *amount > 0) {
        let loser = fields
            .iter()
            .enumerate()
            .filter(|(index, _)| allocations[*index] > 0)
            .min_by(|(left_index, left), (right_index, right)| {
                (left.weight / allocations[*left_index] as f64)
                    .total_cmp(&(right.weight / allocations[*right_index] as f64))
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .expect("at least one allocation");
        allocations[loser] -= 1;
        (projected, rendered, tokens) = materialize(&allocations);
    }
    let compression_level = if !dropped.is_empty() {
        "reduced_batch"
    } else if allocations.iter().all(|amount| *amount == 0) {
        "structural"
    } else {
        "descriptions"
    };
    Ok(ToolProjection {
        projection_sha256: sha256_bytes(
            dumps_canonical(&Value::Array(projected.clone())).as_bytes(),
        ),
        projected,
        full_tools: kept_full,
        dropped_tool_ids: dropped,
        rendered,
        tokens,
        cap,
        profile: profile.to_string(),
        compression_level: compression_level.into(),
        omitted_description_tokens: full_lengths.iter().sum::<usize>()
            - allocations.iter().sum::<usize>(),
        context_unrepresentable: false,
    })
}

fn render_ordinary(request: &Value) -> String {
    let mut parts = Vec::<String>::new();
    if let Some(history) = request.get("history").and_then(Value::as_array) {
        for turn in history {
            let role = turn.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = turn
                .get("content")
                .and_then(Value::as_str)
                .or_else(|| turn.get("text").and_then(Value::as_str))
                .unwrap_or("")
                .trim();
            if !content.is_empty() {
                parts.push(format!("{role}：{content}"));
            }
        }
    }
    let results = request
        .get("tool_results")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .or_else(|| request.get("prior_tool_results").and_then(Value::as_array));
    if let Some(results) = results {
        for result in results {
            parts.push(format!(
                "tool：{}",
                result
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| dumps_canonical(result))
            ));
        }
    }
    for (tag, default) in [
        ("context", json!({})),
        ("evidence", json!([])),
        ("permissions", json!({})),
        ("state", json!({})),
    ] {
        let value = request.get(tag).unwrap_or(&default);
        parts.push(format!("<{tag}>{}</{tag}>", dumps_canonical(value)));
    }
    let empty = json!({});
    let mw = request
        .get("mw")
        .or_else(|| request.get("mw_disposition"))
        .unwrap_or(&empty);
    parts.push(format!("<mw>{}</mw>", dumps_canonical(mw)));
    parts.push(format!(
        "user：{}",
        request
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
    ));
    parts.join("\n")
}

fn minimal_tool_result(result: &Value) -> Value {
    let Some(object) = result.as_object() else {
        return json!("…已裁剪早期工具结果…");
    };
    let mut kept = Map::new();
    for key in ["wire_version", "call_id", "status", "provenance", "error"] {
        if let Some(value) = object.get(key) {
            kept.insert(key.into(), value.clone());
        }
    }
    if object.contains_key("payload") {
        kept.insert("payload".into(), json!({"_mei_omitted":"budget"}));
    }
    Value::Object(kept)
}

fn fit_ordinary(
    vocab: &Vocab,
    request: &Value,
    sink: &str,
    prompt_cap: usize,
) -> Option<(String, String, usize, Value, Value)> {
    let mut visible = request.clone();
    let original_prompt = format!("{sink}\n{}{ASSISTANT_SUFFIX}", render_ordinary(&visible));
    let original_tokens = token_count(vocab, &original_prompt, true);
    let mut dropped_history = 0usize;
    let mut compacted_results = Vec::<String>::new();
    let mut dropped_evidence = 0usize;
    let mut dropped_context = Vec::<String>::new();
    let mut query_clipped = false;
    let snapshot = |value: &Value| {
        let ordinary = render_ordinary(value);
        let prompt = format!("{sink}\n{ordinary}{ASSISTANT_SUFFIX}");
        let used = token_count(vocab, &prompt, true);
        (ordinary, prompt, used)
    };
    let (mut ordinary, mut prompt, mut used) = snapshot(&visible);
    while used > prompt_cap {
        let Some(history) = visible.get_mut("history").and_then(Value::as_array_mut) else {
            break;
        };
        if history.is_empty() {
            break;
        }
        history.remove(0);
        dropped_history += 1;
        (ordinary, prompt, used) = snapshot(&visible);
    }
    let results_key = if visible.get("tool_results").is_some() {
        "tool_results"
    } else {
        "prior_tool_results"
    };
    let result_len = visible
        .get(results_key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    for index in 0..result_len.saturating_sub(1) {
        if used <= prompt_cap {
            break;
        }
        if let Some(results) = visible.get_mut(results_key).and_then(Value::as_array_mut) {
            let id = results[index]
                .get("call_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| index.to_string());
            results[index] = minimal_tool_result(&results[index]);
            compacted_results.push(id);
        }
        (ordinary, prompt, used) = snapshot(&visible);
    }
    while used > prompt_cap {
        let remove_index = visible
            .get("evidence")
            .and_then(Value::as_array)
            .and_then(|items| {
                items
                    .iter()
                    .enumerate()
                    .min_by(|(left_index, left), (right_index, right)| {
                        let priority = |value: &Value| {
                            value.get("priority").and_then(Value::as_f64).unwrap_or(0.0)
                        };
                        priority(left)
                            .total_cmp(&priority(right))
                            .then_with(|| left_index.cmp(right_index))
                    })
                    .map(|(index, _)| index)
            });
        let Some(remove_index) = remove_index else {
            break;
        };
        visible
            .get_mut("evidence")
            .and_then(Value::as_array_mut)
            .expect("evidence array")
            .remove(remove_index);
        dropped_evidence += 1;
        (ordinary, prompt, used) = snapshot(&visible);
    }
    while used > prompt_cap {
        let remove_key = visible
            .get("context")
            .and_then(Value::as_object)
            .and_then(|object| {
                object
                    .iter()
                    .min_by(|(left_key, left), (right_key, right)| {
                        let priority = |value: &Value| {
                            value.get("priority").and_then(Value::as_f64).unwrap_or(0.0)
                        };
                        priority(left)
                            .total_cmp(&priority(right))
                            .then_with(|| left_key.as_bytes().cmp(right_key.as_bytes()))
                    })
                    .map(|(key, _)| key.clone())
            });
        let Some(remove_key) = remove_key else { break };
        visible
            .get_mut("context")
            .and_then(Value::as_object_mut)
            .expect("context object")
            .remove(&remove_key);
        dropped_context.push(remove_key);
        (ordinary, prompt, used) = snapshot(&visible);
    }
    if used > prompt_cap && result_len > 0 {
        if let Some(results) = visible.get_mut(results_key).and_then(Value::as_array_mut) {
            if let Some(last) = results.last_mut() {
                let id = last
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| (result_len - 1).to_string());
                *last = minimal_tool_result(last);
                if !compacted_results.contains(&id) {
                    compacted_results.push(id);
                }
            }
        }
        (ordinary, prompt, used) = snapshot(&visible);
    }
    if used > prompt_cap {
        let original_query = visible
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut low = 0usize;
        let mut high = token_count(vocab, &original_query, false);
        let mut best = None;
        while low <= high {
            let middle = (low + high) / 2;
            let clipped = clip_head_tail(vocab, &original_query, middle);
            visible["query"] = json!(clipped);
            let candidate = snapshot(&visible);
            if candidate.2 <= prompt_cap {
                best = Some((visible["query"].clone(), candidate));
                low = middle + 1;
            } else if middle == 0 {
                break;
            } else {
                high = middle - 1;
            }
        }
        if let Some((query, candidate)) = best {
            visible["query"] = query;
            (ordinary, prompt, used) = candidate;
            query_clipped = visible.get("query").and_then(Value::as_str) != Some(&original_query);
        } else {
            visible["query"] = json!(original_query);
        }
    }
    if used > prompt_cap {
        return None;
    }
    let compression_level = if query_clipped {
        "query"
    } else if !dropped_context.is_empty() || dropped_evidence > 0 {
        "context"
    } else if !compacted_results.is_empty() {
        "tool_results"
    } else if dropped_history > 0 {
        "history"
    } else {
        "none"
    };
    let input_budget = json!({
        "cap": prompt_cap,
        "original_prompt_tokens": original_tokens,
        "dropped_history": dropped_history,
        "compacted_tool_results": compacted_results,
        "dropped_evidence": dropped_evidence,
        "dropped_context_keys": dropped_context,
        "query_clipped": query_clipped,
        "used": used,
        "compression_level": compression_level,
        "visible_query_tokens": token_count(
            vocab,
            visible.get("query").and_then(Value::as_str).unwrap_or(""),
            false,
        ),
    });
    Some((ordinary, prompt, used, visible, input_budget))
}

pub fn render_budgeted_request(
    vocab: &Vocab,
    request: &Value,
    tools: &[Value],
    relevances: &[f64],
    profile: &str,
    output_reserve: usize,
) -> Result<Value, SdkError> {
    if output_reserve == 0 || output_reserve >= MAX_CONTEXT_TOKENS {
        return Err(SdkError::new(
            "invalid_argument",
            "output reserve must be in 1..2047",
        ));
    }
    let prompt_cap = MAX_CONTEXT_TOKENS - output_reserve;
    let mut current_tools = tools.to_vec();
    let mut current_relevances = relevances.to_vec();
    let mut dropped = Vec::<String>::new();
    while !current_tools.is_empty() {
        let projection = project_tool_batch(
            vocab,
            &current_tools,
            &current_relevances,
            profile,
            &format!("{}\n", task_contract()),
        )?;
        dropped.extend(projection.dropped_tool_ids.iter().cloned());
        let selected_count = projection.full_tools.len();
        if projection.context_unrepresentable || selected_count == 0 {
            break;
        }
        current_tools.truncate(selected_count);
        current_relevances.truncate(selected_count);
        if let Some((ordinary, prompt, prompt_tokens, visible, input_budget)) =
            fit_ordinary(vocab, request, &projection.rendered, prompt_cap)
        {
            return Ok(json!({
                "prompt": prompt,
                "sink": projection.rendered,
                "ordinary": ordinary,
                "prompt_tokens": prompt_tokens,
                "visible_request": visible,
                "schema_fingerprint": crate::canonical::schema_fingerprint(&projection.full_tools),
                "schema_projection_sha256": projection.projection_sha256,
                "schema_budget": {
                    "packer_id": CONTEXT_PACKER_ID,
                    "projection_serializer_id": PROJECTION_SERIALIZER_ID,
                    "profile": projection.profile,
                    "cap": projection.cap,
                    "used": projection.tokens,
                    "compression_level": projection.compression_level,
                    "omitted_description_tokens": projection.omitted_description_tokens,
                    "dropped_tools": dropped,
                    "context_unrepresentable": false,
                },
                "input_budget": input_budget,
                "selected_tools": projection.full_tools.iter().map(|tool| tool.get("name").and_then(Value::as_str).unwrap_or("")).collect::<Vec<_>>(),
                "_validation_tools": projection.full_tools,
                "_projected_tools": projection.projected,
            }));
        }
        let dropped_name = current_tools
            .last()
            .and_then(|tool| tool.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        dropped.push(dropped_name);
        current_tools.pop();
        current_relevances.pop();
    }
    Ok(json!({
        "prompt": null,
        "sink": null,
        "ordinary": null,
        "prompt_tokens": 0,
        "schema_projection_sha256": null,
        "schema_budget": {
            "profile": profile,
            "cap": stable_cap(profile)?,
            "used": null,
            "compression_level": "unrepresentable",
            "dropped_tools": dropped,
            "context_unrepresentable": true,
        },
        "input_budget": {"cap":prompt_cap,"used":null},
        "selected_tools": [],
        "_validation_tools": [],
        "_projected_tools": [],
        "error": "context_unrepresentable",
    }))
}
