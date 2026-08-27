# 示例

## 示例 1：build sample

```json
{
  "author_surface": "data_metric",
  "task_intent": "bind/configure",
  "artifact_target": "metric",
  "maturity_tier": "tier_a_proven",
  "source_kind": "example",
  "route_mode": "build",
  "verification_mode": "compile_pass"
}
```

Review 预期：context 只含完成任务所需的精确片段，output 描述稳定声明模式，verifier 优先 compiler 支撑。

## 示例 2：Transcript → 候选任务

```json
{
  "candidate_task": "repair a wrong filter helper pairing on an analytics board",
  "family_hint": "sample.repair_board_and_filters",
  "route_mode": "build",
  "source_paths": [
    "apps/sample/src/scene/analytics.board.mei",
    "apps/sample/src/scene/_shared/list-filters.mei"
  ],
  "needed_context": [
    "board filter_schema fragment",
    "helper definition",
    "accepted final patch pattern"
  ],
  "draft_task_type": "build.repair.fix",
  "why_it_is_worth_distilling": "repeated repair pattern with canonical source support"
}
```

关键规则：不从 transcript 直接发布，先在调用方 canonical sources 上 re-grounding。

## 示例 3：Batch worklist

好的 assessment：

```json
{
  "yield_type": "batch_worklist",
  "route_mode": "build",
  "family_scope": ["sample.analytics", "sample.home_shell"],
  "author_surface": "dashboard_home",
  "maturity_tier": "tier_a_proven",
  "why": "source line is stable and supports repeated authoring variants"
}
```

好的候选行：

```json
{
  "candidate_id": "analytics_popup_link",
  "family_id": "sample.analytics",
  "route_mode": "build",
  "task_type": "build.component.props_fill",
  "source_paths": ["apps/sample/src/scene/analytics.mei"],
  "why_it_is_worth_distilling": "stable card-to-overlay binding path",
  "verification_mode": "compile_pass"
}
```

## 反模式

- 整段 transcript 放入 context：探索与最终答案混杂，真值不清。
- 整篇 markdown 放入 docs context：应只引用支撑任务的段落。
- 同一 sample 同时要求 patch board 与回答 live runtime metric：应拆为 `build.*` 与 `access.*`。
- dry-run 通过后直接声称 publish-ready：还需人工门与 verifier 门。
