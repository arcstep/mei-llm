# 执行规则

## 身份与分工

- authoring skill：创建或修改 `.mei` 源码。
- `meilang-corpus-distill`：维护调用方提供的 `manifests/`、`samples/*.jsonl`、`eval/*.index.json`。
- 不要把语料蒸馏当作 workspace-local 的公开 author skill。

## Canonical sources

调用方必须传入有序 `source_roots`，并为每条候选记录实际 `source_paths`。Transcript 可揭示真实任务与 repair 模式，但不是最终真源，发布前必须在 canonical source 上重新 grounding。

## 语料 Lane

- `build`：source-first authoring、generate、repair、migrate、结构与绑定。
- `access`：runtime 问答、query-state 敏感的 metric 或 dataset 回答。
- `docs`：结构摘要、迁移说明、inventory、benchmark、release 文档任务。

单条 sample 不可混用多个 lane。

## Family 规则

1. 起草 sample 前先注册或选定 `family_id`。
2. 一个稳定的问题闭包 = 一个 family，不是单个文件。
3. 默认 scene/world/board 三件套同属一个 family。
4. split 先在 family 级选定，sample 继承。

## 最小执行 schema

```json
{
  "author_surface": "",
  "task_intent": "",
  "artifact_target": "",
  "maturity_tier": "",
  "source_kind": "",
  "route_mode": "",
  "verification_mode": ""
}
```

这些字段用于执行与 triage，不替代 corpus 自身字段。

## 任务评估 schema

```json
{
  "yield_type": "single_sample|sample_cluster|batch_worklist|governance_only",
  "route_mode": "",
  "family_scope": [],
  "author_surface": "",
  "maturity_tier": "",
  "source_kind": "",
  "why": ""
}
```

## Sample 起草规则

每条 draft 至少包含：

- `sample_id`
- `family_id`
- `task_type`
- `route_mode`
- `instruction`
- `context`
- `output`
- `verifier`

推荐 `sample_id`：`<app>.<task_type>.<slug>.v<version>`。

## Context 切片规则

好的 context 有真源支撑、最小化、足以完成任务，并围绕字段、片段、断言组织。拒绝整篇文档、巨大 transcript 和伪装成 context 的完整答案。

## Verifier 规则

- `build`：compiler 或 patch 级真值检查。
- `access`：runtime 真值检查。
- `docs`：source-trace 或 doc-assertion 检查。

优先使用调用方 `tools_root` 中的 verifier；无可靠自动检查时诚实使用 `manual review`。

## 发布流程

顺序是 generate stub → append dry-run → 人工 review → 正式 append → validate → run verifiers。未就绪时停在正式 append 前。

## 统计与 snapshot

```bash
python3 skills/meilang-corpus-distill/scripts/corpus_stats.py --corpus-root /path/to/corpus
python3 skills/meilang-corpus-distill/scripts/family_snapshot.py \
  --corpus-root /path/to/corpus --family-id <family_id>
```

## Batch worklist

推荐规模：小批 8～12、常规 12～24、大批 24～40。只有 family scope 稳定、真源充足、verifier 风格明确且候选窄而重复时才扩大。

每行至少含 candidate id、`family_id`、`route_mode`、`task_type`、`source_paths`、价值说明和 verifier mode。

## 反模式

- 未经源码复核就把 transcript 当 sample。
- 单 sample 混用 `build` 与 `access`。
- 随意选 split。
- 把未来设计当已实现语法。
- 未思考 verifier 就发布。
- 仅因 taxonomy 有槽位就扩展未成熟 surface。
