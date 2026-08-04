---
name: meilang-corpus-distill
description: 维护 MeiLang 训练语料：从 examples、文档、对话与源码发现候选任务，经 family/sample/verifier 工作流入库。用于语料蒸馏、family 扩展、sample JSONL 起草与语料校验。勿用于直接编辑 `.mei` 源码。Maintain MeiLang training corpus; do not use for editing `.mei` source files.
disable-model-invocation: true
---

# MeiLang 语料蒸馏（Corpus Distill）

本 Skill 用于语料库维护，不是日常 MeiLang 创作，也不捆绑 monorepo 文档或 sibling workspace。

- 创建、编辑、审查或修复 `.mei` 源码 → 使用 authoring skill。
- 发现、分类、起草、验证或扩展训练样本 → 使用本 Skill。

## 调用方输入合同

开始前显式提供：

1. `corpus_root`：包含 `manifests/`、`samples/`、`eval/` 的语料根。
2. `tools_root`：提供 stub、append、validate、verifier 等命令的工具根。
3. `source_roots`：按调用方声明的优先级排列的源码、examples、文档或 transcript 根。
4. 目标 release/catalog 版本与允许的 lane/family 范围。

本 Skill 不猜这些路径；缺失时只产出 assessment 或 worklist，不发布。

## 职责范围

本 Skill 是半自动蒸馏工作台：

- 评估一次任务 realistically 能产出什么。
- 发现候选任务并分类到 `build` / `access` / `docs`。
- 汇总语料覆盖并收集最小、有真源支撑的 context 切片。
- 起草 sample 对象或 batch worklist。
- 调用现有 corpus tools 做发布检查。

它不是一键数据集生成器。

## 术语速查

| 术语 | 含义 |
|------|------|
| family | 问题闭包单元；split 在 family 级分配 |
| sample | 一条 JSONL 训练样本 |
| lane / `route_mode` | `build` / `access` / `docs`，单样本不可混用 |
| dry-run | 仅校验结构与路由，不写库 |
| publish-ready | 可通过质量门并正式 append |
| batch worklist | 批量候选清单，先于最终 JSON draft |
| yield_type | 本次任务产出形态评估 |

Transcript 是任务发现源，不是最终真源。真源优先级完全由调用方传入的 `source_roots` 决定。

## 标准工作流（7 步）

```text
语料蒸馏进度
- [ ] Step 1：评估任务产出（yield_type）与目标范围
- [ ] Step 2：确认 route_mode、family scope、分类
- [ ] Step 3：按需展示当前语料统计
- [ ] Step 4：收集最小 context 切片
- [ ] Step 5：起草单条 sample 或 batch worklist
- [ ] Step 6：人工 review（真源、厚度、泄漏）
- [ ] Step 7：运行语料发布检查
```

### 1. 评估产出

```json
{
  "yield_type": "single_sample|sample_cluster|batch_worklist|governance_only",
  "route_mode": "",
  "family_scope": [],
  "author_surface": "",
  "maturity_tier": "",
  "why": ""
}
```

### 2. 确认路由与 family

先判定 lane，再优先复用已有 `family_id`；仅在任务未被覆盖时新建或扩展 family。

### 3. 展示统计

```bash
python3 skills/meilang-corpus-distill/scripts/corpus_stats.py --corpus-root /path/to/corpus
python3 skills/meilang-corpus-distill/scripts/family_snapshot.py \
  --corpus-root /path/to/corpus --family-id <family_id>
```

### 4. 收集最小 context

优先精确源码片段、特定文档段落、诊断信息、inventory/benchmark 断言和必要 runtime truth。避免整篇 markdown、整目录、巨大源码 dump 和泄漏完整答案的 context。

### 5. 起草 sample 或 batch worklist

机械骨架使用调用方 `tools_root` 中的现有工具。手工填写 `instruction`、`context`、`output`、`verifier.checks` 与真源 metadata。

大规模请求先选 family cluster，定义 12～40 条候选目标，先产出 worklist，再 selective 起草 sample。

每条 worklist 至少含 candidate id、`family_id`、`route_mode`、`task_type`、`source_paths`、价值说明和 verifier mode。

### 6. 人工 review

1. 真实任务，而非答案型假 prompt？
2. context 最小但够用？
3. output 可回溯源码或 runtime 真源？
4. verifier 可执行，或诚实标注 manual review？

### 7. 发布检查

调用 `tools_root` 中现有 append/validate/verifier 工具；未达 publish-ready 时停在正式 append 之前，优先 dry-run。

## Lane 边界

- `build`：authoring、repair、migrate、结构生成等 source-first 任务。
- `access`：runtime 问答、metric/dataset 回答、query-state 敏感任务。
- `docs`：摘要、结构说明、迁移说明、inventory/benchmark 维护。

单条 sample 不可混用多个 lane。

## 停止条件

- 任务跨越 `build` 与 `access`。
- 真源或版本不清。
- context 必须过大才能完成。
- verifier 设计只能猜测。
- 候选属于未成熟 surface。

延伸阅读：[reference.md](reference.md) · [workflows.md](workflows.md) ·
[quality-gates.md](quality-gates.md) · [examples.md](examples.md)
