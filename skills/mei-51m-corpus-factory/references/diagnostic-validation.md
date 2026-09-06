# 便宜诊断验证（正式 pipeline 前的语料信号测试）

## 用途

在 commit 全量多阶段产品化（Base→QAT→retrieval R0→full-call SFT→…→MW→
confidence→narration）之前，用最便宜的方式回答："这份语料对 raw base 是否
携带可学习信号？"。MW 先例：600 步 ≈1 分钟 wall clock，dev acc 0.6975 /
macro-F1 0.3403，test 0.7075 / 0.3464（旧正式管线 reference：0.226 /
0.030093）。这是语料诊断，不是正式训练、不产生 release。

## 模式（模板文件）

`model-factory/diagnostics/mw_disposition_corpus_diagnostic_51m.py`，结构：

1. **加载 raw float base**：`lifecycle._load_runtime(BASE_WEIGHTS,
   quantized=False)`——不经 CQ2、不挂任何已训 head。权重路径显式给
   `exp-000600m/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/…npz`。
2. **行适配**：语料行 remap 成训练器期望的 `source` 形状（sample_id=case_id、
   reason_class_id/raw 标签等）。改语料 schema 时同步改 remap，不反向改训练器。
3. **oracle-mode 固定视图**：`adaptive.iter_mw_visible_batches(runtime, rows,
   catalog, calibration={}, retrieval_mode="oracle", profile=…)`，compact +
   standard 两个 profile 都建；oracle 用行内 `retrieved_tools`，不需要已训
   retrieval head。catalog 工具必须先 portable projection
   （name/description/parameters），否则 schema_subset 拒绝。
4. **训练**：`training.train_mw_disposition_v3(runtime, train_eligible, {}, [],
   steps=600, checkpoint_dir=…)`。
5. **评估**：`evaluation.evaluate_frozen_mw_batches(runtime, dev_views,
   limit=…)` / test。
6. **输出**：run_dir 只写 `artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00600m/runs/
   <run-id>/`（STATUS.json running→complete、run.log、oracle views jsonl、
   predictions jsonl、diagnostic-report.json）。不碰 CURRENT.json、不写 release、
   不写 cycle receipt。

## 报告与措辞纪律

- report 标 `diagnostic_only: true`、`not_a_formal_release: true`、
  `not_representative_of_final_cq2_aligned_checkpoint: true`、
  `current_json_mutated: false`，记录 base weights sha256、source_release_id、
  steps、classes_present_in_train、train/dev/test 报告。
- 旧正式数字只进 `baseline_for_reference_only` 字段并注明
  "measured on CQ2-quantized, retrieval-R2/agent-aligned checkpoint — not
  directly comparable"；正文禁止写成 A/B 结论。
- 训练中 batch-visibility 自动 relabel（class0→10）的计数要记录
  （`relabelled_capability_insufficient`），它是语料规则正确性的旁证。

## 注册与边界

- 脚本放 `model-factory/diagnostics/`，并在
  `model-factory/contracts/CODE_CATALOG.json` 标 `diagnostic_only`——
  诊断代码不得静默进入正式 lineage。
- 诊断失败/指标差 ≠ 语料 release 作废：先查 remap、oracle 视图、类别泄漏，
  再查语料本身；指标差到不可学才标记 `corpus_release_ineligible` 并出证据，
  不无限扩量放水。

## 延伸到其他 head（模板复用点）

- retrieval（no-match / rank>5 retention）可用同款对比头（ContrastiveHead
  探针）：oracle ranking 只需要行内 `retrieved_tools`，同样不需要正式
  retrieval head——是 v8 深 bank 就位的下一个候选验证（见
  `topic-priority.md`）。
- confidence：label=null 等待真实 runtime harvest，post-SFT 校准后才有可学
  标签，诊断前移价值有限。
- narration：只吃 verified ToolResult，瓶颈更可能在 post-SFT 行为，不是语料
  叶密度——扩叶前先跑 post-SFT 度量。
