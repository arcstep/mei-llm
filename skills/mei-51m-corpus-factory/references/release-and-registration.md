# 铸币与登记：语料 release → eval-lock → cycle 账本

## 铸币（mint）流程

1. **build.py 编排**：每个 family 写 semantic + compiled
   `{train,valid,dev,test}.jsonl`，落
   `artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00600m/corpus/sft-suite/<RELEASE_ID>/`；
   split 由 `cf_group` sha256 分桶决定（与 eval-lock 同机制隔离）。
2. **release ID**：`mei-1.0-51m-exp-000600m-sft-zh-rebuild-<v<n>>`。已有同 ID
   产物 = 已是持久化发布 → 禁止覆盖，直接 bump 版本并写 supersede 链；
   旧版目录保留为证据（v1 保留、不删除、不重写）。
3. **manifest**：`build_manifest` 输出 artifact merkle root；verifier 校验
   root 与 CURRENT.json 绑定的 base 权重 sha256
   （`6d55a61773cdd0a6713c43fa7565f8c2fa64414c3e4a1beadfb6507e0d91752a`）不变。
4. **eval-lock 同步**：`build_eval_lock.py` 把新 SFT release 的 dev/test 分区
   拷进 eval-lock bank；ID 同步 bump（`mei-51m-longitudinal-eval-v8-…-v2`），
   supersedes 块写明 reason（如"mw dev/test 行数与 SFT v2 对齐 1023/1050"）；
   `locked_test_used_for_threshold_tuning=false`。
5. **known_quality_gaps 如实写**：如"free-text 字符串仍有占位填充值；部分
   transform 命中率 66-79%；undesigned exact-dup ~16.3%；未重建
   schema_generalization/跨语言-BFCL 层"。缺陷不藏，但也不因此无限扩量。

## 登记（registration）流程

- `cycles/mei-1.0-51m/exp-000600m/corpus/sft.json`：
  `rebuild_candidate` → 新 ID + supersedes 块 + row_totals（每 family 与 split
  行数）+ split counts + known_quality_gaps + 状态字面量
  `prepared_not_yet_consumed` / `corpus_release_eligible` / `ready_for_600m_sft`
  / `training_started=false`；真实模型指标未测一律
  `model_quality_metrics_pending_sft`（禁止用语料统计冒充）。
- `cycles/mei-1.0-51m/exp-000600m/corpus/eval.json`：
  `rebuild_candidate_eval_lock` → 新 lock ID + supersedes + extends
  （v8 additive-retrieval-depth-only，v7 各 bank 不动）+ gate_targets。
- `CORPUS.md`：中文叙事**追加** v<n+1> 小节（旧小节只留 supersede 指路，不改
  写历史结论）；写明绑定 base、SHA、审计证据路径、诊断结果与限定措辞。
- verifier（`corpus-factory/verifiers/verify_sft_zh_rebuild_v1.py`）：
  新版本发布后更新为校验**新旧双版本** manifest + 各自 eval-lock + CURRENT.json
  base 仍为原值 + `sft: null`；跑通报 `ok: true` 再宣布完成。

## 状态字面量对照（拼错即泄漏进训练门）

| 语境 | 合法字面量 |
|---|---|
| 语料候选状态 | `prepared_not_yet_consumed` / `corpus_release_eligible` / `corpus_release_ineligible` |
| 是否可上 600M SFT | `ready_for_600m_sft` + `training_started=false` |
| 模型侧度量 | `pending_sft` / `pending_sft_not_yet_run`（eval-lock 的 `model_measurement_status`） |
| 旧版本状态 | `superseded_kept_immutable_as_evidence_not_deleted` |

## 禁止清单

- 覆盖/删除旧 release 或旧 eval-lock；改老 receipt 的哈希；
- 用一个 release ID 重跑不同内容（先例：同 ID 幂等重跑仅限
  保留 diff、未持久化场景）；
- CURRENT.json 任何修改（含 `finalize_current` 之外的所有路径）；
- 用行数反驳质量审计、用 hash 干净宣称自然多样、把诊断数字写成正式 A/B。
