# 600M 周期语料

本轮从 300,000,485-token Base 继续至实际累计 600,001,765 tokens。计划增量为
299,999,515 tokens，来自 `lm-v2-cpt-600m` 的 wiki、hq、structure、colloquial 四角色。

完整性、去重和污染检查通过，但合成 structure/colloquial 的模板多样性证据降级，因此：

- checkpoint 数值上可以继续 CPT；
- 不能自动晋升为下一轮正式 parent；
- 本轮增量语料不能原样复用到 900M；
- 900M 应使用修复后的增量语料，而不是抛弃数值完整的 600M checkpoint。

adaptive-v5 SFT 与 300M 使用同合同。600M 在 Float Task Control 和 natural exact 上明显
提高，但 no-match/rank>5、MW macro-F1、confidence ECE 和 learned narration 仍未达到
发布门。`mei-1.0-51m-sft-gap-pilot-v1` 是后续候选语料，不得倒写成已被本轮消费。

## SFT 语料重建候选：`mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1`

针对上述缺口新生成的候选 SFT 套件，状态为 `prepared_not_yet_consumed`（未启动
SFT，未训练，不改 CURRENT.json）。绑定 Base 为
`mei-1.0-51m-base-cpt600m-clean-source-v3-v1`（weights sha256
`6d55a61773cdd0a6713c43fa7565f8c2fa64414c3e4a1beadfb6507e0d91752a`）。位置：
`.local/artifacts/mei-1.0-51m/exp-000600m/corpus/sft-suite/mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1/`。

- 生成器：`corpus-factory/generators/rebuild_zh_v1/`（新增模块，六个 family 各一
  个文件 + 共享 `common.py`/`text_variants.py`/`mw_scenarios.py`，`build.py` 编排）。
- 工具目录：动态读取，147 deploy tools / 14 families，211 training tools / 22
  families（与 eval-v7 bank 校验一致，未硬编码）。
- 六个 family 合计 11,910 行：retrieval 2,760、full_call 2,340、agent 1,080、
  mw_disposition 3,680、narration 900、confidence 1,150（confidence 的
  `label` 全部为 `null`，等真实 SFT 后由 runtime harvest）。
- 冻结评测扩展：`mei-51m-longitudinal-eval-v8-retrieval-depth`，在 v7 基础上补齐
  no-match（v7 仅 60 行 → 新增 dev/test 各 ~190 行）与 rank>5（v7 仅 6-7 行 → 新增
  dev/test 各 ~190 行）的统计覆盖，`locked_test_used_for_threshold_tuning=false`。
- 已知质量缺口（详见 `corpus/sft.json` 的 `rebuild_candidate.known_quality_gaps`）：
  自由文本字符串仍有部分占位填充值；unit_conversion/chinese_numeral 场景转换命中率
  约 66–79%（非 100%）；跨 family 的非设计性精确重复率约 11–21%；未重建
  schema_generalization 与跨语言/BFCL 基准层。
- 未消费：本候选未被任何训练 run 引用，`training_started=false`，
  `corpus_release_eligible=true`，`ready_for_600m_sft=true`。检索/MW/confidence/
  narration 的实际模型质量指标全部 `pending_sft`，不得用语料统计冒充。

### v2 更新：审计旧 MW 语料后扩量（当前候选）

`mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2` 取代上面 v1 成为当前候选（v1 保留、不删除、不覆盖）。
起因：审计旧 `mei-1.0-51m-tool-sft-v4-300m-v4` 的 mw-disposition.train.jsonl（13,763 行）发现
**20 类中有 12 类 `candidate_tool` 字段 100% 为 null**，包括 `capability_insufficient` 本身
（375 行全部为 null）——完全没有工具级 ground truth，无法验证甚至无法对旧数据做"批次可见性重标注"
式的原地修复。详见
`.local/artifacts/mei-1.0-51m/exp-000600m/corpus/sft-suite/mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2/governance/old-mw-corpus-audit.json`。

这确认了 v1 的 "replace" 决策方向是对的，但也说明 v1 的 mw_disposition 规模（140/类）相对旧语料
（294–3263/类）偏薄。v2 将 mw_disposition 扩到 500/类，总量 11,800 行（train 8,320），
每一行都有明确 grounding 的 `candidate_tool`；其余五个 family 与 v1 保持一致未变。同步扩量前修复了
一个多样性 bug（部分 MW 场景模板不随 seed 变化，扩量后会造成大量精确重复），修复后
undesigned exact-duplicate 率约 16.3%（原始估算若不修复会到 ~39%）。冻结评测同步生成
`mei-51m-longitudinal-eval-v8-retrieval-depth-v2`。
