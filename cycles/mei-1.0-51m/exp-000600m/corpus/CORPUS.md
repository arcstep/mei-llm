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
