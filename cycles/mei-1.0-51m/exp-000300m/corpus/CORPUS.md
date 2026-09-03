# 300M 周期语料

本轮从随机初始化训练到实际累计 300,000,485 tokens。CPT 使用 `lm-v1` 四角色语料：
wiki、hq、structure、colloquial，计划配额为 166,657,644 / 98,372,582 / 4,861,158 /
30,108,616 tokens。

后续 adaptive-v5 产品化使用统一的 retrieval、full-call、Agent、MW disposition、
confidence 和 narration 合同。该套 SFT 已完成机制闭环，但 no-match、rank>5 retention、
MW macro-F1 与 learned narration 仍需补强，因此下一周期不得无条件照搬。

300M 的旧 CPT release 在 hash/dedup/isolation 上已有历史证据，但缺少与当前工厂同口径的
模板多样性和语义一致性审计，故 `corpus_reuse_eligible` 暂不推断为 true。
