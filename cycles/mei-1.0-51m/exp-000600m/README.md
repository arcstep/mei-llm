# exp-000600m

从 300M Base 继续训练到实际累计 600,001,765 tokens，并在同一 adaptive-v5 合同下完成
产品化。Base loss 和工具调用主任务明显改善，但语料多样性与 hybrid recovery 构成
confound，MW、confidence、narration 和 runtime 性能仍有缺口。

终态为 `process_complete=true / release_eligible=false`。checkpoint 可继续训练，但不
自动晋升为正式 parent，`CURRENT.json` 未改变。

- [本轮语料](corpus/CORPUS.md)
- [本轮训练与评估流水线](pipeline/PIPELINE.md)
- [Base](model/BASE.md)
- [产品](model/PRODUCT.md)
- [真实模型文件](../../../models/mei-1.0-51m/releases/exp-000600m/)
- [与 300M 比较](evaluation/COMPARISON.md)
- [决策](decision/DECISION.md)
