# exp-000600m 流水线

本轮从正式 300M parent 继续预训练至实际 600,001,765 token，再以与 300M 相同的
adaptive-v5 合同执行 QAT 后续产品化和配对评估。

## Base 段

- 历史方法：continued pretraining。
- 实际 run：`training/runs/mei-1.0-51m/20260902-cpt-600m-clean-source-v3`。
- 正式产物：[Base RELEASE](../../../../models/mei-1.0-51m/releases/exp-000600m/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/RELEASE.json)。
- 启动时有 source manifest，但 lineage 是 `hybrid_recovery`，且语料审计为
  `corpus_diversity_degraded`；checkpoint 可续训，不得自动晋升 parent。

## 产品化段

产品化因历史恢复分成三个不可变 run：训练前缀、heads/package，以及基于已打包候选重跑
Python/Browser-WASM 门禁。精确 run ID、指纹、plan hash 和阶段顺序见
[`PIPELINE.lock.json`](PIPELINE.lock.json)。最终配对审计确认流程完成，但质量与性能门退化，
所以产物不可发布。

## 源码可恢复性

历史 plan 各自保存 242 或 244 条源码哈希。最弱的一段能恢复 213/242 个精确字节，29 个
仅剩哈希；这不否定模型文件和训练 receipt，却意味着旧产品化链不能宣称源码逐字节完全
可复现。未来 cycle 必须在启动前冻结完整 source bundle 和逐 stage 闭包。

