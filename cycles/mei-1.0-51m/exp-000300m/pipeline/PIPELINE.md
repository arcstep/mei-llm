# exp-000300m 流水线

本轮包含两段事实：历史 scratch 预训练得到正式 300M Base，以及 adaptive-v5 产品化得到
QAT、工具调用 LM、retrieval R2、MW、confidence、narration 和 CQ2 package。

## Base 段

- 历史方法：scratch curriculum pretraining。
- 实际 run：`training/runs/pretrain-mei-1.0-51m-base-scratch300m-v1`。
- 正式产物：[Base RELEASE](../../../../models/mei-1.0-51m/releases/exp-000300m/base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json)。
- 源码保证：旧 release 未记录完整启动前 source capture，因此是 `legacy_unspecified`；当前
  `orchestration.lifecycle_51m` 只是未来等价入口，不冒充当时字节。

## 产品化段

产品化因历史恢复分成三个不可变 run：训练前缀、heads/package，以及基于已打包候选重跑
Python/Browser-WASM 门禁。精确 run ID、指纹、plan hash 和阶段顺序见
[`PIPELINE.lock.json`](PIPELINE.lock.json)。最终配对审计确认流程完成，但 F01、F04、F06、
F07、F10、F11 退化，所以产物不可发布。

## 源码可恢复性

历史 plan 各自保存 242 或 244 条源码哈希。最弱的一段能恢复 213/242 个精确字节，29 个
仅剩哈希；因此模型/receipt 可信，但这条旧流水线不能宣称“源码逐字节完全可复现”。详见
`SOURCE_RECOVERY*.json`。未来 cycle 必须在启动前冻结完整 source bundle 和逐 stage 闭包。

