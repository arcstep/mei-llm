# exp-000300m 模型资产

本目录保存 300M exposure 周期的真实二进制成果：

- `base/`：正式冻结 Base 与可续训优化器状态；
- `qat/`：CQ2 主线 QAT 权重和 Q4 对照权重；
- `sft/`：最终 Agent/full-call 主干，以及相互独立的 retrieval R2、MW、confidence、narration；
- `package/`：Python/MLX 与 Browser-WASM 直接加载的统一 CQ2 v2 package。

模型流程已经完成，但质量/性能门禁未全部通过，因此这里只是不可变候选资产，不代表可公开
发布。指标与决定见 [`../../../../cycles/mei-1.0-51m/exp-000300m/`](../../../../cycles/mei-1.0-51m/exp-000300m/)。
完整路径、大小和哈希见 [`ASSETS.json`](ASSETS.json)。
