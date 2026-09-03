# exp-000600m 模型资产

本目录保存 600M exposure 周期的真实二进制成果：

- `base/`：600M Base 候选与可续训优化器状态；
- `qat/`：CQ2 QAT 主权重与 Q4 diagnostic；
- `sft/`：最终 Agent/full-call 主干，以及相互独立的 retrieval R2、MW、confidence、narration；
- `package/`：Python/MLX 与 Browser-WASM 直接加载的统一 CQ2 v2 package。

600M Base 具有 hybrid-recovery/source-diversity confound，未被自动提升为 `CURRENT`；产品流程
已经完成但质量/性能门禁未全部通过。详情见
[`../../../../cycles/mei-1.0-51m/exp-000600m/`](../../../../cycles/mei-1.0-51m/exp-000600m/)，
完整路径、大小和哈希见 [`ASSETS.json`](ASSETS.json)。
