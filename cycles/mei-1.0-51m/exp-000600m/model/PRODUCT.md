# 600M 产品候选

最终候选 `mei-1.0-51m-cpt600m-tool-sft-cq2-v2-adaptive-v5` 使用与 300M 相同的
adaptive-v5 数据、评测与 runtime 合同。

- 410 个 tensors，其中 LM 400；MTP tensors 为 0。
- package 18,858,025 bytes，满足 18 MiB 合同。
- natural exact 高于 300M，但 MW macro-F1、confidence ECE、learned narration 与性能门仍降级。
- 终态为 process complete、release ineligible。
- 主权重与四个 heads：[`models/.../exp-000600m/sft/`](../../../../models/mei-1.0-51m/releases/exp-000600m/sft/)。
