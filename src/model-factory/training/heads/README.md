# Auxiliary heads

这里保存与 LM 主干区分管理的功能头训练：

- `train_contrastive_51m.py`：retrieval/contrastive head；
- `train_confidence_51m.py`：执行 confidence head 与校准输入；
- `mtp_ablation_51m.py`：训练期 MTP 辅助目标的受控消融，不进入部署包。

MW disposition 仍是独立 20 类 head，但其当前训练原语位于
`training/tool_use/sft_v3_training_51m.py`，由产品化 DAG 单独调用；它不等于 retrieval 或
confidence。head 的最终身份、输入权重与指标必须记录在 cycle lock 和 scorecard 中。
