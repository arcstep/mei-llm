# Training

仅存放会改变权重或训练 sidecar/head 的实现：

- `cpt/`：Base continued/scratch pretraining 与 CPT 门禁；
- `qat/`：CQ2/Q4 quantization-aware 训练；
- `tool_use/`：full-call、Agent、检索和联合上下文对齐；
- `heads/`：retrieval/contrastive、MW、confidence、MTP 等独立训练部件。

正式运行从 `orchestration/` 发起，不能通过挑选本目录中“版本号最大”的文件启动。

