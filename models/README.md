# Models

这里同时回答“模型是什么”和“正式模型资产在哪里”。当前唯一主产品是
[`mei-1.0-51m`](mei-1.0-51m/README.md)，其不可替代的 Base、QAT、SFT/heads 与端侧
package 统一进入 [`mei-1.0-51m/releases/`](mei-1.0-51m/releases/)，并由逐周期
`ASSETS.json` 固定大小、SHA-256 与训练 lineage。

`cycles/` 解释每轮语料、过程、指标和决策；`models/*/releases/` 保存被这些周期产出的正式
二进制成果。源码可以重建程序，却不能无成本重建训练权重，因此权重不再埋在 run 或泛化
的 artifacts 目录里。

在上层组合 ASR/TTS 不会产生新的模型成员；只有原生接收图像、音频等模态的新模型，才在
拥有明确架构合同、tokenizer/input contract 和首个 cycle 后创建，避免空目录冒充路线图。
