# corpora（共享语料面）

原始/蒸馏语料按 **corpus id** 存放，训练任务订阅，不按模型复制整树。

```text
corpora/
  zh-vocab-v0/          # 词表用汉字/文本；不是语义预训练
  zh-pretrain-v0/       # 冻结中文维基 unique（100M/300M 默认槽）
  zh-pretrain-v1/       # 1.04B 盘点快照：引用 v0 + FineWeb2-HQ/Schema/OAI（只读）
  zh-pretrain-v2/       # 1.0–1.5B 多源 CPT：v0 wiki + HQ + CWT2 口语 + 结构/技术
  shared-tool-traces-v0/# 多任务可复用的工具调用轨迹（SFT 池；空槽，不是风格卡）
  sft-style-v0/         # CrossWOZ/KdConv train 脱敏风格卡（无产品 gold）
```

本目录默认 **不入库大文件**（见根 `.gitignore`）。只提交 README、清单与小型样例。

调用方必须显式传入 `corpus_root` 与版本；skill `meilang-corpus-distill` 不捆绑外部语料树。

禁止：把 `eval/banks/**` 或 `EVAL-*` 题面写入任何 corpus shard。
