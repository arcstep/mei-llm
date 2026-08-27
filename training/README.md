# training/

正式训练主路径。现行指针：`CURRENT.training`。

- [`mei-1.0-58m-train-v1/`](mei-1.0-58m-train-v1/) — trainer、data packing、checkpoint I/O、recipes
- [`runs/`](runs/) — 正式 run 的日志与中间 checkpoint；promote 后进入 `base/` 或 `sft/`

只消费根上的 `tokenizer/`、`corpus/`、`architecture/`。不从 notebook 读实现真源。语料生产、审计、隔离检查仍在 notebook；被接受后才进入 `corpus/`。
