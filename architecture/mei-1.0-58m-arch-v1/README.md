# mei-1.0-58m-arch-v1

Needle 2 对齐的 MLX 学生网络。独立再实现，不调用官方 libneedle / `.cact`。

```text
architecture.py          # GQA / RoPE / HadamardMLP / engram / 4-lane mHC / tied LM
hidden_cells.py
contrastive_head.py
confidence_v2.py
config.py                # 从 spec/model.json 与 model-target-v2.json 加载
tokenizer.py             # 读取根 tokenizer/zh-24k-v1
parity.py
spec/model.json          # 冻结几何
spec/model-target-v2.json
```

参数量约 58,541,901。正式 LM pretrain 使用 `NeedleZhConfig.from_spec()`（`spec/model.json`）。`from_target_v2()` 才打开 ContrastiveHead / ConfidenceV2；`implemented_in_code` 不是已训通或可发布。

长度分账：`max_seq_len=2048` 为位置上限；从零 300M 课程为 150M@512 → 100M@1024 → 50M@2048；runtime ordinary KV window 目标为 256 + tool sinks。量化目标为 CQ2-first QAT，当前仅有诊断用 4-bit PTQ 扫描，不得写成已 CQ2。
