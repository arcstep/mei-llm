# sft-style-v0

Needle-zh 家居 SFT 的**风格源**，不是产品 gold，也不是 `shared-tool-traces-v0` 轨迹池。

- 只读 CrossWOZ / KdConv 的 **train** split。
- 卡片只有脱敏口吻标签与短参考；不含原始 dialog_act / KG / 产品 `answers`。
- 近重复哈希用于挡住教师抄原句。raw / cards jsonl / hashes 默认 gitignore；本 README、SOURCES、manifest 入库。

构建：

```bash
python3 scripts/build_needle_zh_style_bank.py
```
