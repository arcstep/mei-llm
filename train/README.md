# train（草案）

> **阶段**：仅 `Qwen/Qwen3.5-0.8B` SFT-first  
> **隔离**：`sample_id` 不得为 `EVAL-*`；改题先改 seed，再 export

## 命令

```bash
cd mei-llm

# 隔离门禁
python3 scripts/check_train_eval_isolation.py

# → mlx/exports/sft-smoke-v0/{train,valid}.jsonl
python3 scripts/export_mlx_sft_seed_v0.py

# 冷基线（需已能 load 权重）
python3 scripts/run_eval_mlx_v0.py --face face.edge --limit 5 --judge none

# SFT 烟测（在 mlx/ 下，需权重与 export）
# cd mlx && python3 -m mlx_lm lora --config config/qwen35-0.8b-sft-smoke.yaml
```

种子：`seed/sft-smoke-v0.jsonl`（合成对话，非题库抄录；现行约 30 条）。

```bash
python3 scripts/export_mlx_sft_seed_v0.py \
  --upsample-topics tp.bucket_archive,tp.archive_as_current,tp.bucket_ssot,tp.three_buckets \
  --upsample-copies 3
```

对照：

```bash
python3 scripts/compare_mlx_base_lora_v0.py --limit 8 --faces face.edge,face.dev
# 读 experiments/runs/*-compare/summary.json
# 笔记仍在 docs/draft/mei-llm/2026-07-31-mlx-0p8b-base-lora-compare-note.md
```
