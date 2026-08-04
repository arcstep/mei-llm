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

Evidence 派生的领域样本必须同时记录：

- `task_catalog_release`：任务域知识、Catalog 与 contracts 的版本；
- `evidence_release`：求解事实/代码证据版本；
- `source_refs` 与 `transform_recipe`。

缺任一项时 `check_train_eval_isolation.py --domain-train ...` 必须失败。

```bash
python3 scripts/export_mlx_sft_seed_v0.py \
  --upsample-topics tp.bucket_archive,tp.archive_as_current,tp.bucket_ssot,tp.three_buckets \
  --upsample-copies 3
```

对照：

```bash
python3 scripts/compare_mlx_base_lora_v0.py --limit 8 --faces face.edge,face.dev
# 读 experiments/runs/*-compare/summary.json
```
