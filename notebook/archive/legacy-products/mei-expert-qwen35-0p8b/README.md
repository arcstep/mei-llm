# mei-expert-qwen35-0p8b

既有 MEI 专家行为线：`Qwen/Qwen3.5-0.8B` SFT-first（MLX）。

共享评测：`notebook/evaluation/banks/mei-expert-v0/`。  
本 task 种子：`train/seed/sft-smoke-v0.jsonl`。  
MLX 工作目录：`notebook/archive/legacy-products/mei-expert-qwen35-0p8b/mlx/`。

```bash
cd mei-llm
python3 notebook/_tooling/scripts/check_train_eval_isolation.py
python3 notebook/_tooling/scripts/export_mlx_sft_seed_v0.py
python3 notebook/_tooling/scripts/run_eval_mlx_v0.py --face face.edge --limit 5 --judge none
# cd notebook/archive/legacy-products/mei-expert-qwen35-0p8b/mlx && python3 -m mlx_lm lora --config config/qwen35-0.8b-sft-smoke.yaml
```

隔离：`sample_id` 不得为 `EVAL-*`。Evidence 派生行必须带 `task_catalog_release` + `evidence_release` + `source_refs` + `transform_recipe`。
