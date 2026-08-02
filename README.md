# mei-llm

MEI / MeiLang 基座模型工作线的**代码根**（与 `mei-agent`、`rubble` 同级独立仓）。

设计合同与题库 MD 正文仍在文档仓：`docs/draft/mei-llm/` · 薄入口 `docs/mei-llm/README.md`。

## 目录

```text
scripts/              # 训评 / 导出 / 隔离检查
train/                # SFT 种子与 ops README
mlx/                  # config · exports · adapters（大权重见 .gitignore）
data/eval/            # 评测集机器面（如 eval-bank-v0.pending.jsonl）
experiments/runs/     # 跑分产物（默认 gitignore）
```

## 常用命令

```bash
cd mei-llm

python3 scripts/check_train_eval_isolation.py
python3 scripts/export_mlx_sft_seed_v0.py
python3 scripts/run_eval_mlx_v0.py --help
python3 scripts/run_eval_qwen_cloud_v0.py --help
```

稳定背景包默认读 monorepo：`docs/draft/mei-llm/2026-07-31-ctx-stable-v0.md`（可用 `--ctx` 覆盖）。

## 依赖

见 `requirements.txt`。MLX 本机评测需 `mlx` / `mlx_lm`；云 / Ollama runner 可选用旁路 `tools/mei-eval/python` 的 `mei_eval` 加载 `.env`。
