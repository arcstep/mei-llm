# mei-llm

MEI / MeiLang 基座模型工作线的代码根（独立 Git 仓；与 `mei-agent`、`rubble` 并列）。

本仓只含可运行的脚本、数据与训练旁路。**设计主张与过程笔记不在本仓发布**；正式发行文档将另行写入本仓自有说明，而不是链到外部文档树。

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

背景包请用 `--ctx /path/to/ctx.md` 显式传入（仓库不捆绑外部文档路径）。

领域语料组合必须钉两个独立版本：`task_catalog_release`（Evidence Task
Model：L0–L9/LX knowledge + Catalog + contracts）与 `evidence_release`
（求解事实/代码）。训练样本不得从 sibling docs 或 bench 临时拼装任务定义。

跑分结束后同目录可生成客观报告 `report.md`（由 `summary.json` + `predictions.jsonl` 派生；需已安装旁路包 `mei_eval`）。

## 依赖

见 `requirements.txt`。MLX 本机评测需 `mlx` / `mlx_lm`。云 / Ollama runner 可选用已发布的 [`mei-eval`](https://github.com/arcstep/mei-eval)（`pip install -e` 其 `python/`），用于加载 `.env` 与写报告。
