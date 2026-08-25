# mei-llm

MEI 专家模型及其训练、评测工作线的代码根（独立 Git 仓；与 `mei-agent`、`rubble`
并列）。本仓不是在线模型 provider SDK；运行时模型连接由消费它的 Host 显式配置。

本仓只含可运行的脚本、数据与训练旁路。**设计主张与过程笔记不在本仓发布**；正式发行文档将另行写入本仓自有说明，而不是链到外部文档树。

布局合同见 `DESIGN.md`：语料与评测共享，模型线按 task 拆分。

## 目录

```text
corpora/              # 共享语料槽（大文件默认不入库）
eval/banks/           # 共享评测题库（多 task 订阅）
eval/shared/          # 共享 tool schema 等
tasks/                # 每条可训练模型线（配方 / seed / mlx）
  mei-expert-qwen35-0p8b/
  needle-zh/
scripts/              # 训评 / 导出 / 隔离检查
experiments/runs/     # 跑分产物（默认 gitignore）
```

## 常用命令

```bash
cd mei-llm

# 所有 task 的 train↔eval 隔离
python3 scripts/check_train_eval_isolation.py --all

# 既有 0.8B 专家线（默认仍指向该 task）
python3 scripts/check_train_eval_isolation.py
python3 scripts/export_mlx_sft_seed_v0.py
python3 scripts/run_eval_mlx_v0.py --help

# 中文 Needle 题库 schema / exact-match
python3 scripts/eval_needle_toolcall_v0.py
python3 scripts/eval_needle_toolcall_v0.py --bank eval/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl --split eval

# needle-zh 主线（规格 / 学生 / 词表 v1 / 预训阶梯 / SFT）
python3 tasks/needle-zh/model/check_student.py
python3 scripts/train_zh_vocab_spm.py --freeze-v1
python3 scripts/build_needle_vrm_holdout_v1.py --tier 2k
python3 scripts/check_train_eval_isolation.py --all
python3 scripts/build_zh_pretrain_v0.py --smoke
python3 scripts/train_needle_zh_pretrain.py --rung 100m --smoke
python3 scripts/train_needle_zh_sft.py --init pretrained --tier 2k --smoke
python3 scripts/ablate_needle_zh.py --require-baseline
```

背景包请用 `--ctx /path/to/ctx.md` 显式传入（仓库不捆绑外部文档路径）。

领域语料组合必须钉两个独立版本：`task_catalog_release`（可选；钉定所用
Evidence knowledge / contracts，以及若使用则钉定 `construction/task-seeds`
抽样 catalog）与 `evidence_release`（求解事实/代码）。训练样本不得从 sibling
docs 或 bench 临时拼装任务定义；不得把 seeds 称作 Task Model 或 Universe 真源。

跑分结束后同目录可生成客观报告 `report.md`（由 `summary.json` + `predictions.jsonl` 派生；需已安装旁路包 `mei_eval`）。

## 依赖

见 `requirements.txt`。MLX 本机评测需 `mlx` / `mlx_lm`。云 / Ollama runner 可选用已发布的 [`mei-eval`](https://github.com/arcstep/mei-eval)（`pip install -e` 其 `python/`），用于加载 `.env` 与写报告。
