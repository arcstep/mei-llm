# mei-llm

MEI 1.0 58M 及其训练、评测工作线的代码根（独立 Git 仓）。本仓不是在线模型 provider SDK。

现行指针见 `CURRENT.json`。布局合同见 `DESIGN.md`。

## 目录

```text
CURRENT.json
tokenizer/zh-24k-v1/     # 冻结词表
corpus/lm-v1/            # 冻结消费面（mix + 各包 tokens）
architecture/            # 模型结构（Needle 主干与 heads）
training/                # 正式 trainer 与 runs
runtime/                 # Route-ID v1 与 Needle2 v2
base/                    # 正式 base（当前为空）
sft/                     # 正式 SFT 模型（当前为空）
notebook/                # 语料准备、测试、验证、评测、归档
```

当前阶段：`scratch-pretrain-ready`。`CURRENT.corpus` 指向 `corpus/lm-v1`。四角色 300M 从零预训练已冻结：口语 30,108,616 与 structure 4,861,158 完整消费，课程 150M@512 → 100M@1024 → 50M@2048。正式架构与 trainer 已升格；`CURRENT.base` / `CURRENT.runtime` 在 300M final 过门前保持 `null`。

从零预训练默认走 `from_spec()` 主干（58,541,901）+ 冻结 `zh-24k-v1`。默认训练课程从 `seq_len=512` 起；`max_seq_len=2048` 是位置上限，末段 50M 证明 2048。v2 retrieval/confidence 头不进默认 LM pretrain。CQ2/QAT 未落地。

## 常用命令

```bash
cd mei-llm

.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --count-params
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --smoke
.venv/bin/python training/mei-1.0-58m-train-v1/check_pretrain_readiness.py --require-formal
.venv/bin/python training/mei-1.0-58m-train-v1/run_scratch_curriculum.py --dry-run
.venv/bin/python training/mei-1.0-58m-train-v1/run_scratch_curriculum.py --pilot-5m

.venv/bin/python notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2
.venv/bin/python notebook/_tooling/scripts/check_train_eval_isolation.py --scope sft-v2

.venv/bin/python notebook/_tooling/model/mei-1.0-58m/check_student.py
.venv/bin/python notebook/_tooling/scripts/test_mei_route_runtime.py
.venv/bin/python notebook/_tooling/scripts/test_mei_v2_runtime.py
```

依赖见 `requirements.txt`（转发到 `notebook/_tooling/requirements/`）。请用仓内 `.venv`；系统 `python3` 通常没有 `mlx` / `sentencepiece`。
