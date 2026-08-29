# mei-llm

MEI 1.0 51M 及其训练、评测、场景运行时工作线的代码根（独立 Git 仓）。本仓不是在线模型 provider SDK。

现行指针见 `CURRENT.json`。布局合同见 `DESIGN.md`。

## 目录

```text
CURRENT.json
tokenizer/zh-24k-v1/     # 冻结词表
corpus/lm-v1/            # 冻结消费面（mix + 各包 tokens）
architecture/            # 模型结构（Needle 主干与 heads）
training/                # 正式 trainer 与 runs
runtime/                 # Route-ID v1；历史 Python+MLX 参考实现（非 SDK 产品名）
sdk/                     # 实验性 MEI Runtime 嵌入式 SDK
base/                    # 正式 base
sft/                     # 正式 SFT 模型（当前为空）
notebook/                # 语料准备、测试、验证、评测、归档
```

当前阶段：`scratch-300m-promoted`。`CURRENT.corpus` 指向 `corpus/lm-v1`，`CURRENT.base` 指向 immutable `base/mei-1.0-51m-base-scratch300m-v1`。四角色 300M 从零预训练已完成并晋级，课程为 150M@512 → 100M@1024 → 50M@2048。`CURRENT.sft` / `CURRENT.runtime` 仍为 `null`。

现行主干是 `architecture/mei-1.0-51m-arch-v1`（51,463,797 参数）+ 冻结 `zh-24k-v1`；与 58M checkpoint 不兼容。`max_seq_len=2048` 是位置上限，末段 50M 已训练到 2048。tool retrieval/MW/confidence 与 51M runtime 尚未发布。PTQ 只做分组件诊断；CQ2-first QAT 仍是强制产品路径，短 QAT pilot 入口目前 fail-closed，未开训。

产品场景是 `用户意图 → 工具调用 → 工具执行 → 已验证结果 → 用户可读解说`。工具调用由 51M 核心承担；解说走后端无关 NarrationProvider。51M 派生、Qwen、模板或其他后端均只是候选。现行 LM head 与输入 embedding 绑权，仓内尚无正式 51M head-only/LoRA/adapter 路由；任何 tool/narration 后训都必须使用新的 model ID，禁止覆盖 base。

## 常用命令

```bash
cd mei-llm

PYTHONPATH=architecture/mei-1.0-51m-arch-v1:training/mei-1.0-58m-train-v1 \
  .venv/bin/python -m pytest training/mei-1.0-58m-train-v1/test_arch_51m_readiness.py \
  training/mei-1.0-58m-train-v1/test_ptq_51m_scan.py
# 若未装 pytest，可直接跑：
# .venv/bin/python training/mei-1.0-58m-train-v1/test_ptq_51m_scan.py
.venv/bin/python training/mei-1.0-58m-train-v1/run_scratch_curriculum_51m.py --dry-run
.venv/bin/python training/mei-1.0-58m-train-v1/promote_scratch_base_51m.py
.venv/bin/python training/mei-1.0-58m-train-v1/freeze_float_baseline_51m.py
.venv/bin/python training/mei-1.0-58m-train-v1/scan_ptq_51m.py
.venv/bin/python training/mei-1.0-58m-train-v1/check_qat_pilot_readiness.py

.venv/bin/python notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2
.venv/bin/python notebook/_tooling/scripts/check_train_eval_isolation.py --scope sft-v2

.venv/bin/python notebook/_tooling/model/mei-1.0-58m/check_student.py
.venv/bin/python notebook/_tooling/scripts/test_mei_route_runtime.py
.venv/bin/python notebook/_tooling/scripts/test_mei_v2_runtime.py
```

依赖见 `requirements.txt`（转发到 `notebook/_tooling/requirements/`）。请用仓内 `.venv`；系统 `python3` 通常没有 `mlx` / `sentencepiece`。
