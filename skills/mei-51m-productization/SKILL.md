---
name: mei-51m-productization
description: >-
  Orchestrates the four independently bound mei-1.0-51m product phases: QAT,
  SFT alignment, model evaluation, and runtime/release. Use for cross-phase
  planning and status; delegate phase execution to the specialized Skills.
---

# mei-1.0-51m 产品化编排

## 正式管线

本 Skill 不再直接拥有 QAT/SFT/评测/发布实现。它从
`model-factory/contracts/PIPELINES.json` 读取 current child pipelines，检查
相邻阶段 binding 和 receipt 后路由：

1. `mei-51m-qat-training`
2. `mei-51m-sft-alignment`
3. `mei-51m-model-evaluation`
4. `mei-51m-runtime-release`

`mei-51m-adaptive-productization-v5` 暂时保留为历史兼容编排 ID；新 cycle
必须走 phase binding，不得直接依赖其 300M 默认路径。

## 硬边界

- 每阶段使用新的 `mei-51m-phase-binding-v1`，显式冻结 cycle、pipeline/recipe
  SHA、源码 closure、输入 artifact SHA、参数和输出目录。
- 上游 receipt 不存在、hash 漂移、pipeline 不再 current、架构/源码/参数漂移时
  fail closed；禁止自动回退到上一周期。
- SFT/evaluation/runtime 可以复用同一 run，但必须将该 run 的 `plan.json`
  作为 hash-bound input。
- 编排不修改 `CURRENT.json`，不训练、不发布。

## 入口

```bash
python scripts/doctor.py
python scripts/plan.py --help
python scripts/status.py --run-dir <path>
python scripts/compare.py --help
```

旧的 `run.py`、`resume.py`、`adopt.py`、`final_audit.py` 仅保留历史 run
修复兼容；新 cycle 不使用它们启动执行。
