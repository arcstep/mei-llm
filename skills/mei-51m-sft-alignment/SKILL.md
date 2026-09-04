---
name: mei-51m-sft-alignment
description: >-
  Runs cycle-bound supervised alignment for mei-1.0-51m from an approved CQ2
  QAT receipt: bootstrap full-call/Agent/retrieval training, then adaptive R2,
  MW, confidence and narration alignment. Every run is selected by an explicit
  phase binding rather than exposure-specific defaults.
---

# mei-1.0-51m SFT / Alignment

## 何时使用

QAT 已完成，需要训练工具调用、Agent、多批检索、MW disposition、confidence
或 narration 时使用。

## 两个连续子阶段

1. `mei-51m-sft-bootstrap-v4`：Float task control、CQ2 import、R0、
   full-call、Agent、R1 和 index，形成 adaptive 所需的最小 seed prefix。
2. `mei-51m-sft-adaptive-v5`：fixed-five adaptive full-call/Agent、R2/index、
   MW、confidence 和 narration 增量对齐。

两阶段各自使用独立 binding；第二阶段把第一阶段 run 的 `plan.json` 作为
hash-bound input，不从固定 300M seed 路径继承。

## 入口

```bash
python scripts/phase.py template --cycle-id <cycle> --pipeline-id <sft-pipeline>
python scripts/phase.py doctor --binding <binding.json>
python scripts/phase.py plan --binding <binding.json>
python scripts/phase.py run --binding <binding.json> --confirm-training
python scripts/phase.py resume --binding <binding.json> --confirm-training
python scripts/phase.py status --binding <binding.json>
```

训练预算必须全部写进 binding。adaptive generation 与 sidecar 评测会被
`--phase-scope sft-alignment` 跳过；Skill 在 narration alignment 结束后停下；
锁定模型评测交给 `mei-51m-model-evaluation`，SDK/资源验收交给
`mei-51m-runtime-release`。
