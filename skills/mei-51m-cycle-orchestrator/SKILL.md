---
name: mei-51m-cycle-orchestrator
description: >-
  Orchestrates one cumulative-exposure cycle of mei-1.0-51m, normally a 300M
  token increment. Use when planning, starting, inspecting, comparing, or
  closing an exp-* cycle across corpus sourcing, corpus quality, CPT, QAT/SFT,
  evaluation, packaging, and eligibility decisions.
---

# mei-1.0-51m Cycle 编排

## 职责

只协调阶段和证据，不实现语料生成或训练算法：

`source → corpus quality → CPT → Base candidate → QAT → SFT → model evaluation
→ runtime/release → final audit`

## 硬边界

- exposure 是累计 token；增量 = target − parent actual exposure。
- planned cycle 只登记 registry，真正开工才物化 cycle/run。
- 启动前冻结 corpus Merkle、pipeline、源码 bundle/dirty patch、环境和 parent。
- `process_complete`、`release_eligible`、`continuation_checkpoint_eligible`、
  `automatic_parent_promotion_eligible`、`corpus_reuse_eligible` 分开记录。
- 不下载、不生成语料、不直接训练、不写 `CURRENT.json`。
- 300M/600M 是回归证据，不是未来 cycle 的硬编码参数。

## 执行入口

```bash
python scripts/doctor.py
python scripts/plan.py --cycle-id exp-000900m
python scripts/status.py --cycle-id exp-000900m
python scripts/verify.py
```

实际阶段分别交给 `mei-51m-corpus-sourcing`、
`mei-51m-corpus-factory`、`mei-51m-corpus-quality`、
`mei-51m-cpt-training`、`mei-51m-qat-training`、
`mei-51m-sft-alignment`、`mei-51m-model-evaluation` 和
`mei-51m-runtime-release`；`mei-51m-productization` 只做产品阶段编排。

完整流程与终态投影见 `references/workflow.md`。
