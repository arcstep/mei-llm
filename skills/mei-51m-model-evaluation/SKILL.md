---
name: mei-51m-model-evaluation
description: >-
  Evaluates a frozen mei-1.0-51m alignment run against hash-bound dev and
  locked-test banks across LM, retrieval, full-call, Agent, MW, confidence and
  narration. Use after SFT and before packaging or runtime release.
---

# mei-1.0-51m 模型评测

## 何时使用

权重训练已停止，需要判断模型质量、退化项、纵向变化或 release eligibility
时使用。它不训练权重，也不把 SDK、包体积和浏览器性能混入模型质量分。

## 约束

- binding 必须指向同一个 adaptive run，并校验其 `plan.json` SHA；
- 必须已有 `narration_adapter_v5` 可复用 receipt；
- eval lock 与数据集 manifest 必须 hash-bound；
- locked test 只用于最终判定，禁止调阈值；
- upstream recipe/source/input fingerprint 漂移时拒绝 resume。
- `--phase-scope model-evaluation` 只允许 adaptive generation 与 sidecar
  evaluation；任一训练 receipt 缺失时直接拒绝，绝不补跑训练。

## 入口

```bash
python scripts/phase.py template --cycle-id <cycle> --pipeline-id mei-51m-model-evaluation-v5
python scripts/phase.py doctor --binding <binding.json>
python scripts/phase.py plan --binding <binding.json>
python scripts/phase.py run --binding <binding.json>
python scripts/phase.py resume --binding <binding.json>
python scripts/phase.py status --binding <binding.json>
```

输出为模型质量 receipts，不打包、不发布、不修改 `CURRENT.json`。
