---
name: mei-51m-runtime-release
description: >-
  Packages an evaluated mei-1.0-51m run and executes Python SDK, Browser-WASM,
  resource, integrity and final release gates. Use only after model evaluation;
  this Skill never changes model weights.
---

# mei-1.0-51m Runtime / Release

## 何时使用

模型质量评测已经形成可复用 receipt，需要验证可交付包、SDK、浏览器、资源预算
和最终 release eligibility 时使用。

## 前置

- phase binding 指向与评测相同的 run dir 和 `plan.json` SHA；
- `sidecar_runtime_eval_v5` 已 passed/degraded 且 receipt 完整；
- Base、QAT、SFT、eval lock、recipe、源码 closure 未漂移；
- package ID 和运行预算显式冻结。
- `--phase-scope runtime-release` 只允许 package、Python、Browser-WASM
  和 final audit；缺少任一训练/评测 receipt 即拒绝，不回补上游 action。

## 入口

```bash
python scripts/phase.py template --cycle-id <cycle> --pipeline-id mei-51m-runtime-release-v5
python scripts/phase.py doctor --binding <binding.json>
python scripts/phase.py plan --binding <binding.json>
python scripts/phase.py run --binding <binding.json>
python scripts/phase.py resume --binding <binding.json>
python scripts/phase.py status --binding <binding.json>
```

本 Skill 只打包并验收，不训练权重。通过 final audit 也不会自动修改
`CURRENT.json` 或推送远端。
