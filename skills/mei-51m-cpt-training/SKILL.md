---
name: mei-51m-cpt-training
description: >-
  Plans, starts, observes, resumes, gates, and registers continued pretraining
  for mei-1.0-51m from a frozen parent Base and an approved CPT delta. Use for
  cumulative exposure, live-run reconciliation, checkpoints, optimizer and
  sampler state, Base candidates, and freeze proposals.
---

# mei-1.0-51m CPT

## 正式入口

只通过 `orchestration.lifecycle_51m` 的 `track=cpt` 执行：

`corpus_freeze → cpt_readiness → cpt → cpt_gate`

## 硬边界

- target 是累计 exposure，必须大于 parent actual exposure。
- 先核验 parent、corpus quality/Merkle、PID/lock/heartbeat/checkpoint。
- tokenizer、tensor identity、optimizer、sampler/cursor 或 hash 漂移时 fail closed。
- OOM 只能调 microbatch/accumulation，不能改变 exposure 和有效 batch。
- 三个资格轴独立；可续训不等于可自动晋升，也不等于语料可复用。
- 只生成 Base candidate 和 freeze proposal；不修改 `CURRENT.json`。

## 入口

```bash
python scripts/doctor.py
python scripts/plan.py --run-id <id>
python scripts/status.py --run-id <id>
python scripts/run.py --run-id <id> --confirm-training
python scripts/resume.py --run-id <id> --confirm-training
python scripts/register_base.py --run-id <id>
python scripts/propose_freeze.py --candidate <path>
```

恢复和终态语义见 `references/cpt-control.md`。
