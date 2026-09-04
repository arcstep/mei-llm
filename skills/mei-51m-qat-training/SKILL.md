---
name: mei-51m-qat-training
description: >-
  Runs primary CQ2 quantization-aware training for mei-1.0-51m from a
  cycle-bound Float anchor and explicit hash-bound phase binding. Use after
  CPT has produced a frozen Base and before any SFT alignment.
---

# mei-1.0-51m QAT

## 何时使用

已经有冻结 Base 与 Float anchor，需要执行 CQ2 QAT 时使用。Q4 诊断是可选的
旁路证据，不属于当前 `mei-51m-qat-cq2-v1` 执行图。

## 输入

只接受 `mei-51m-phase-binding-v1`：

- 当前 cycle；
- Base release/weights 及 SHA；
- Float anchor；
- replay corpus；
- QAT recipe SHA、源码 revision/manifest/closure；
- target tokens、seq len、batch、accumulation、LR、checkpoint cadence。

禁止从脚本默认值猜 300M/600M Base 或语料。

## 输出

CQ2 master、bit map、完整 checkpoint state、QAT import receipt 和资格门。
Q4 只作诊断，不能冒充 CQ2 产品主线。

## 入口

```bash
python scripts/template.py --cycle-id <cycle> --pipeline-id mei-51m-qat-cq2-v1
python scripts/doctor.py --binding <binding.json>
python scripts/plan.py --binding <binding.json>
python scripts/run.py --binding <binding.json> --confirm-training
python scripts/resume.py --binding <binding.json> --confirm-training
python scripts/status.py --binding <binding.json>
```

本 Skill 不执行 SFT、模型综合评测、SDK 验收或发布。
