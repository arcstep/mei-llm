---
name: mei-51m-corpus-quality
description: >-
  Audits and decides reuse eligibility for mei-1.0-51m natural, synthetic,
  SFT, and Eval corpora. Use for provenance, license, hash, dedup, split and
  leakage checks, template diversity, grounded schema checks, named human
  semantic review, natural holdouts, diagnostic learnability, A/B comparison,
  and reuse/replace/retire decisions.
---

# mei-1.0-51m 语料质量门

## 核心判断

`hash/dedup/leakage passed` 只证明完整性与隔离，不证明自然度、语义一致性、
可学习性或模型质量。

## 必须分轴

- source：provenance、license、bytes/hash；
- isolation：document/exact/near dedup、family/group split、eval leakage；
- representation：token budget、schema、grounding、template diversity；
- semantics：具名人工抽检与缺陷分类；
- learnability：自然 holdout 和便宜 diagnostic；
- outcome：正式训练后的 locked model metrics；
- decision：`reuse`、`replace` 或 fail-closed `retire`。

## 入口

```bash
python scripts/audit_source.py --help
python scripts/audit_synthetic.py --help
python scripts/audit_sft.py --help
python scripts/diagnose_signal.py --help
python scripts/compare.py --help
python scripts/decide_reuse.py --help
```

所有质量 receipt write-once。人工评审缺失不得以自动统计替代。阈值和
300M/600M 失败经验见 `references/quality-gates.md`。
