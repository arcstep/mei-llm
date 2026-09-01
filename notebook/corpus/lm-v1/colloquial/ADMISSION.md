# Colloquial admission (four-role scratch)

The mixed-fleet pack `zh-pretrain-colloquial-synth-pooled-v1` is admitted for **internal unlabeled scratch pretraining only**.

Frozen facts:

- 30,108,616 unique train tokens, consumed once in the 300M scratch mix
- generator: `mixed-bailian-fleet`
- `public_distribution_clearance_asserted=false`
- not a qwen-plus-only or public-distribution claim

Machine gates already required: quality hard gates, isolation, exact token count, product-owner approval in `corpus/lm-v1/colloquial/contract-mixed-v1.json`.

Rebuild / re-admit:

```bash
.venv/bin/python notebook/_tooling/scripts/admit_colloquial_mixed.py
.venv/bin/python training/mei-1.0-51m-train-v1/check_pretrain_readiness.py --require-formal
```

Do not restore `schedule.json` or qwen-only admission onto the serving root. Formal public CPT remains a later, separate contract.
