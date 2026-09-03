# 600M 决策

- checkpoint 数值与训练状态完整，`continuation_checkpoint_eligible=true`。
- 因 hybrid recovery 和语料多样性缺口，`automatic_parent_promotion_eligible=false`。
- `lm-v2-cpt-600m` 不得原样滚入下一 exposure，`corpus_reuse_eligible=false`。
- 可以从该 checkpoint 配合修复后的新语料继续训练，但必须以新 cycle/run ID 和新 fingerprint 执行。
- adaptive-v5 产品候选机制完整、质量不合格，不公开发布、不写 CURRENT。
