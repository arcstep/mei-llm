# Cycle workflow

## 输入真源

- `.internal/registry/cycles.json`
- `model-factory/contracts/PIPELINES.json`
- 目标 cycle 的 `CYCLE.json`、`corpus/*.json`、`pipeline/PIPELINE.lock.json`
- parent `RELEASE.json` 与 artifact hashes
- `CURRENT.json`（只读）

## 顺序

1. 选择 planned target，核验 parent、实际累计 exposure 和增量。
2. sourcing 生成天然池 inventory 与 candidate mix。
3. synthesis 生成必要的 CPT gap-fill/SFT family；quality 独立裁决。
4. quality receipt 全部通过后冻结 CPT delta、SFT suite 和 eval lock。
5. CPT Skill 完成 readiness、训练/恢复、gate、Base candidate 和 proposal。
6. productization Skill 对冻结 Base 执行 current pipeline。
7. 将稳定结论投影到 cycle 的 corpus/model/evaluation/decision。

质量未过门可形成 `process_complete=true/release_eligible=false`；污染、hash、
身份或 CURRENT 漂移必须 blocked。
