# notebook/

外围实验与治理基地，不是正式模型主路径。

| 目录 | 职责 |
|------|------|
| `corpus/` | 语料生产、审计、候选；`work/` 为加工面；`outbox/accepted` 只读指向根 `corpus/`；assemble 为 receipt，不是成果树可写别名 |
| `evaluation/` | 测试、验证、评测 banks 与证据 |
| `sft/` | SFT pack 过程面（未 promote 的 packs/reviews） |
| `archive/` | 历史权重、脏语料、legacy 产品、旧 runs |
| `_tooling/` | 外围脚本、兼容入口、skills |
| `_migration/` | 迁移记录 |

正式架构、trainer、runtime 在仓根 `architecture/` `training/` `runtime/`。`notebook/_tooling/model/mei-1.0-58m` 只保留指向正式树的 symlink。
