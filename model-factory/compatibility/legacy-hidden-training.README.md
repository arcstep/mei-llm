# Training control plane

`pipelines/` 暂存迁移后的现有 51M trainer 实现；稳定入口逐步收敛为 CPT、QAT、SFT、
heads 与 productize 五类生命周期动作。不同 exposure 和数据版本由 cycle/recipe 驱动，
不得再复制训练脚本。
