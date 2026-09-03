# 旧 training/ 路径说明

此文件只解释迁移前路径，不是活跃入口。`CURRENT.training` 保留旧字符串并通过 migration map 解析。

- `mei-1.0-51m-train-v1/` → `.internal/src/mei_llm/training/pipelines/`
- `runs/` → 对应 cycle 的 Gitignored `.local/artifacts/mei-1.0-51m/<cycle>/runs/`

当前训练控制面从 `models/`、`corpus-factory/`、`cycles/` 与 `.internal/registry/` 解析输入；历史 notebook 仅作证据。
