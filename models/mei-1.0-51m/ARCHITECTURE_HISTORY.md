# 架构迁移说明

旧目录 `architecture/` 已退出活跃结构；`CURRENT.architecture` 中的历史字符串由
`.internal/registry/migrations/2026-09-four-domain.json` 解析，不改写原始指针。

当前唯一真源是 [`architecture/`](architecture/)。它包含精确 51,463,797 参数的
SAN-like 主干、独立权重几何/runtime/training-aux 契约与冻结规范，不含权重、语料或训练循环。
