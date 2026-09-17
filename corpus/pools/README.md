# 语料池：按用途分组，批次按日期排序

- [frozen](frozen/)：当前锁定的 CPT 训练输入（被 v1.3 adoption 引用为 `current_locked_inputs.cpt`）。
- [raw](raw/)：原始下载。
- [candidates](candidates/)：候选正文，内分基础语言、口语、文学、代码结构、任务工具及混合来源。
- [task-trials](task-trials/)：任务工程案例、转换试批及旧准备修订。
- [frozen-history](frozen-history/)：历史冻结输入；本次简英方案已取代，不能直接作为当前开训依据。
- [incomplete](incomplete/)：未完成导出、中断下载和失败 staging。
- [history](history/)：汇总、去重等过程记录。
- [zh-v2-pool](zh-v2-pool/)：旧训练链仍引用的混合池，本次保留原路径。

批次目录形如 `2026-09-14-mei-51m-v1.3-task-preparation-r06`。日期前置便于名称排序；r06 只表示产物修订，不是模型第六代。包内旧 ID 与凭据不改写。

48 个原平铺条目已实际迁入上述分组，原位置不保留软链接。旧路径由 [.internal 迁移表](../../.internal/registry/migrations/2026-09-corpus-pools.json) 经控制面解析。直接使用旧绝对路径的外部脚本须改为新路径或调用仓库 resolver。

返回[语料总入口](../README.md)查看各代采用关系。
