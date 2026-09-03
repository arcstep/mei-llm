# Orchestration

这里是正式生命周期入口：组合阶段 DAG、指纹、resume、supervisor 和终态 receipt。
当前入口由 [`../contracts/PIPELINES.json`](../contracts/PIPELINES.json) 指定；辅助模块不能凭
文件名自行升级为正式 pipeline。

