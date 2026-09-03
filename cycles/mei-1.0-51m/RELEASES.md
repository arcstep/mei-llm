# mei-1.0-51m 发布队列

当前没有可公开发布候选。300M 与 600M adaptive-v5 均为
`process_complete / release_ineligible`；具体 blocker 见各自 cycle 的
`decision/DECISION.md`。

未来候选必须从 cycle registry 解析，不得手工从 run 目录拷贝。每个发布目录应包含
`RELEASE.json`、模型卡、评测卡、license/provenance 摘要与 SHA256SUMS；实际权重只输出到
Gitignored `.local/dist/`。
