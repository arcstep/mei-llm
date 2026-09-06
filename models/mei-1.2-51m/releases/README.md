# mei-1.0-51m 模型资产

这里是 Base、QAT、SFT 与端侧 package 的唯一正式文件入口。`exp-*` 表示累计训练
exposure；进入某个周期后即可直接看到权重，而不必进入训练日志寻找。

| 周期 | Base | QAT | 最终 SFT + heads | CQ2 package | 发布状态 |
|---|---|---|---|---|---|
| [`exp-000300m/`](exp-000300m/) | 已冻结 | CQ2 + Q4 | 已保存 | 已保存 | process complete / release ineligible |
| [`exp-000600m/`](exp-000600m/) | 候选已保存 | CQ2 + Q4 diagnostic | 已保存 | 已保存 | process complete / release ineligible |

每个周期的 `ASSETS.json` 固定关键文件大小、SHA-256、原训练证据路径与备份状态。大二进制
不进入 Git，但绝不视为可由源码随时重建的缓存；迁移、备份和发布必须先校验这些哈希。

`.local/artifacts/.../runs/` 中的旧文件目前作为第二份本机训练证据保留。本机双路径不等于
灾难恢复备份；`off_device_backup_status=open` 在完成独立存储备份前不得改为 completed。
