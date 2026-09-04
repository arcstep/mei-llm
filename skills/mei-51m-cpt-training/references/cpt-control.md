# CPT control

状态判定顺序：parent release/identity → PID 与 run lock → heartbeat/progress →
checkpoint weights/optimizer/sampler/cursor → ledger。陈旧 ledger 不能证明进程停止。

恢复仅在 run fingerprint 与完整 checkpoint 一致时原地继续；不完整或不可信对象
隔离后从最后可信 Base 使用新 run ID。`hybrid_recovery` 可做明确标注的实验，但不能
宣称纯 exposure 对照。

正常结束后先 `register-base-candidate`，再 `propose-freeze`。模型文件提升必须独立
copy/clone 并核验 bytes/SHA，原 run 继续保留。最终结论同步到 cycle 的
`model/BASE.md`、`evaluation/SCORECARD.md`、`decision/DECISION.md`。
