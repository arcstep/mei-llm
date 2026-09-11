# Base evaluation

评估未产品化 Base 的语言损失和固定 probes，不训练权重。

- `freeze_float_baseline_51m.py`：冻结 Float Base-LM anchor/control；
- `pretrain_probes.py`：预训练能力 probes。
- `paired_cpt_diagnostic.py`：独立父子底座复评，完整张量加载、有效 token 加权与输入/源码漂移检查。

配对诊断入口：`PYTHONPATH=src .venv/bin/python -m mei_llm cpt diagnose --config <config.json>`。
配置显式绑定两份 weights/SHA256、layout、tokenizer SHA256、probe 文件和独立输出目录。
`windows_per_role=0` 使用全部窗口，正数使用确定性跨范围抽样。执行前保存源码字节归档与
输入哈希；结束生成独立 receipt，不改变旧 gate、不训练、不授权 parent 晋升。

输出只有在绑定 Base hash、eval lock 与源码 manifest 后，才能进入 cycle scorecard。

恢复实验可显式指定 `recovery_gate_policy`：配置冻结六角色权重和容差，要求
`windows_per_role=0`，同一次复评的第一份 checkpoint 为对照、第二份为候选。
`recovery-gate.json` 单独记录配额加权 valid、逐角色保护和 probe 保护；计算完成
不等于质量通过，通过也只支持继续有界诊断，不授权完整 1800M 或自动晋升。

真实 51M 累积资源探针：`PYTHONPATH=src .venv/bin/python -m mei_llm cpt resource-probe --config <config.json>`。
只加载哈希绑定的完整 Adam 状态，在合成 token ID 上做 1–4 次 1×8、2048 长度更新，
不保存权重、不消费自然语料、不增加 CPT 链 exposure；资源结果不代表质量改善。
源码 v2 快照包含训练、公共模块、语料工厂、控制平面与架构的可恢复字节和逐阶段闭包。

有界恢复入口：`PYTHONPATH=src .venv/bin/python -m mei_llm cpt recover --config <binding.json> --confirm-training`。
只接受完整性通过、收益门 blocked 的显式诊断 parent，最多 611 次更新；batch/LR 配置变化、
旧池有界 reuse、全部输入哈希与资源结果必须显式绑定。旧正式续训门不会因此放宽。
训练结束自动执行两份 checkpoint 的六角色全量复评和独立恢复门，然后停止。
每 128 次更新保存独立状态；`STOP` 文件可在安全更新边界停止。任何源码/CURRENT/输入漂移
均拒绝将候选用于后续步骤。诊断产物没有自动 base 注册、发布或继续 1800M 的权限。

原切片重训入口：`PYTHONPATH=src .venv/bin/python -m mei_llm cpt replay --config <binding.json> --confirm-training`。
从 canonical 1200M 完整状态出发，保留原 1500M 配额抽样顺序和 token cosine LR，使用
microbatch=1/accumulation=8。目标按旧 run 实际终态 1,500,001,792 token 对齐，
末次可为不足八个微批的更新；预检与终检核对原终态六角色消费及游标。
每 2048 次更新保存新状态，结束后自动对比 1200M、旧 1500M、1510M 恢复与重训四份底座。
每两小时输出 `progress-reports.jsonl` 和标准输出进度，结束或失败另发终态报告。
这只是本机报告，不构成聊天唤醒订阅；CLI 未提供通知工具时不能声称会主动推送到对话。
