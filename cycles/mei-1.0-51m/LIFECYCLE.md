# mei-1.0-51m 工作指南

> 角色：公开仓内、自包含的产品与生命周期简明合同。实时成果以 `cycles/`、registry 和不可变 artifact 为准。

## 1. 唯一主线

`mei-1.0-51m` 是固定 51,463,797 个部署 LM 参数的文本模型身份。`exp-000300m`、`exp-000600m`、`exp-000900m` 等表示累计训练 token exposure，不是参数规模。上层组合 ASR/TTS 不属于本 cycle；未来原生接收音频或图像的新模型必须使用新的 model ID 和独立 cycle 索引。

标准产品不是单个权重文件，而是：不可变 Base、CQ2-QAT 后的工具调用主干、retrieval/MW/confidence/narration 独立 heads、tool index、统一 package、Python/MLX 与 Browser-WASM Runtime/SDK，以及冻结数据、评测和决策证据。

## 2. 五域入口

| 入口 | 职责 |
|---|---|
| `models/` | 模型身份、架构、tokenizer、recipe；历史由 Git 追踪 |
| `corpus-factory/` | 当前 CPT/SFT/Eval 生产、审计和 release 系统 |
| `model-factory/` | 当前训练、评估、编排和发布方法；每轮由 pipeline lock 固化 |
| `cycles/` | 每个 exposure rung 的永久成果、指标与决定 |
| `platform/` | 当前唯一 Runtime、SDK 与 Packaging 实现 |

轻量 CLI/registry 门面位于 `src/mei_llm/`，机器索引位于 `.internal/registry/`；后者不得
保存唯一算法源码。大文件只在 Gitignored `.local/artifacts/<model>/<cycle>/`。默认阅读路径：
`README.md → cycles/mei-1.0-51m/INDEX.md → cycle README → CORPUS/PIPELINE/SCORECARD/DECISION`。

## 3. 每轮 cycle 的完成定义

每个已执行 rung 必须同时冻结：

- 父 cycle、目标/增量 exposure 与累计 CPT lineage；
- 本轮 CPT delta、完整 SFT suite、Eval lock 及 corpus factory revision；
- Base 方法与本轮训练/评估 `PIPELINE.lock.json`，包括源码 capture 和 stage 闭包；
- Base、QAT、Product、heads、tool index 与 package 的稳定 URI/hash；
- Base/QAT/retrieval/full-call/Agent/MW/confidence/narration/Runtime 逐项指标；
- `process_complete`、`release_eligible`、`parent_eligible`、`corpus_reuse_eligible` 四个独立结论；
- 已知 confound、blocker、下一轮的 `reuse / replace / retire` 决定。

质量不足时可以 `process_complete=true / release_eligible=false`，但不能把低分隐藏为流程未完成，也不能无限扩大训练预算。`CURRENT.json` 只在用户显式 finalize 时修改。

## 4. 训练目标边界

| 目标 | 职责 |
|---|---|
| Base CPT | 中文语言与任务理解底座 |
| CQ2 QAT | 让主干适应最终 Q2/Q4 量化数学 |
| Full-call SFT | 在当前五工具批次中生成完整工具名与参数 |
| Agent SFT | call → verified result → 继续调用或终止 |
| Retrieval R0/R1/R2 | 冻结 LM 上的全目录候选排序；最终产品使用 R2 |
| MW disposition | 独立 20 类不能继续原因；不是检索或置信度 |
| Confidence | 用真实 Runtime outcome 判断调用执行风险 |
| Narration rank-16 sidecar | 仅在终态把已验证结果表达为短中文 |
| MTP | 训练旁路消融；部署 package 不携带 MTP tensor |

固定门序为 `retrieval → 五工具分批 → constrained generation → grammar/schema → provenance/permission/state → MW → confidence → execute/refuse → terminal narration`。learned head 不得覆盖确定性安全拒绝。

## 5. 跨 cycle 可归因比较

300M、600M、900M 等必须用相同冻结评测和 Runtime 合同并列展示 Base valid loss、CQ2 loss 税、retrieval recall/no-match/rank>5、工具名及参数 exact、Agent 多步成功、MW macro-F1、confidence AUROC/ECE、narration learned/delivered、包体、Python 与 Browser-WASM 性能。

若同时改变 Base、语料、SFT 或 Runtime，报告必须把差异标为 confound，不能把全部变化归因于 exposure。现有 300M/600M 的真实结果与限制见 `cycles/mei-1.0-51m/INDEX.md`。

## 6. 固定安全边界

- Base、历史 run 和 receipt 不覆盖；损坏对象隔离后恢复。
- 每项 corpus slice 都记录来源、许可、hash、split、leakage 和复用决定。
- 公开仓不提交权重、大语料、隐私审查正文或私有文档链接。
- 真实红队证据和备案材料进入独立私有仓；公开 release 只携带不可逆摘要和 clearance 状态。
- 未经明确授权不训练、不发布、不 commit/push、不修改 `CURRENT.json`。
