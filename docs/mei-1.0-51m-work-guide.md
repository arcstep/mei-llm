# mei-1.0-51m 工作指南与状态基线

> 文档版本：2026-09-01 v2
>
> 适用产品：`mei-1.0-51m`
>
> 文档角色：长期目标、能力边界、工作图、状态判定与跨语料规模比较的工程 SSOT

## 1. 如何使用这份文档

本文回答四个问题：这个模型最终要成为怎样的产品；完整工作应包含什么；目前哪些机制已经实现、哪些能力尚未成立；后续 600M、900M、1.2B、1.5B、2.1B 等累计语料版本如何做可归因比较。

它不替代下列事实来源：

1. `base/<base-id>/RELEASE.json` 是冻结 Base 的事实来源。
2. `training/runs/.../run.json`、stage receipt、heartbeat 和 artifact hash 是单次运行的事实来源。
3. [`AGENTS.md`](../AGENTS.md) 保存 Codex/Agent 必须遵守的稳定约束。
4. workspace skill `mei-51m-cpt-lifecycle` 保存具体命令、恢复流程和 receipt/gate 操作方法。

发生冲突时，以不可变 artifact、hash 和 receipt 为准；本文中的运行状态快照只用于说明，不代替实时检查。历史 receipt 不回写，后续发现的新问题通过新评估或 erratum 补充。

状态词统一如下：

| 状态 | 含义 |
|---|---|
| `已验证（机制）` | 实现、契约、测试或实际数值路径已经跑通；不等于模型质量可用 |
| `待质量` | 机制存在，但冻结评测未达到产品要求 |
| `待补证` | 缺少足够、平衡或可比较的冻结评测，不能得出质量结论 |
| `训练中` | 运行仍是可变状态，不能作为冻结父版或产品结论 |
| `暂缓` | 有意不列入当前交付范围，不冒充已验证 |
| `有意不兼容` | 明确不追求该兼容目标，不属于缺陷 |

## 2. 产品目标

`mei-1.0-51m` 的目标不是一个只会续写文本的 51M 模型，也不是简单复制 Needle2。目标是得到一个中文增强、体积受控、可重复扩展训练的轻量工具 Agent 产品：

- 一个保持 51,463,797 个部署语言模型参数的中文 Base，并形成 300M → 600M → 900M → 1.2B → 1.5B → 2.1B 等累计 exposure 父链。
- 一个能从工具目录检索 top-5、生成受约束工具调用、接收可信工具结果并继续多步决策的产品模型。
- retrieval、MW disposition、confidence 和 narration 各自独立、语义清楚、可单独训练和评测；full-call SFT 则负责主干工具调用生成能力。
- 一个真正的 CQ2 产品包，统一容纳最终 LM、全部 sidecar、tokenizer 和 tool index，并验证量化没有明显破坏模型。
- Python/MLX 服务端路径和浏览器路径。浏览器以 WASM 负责 portable 控制面，以 WebGPU/等价 GPU 计算作为性能目标；CPU-WASM 只作为正确性回退，不作为最终性能基线。
- 一套可冻结、可复用、可归因的评测合同，用同一数据、预算、seed、runtime revision 比较不同 exposure 的 Base 与最终产品模型。

最终交付不是“Base + 若干互不相干的模型”。标准交付应包括：

1. 不可变 Base release。
2. 一个 `mei-model-package-v2` 产品包，包含 CQ2 LM、retrieval、MW disposition、confidence、narration sidecar 和完整 tensor directory。
3. 与最终 LM/检索头严格绑定的 tool index。
4. Python/MLX runtime/SDK 和 Browser-WASM runtime/SDK。
5. 冻结数据、评测合同、阶段 receipt、资源报告和对齐报告。

### 非目标

- exposure 名称不是参数规模；600M、900M、1.2B 等不产生 600M、900M、1.2B 参数模型。
- 不以单轮追求最佳分数为目标；分数不足时保存证据并结束该轮，不无限调参。
- 不声明 Needle2 分数复现，也不声明 `.cact`、`libneedle`、Cactus API/ABI 兼容。
- MTP 只做训练旁路消融，部署包不得携带 MTP tensor。
- 在 Python/浏览器质量尚未成立前，不把独立 Rust SDK、Node SDK、C/FFI 扩展作为当前主工期。
- 未获得明确授权时，不公开发布、不 commit/push、不修改 `CURRENT.json`。

## 3. 不可变身份与产品合同

| 项目 | 固定合同 |
|---|---|
| 产品 ID | `mei-1.0-51m` |
| 部署 LM 参数 | 51,463,797，sidecar 单独统计，不改变 LM 身份 |
| tokenizer | `zh-24k-v1`，必须以 hash 锁定 |
| SDK | `0.2.0-experimental` |
| wire | `mei-runtime-wire-v2` |
| package | `mei-model-package-v2` |
| ABI | `mei-runtime-abi-2` |
| 量化 | `mei-cq-v2-g128-wht-codebook`；group 128、固定 WHT、codebook、逐 group Q2/Q4 |
| runtime context | 最大 2048；稳定前缀最多 1024；普通滚动窗口 256；默认输出最多 128 |
| Agent 步数 | 默认最多 4，硬上限 8；每一步最多一个工具调用 |
| 产品包上限 | 18 MiB |
| Rust session 目标 | 64 MiB；独立 Rust 交付暂缓时仍保留为扩展债务 |
| Browser WASM heap | 96 MiB；WebGPU buffer 必须另行报告，不能藏入 heap 指标 |
| `CURRENT.json` | 只读；仅显式 compare-and-swap `finalize_current` 可修改 |

架构身份必须拆成三个合同，不能再用整个 `model.json` 原始字节 hash 代替兼容性：

- `weight_contract_sha256`：有序 tensor 名称、形状以及影响 tensor 几何的结构常量。
- `runtime_profile_sha256`：上下文、稳定前缀、滚动窗口、KV/activation dtype 等部署策略。
- `training_aux_sha256`：MTP 等只在训练期间存在的辅助配置。

Base release 一旦冻结即不可变。任何 SFT、QAT、量化、head 或 package 都必须产生新 run/package ID，不能覆盖 Base 或旧候选。

## 4. 模型组件与语义边界

| 组件 | 训练对象 | 职责 | 不能被误解为 |
|---|---|---|---|
| Backbone + Engram + mHC | CPT/QAT/full-call 时的 LM 主干 | 中文语言建模、上下文表征和工具调用 token 生成 | 任一分类 head |
| Retrieval R0/R1 | 独立 contrastive head；LM 冻结 | 将用户请求与工具 schema 编码并检索 top-5 | full-call、confidence 或 MW |
| Full-call | quant-aware LM SFT | 在 oracle/learned top-5 schema 条件下输出调用或拒绝 | retrieval head |
| MW deviation | 无 tensor、无训练 | 评测/治理层的 evidence-boundary 偏移判定 | MW disposition head |
| MW disposition | 独立 20 类 sidecar；LM 冻结 | 对不能继续执行的原因分类；只有 class 0 可以继续 | retrieval/confidence 或 MW deviation |
| Confidence | 独立二分类/校准 sidecar；LM 冻结 | 基于真实 runtime outcome 判断调用正确性/风险 | 允许绕过确定性安全门的裁决器 |
| Narration | 冻结主干上的 rank-16 logit-residual sidecar | 仅在终态消费已验证结果，生成简短中文解释 | 标准 LoRA、独立大模型或工具执行器 |
| MTP | 训练旁路 | 比较 t+1 与 t+2 辅助目标是否有益 | 部署能力或包内参数 |

Narration sidecar 当前为 392,192 个附加参数，不计入 51,463,797 个 LM 参数。它只应在 Agent 得到终态可信结果后运行一次；中间工具结果先回到主 Agent 循环，用于决定继续调用、拒绝或结束，不能每一步都生成面向用户的解释。

确定性门序必须先于 learned sidecar：

`retrieval → byte grammar → JSON Schema → provenance/permission/state → MW disposition → confidence → call/refuse/error`

任何 learned head 都不能覆盖 schema、provenance、permission、state 或其他 fail-closed 拒绝。

## 5. 标准 CPT 与产品化工作图

```text
不可变 Base
  ├─ Float Base 语言能力评测
  ├─ Float Task Control（独立对照分支；不是 QAT 父版）
  ├─ Q4 诊断（独立诊断分支）
  └─ CQ2 QAT → Retrieval R0
          ↓
oracle top-5 quant-aware full-call SFT
          ↓
Retrieval R1 → 重建 tool index → learned top-5 端到端评测
          ↓
冻结最终 LM → MW disposition → confidence → narration
          ↓
统一 package v2 → 重建 index → Python/Browser gates
          ↓
MTP 消融 → longitudinal eval → 资源/性能 → 对齐与冻结建议
```

顺序不能随意交换：

1. 先冻结 serializer、grammar、schema subset、scorer、数据 split 与 eval hash。
2. Float Base Anchor 与同数据 Float Task Control 必须都从同一冻结 Base 起步。
3. Q4 只用于诊断；主线是 CQ2 QAT。
4. R0 在 QAT 后主干上建立初始检索能力。
5. full-call SFT 使用 oracle top-5，避免早期 retrieval 误差污染 LM 学习。
6. LM 变化后必须重新训练 R1，并重建最终 tool index。
7. learned top-5 E2E 同时衡量 retrieval 与 full-call 的组合效果。
8. 冻结最终 LM 后再分别训练 MW disposition、confidence、narration，避免 sidecar 相互拉扯主干。
9. 任何最终 LM 或 retrieval head 变化后必须重建 index；任何 tensor 变化后必须重打包和重跑 gates。
10. narration 必须在最终 CQ2 包上做完整冻结生成评测；模板 fallback 通过不等于 learned narration 通过。

## 6. 当前状态基线

### 6.1 Exposure 父链

| 累计 exposure | 当前状态 | 结论 |
|---|---|---|
| 300M | Base 已冻结；完整产品化机制基线已保存 | 可作为后续比较起点，但产品候选不具备发布资格 |
| 600M | `训练中` | 存在 live continuation；动态进度只由 PID、held lock、heartbeat 与 checkpoint 联合判断。它不是冻结 artifact，也不属于本轮 300M 产品化 lineage |
| 900M | 计划能力已支持，尚无真实 artifact | 必须从正式冻结 600M 父版只增加约 300M exposure |
| 1.2B / 1.5B / 2.1B | 计划能力已支持，尚无真实 artifact | 使用任意正整数 `target_exposure_tokens`，不得建立空占位 release |

旧的 58.5M 参数 legacy run 不属于当前 51,463,797 参数产品父链，不能作为 600M/1B 证据。

生命周期控制面已经从固定 rung allowlist 改为累计 exposure 合同；不过当前 arbitrary-entry dry-run fixture 主要覆盖 600M/900M，仍需增加 1.2B、1.5B、2.1B 的回归用例。

### 6.2 300M 不可变证据

| 项目 | 值 |
|---|---|
| Base | `mei-1.0-51m-base-scratch300m-v1` |
| 实际 exposure | 300,000,485 tokens |
| Base weights SHA-256 | `97f4b537f2869ab315245fd8769eff8a61117b112eb7dce40ebfbd39e2afbd48` |
| 产品化 run fingerprint | `ff204182428e291f673cf7297efeb8d5e62069a7350040a98252f04dbfb791f1` |
| 最终 package | `mei-1.0-51m-scratch300m-agent-tool-sft-cq2-v2` |
| package manifest SHA-256 | `977d11ac2fd6b6aecfc37d77bfe362a4fa16421abf96e13bef38e07bc58646d1` |
| tensor container SHA-256 | `9553f52b2280d564a0e9f84cc47218d9ea611ff181dbc3c838aad796b4fb900b` |
| `CURRENT.json` 前后 SHA-256 | `5b0b68eeb8322bb9cdbef112777b1b346b69a91f3bce7234b1c6370389a42607`，未修改 |

历史审计记录为 `process_complete=true / release_eligible=false`。这应理解为“当时定义的机制流程已经闭环”，不能解释为质量可用。新的主曲线已经冻结为 `mei-51m-longitudinal-eval-v6`；旧候选保持原 receipt，不回写分数。300M 的新 SFT-v3 产品化将成为该冻结合同下的首条完整产品记录。

### 6.3 已实现与仍需优化

| 工作项 | 机制状态 | 当前产品判定 | 下一项工作 |
|---|---|---|---|
| 51M backbone、Engram、mHC、中文 tokenizer | 已验证（机制） | 300M Base 语言能力偏弱 | 用统一 Base scorecard 观察 exposure 曲线 |
| MTP t+2 旁路 | 已验证（机制） | 不进入部署包 | 保留同预算消融，只给未来 CPT recipe 提建议 |
| 架构三合同与历史兼容映射 | 已验证（机制） | 可复用 | 每个新 Base/package 强制校验 |
| 任意累计 exposure lifecycle | 已验证（机制） | 真实 600M 正在运行 | 补 1.2B/1.5B/2.1B fixtures 和 scale registry |
| CQ2 + QAT + activation/KV int8 | 已验证（机制） | QAT 税较小，量化不是当前首要质量瓶颈 | 保持数学/bytes golden，优化真实量化计算速度 |
| package v2 单容器与安全 loader | 已验证（机制） | 18 MiB 包门已通过 | 保持完整 heads，不靠删能力压体积 |
| runtime v2 schema/grammar/安全门 | 已验证（机制） | 确定性 fail-closed 路径成立 | 继续做真实 learned-flow 回归 |
| 多步 Agent 状态机 | 已验证（机制） | SFT-v3 已冻结 1,388 条轨迹、2–4 步与 147-tool 终态覆盖；最终 learned package 尚待重训 | 用冻结 multistep dev/test 报告 task success 与 call-id integrity |
| Retrieval R0/R1 | 已验证（机制） | SFT-v3 结构覆盖和自然补充已通过 preflight；新质量尚未测量 | 按固定 1,200-step R0/R1 重训并报告完整 held-out/cross-generator 指标 |
| quant-aware full-call SFT | 已验证（机制） | 147 工具 execute/refuse 平衡、自然补充和固定预算均已通过 preflight；新质量尚未测量 | 执行 300M quant-aware 重训并拆分调用名/参数/拒绝指标 |
| MW deviation gate | 已验证（机制） | 治理边界成立 | 继续作为 deterministic receipt，不训练成 head |
| MW disposition 20 类 head | 已验证（机制） | 13,387 train / 2,197 valid、20 类和 class-uniform sampler 已冻结；质量尚待重训 | 在冻结 v6 MW bank 上报告 macro-F1、逐类指标和 class-0 false-continue |
| Confidence head | 已验证（机制） | 3,528/1,176 个无合成标签候选已冻结；必须等待最终 runtime 采集真实 outcome | 获得正负双类后再训练并报告 AUROC/AUPRC/ECE/Brier |
| Narration rank-16 sidecar | 已验证（机制） | `待补证`；有 train/valid loss，无最终包全量生成评测 | 补事实、数字、极性、幻觉与 fallback 评测；必要时再扩语料/预算 |
| Python/MLX runtime | 已验证（机制） | `待优化`；数值路径可跑，速度未过门，真实 turn 仍出现 error | 修复产品调用结果，再做可迁移 kernel/profile 优化 |
| Browser-WASM runtime | 已验证（机制） | `待优化`；CPU-WASM 只证明可执行，速度和 heap 均不合格 | WASM 控制面 + WebGPU/packed compute；CPU-WASM 保留 fallback |
| 独立 Rust/Node/C SDK | 暂缓 | 不属于当前交付 | Python/浏览器质量成立后再扩展并重新选 validation scope |
| Needle2 机制对齐 | 部分已验证 | 核心结构和流程已实现，关键 learned 质量/资源未对齐 | 使用本文 scorecard 报告，不再以“代码存在”代替“能力成立” |
| `.cact`/`libneedle`/Cactus ABI | 有意不兼容 | 非缺口 | 不开展 |

### 6.4 当前冻结的 SFT-v3 输入基线

300M 新产品化不再使用历史 v2 数据作为直接训练真源，而是组合一个完整结构骨架和一个保守准入的自然中文增广层。两层各自不可变、各自有 hash；自然层不能替代 147-tool 结构覆盖。

| 输入 | 冻结身份 | 规模与角色 |
|---|---|---|
| 结构 SFT | `mei-1.0-51m-tool-sft-v3-300m-v7`；fingerprint `bf65e272…d11a2` | retrieval 2,352；full-call 3,528；Agent 3,736 行/1,388 轨迹；MW 13,387；confidence 候选 3,528；覆盖 147 工具 |
| 自然中文增广 | `mei-1.0-51m-tool-sft-natural-aug300m-v1`；fingerprint `b12ff1a3…d23f52` | retrieval 297、22 工具；full-call 295、19 工具；三个教师只改写 query；另有隔离的 cross-generator dev/test |
| 纵向评测 | `mei-51m-longitudinal-eval-v6`；fingerprint `311c76c4…e9433` | retrieval/full-call/confidence 各 588 dev + 588 test；multistep 各 614；MW 各 1,000；narration 各 600 |
| CPU 全量预检 | `sft-v3-v7-natural-v1-preflight-bab9fe672915.json`；fingerprint `bab9fe67…a37b7` | run fingerprint `84483d0a…68378`；全部 hash、编码、token budget、隔离和固定预算采样暴露通过；未导入 MLX、未占用 Metal |

训练合并后的固定预算为：retrieval 2,649 行，R0/R1 各 1,200 步、batch 8，全部行至少暴露一次；full-call 3,823 行、4,000 步，全部行暴露；Agent 3,736 行、4,000 步，全部行暴露。retrieval sampler 使用每工具独立确定性轮转，断点恢复从 step 0 重放相同 schedule，避免自然层加入后静默漏样本。

数据生成与准入边界如下：

1. 147-tool 结构骨架由本地 schema program 产生；工具、参数、拒绝理由、top-5、hard negative 和目标 serializer 全部是确定性 gold。
2. `deepseek-v4-flash-0731`、`qwen-plus-2025-12-01`、`qwen3.7-plus` 只提供历史 query 改写；本轮不调用 provider。工具、参数和 schema 在本地重新编译，无法证明的数字、时间、参数补造、延迟执行和主语漂移 fail closed。
3. Agent 只使用非空、provenance verified 的 ToolResultV2；生成轨迹用于同工具终态覆盖，跨工具链只保留有审查来源的历史轨迹，不穷举虚假组合。
4. MW disposition 是独立 20 类 sidecar 数据；MW deviation 仍只属于治理 gate。confidence 文件只包含待实跑候选，禁止合成正确率标签。narration 只从 verified terminal result 生成受限中文目标。

未来增广使用单独 fingerprint 的 adaptive branch：先补自然 query 到全部 147 工具，再按适用的 `reason × tool-family × state` 单元补 MW、按真实兼容关系补多步轨迹、按最终 runtime 采集 confidence。主 exposure 曲线始终复用上述冻结数据、预算和 eval；否则无法把 Base exposure 收益与换数据的收益分开。

### 6.5 长期可比历史账本

本节是人可读的只追加比较索引；底层事实仍是 Base release、run plan、stage receipt 和 artifact hash。后续 Base、同一 Base 的改进版 SFT、重新量化、runtime 优化或新评测都追加新 record，不改写旧 record。

可比等级统一为：

| 等级 | 含义 |
|---|---|
| `L` | 纵向可比：artifact 冻结，并使用同一 longitudinal eval、数据/预算和 runtime profile |
| `C` | 条件可比：只允许与使用相同旧 eval lock、设备或 backend revision 的记录比较 |
| `W` | 单次运行内可比：例如同一 Base 的 Float/QAT loss；不能直接跨不同合同使用 |
| `D` | 诊断数据：train loss、单类 calibration 或小样本 smoke；不得用于版本排名 |
| `P` | 尚未评测或证据不足 |

#### 历史索引

| Record ID | Base exposure | 产品化变体 | 评测合同 | 可比等级 | 终态 |
|---|---:|---|---|---|---|
| `300m-product-v2-ff204182428e-legacy` | 300,000,485 | CQ2 + R0/R1 + full-call + MW + confidence + narration | `sft-v2-eval-lock-v3-20class` | `C/W/D` 混合，见下表 | 历史 `process_complete=true`；`release_eligible=false` |

只有真实 artifact 或对既有 artifact 完成的新冻结评测才能新增一行。600M、900M、1.2B、1.5B、2.1B 尚未形成对应 artifact 时，不预填虚假成绩行。

#### Record：`300m-product-v2-ff204182428e-legacy`

冻结身份与比较条件：

| 字段 | 值 |
|---|---|
| Base / exposure | `mei-1.0-51m-base-scratch300m-v1` / 300,000,485 |
| 完整冻结计划 | [`plan.json`](../training/runs/mei-1.0-51m/productize-scratch300m-agent-cq2-v2-ff204182428e/plan.json) |
| Base release / weights SHA-256 | `6314732ee6eab747021ba6a20598cb6a431545dcd5247def03567c130daad1f4` / `97f4b537f2869ab315245fd8769eff8a61117b112eb7dce40ebfbd39e2afbd48` |
| weight / runtime / aux contract | `c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b` / `74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43` / `83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac` |
| productization run fingerprint | `ff204182428e291f673cf7297efeb8d5e62069a7350040a98252f04dbfb791f1` |
| SFT release / manifest SHA-256 | `mei-1.0-51m-tool-sft-v2-agent300m-v1` / `9dd4adf0de8c34ab9cfe01ea79fff6871a7d658ce8c7f7c153ac85a3bfa0277b` |
| narration release / manifest SHA-256 | `mei-1.0-51m-narration-sft-agent300m-v3` / `639fdd4e196c88172760ffd035a68351c5ce5a34d136dc6364452256e8cb460d` |
| tool catalog SHA-256 / count | `c427738ed0e2b4ae5ae1588b5fb00a91bfeea4ee34e1844b3130ed6ade716106` / 147 |
| historical eval lock / SHA-256 | `sft-v2-eval-lock-v3-20class` / `763d50b92e84837e97dec2e8540cb827bfe26fcf7fbfb88a14dc2620a254aa38` |
| preregistered thresholds SHA-256 | `3fe41e812061f7a3f79b293401d504f0d323934c888a9e59fc6645efb70308d4` |
| seed / validation scope | 51 / `python-browser-wasm` |
| package manifest / tensor SHA-256 | `977d11ac2fd6b6aecfc37d77bfe362a4fa16421abf96e13bef38e07bc58646d1` / `9553f52b2280d564a0e9f84cc47218d9ea611ff181dbc3c838aad796b4fb900b` |

Base 与量化 scorecard：

| 指标 | 结果 | 等级 | 解读 |
|---|---:|---|---|
| Base valid loss | 3.170981 | `C` | 以后须用同一 validation corpus/hash 重算才可横比 |
| HQ / colloquial / structure valid loss | 3.190563 / 2.612144 / 0.054195 | `C` | 保留分域结果，不能只看 global loss |
| Base probe mean NLL / exact | 4.306869 / 0 of 7 | `C` | 固定 probe bank 下的语言/结构基线 |
| probe NLL：UTF-8 / copy / number | 4.770769 / 2.401881 / 6.865310 | `C` | 数字是当前最弱 family |
| probe NLL：date / JSON / instruction | 3.948805 / 4.340512 / 3.480294 | `C` | 后续 Base 使用同 bank 对比 |
| Float Task Control | 200/12,090 rows；24-row call exact 0.25 | `D` | 覆盖和评测太小，不能当正式 float 对照分数 |
| Q4 diagnostic size | 26,819,741 bytes | `W` | 全 Q4 诊断包，不是 CQ2 产品候选 |
| CQ2 QAT | 5,001,216 tokens；2,442 steps | `W` | 同一 Base 内量化控制 |
| CQ2 QAT valid loss | 3.191214 | `W` | 对 Base 税约 +0.020233（+0.64%），未见明显量化破坏 |
| MTP t+1 / t+2 final loss | 6.926657 / 7.094016 | `D` | 每支仅 8,192 tokens；只作未来 recipe 线索，MTP 未部署 |

任务训练覆盖：

| 组件 | 冻结数据规模 | 本轮实际训练 | 等级 | 缺口 |
|---|---:|---:|---|---|
| Retrieval R0 | 8,643 pairs；25 个 gold tools；33 个 cached schemas | 400 steps；last loss 0.176693 | `D` | 最终 index 有 147 tools，训练覆盖不匹配；train loss 不是 retrieval 质量 |
| Full-call LM | 12,090 train pairs；31 个 gold tools | 4,000 steps；4,000 unique rows | `D` | 未覆盖完整训练集；execute/refuse 拉扯未单独计分 |
| Retrieval R1 | 8,643 pairs；25 个 gold tools；33 个 cached schemas | 400 steps；last loss 0.169686 | `D` | R1 已在最终 LM 上重训，但 final 147-tool held-out 效果差 |
| MW disposition | 9,859 train / 1,021 valid | 400 单样本 steps；last loss 5.271305 | `D/P` | valid 未形成 macro-F1、逐类和 class-0 false-continue receipt |
| Confidence | release 9,530 train / 470 valid | 实际 runtime outcomes 128；102 train、26 calibration | `D` | 0 positive / 128 negative，校准无效 |
| Narration | 4,800 train / 600 valid / 600 eval | 1,200 steps；valid loss 0.737158 | `D/P` | 缺最终 CQ2 package 的完整 600-row generation eval |
| Agent continuation | 2,560 train / 320 valid / 320 eval | 无独立训练/冻结 task-success receipt | `P` | 数据已冻结，但不能据此宣称 learned 多步能力 |

最终产品 scorecard：

| 指标 | 结果 | 等级 | 解读 |
|---|---:|---|---|
| Retrieval learned recall@5 | 0.090 | `C` | 低于 lexical 0.591，检索能力未成立 |
| Oracle top-5 full-call exact | 0.3033 | `C` | 等于 364/1,200 的全拒绝基线 |
| Learned top-5 E2E exact | 0.3033 | `C` | 同样等于全拒绝基线，不是组合能力成功 |
| unsupported / unprovenanced accepted | 0 / 0 | `C` | 冻结安全门通过 |
| MW held-out macro-F1 / class-0 false-continue | 未评测 | `P` | 不能给 MW head 做版本排名 |
| Confidence AUROC/AUPRC/ECE/Brier | 双类无效 | `P` | 单类 outcome 的低 ECE/Brier 不记录为能力收益 |
| Narration full generation | 未评测 | `P` | 模板 fallback 测试不等于 learned generation 通过 |
| Learned multi-step task success | 未评测 | `P` | 状态机测试通过，但最终 package 无真实任务成功率 |
| Python unit/golden | 84 passed | `C` | 证明 API、schema、CQ2 和状态机机制 |
| Browser wrapper / Rust core dependency | 27 passed / core tests passed | `C` | 证明 portable 机制，不证明端侧性能 |
| Package size | 18,857,111 bytes | `C` | 比 18 MiB 上限少 17,257 bytes，刚好通过 |
| Python/MLX raw-128 decode p50 | 265.79 tok/s | `C` | MLX 0.32.1、Apple GPU、backend revision `da3c60cd644526b4d99f88802b5d9ee164f115b8643de350ace4c888965e0da4`；低于 280/300 门 |
| Python RSS / Metal peak | 441.9 / 230.3 MiB | `C` | 仅与同 host profile 比较 |
| CPU-WASM decode / heap | 0.0768 tok/s / 419,889,152 bytes | `C` | 真实 numeric forward；速度和 96 MiB heap 门均失败 |

本 record 的结论是：CQ2 量化损伤较小，而 retrieval、full-call、MW、confidence、narration 的训练覆盖和评测完整性不足。300M Base 也偏弱，但在完成统一 exposure 曲线前，不能把全部失败归因于 Base；更没有证据直接归因于 Engram、mHC 或 MLP 等结构错误。

#### 后续追加规则

- 同一 Base 做新的 SFT、head、量化或 runtime 优化时，新增 record，保留相同 `base_id/weights_sha256`，并使用新的 run/package/backend hash；这用于分离“Base 变化”和“下游流程变化”。
- 同一 artifact 只换新评测合同重评时，新增 evaluation record，不能覆盖旧分数；记录 `re_evaluates_record_id`。
- 新 exposure 必须先增加 Base-only record，再增加该 Base 的产品化 record；这样可分别比较 CPT 收益与 SFT/runtime 收益。
- 只有相同 `mei-51m-longitudinal-eval-v6` hash、SFT release/augmentation hash、budget hash 和 runtime profile 的记录进入主 scale curve；其他结果保留在历史表，但标为 `C/W/D/P`。
- 每条新 record 必须给出所有指标；没有证据写 `未评测`，不得留空或沿用上一版本数值。

## 7. 下一阶段优先级

### P0：执行已冻结的 300M SFT-v3 产品化

- 数据合同、147-tool 结构覆盖、自然中文准入、cross-generator holdout、`mei-51m-longitudinal-eval-v6` 和 CPU 全量 preflight 已冻结。
- Metal 可用后按 run fingerprint `84483d0a…68378` 执行 Float Control、CQ2 import、R0、full-call、Agent、R1/index、全部 heads 与 package；不得改阈值或训练预算。
- Retrieval 报告 Recall@1/5、MRR/NDCG、lexical delta 和 cross-generator；full-call 分别报告工具名、参数、schema、拒绝和 oracle→learned gap。
- Confidence 从最终 runtime 的真实正确/错误调用构建正负双类；Narration 在最终 CQ2 package 上跑完整 600-row generation eval。
- 每产生一个真实冻结 artifact 或重评结果，就向第 6.5 节追加 hash-bound record；禁止通过空目录或手填状态声称未来模型存在。

### P1：根据冻结分数做有限诊断

- 分数不足仍完成全部独立阶段并标记 `release_ineligible`，不无限调参。
- 先用 Base、Float Control、oracle/learned gap、cross-generator 和各 sidecar held-out 判断瓶颈属于 Base、数据、retrieval、full-call 还是 runtime。
- 需要新增语料或预算时使用单独 adaptive release/fingerprint，不覆盖主 scale curve。

### P2：完成两个真实产品 runtime

- Python/MLX：先保证最终 learned package 能稳定得到 `call/refuse/error` 的预期终态，再达到至少 280 tok/s、目标 300 tok/s 的冻结 profile。
- Browser：保留 WASM 语义/状态机，改用 WebGPU 或等价 packed GPU kernel 承担 51M 主干计算；CPU-WASM 不再作为速度资格基线。
- 所有性能优化必须进入 runtime profile、backend revision 和 benchmark receipt，不能只依赖本机未记录的环境变量或临时缓存。
- 不同时扩展独立 Rust/Node/C 产品面，直到 Python/浏览器质量和性能成立。

### P3：逐 exposure 产品化并做因果比较

- 600M CPT 完成且冻结后，使用同一 productizer、SFT 数据/自然增广/预算和 `mei-51m-longitudinal-eval-v6` 运行，不改门槛。
- 900M 必须从正式 600M 父版继续；1.2B、1.5B、2.1B 同理使用最近可信父版。
- 只有当更大 Base 与充分 SFT 都出现平台期，才安排同语料、同 exposure、同训练预算的结构消融。

## 8. `mei-51m-longitudinal-eval-v6` 纵向比较合同

当前状态为 `frozen`。evaluation fingerprint 为 `311c76c47e7b53fb72c1357cb3ea8dc55480ddf407f4214bd12d5c37a7be9433`，lock SHA-256 为 `71fe42297d0df0d69b24fb7cfd7b9cfc0ca6e518fd47af5fd9d774178b467ec8`。v1–v5 保持为不可变历史修订，不再作为新主曲线入口。当前尚无在 v6 下完成全部产品 stage 的 `L` 级记录；冻结合同存在不等于模型已经评测完成。

每个 exposure 必须保存两组 scorecard：冻结 Base 和最终产品包。至少包含以下指标。

| 层次 | 必须冻结的指标 |
|---|---|
| Base | global/family valid NLL、固定中文/UTF-8/复制/数字/日期/JSON/instruction probes、NaN/Inf、参数与 tokenizer identity |
| QAT | Float Base、Float Task Control、CQ2 QAT、post-SFT LM retention；分别报告 QAT tax 与 SFT gain |
| Retrieval | Recall@1/5、MRR、NDCG、lexical baseline、learned-minus-lexical、稳定 tool-ID tie break |
| Full-call | execute rows 的 tool-name exact、arguments exact、schema valid；refuse accuracy、false-refuse、false-execute、balanced accuracy |
| 组合效果 | oracle top-5、learned top-5 和 no-retrieval 三组；报告 oracle→learned gap |
| 安全 | unsupported accepted、unprovenanced accepted、permission/state violation、schema escape、MW class-0 false-continue |
| MW disposition | macro-F1、balanced accuracy、逐类 precision/recall、混淆矩阵、class-0 false-continue |
| Confidence | 正负样本数、AUROC、AUPRC、ECE、Brier、risk-coverage；单类直接标记 invalid |
| Narration | exact/fact coverage、数值与单位、动作极性、失败语义、幻觉率、fallback 率、delivered correctness |
| Agent | 1/2/3/4-step task success、stale call、executor error、max-step、cancel、结果过大和 provenance continuation |
| 性能资源 | package bytes、load/prefill/decode、RSS/Metal、WASM heap、GPU buffer、冷/热启动、设备与 backend revision |

跨 exposure 比较时必须保持不变：

- tokenizer、weight contract、runtime profile 和 productization stage graph；确需变化时另开比较分支，不能混入同一曲线。
- CPT 之外的 SFT 数据 manifest、训练 token/step 预算、batch、seed、eval rows 与门槛。
- tool catalog、schema、serializer、codebook、CQ2 math ID、tool-index 构建规则。
- Python/Browser backend revision 与设备 profile；不同设备只能作单独 profile，不能直接画在同一性能曲线上。

scale registry 的每一行至少记录：`exposure_tokens`、`base_id`、parent、Base weights hash、productization run fingerprint、package manifest hash、eval contract hash、SFT manifest/budget hash、runtime revision、各 scorecard 路径和资格结论。

### 如何判断问题来自哪里

| 对比信号 | 优先结论 | 下一步 |
|---|---|---|
| exposure 增加后 Base NLL 与任务指标同步改善 | Base/CPT exposure 是主要限制 | 保持 SFT 不变继续画 scale curve |
| Base NLL 明显改善，但 oracle full-call 仍不改善 | SFT、serializer、decode 或标注是主要限制 | 在同一 Base 上做 SFT/数据消融 |
| 同一 Base 上充分 SFT 显著优于当前 SFT | 当前训练覆盖/预算不足 | 固定新预算供所有 exposure 使用 |
| 同 exposure、不同 corpus mix 出现稳定差异 | 语料组成问题 | 固定总 token 做 mix ablation |
| oracle 好、learned 差 | retrieval/index 问题 | 只优化 retrieval，不改 LM |
| oracle 与 learned 都差 | full-call LM 或 Base 问题 | 对比 Float Control、CQ2 SFT 与更大 Base |
| Base、充分 SFT、量化控制都改善后仍平台 | 才可能是容量/结构限制 | 做同 corpus/exposure 的结构 ablation |

仅凭速度低、训练 loss 或一次 300M 分数，不能断言 mHC、Engram、MLP/WHT 结构实现错误，也不能断言必须重训 Base。结构问题需要相同语料、exposure、预算和评测下的 matched ablation。

## 9. Definition of Done

### 单个 Base CPT 完成

- 累计 exposure、parent exposure、增量 exposure 和唯一 schedule 一致。
- PID/heartbeat/run lock 已终止或关闭，checkpoint、优化器 state、tensor contract 和 source manifest 完整。
- 无 NaN/Inf，LM validation 有冻结结果，权重与 receipt hash 可复核。
- 以新 Base ID 原子注册，不覆盖父版，不隐式修改 `CURRENT.json`。

### 单个 exposure 的产品化流程完成

- 所有标准 stage 有 terminal receipt；跳过项必须在 validation scope 中明确为 deferred。
- Float Base、Float Task Control、CQ2 QAT、R0、full-call、R1/index、E2E、cross-generator、MW、confidence、narration、package、runtime、MTP 和 `mei-51m-longitudinal-eval-v6` 均有证据。
- confidence 必须有正负双类，否则是 `待补证`，不是 calibration complete。
- narration 必须跑最终 CQ2 package 的完整冻结 generation eval，否则是 `待补证`。
- `CURRENT.json` 未变化；旧 artifact 和旧 receipt 未覆盖。

### 发布资格

- `process_complete` 只表示所选合同下流程完整；不等于发布资格。
- `release_eligible` 还要求质量、安全、hash/parity、Python/Browser 功能以及所选 runtime profile 的资源/性能门全部通过。
- 当前 300M 候选是机制基线和诊断资产，不是发布候选。

## 10. 每次工作的依据清单

### 开始任何工作前

- [ ] 读取本指南、[`AGENTS.md`](../AGENTS.md)、[`CURRENT.json`](../CURRENT.json) 和目标 Base 的 `RELEASE.json`。
- [ ] 检查所有 live run 的 PID、heartbeat、lock、checkpoint progress 和 artifact hash。
- [ ] 记录 Git 基线与脏工作树；不 reset、checkout、删除或覆盖用户修改。
- [ ] 冻结本轮 validation scope；默认只要求 Python/MLX 与 Browser-WASM。
- [ ] 冻结 source/data/config/eval fingerprint；确认没有 gold/eval 污染。

### 开始或恢复 CPT 前

- [ ] parent 是同一 weight contract 的真实冻结 Base，而不是目录名或过期 ledger。
- [ ] `parent_exposure + incremental_exposure = target_exposure_tokens`。
- [ ] schedule 唯一、corpus Merkle root 完整、tokenizer/source/environment 有 hash。
- [ ] resume state 同时通过 step、tokens、optimizer、tensor 和 fingerprint 校验。
- [ ] 新 run 使用新 ID；任何不可信 run 只隔离，不覆盖。

### 开始产品化前

- [ ] 明确 `--base-release`、`--base-weights`、`--package-id` 和独立 `--run-dir`。
- [ ] 冻结 SFT train/valid/eval manifest、family/group split、serializer、schema、scorer、seed 与预算。
- [ ] 先运行 Float Base Anchor 与同数据 Float Task Control，再开始 QAT。
- [ ] 按第 5 节固定顺序训练；R1 后和最终打包后都重建 tool index。
- [ ] MW deviation、MW disposition、retrieval、confidence、narration 分别记账，绝不合并语义。

### 评测与报告前

- [ ] 使用与 300M 相同的 `mei-51m-longitudinal-eval-v6` hash、结构 SFT/natural augmentation hash、budget hash 和 runtime/backend revision。
- [ ] execute/refuse 分开计分，并报告 trivial baseline。
- [ ] 检查 confidence 正负类、MW held-out、narration final-package generation 和多步 Agent success 是否齐全。
- [ ] 区分 kernel benchmark、完整 request latency、不同 head 的分类延迟和 Agent 端到端延迟。
- [ ] 分别报告 RSS、Metal/GPU、WASM heap 与 GPU buffer；不把不同口径混成一个“内存”。
- [ ] 结论按“机制、质量、资源、兼容”四层书写，不用“implemented”代替“validated quality”。

### 冻结或发布前

- [ ] 所有输入/输出 hash、stage receipt、package tensor directory 和安全检查可复核。
- [ ] package 内没有 MTP tensor，LM 仍是 51,463,797 参数，sidecar 参数单独列出。
- [ ] Python/Browser 实际加载最终 package 并执行数值 forward、head 和 Agent loop。
- [ ] 所选 runtime profile 的质量、性能和资源门全部通过；deferred binding 不冒充通过。
- [ ] 只生成 freeze proposal；没有单独授权不得 `finalize_current`、commit、push 或公开发布。

## 11. 当前证据入口

- 300M Base：[`base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json`](../base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json)
- SFT-v3 结构 release：[`mei-1.0-51m-tool-sft-v3-300m-v7/manifest.json`](../notebook/sft/mei-1.0-51m/releases/mei-1.0-51m-tool-sft-v3-300m-v7/manifest.json)
- 自然中文增广 release：[`mei-1.0-51m-tool-sft-natural-aug300m-v1/manifest.json`](../notebook/sft/mei-1.0-51m/releases/mei-1.0-51m-tool-sft-natural-aug300m-v1/manifest.json)；准入证据见同目录 `admission-receipt.json`
- 冻结纵向评测：[`mei-51m-longitudinal-eval-v6/lock.json`](../notebook/evaluation/banks/mei-51m-longitudinal-eval-v6/lock.json)
- 当前 CPU 全量 preflight：[`sft-v3-v7-natural-v1-preflight-bab9fe672915.json`](../notebook/evaluation/jobs/mei-1.0-51m/sft-v3-v7-natural-v1-preflight-bab9fe672915.json)
- 300M 产品化终局索引：[`post-run-audit/README.zh-CN.md`](../training/runs/mei-1.0-51m/productize-scratch300m-agent-cq2-v2-ff204182428e/post-run-audit/README.zh-CN.md)
- 300M corrected final audit：[`final-audit.erratum-v1.json`](../training/runs/mei-1.0-51m/productize-scratch300m-agent-cq2-v2-ff204182428e/post-run-audit/final-audit.erratum-v1.json)
- 300M Needle2 对齐矩阵：[`needle2-alignment.erratum-v1.zh-CN.md`](../training/runs/mei-1.0-51m/productize-scratch300m-agent-cq2-v2-ff204182428e/post-run-audit/needle2-alignment.erratum-v1.zh-CN.md)
- Python/MLX 性能：[`python-mlx-cq2-performance-v2.json`](../training/runs/mei-1.0-51m/productize-scratch300m-agent-cq2-v2-ff204182428e/post-run-audit/python-mlx-cq2-performance-v2.json)
- live CPT 的动态状态不在本文固化；每次操作前按 lifecycle skill 重新核验 PID、held lock、heartbeat 和 checkpoint

维护本文件时，只更新稳定目标、当前已核验证据与优先级。单次命令、PID、临时 ETA 和详细恢复步骤应留在 run receipt 或 lifecycle skill 中，避免再次形成多个互相漂移的事实源。
