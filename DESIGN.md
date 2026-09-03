# mei-llm 五域布局合同

## 1. 五个域

| 域 | 事实 | 历史 |
|---|---|---|
| `models/` | 模型身份、架构、tokenizer 和通用 recipe | Git |
| `corpus-factory/` | 数据生产、验证和 release 方法 | Git |
| `model-factory/` | 训练、评估、编排和发布方法 | Git + cycle pipeline lock/source capture |
| `cycles/` | 每轮实际使用的语料、模型、指标与决策 | 永久 cycle 记录 |
| `platform/` | 当前唯一 Runtime/SDK/Packaging 实现 | Git |

`models`、`corpus-factory` 与 `model-factory` 分别保存模型身份、数据生产能力和模型生产
能力；`cycles` 是三者每轮汇合的地方。源码目录不得保存权重，cycle 人类入口不得堆放
checkpoint，`platform` 不得复制 exposure 历史版本。

根目录的五个业务目录是唯一人类主导航。`src/` 只提供轻量 CLI/registry 门面；
`.internal/` 只保存派生 registry、迁移映射和非主线说明，不得隐藏唯一算法源码。本机
大文件、run 与恢复包统一进入 `.local/`；不得再新增与五域平级的工程辅助目录。

## 2. Cycle 合同

每个已执行 cycle 固定包含 `README.md`、`CYCLE.json`、`corpus/`、`pipeline/`、`model/`、
`evaluation/` 和 `decision/`。`CYCLE.json` 必须绑定 parent、目标/实际/增量 exposure、
工厂 revision、CPT delta、累计 corpus lineage、SFT suite、eval lock、Base/Product URI、
训练/评估 pipeline lock、质量指标和三个 eligibility 轴。

实际大文件进入 `.local/artifacts/mei-1.0-51m/<cycle>/`。人类结论只在 cycle 页面出现，机器通过
`mei-artifact://` URI 和 registry 定位。历史 receipt 保持原字节；旧路径由 migration map
解释。

## 3. Corpus 合同

- CPT delta 只表示本轮新增 consumption；累计 lineage 展开所有父轮来源。
- SFT suite 必须逐轮绑定 retrieval/no-match、fixed-five scan、full-call、Agent、MW、
  outcome/confidence 和 narration，不能写成一个模糊的 “SFT”。
- Eval release 独立冻结；locked test 不能参与阈值或语料生成。
- 每个旧 slice 进入下一轮前必须决定 `reuse`、`replace` 或 `retire`。
- 完整性、去重、污染和质量是不同门；合成语料必须额外检查多样性与语义一致性。

## 4. Platform 合同

Runtime、SDK、Packaging 各只有当前一份。每个 cycle 通过 Git revision、source manifest 和
runtime profile hash 记录其使用版本。要复现历史实现，应 checkout 相应 Git revision，
不得创建 `runtime/300m`、`runtime/600m` 或 `old-sdk`。

当前主要交付范围是 Python/MLX 与 Browser-WASM；Rust core 可作为 WASM 内部实现。独立
Rust、Node、C/FFI 在扩展 profile 明确启用前保持 deferred。

## 5. Model Factory 合同

`model-factory/` 只维护当前实现，并按 `contracts / training / evaluation / orchestration /
release / recipes / common / tests / diagnostics / compatibility` 分类。正式入口来自
`contracts/PIPELINES.json`，不能凭文件版本后缀猜测。

每个执行过的 cycle 必须冻结 `pipeline/PIPELINE.lock.json`，记录 Base 方法、产品化阶段、
实际 run 指纹、plan/source manifest hash 和源码可恢复性。未来正式 run 启动前必须冻结
完整 source bundle 与逐 stage 源码闭包；历史缺失字节应标记为 reconstructed，不得补写。

## 6. 语音集成边界

`mei-1.0-51m` 是文本到可信工具行为及终态文字解说的核心，不包含声学编码器、ASR head、
TTS head、麦克风或音频播放。`mei-agent` 可组合 ASR/TTS provider、Tool Host、会话确认与取消；
`mei-avatar` 可进一步管理实时打断、声音和视觉呈现。上游转写文本可以进入 51M，原始音频和
ASR/TTS 权重不进入 51M package 或 cycle。

“中文口语理解”只表示口语化文本能力。mei-llm 的语料工厂可以维护带来源的 ASR 噪声文本
slice，但声学训练属于独立模型/上层项目。未来原生音频模型必须使用新的 model ID、架构与
cycle，不得作为 51M 的普通 head。

## 7. 不变量

- `CURRENT.json` 只读，只有另行授权的 CAS `finalize_current` 可改变。
- Base、release 和历史 receipt 不可覆盖。
- `process_complete` 与 `release_eligible` 分开。
- 300M/600M 等是累计 exposure，51M 是部署 LM 参数身份。
- Needle2 仅是机制参考，不声明 `.cact`、`libneedle` 或 Cactus ABI 兼容。
- 未获授权时不下载、公开发布、部署、commit、push 或补写 distribution clearance。
