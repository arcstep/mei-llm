# mei-llm agent contract

本仓有且仅有一个现役模型产品：`mei-1.2-51m`（部署 LM 参数固定 51,463,797；
旧链归档名 `mei-1.1-51m`）。300M/600M/900M 等均为累计训练 token exposure，不是模型规模。

## 四分离导航（源码/语料/过程/成果）

- `src/`：源码。`architecture/mei-1.2-51m/` 现役架构；`model-factory/` 训练/评测/编排/发布；
  `corpus-factory/` 语料生产；`platform/` runtime 源码（python-sdk/browser-sdk/_shared/rust）；
  `mei_llm/` 是唯一控制平面入口（`python -m mei_llm`），已取消 skills 体系。
  唯一实现一律放 `src/`，禁止藏进 `.internal/`；历史实现从 Git 取，不复制 exposure 目录。
- `corpus/`：语料成果（pools/sft-suite/eval-lock/archive），不按产品版本划分，训练按需引用。
- `cycles/`：过程证据。`mei-1.2-51m/exp-XXXm` 现役序列；`mei-1.1-51m/` 旧链归档（只读）。
  每个 executed rung 绑定 corpus/model/evaluation/decision 证据。
- `models/`：成果（发布面）。`mei-1.2-51m/{tokenizer,runtime,exp-XXXm/{base,products}}`；
  `mei-1.1-51m/` 旧链成果。资产经 `mei-artifact://` URI 寻址。
- 旧路径只经 `.internal/registry/migrations/` 路由表解析（历史前缀：`.local/artifacts/`、
  `artifacts/…`、`notebook/`、`sdk/…`、`training/…`），永不因文件搬家而改写旧 receipt。
- `docs/` 是独立文档仓（自有 `.git`，私有 SSOT：`draft/` 过程稿、`archive/` 归档、
  `01..07` 主题目录）；产品仓不跟踪 `docs/`。

## 文档写作与提交

- 默认过程稿：`docs/draft/YYYY-MM-DD-短题.md`；改 SSOT / 归档须明示。
- 产品仓（根 `.git`）与文档仓（`docs/.git`）分别提交。
- 产品仓内不得引用 `../docs` sibling 路径；文档仓内链接用相对路径。
- 跨产品治理：`../mei-world/mei-world/docs/AGENTS.md`；
  工作区拓扑：`../mei-env/docs/03-workspace-topology.md`。

## Model-factory 源码治理

- 正式入口只从 `src/model-factory/contracts/PIPELINES.json` 选；文件名里的 v3/v4/v5 不代表最新。
- 每个正式 run 前冻结 Git revision、dirty-tree patch 或源码包、source manifest 与逐阶段闭包；
  只有哈希没有可恢复字节不等于可复现。
- 工具分类走 `src/model-factory/contracts/CODE_CATALOG.json`；诊断/兼容代码不得静默进入正式血缘。
- 历史重建的 lock 必须诚实声明缺失的源字节，不得用现源码顶替旧哈希。

## 动手之前

- 先看 `CURRENT.json`、所选 base `RELEASE.json` 与活体 run；账本过时不等于 run 已停
  （对账 PID/lock/heartbeat/checkpoint/哈希）。
- 保留既有 dirty worktree，不 reset/checkout/删除/覆盖无关改动。
- 用仓内 `.venv` 与离线模式；不下载模型/数据、不调 provider、不发布、不部署、不 push。

## 不可变边界

- `models/*/exp-XXXm/` 下一切 canonical 资产不可变、哈希锁定；权重是不可替代的产品资产，
  不是可重建缓存。
- 晋升 run 用独立拷贝 + 字节/哈希校验；原 run 证据保留；本机双份不等于仓外备份。
- 禁止原地训练或覆盖 checkpoint/package/run；一律新 ID。
- `CURRENT.json` 只读，只有显式 compare-and-swap 的 `finalize_current` 在单独授权后可改。
- 权重兼容以 canonical weight contract 为准，不用裸 model.json 字节哈希；
  runtime profile 与 training-aux 契约分开版本化。
- Needle 2 只是机制参考，不得宣称 `.cact`/libneedle/API/ABI 兼容。

## 产品化固定顺序

1. 先冻结 serializer/grammar/schema 子集/scorer/数据切分/eval。
2. 任务 SFT 之前先实现并验证 runtime 行为。
3. QAT 之前记录 Float Base-LM Anchor 与同数据 Float Task Control。
4. Q4 仅作诊断，主路径为 CQ2 QAT。
5. 按序训练 retrieval、oracle-top5 full-call、learned-top5 E2E、MW disposition、confidence。
6. 最终 LM 或检索头每次变更后重建 tool index。
7. 全部 heads 随最终 LM 重打包并重跑跨 runtime 与资源门；narration 只消费已验证结果，不能执行工具。
8. narration 是终端可选的 sidecar：冻结 LM、只训 rank-16 residual、在最终 CQ2 包上跑完整冻结
   generation eval split，grounding 失败时确定性回退。

## 语料工厂与 cycle 不变量

- 语料生产与架构工作同级；每个 cycle 绑定自己的 CPT delta、累计血缘、SFT suite、eval lock、
  工厂 revision、质量门与复用裁决。
- 不隐式继承语料切片：显式标 `reuse`/`replace`/`retire`；多样性退化默认下一轮 `retire`。
- CPT/SFT/Eval release 分开；retrieval/full-call/Agent/MW disposition/confidence/narration
  各自独立可审计的 SFT binding。
- 哈希/去重/泄漏通过只证明完整性与隔离，不证明自然度与语义多样性。

## 工具上下文与候选扫描不变量

- retrieval 按 rank 稳定批次处理，每批至多 5；不得因单个高分把首批砍到 1-2。
- `retrieval_discard_threshold` 淘汰不可用候选；`retrieval_expand_threshold >= discard`
  控制 rank>5 候选进入后续批次；锁定测试数据永不用于调这两个阈值。
- 只有空调用 + MW disposition 第 10 类 `capability_insufficient` 可扫描下一批。
- 候选扫描是内部推理 pass，不是外部工具步骤；收到已验证 ToolResult 后 retrieval 从首批重启。
- 固定 prompt、投影 schema、query/context/evidence、history/results、output reserve 共占
  2048-token 契约；compact/standard stable cap 为 1024/1536 软预算，普通 KV 区吃动态余量。
- 可见投影 schema 可缩短，但名字、参数结构、校验约束与完整执行 schema 永不削弱。
- MW disposition 训练只消费不可变的 per-batch visible view，优化器循环内不得做活体 retrieval。

产品化必须接受任何真实冻结 51M base（显式 release/weights 路径）；exposure 标签改变指纹但
绝不更换硬编码 stage graph；schema fixture 不是未来 600M/900M 产物已存在的证据。

## Runtime 验收档案

- 每个 run 冻结 `validation_scope` 于指纹与 receipt；续跑不得静默增删 runtime 目标。
- 模型质量未证前，机制验证 run 可用 `python-browser-wasm` profile（Python/MLX 与真实
  Browser-WASM 必须执行；Rust 核心可作为 WASM 实现依赖；独立 Rust SDK/Node SDK/C-FFI 保持
  显式 defer）。
- 要发 defer 绑定的 release 必须选扩档 profile，并把该绑定的 API/parity/集成/资源门设为必选；
  不得把 defer 绑定报成已验证。

## MW 术语边界

- **MW deviation**：评测/治理发现的证据边界漂移，由 gates/receipts 记录；不是模型头/张量/
  训练标签/包能力/任务评测替代品。
- **MW disposition**：产品流要求的独立闭集原因码 sidecar，可约束执行但不能替代 MW deviation 门。
- retrieval 与 confidence 是独立头（独立数据/checkpoint/校准/receipt/runtime 职责），
  不得归入 "MW" 名下。
- confidence 校准只在冻结结果集同时含正负类时验证；单类校准已实现但降级，仍是 release gap。

## 完成与失败语义

- 稳定完成边界见 `cycles/mei-1.1-51m/LIFECYCLE.md`（现役线对应
  `cycles/mei-1.2-51m/`）；子缺口加在受影响 cycle 能力上，不替换生命周期。
- `process_complete` = 每个计划 stage 都跑完且有终端 receipt；`release_eligible` 另需
  完整性/安全/质量/parity/资源硬门全过；拿到分数后不得调阈值。
- 质量不达标 = 不合格候选，不是无界优化许可；保留证据、继续独立诊断 stage。
- 哈希漂移/数据污染/checkpoint 损坏/CURRENT 变更一律 fail closed，不得用作 parent。
- package/Rust-session/WASM-heap 目标 18 MiB / 64 MiB / 96 MiB，除非后续契约显式取代。
- 包 receipt 完成后只改 runtime 门/报告/控制面代码时，用新指纹 packaged adoption，
  只重跑下游门/审计；单 token 功能探针不得把 `max_new=1` 漏进 128-token-reserve 联合预算门。

## 工作流入口

从 [cycles/mei-1.2-51m/exp-00300m/](cycles/mei-1.2-51m/exp-00300m/)（现役第 1 轮）或
[cycles/mei-1.1-51m/INDEX.md](cycles/mei-1.1-51m/INDEX.md)（旧链纵向历史）开始，
再读所选 cycle 的 corpus、scorecard、decision 页与该 cycle `pipeline/PIPELINE.md`。
每次不可变 Base 注册、产品化或冻结复评后，向纵向历史追加哈希绑定记录；不重写旧记录，
不把未测量值带入未来。

训练只经 `python -m mei_llm` 控制平面与 src/ 管线模块驱动，无 skills 层；不得从文件名
重建管线。新产品阶段需 `mei-51m-phase-binding-v1`；exposure 专用脚本默认值仅兼容用途，
不得为新 cycle 选择输入。
