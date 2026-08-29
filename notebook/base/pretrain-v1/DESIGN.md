# mei-1.0-51m tool core / scene design boundary

正式族名是 **`mei-1.0-51m`**。`needle-zh`、58M heads/runtime 与训练目录名均为历史或迁移参考，不代表现行产品身份。

## 状态

| 面 | Legacy/58M reference | 51M product target |
|----|-----------|------------|
| 协议 | Route-ID：`[]` / `{"route_id":N}` | 完整工具 JSON / `[]` |
| 工具选择 | Python 编译 `<routes>` | ContrastiveHead + top-5（可关） |
| 参数 | 预编译候选 | byte grammar 约束解码 |
| 来源安全 | RouteManifest provenance | 生成后 MEI validator |
| KV | 无界普通 cache | tool/system sinks + ordinary 256 |
| Confidence | execute/refuse probe | min(head, decode)；未校准 |
| 实现状态 | `legacy_frozen` / 58M smoke | 51M base 已晋级；tool runtime 未发布 |

机器规格索引：[`spec/README.md`](spec/README.md)。

## 当前可复现 legacy

v1 runner/pack/bank/promote 仍只消费：

- `spec/model.json`；
- `spec/route-protocol-v1.json`；
- `spec/schema-subset-v1.json`；
- `spec/source-grounding-v1.json`；
- `spec/runtime.md`；
- `spec/special-tokens.json`；
- `spec/gates.json`。

入口：`runtime/mei-1.0-58m-route-v1/` 的 `route_compiler.py`、`route_protocol.py`，`runtime/_shared/decode.py` 的 `greedy_route_id`、`scripts/promote_mei_58m_release.py`。禁止改 v1 protocol/hash/promote 语义。

## 58M v2 runtime（迁移参考，非 51M tool-call 发布）

```text
catalog → ContrastiveHead / index → top-5 or ≤5 全量
→ serializer v2 + sink/ordinary spans
→ byte grammar constrained decode
→ provenance / permission / state validator
→ confidence gate
→ complete() / run(max_steps)
```

代码：`architecture/mei-1.0-58m-arch-v1/` 的 heads；`runtime/mei-1.0-58m-needle2-v2/` 的 `prompt_v2.py`、`byte_grammar.py`、`kv_manager.py`、`runtime_v2.py`、`provenance_validator_v2.py`、`tool_index.py`。

分层测试：`scripts/test_mei_v2_runtime.py`、`eval_mei_retrieval_v2.py`、`run_eval_mei_toolcall_v2_oracle.py`。

现行 parent 是 immutable `mei-1.0-51m-base-scratch300m-v1`，与 58M checkpoint 不兼容。新 heads 移植到 51M 时必须显式列出随机初始化 tensors，并证明关闭时 logits/base parity。Route-ID SFT checkpoint 不是新路线 parent。

## 51M base 后的 tool/MW SFT 边界（目标，未实现）

后续先冻结纯工具 full-call SFT checkpoint，再从同一合格 base/tool parent 建立 `MWDispositionHead` 分支。MWHead 读取共享 hidden cells，输出版本化闭集 `reason_code` logits；runtime 确定性映射 `act` / `cell`，compiler/validator 可求的 `gaps` 不由模型重复预测。

- LM head 仍只生成 grammar 约束的完整 call / `[]`；
- 一次 `complete()` 可同时返回 call、MW 分类 sidecar 与 confidence，不要求调用方三选一；
- MW 分类码、gold `reason_code`、`act`、`cell`、`gaps` 不进入部署 prompt 或工具 token target；
- 先保留 pure-tool、MW-head-only、tool+MW joint 三组 checkpoint；joint 只有在 retrieval、full-call、refuse、grounding 与 calibration 均不退化时才可晋级；
- `stop_gradient` 只证明 hidden cells 可分类；若要帮助工具生成，MW loss 只能以小权重进入上层主干或工具 adapter，并持续 replay 工具样本；
- 核心工具 SFT 不训练工具结果中文解说，也不生成 MW 文本解释；用户可读说明由隔离、后端无关的 NarrationProvider 承担。

该段只固定工程边界，不表示 MWHead 已进入 `model-target-v2.json`、trainer 或 runtime。

## 已晋级 300M scratch

正式 51M 入口已从随机初始化消费 `corpus/lm-v1/schedule-scratch.json`：四角色 300M quota plan，课程为 150M@512 → 100M@1024 → 50M@2048。`base/mei-1.0-51m-base-scratch300m-v1` 已晋级；它不含已发布 retrieval/MW/confidence/tool SFT。旧 300M 后续 CPT 计划已归档，不得由正式 trainer 静默复用。

## 量化

产品目标是 CQ2-first mixed Q2/Q4 QAT。51M 诊断入口是 `training/mei-1.0-58m-train-v1/scan_ptq_51m.py`；旧的 `quantize_needle_zh.py --scan` 只服务 58M 考古，不得当作 51M 现行结果。没有 product fake-quant、STE、activation/KV 量化、product mixed-bit map 或真实 Q2/Q4 kernel。PTQ 只能产生 candidate map，不能决定最终位宽，更不是发布内核。QAT 不因 PTQ 好看而取消。`check_qat_pilot_readiness.py` 在 STE/kernel/阈值未就绪时必须 fail-closed。

## 51M base 后的冻结路线

```text
float base 晋级并冻结 immutable 对照
→ Q4/PTQ 分组件扫描
→ Q2/Q4 candidate map
→ fake-quant / STE / activation-KV / kernel reference
→ 短 QAT pilot
→ product mixed-bit map
→ LM QAT continuation
→ quant-aware RetrievalHead
→ oracle-top5 纯工具 full-call SFT
→ learned-top5 E2E
→ MWHead 独立分支 / 可选联合
→ 最终 confidence calibration
→ tool index rebuild
→ Q2/Q4 pack / kernel parity / 全量评测
→ NarrationProvider 同协议选型
→ call / execution / verified result / narration 分段场景验收
```

float reference 至少保留一个与 QAT 产品分支同数据 release、seed、serializer、grammar 和 scorer 的 task candidate。不要把 runtime smoke、300M base、QAT loss 可降或 MW 分类单项分数写成 tool-call 发布模型。

NarrationProvider 可使用确定性模板、51M 独立派生、Qwen 或其他后端。它只读取已验证、已裁剪的结果视图，不得修改 call 或授权执行。当前 51M 输入 embedding 与 LM projection 绑权，仓内没有正式 head-only/LoRA/adapter runtime；若训练 51M 解说模型，必须使用新的 model ID，禁止覆盖 base/tool checkpoint。

## 当前缺失任务

量化：

- [x] 51M 分组件 Q4/PTQ 扫描脚本与 jobs 报告入口（阈值仍 unregistered）；
- [x] Q2/Q4 candidate map 入口；短 QAT pilot **合同 + readiness**，未开训；
- [ ] 分组件质量-delta 阈值注册与短 QAT pilot 实跑；
- [ ] product fake-quant、STE、activation/KV reference；
- [ ] mixed-bit map 冻结、packed format、真实 kernel 与 parity harness。

训练数据：

- [ ] RetrievalHead 正式 train/valid release；当前仅有 6 条 retrieval smoke fixture，无 train pairs；
- [ ] v2 full-call 正式 train/valid release；当前仅有 2 条 isolation smoke；
- [ ] 旧 home 2k/10k 是 Route-ID v1、单 toolset 回归原料，须经 v2 compiler 重算 top-5 prompt、full-call target 与 provenance，禁止改名冒充；
- [ ] MWHead 正式 train release；现有 2k 包可作起始种子，但 `not_training_entry=true`，须冻结唯一 codebook、扩 schema/tool family 并保持 1k holdout 隔离；
- [ ] Confidence 数据须从最终 QAT 候选的实际生成成功/失败中采集，不能预先用 execute/refuse 静态标签代替。

实现与评测：

- [ ] MWDispositionHead、loss、checkpoint 与 runtime sidecar 接线；
- [ ] quant-aware retrieval/full-call/MW trainer 与 float 对照 runner；
- [ ] serializer/grammar/KV/validator 从 smoke 晋级到正式 correctness/parity 门；
- [ ] 最终量化模型重建 tool index，并重跑 retrieval/MW/full-call/provenance/confidence/性能。
- [ ] NarrationProvider result view、output/error envelope、模板地板、忠实度/隐私门与正式 provider。

## SSOT

本仓可执行 machine contract 以 `spec/README.md` 及其索引文件为准。产品方法、训练批准和评测治理在私有 monorepo 另行维护，不作为公开仓依赖。
