# needle-zh / mei-1.0-58m design boundary

正式族名是 **`mei-1.0-58m`**；`needle-zh` 仅为历史目录名。

## 状态

| 面 | Legacy v1 | v2 runtime |
|----|-----------|------------|
| 协议 | Route-ID：`[]` / `{"route_id":N}` | 完整工具 JSON / `[]` |
| 工具选择 | Python 编译 `<routes>` | ContrastiveHead + top-5（可关） |
| 参数 | 预编译候选 | byte grammar 约束解码 |
| 来源安全 | RouteManifest provenance | 生成后 MEI validator |
| KV | 无界普通 cache | tool/system sinks + ordinary 256 |
| Confidence | execute/refuse probe | min(head, decode)；未校准 |
| 实现状态 | `legacy_frozen` | `implemented_in_code`（非发布模型） |

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

## v2 runtime（已接线，非 tool-call 发布）

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

300M 主干 `mei-1.0-58m-base-cpt300m-v1` 可 strict 迁移；新 heads 默认可关，关闭时 logits 与 base parity。Route-ID SFT checkpoint 不是 v2 parent。

## 300M base 后的 MWHead SFT 边界（目标，未实现）

后续先冻结纯工具 full-call SFT checkpoint，再从同一合格 base/tool parent 建立 `MWDispositionHead` 分支。MWHead 读取共享 hidden cells，输出版本化闭集 `reason_code` logits；runtime 确定性映射 `act` / `cell`，compiler/validator 可求的 `gaps` 不由模型重复预测。

- LM head 仍只生成 grammar 约束的完整 call / `[]`；
- 一次 `complete()` 可同时返回 call、MW 分类 sidecar 与 confidence，不要求调用方三选一；
- MW 分类码、gold `reason_code`、`act`、`cell`、`gaps` 不进入部署 prompt 或工具 token target；
- 先保留 pure-tool、MW-head-only、tool+MW joint 三组 checkpoint；joint 只有在 retrieval、full-call、refuse、grounding 与 calibration 均不退化时才可晋级；
- `stop_gradient` 只证明 hidden cells 可分类；若要帮助工具生成，MW loss 只能以小权重进入上层主干或工具 adapter，并持续 replay 工具样本；
- 不训练工具结果中文播报，也不生成 MW 文本解释；用户可读说明由 runtime 模板或另线后端承担。

该段只固定工程边界，不表示 MWHead 已进入 `model-target-v2.json`、trainer 或 runtime。

## 当前 300M scratch

正式入口从随机初始化消费 `corpus/lm-v1/schedule-scratch.json`：四角色 300M quota plan，课程为 150M@512 → 100M@1024 → 50M@2048。默认 pretrain 只使用 `from_spec()` 主干，不挂 retrieval/MW/confidence heads。旧 300M 后续 CPT 计划已归档，不得由正式 trainer 静默复用。

## 量化

产品目标是 CQ2-first mixed Q2/Q4 QAT。当前只有 `quantize_needle_zh.py --scan` 的 fake 4-bit PTQ 敏感度诊断；没有 product fake-quant、STE、activation/KV 量化、mixed-bit map 或真实 Q2/Q4 kernel。PTQ 只能产生 candidate map，不能决定最终位宽，更不是发布内核。

## 300M final 后的冻结路线

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
```

float reference 至少保留一个与 QAT 产品分支同数据 release、seed、serializer、grammar 和 scorer 的 task candidate。不要把 runtime smoke、300M base、QAT loss 可降或 MW 分类单项分数写成 tool-call 发布模型。

## 当前缺失任务

量化：

- [ ] 分组件 Q4/PTQ 扫描阈值与报告；
- [ ] Q2/Q4 candidate map 与短 QAT pilot；
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

## SSOT

本仓可执行 machine contract 以 `spec/README.md` 及其索引文件为准。产品方法、训练批准和评测治理在私有 monorepo 另行维护，不作为公开仓依赖。
