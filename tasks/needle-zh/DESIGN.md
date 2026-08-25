# needle-zh / mei-1.0-58m design boundary

正式族名是 **`mei-1.0-58m`**；`needle-zh` 仅为历史目录名。

## 状态

| 面 | 当前 | 目标 |
|----|------|------|
| 协议 | Route-ID v1：`[]` / `{"route_id":N}` | Needle2-aligned 完整工具 JSON / `[]` |
| 工具选择 | Python 编译 `<routes>` | ContrastiveHead + top-5 |
| 参数 | Python 候选编译并确定性渲染 | 模型抽取 + byte grammar |
| 来源安全 | RouteManifest provenance | 生成后 MEI validator |
| KV | 无界普通 cache | tool sinks + bounded sliding |
| Confidence | 历史 execute/refuse probe | 完整调用 correctness + decode probability |
| 实现状态 | `implemented_in_code/legacy_frozen` | `not_implemented` |

机器规格索引：[`spec/README.md`](spec/README.md)。

## 当前可复现 legacy

当前模型/runner/pack/bank 使用：

- `spec/model.json`；
- `spec/route-protocol-v1.json`；
- `spec/schema-subset-v1.json`；
- `spec/source-grounding-v1.json`；
- `spec/runtime.md`；
- `spec/special-tokens.json`；
- `spec/gates.json`。

代码入口包括 `model/route_compiler.py`、`model/route_protocol.py`、`model/decode.py`、`scripts/mei_tool_grounded_lib.py` 和 `scripts/promote_mei_58m_release.py`。

Legacy 行为保持冻结，仅用于复现历史 checkpoint/runner/题库。禁止原地改变 protocol/hash/promote semantics。

## 经批准的目标合同

目标 v2：

```text
中文 query + 完整工具目录
→ query/tool ContrastiveHead embeddings
→ top-5 canonical schemas
→ canonical prompt + tool KV sinks
→ byte grammar constrained full-call generation
→ MEI provenance / permission / state validator
→ confidence gate
→ execute / escalate / stop
```

目标 specs：

- `spec/model-target-v2.json`；
- `spec/tool-call-protocol-v2.target.json`；
- `spec/schema-subset-v2.target.json`；
- `spec/source-grounding-v2.target.json`；
- `spec/special-tokens-target-v2.json`；
- `spec/runtime-target-v2.md`；
- `spec/gates-v2.target.json`。

这些文件是实现输入，不是 runner 配置。当前脚本不得自动消费它们。

## 保留主干资产

- `zh-24k-v1` tokenizer；
- 512×27、8Q/4KV GQA、RoPE；
- HadamardMLP、2/3-gram engram、4-lane mHC；
- tied embedding/LM head；
- `mei-1.0-58m-base-cpt300m-v1` warm start。

Route-ID SFT checkpoint 不作为 v2 parent。

## 实现前置

1. 用户确认缺槽、MTP、滑窗、量化/QAT、confidence 风险和首发平台。
2. 迁移 300M 主干并证明 logits/NLL parity。
3. 实现 grammar/KV/retrieval/confidence 与分层 eval。
4. 通过 query-blind、unseen/hard-negative 和 provenance hard gates。
5. 再由用户决定是否创建 CPT/SFT。

## SSOT

本仓可执行 machine contract 以 `spec/README.md` 及其索引文件为准。产品方法、训练批准和评测治理在私有 monorepo 另行维护，不作为公开仓依赖。

**本轮禁止**：实现 v2、改现有 runner/pack/bank、生成数据、启动训练。
