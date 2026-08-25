# mei-1.0-58m machine specs

> **职责**：区分当前可复现 Route-ID v1 与 Needle2-aligned v2 目标合同。  
> **SSOT 边界**：本仓只承载可执行 machine contract；产品批准与训练治理在私有 monorepo 另行维护。  
> **硬规则**：`*.target.*` 均未实现，不得被 runner、promote 或训练脚本当作现行协议。

## 状态

| 文件 | 角色 | implementation_state |
|------|------|----------------------|
| `model.json` | 当前 58m 主干几何 | `implemented_in_code` |
| `route-protocol-v1.json` | Route-ID 内部输出 | `legacy_frozen` |
| `schema-subset-v1.json` | Route-ID schema 子集 | `legacy_frozen` |
| `source-grounding-v1.json` | 预编译 route provenance | `legacy_frozen` |
| `runtime.md` | Route-ID 运行时说明 | `legacy_frozen` |
| `special-tokens.json` | 当前已冻结 token/phase1 声明 | `legacy_frozen` |
| `gates.json` | 当前 Route-ID promote 门 | `legacy_frozen` |
| `execute-threshold.json` | 历史 confidence smoke 阈值 | `legacy_frozen` |
| `schema-subset-v0.json` | 旧 schema 诊断 | `invalid_for_publish` |
| `model-target-v2.json` | Needle2-aligned 模型目标 | `not_implemented` |
| `tool-call-protocol-v2.target.json` | 完整工具 JSON 协议 | `not_implemented` |
| `schema-subset-v2.target.json` | byte grammar schema 目标 | `not_implemented` |
| `source-grounding-v2.target.json` | 生成后 provenance validator | `not_implemented` |
| `runtime-target-v2.md` | retrieval/grammar/KV/loop 目标 | `not_implemented` |
| `special-tokens-target-v2.json` | 目标协议 token 使用 | `not_implemented` |
| `gates-v2.target.json` | 新发布门类型 | `not_implemented` / `blocking_for_publish=false` |

## 当前实现

代码仍硬编码：

- `model/route_protocol.py`；
- `model/route_compiler.py`；
- `model/schema_render.py`；
- `model/decode.py`；
- `scripts/mei_tool_grounded_lib.py`；
- `scripts/promote_mei_58m_release.py`。

这些入口使用 `mei-route-protocol-v1`、`<routes>` 和 Route-ID trie。保留 v1 文件以复现历史 pack/bank/checkpoint；不得原地改写其语义或 hash。

## 目标合同

```text
tool registry
→ ContrastiveHead / index
→ top-5 schemas
→ canonical prompt + tool sinks
→ byte grammar constrained full call / []
→ post-generation provenance validator
→ confidence execution gate
```

`*.target.*` 仅是实现输入。只有代码、测试、bank、runner 和发布门全部接线并经用户批准，才能将其升为 `implemented_in_code`。

## 禁止

- 用 target spec 启动当前 Route-ID trainer；
- 修改 `gates.json` 放宽历史 promote；
- 把 `kv_window` 配置字段写成已实现 sliding；
- 把 `<tools>` 出现在 prompt 开头写成已实现 KV sink；
- 用旧 Route-ID 1.0 证明 v2 工具能力；
- 未重新构建 tokenizer 就假定 target token 已冻结。
