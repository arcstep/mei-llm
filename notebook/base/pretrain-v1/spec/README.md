# mei-1.0-58m machine specs（notebook 索引）

正式规格已升格。本目录只保留过程合同与指向正式树的 symlink。

| 正式位置 | 内容 |
|----------|------|
| `architecture/mei-1.0-58m-arch-v1/spec/` | `model.json`、`model-target-v2.json` |
| `runtime/mei-1.0-58m-route-v1/spec/` | Route-ID v1 frozen protocol / gates |
| `runtime/mei-1.0-58m-needle2-v2/spec/` | Needle2 v2 protocol / grammar / KV / gates |
| 本目录 `colloquial-synth-v1.json` | 口语合成合同（过程面） |
| 本目录 `schema-subset-v0.json` | 旧 schema 诊断，`invalid_for_publish` |

v1 promote 仍不得读取 v2 target specs。v2 代码已接线，产品阈值仍为 null，`not_a_toolcall_model_release`。
