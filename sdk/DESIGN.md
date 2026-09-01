# MEI Runtime SDK 设计

本文是 `mei-llm/sdk/` 的项目内设计真源。不复制 `docs/mei-llm/` SSOT 正文。长期治理笔记见 `docs/draft/mei-llm/`。

## 1. 命名与边界

产品是 **`mei-1.0-51m Runtime`** / **MEI Runtime**。

- `Needle 2` 只作机制参考：KV sink、grammar、retrieval-before-call 等。
- 不进入 SDK 产品名、crate 名、C 符号、npm 包名、错误码或兼容承诺。
- 不实现、不承诺 `.cact` / `libneedle` / 官方 Needle ABI。
- 历史实现路径 `runtime/mei-1.0-51m-needle2-v2/` 保留；**新增公共 API 不暴露 `needle2` 名称**。Python 参考后端通过显式 `mei_sdk.reference` 加载该目录，调用方仍只看见 MEI Runtime 类型。

## 2. 分层

```text
应用 (Python / Node / 其它)
        │
语言绑定（薄；禁止重写 retrieval / grammar / provenance 语义）
        │
稳定 C ABI  （opaque handle + int32 错误码 + UTF-8 JSON）
        │
Rust mei-sdk-core
        │
模型包（架构 + tokenizer + 权重/量化 + head 清单 + 哈希）
```

Rust 是可移植核心。C ABI 是跨语言稳定底座。Python/JS/WASM 不得各写一套协议语义。

Python+MLX 是当前 **golden oracle**，不是 ABI。

## 3. 版本

单一 SDK semver，另独立版本化：

| 轴 | 当前值 | 何时 bump |
|----|--------|-----------|
| `sdk_semver` | `0.1.0-experimental` | 绑定或工具链 |
| `wire_version` | `mei-runtime-wire-v1` | request/result JSON |
| `model_package_version` | `mei-model-package-v1` | 模型包清单 |
| `runtime_abi_version` | `mei-runtime-abi-1` | C 头文件与符号 |

四轴禁止耦死：换权重不必改 ABI；改 C 符号不必改 wire。

`CURRENT.runtime` 仍为 `null`、SFT/head/量化未过门期间，只发布 **experimental** SDK，不宣称产品 Runtime release。

## 4. 冻结 API

从参考实现 `RuntimeV2.complete` / `run` 提炼，去掉产品外名称：

- `load_model` → 校验模型包，报告能力（含缺失 head）
- `create_session`
- `complete` → `TurnResult`
- `run` → `LoopResult`
- `cancel`
- `close`

`TurnResult` 必须覆盖：selected tools、call/respond/refuse/error、confidence、provenance、stats。`respond` 仅能出现在 Session 已验收至少一个成功 ToolResult、且模型输出空 action 后；它不是 refusal 的别名。

`complete` 在选中 schema 上最多 5 个工具。`candidate_text` 仅用于协议金样，跳过引擎。

禁止字段（gold leak）：`<routes>`、`gold_route_id`、`gold_provenance`、`compiled_call_candidates`、`gold_entity_link`、`holdout_only_relation`、`scenario_id`。

## 5. 模型包

清单见 `spec/model-package.schema.json`。必须显式列出四个 head：

| Head | 职责 |
|------|------|
| `lm` | 词预测 / 工具 JSON 生成 |
| `contrastive` | 稠密检索 |
| `mw_disposition` | 闭集 MW reason_code |
| `confidence` | 调用置信 |

缺失或未训必须写 `status: missing|untrained`，加载结果原样报告。调用方不得把「清单没写」当成「已就绪」。

## 6. 移植顺序

1. tokenizer、prompt/schema、retrieval 合同、grammar 形状、KV 窗口声明、validator
2. 对照 Python golden vectors
3. Python+MLX 保留 `backend=mlx-reference` 数值 oracle，并提供可回退的 `backend=mlx-fused` Metal 热路；量化仍未开始。协议路径无引擎时 `engine_unavailable` 仍合法。

WASM 两级：tier-0 协议/校验；tier-1 完整 51M 推理须通过量化包、峰值内存、浏览器算子门。

## 6.1 Python+MLX reference / fused 后端性能（experimental）

两种 MLX 后端都通过 `Engine.load` → `session.complete/embed`；`mlx-reference` 是 oracle，`mlx-fused` 对固定 decode shape 使用 fused SDPA、ZCRMS、mHC、Engram 与任意训练后 `H` kernel，不支持的 shape 自动回退。评测不得绕过 SDK 直调 `RuntimeV2`。

- 隔离基准：`sdk/tools/bench_mlx_complete.py`，矩阵为 `decode_mode=raw|constrained × max_new=32|128`。
- traces 带 `sdk_backend_revision` 与 `performance_profile`；跨实现禁止 resume。
- 默认 `run_gates.py` 只跑协议/ABI/golden。性能门仅 `MEI_SDK_PERF_GATE=1`。
- `mlx-fused` 只有通过空闲 M4 Max 未量化 raw128 p50 ≥ 300 tok/s 且相对 reference ≥ 2× 后，才允许成为 scorecard 后端；未通过时继续保留 `mlx-reference` 评测入口。
- 本轮不做量化、不改训练权重、不改 `CURRENT.sft/runtime`。

## 7. 拆仓条件

见 `spec/split-criteria.md`。未达标禁止迁独立 `mei-sdk` Git 仓。拆仓时保持 wire/ABI 版本不变，`mei-llm` 以 revision/manifest 依赖，禁止复制源码双轨。
