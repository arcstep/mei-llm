# MEI Runtime SDK 当前设计

本文只描述仓内当前实现。历史 Runtime/SDK 设计从 Git 获取；某次训练实际使用的实现由
对应 cycle 的 source manifest 和 runtime profile hash 固定。

## 产品边界

- 产品名为 `mei-1.0-51m Runtime` / MEI Runtime。
- Needle2 只作机制参考，不承诺 `.cact`、`libneedle`、Cactus API 或 ABI 兼容。
- 当前产品验证范围是 Python/MLX 服务端与 Browser-WASM。
- Rust core 是 Browser-WASM 的内部实现；独立 Rust、Node、C SDK 产品资格暂缓。
- 当前公共请求边界是文本。麦克风、PCM、ASR、TTS 和音频播放由上层
  `mei-agent` / `mei-avatar` 通过 provider 组合，不属于 51M Runtime 或 package。
- ASR 转写置信度、retrieval relevance、MW disposition 与 execution confidence 是四条
  独立信号；上层不得把任一 learned score 当作绕过确定性校验的授权。

## 当前合同

| 轴 | 当前值 |
|---|---|
| SDK | `0.2.0-experimental` |
| Wire | `mei-runtime-wire-v2` |
| Package | `mei-model-package-v2` |
| ABI | `mei-runtime-abi-2` |
| Runtime context | 2048 token |
| 默认输出预留 | 128 token |
| 工具候选 | 排序后固定最多 5 个一批，可扫描后续批次 |

公共操作为 `version`、`load_model`、`register_tools`、`create_session`、`complete`、
`submit_tool_result`、`run`、`cancel`、`close_session` 和 `close_engine`。`TurnResultV2`
明确区分 `call | refuse | error`；工具结果必须绑定 session 的 `call_id`、状态、结构化
payload 与 provenance。

## 语义真源与执行顺序

语言无关语义位于 `platform/_shared/runtime/`，Python/MLX 是数值 oracle，Browser-WASM
复用 Rust core。每个 Agent 步骤固定执行：

```text
retrieval → 五工具分批/联合预算 → constrained generation
→ grammar/schema → provenance/permission/state → MW disposition
→ execution confidence → host tool → submit result → 下一步
```

MW disposition、retrieval relevance 和 execution confidence 是三个独立 head/合同。终态
narration sidecar 只读取已验证结果视图，不持有 executor，也不在中间工具步骤生成解说。

## 联合上下文预算

固定任务提示、当前批工具 projection、query/context/evidence/permissions/state、历史、工具
结果和输出预留共同受 2048 token 限制。Runtime 优先保留不可裁剪 schema 约束，再压缩
注释和旧输入；正常超预算不得抛异常。单工具最小结构仍不可表示时返回结构化
`context_unrepresentable`。

完整注册 schema 始终用于 constrained decoder、JSON Schema 和执行校验；模型只看到
预算化 projection。`standard` 工具稳定区软上限 1536，`compact` 为 1024，ordinary/KV
容量随实际稳定前缀动态变化。

## Package 与发布状态

CQ2 主格式为 `mei-cq-v2-g128-wht-codebook`，LM、retrieval、MW、confidence 和 rank-16
narration adapter 统一进入 portable tensor container。旧 v1 只读降级加载，不能冒充完整
v2。

当前 `CURRENT.runtime` 仍为 `null`，所以 SDK 保持 experimental。每个 cycle 的功能、
质量和资源结论见 `cycles/mei-1.0-51m/<cycle>/evaluation/SCORECARD.md`，不能用“代码已实现”
替代发布资格。
