# MEI Runtime 公共 API（冻结）

产品名：`mei-1.0-58m Runtime` / `MEI Runtime`。公共符号、头文件、包名不得包含 `needle2`。

语言无关入口（各绑定必须同语义）：

| 方法 | 作用 |
|------|------|
| `version()` | 返回 `sdk_semver`、`wire_version`、`model_package_version`、`runtime_abi_version` |
| `load_model(package_dir)` | 加载并校验模型包；返回能力清单（含缺失 head） |
| `create_session(engine, options?)` | 创建推理/协议会话 |
| `complete(session, request)` | 单轮：检索后的 top-5 schema → `TurnResult` |
| `run(session, request)` | 多轮循环直到 refuse/`[]` 或 `max_turns` → `LoopResult` |
| `cancel(session)` | 协作式取消进行中的 `complete`/`run` |
| `close_session` / `close_engine` | 显式释放 |

C ABI 只暴露 opaque handle、定长 `int32_t` 错误码、调用方提供或 `mei_sdk_string_free` 释放的 UTF-8 JSON。禁止泄漏 Rust 类型。

Python+MLX 参考实现位于 `runtime/mei-1.0-58m-needle2-v2/`，是 golden oracle，**不是**跨平台 ABI。
