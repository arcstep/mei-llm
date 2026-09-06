# Platform shared internals

这是 Python SDK 与 Browser SDK 共用的实现层，不是第三种用户入口。

- `runtime/`：检索、上下文预算、grammar/schema、KV、provenance 与 narration 语义。
- `spec/`：wire、package、错误码、能力与 golden 合同。
- `rust/`：Browser-WASM 使用的 portable core；CLI/FFI 代码仍保留但不属于当前产品门禁。
- `fixtures/`：跨运行时一致性测试包。
- `tools/`：统一门禁、golden 生成和性能基准。
- `Cargo.toml`：Rust/WASM workspace。

CQ2 package 的正式模型文件位于 `models/mei-1.2-51m/releases/<cycle>/package/`，不放在
这里。
