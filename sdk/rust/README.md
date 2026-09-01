# Rust crates

| crate | 职责 |
|-------|------|
| `mei-sdk-core` | v2 协议、单容器 package/CQ2、调用—结果—继续状态机 |
| `mei-sdk-ffi` | `libmei_sdk` C ABI |
| `mei-sdk-cli` | `mei-sdk` 同语义 CLI（parity 门） |
| `mei-sdk-wasm` | WASM tier-1（复用 Rust core，CQ2 v2 + legacy q4 read-only） |

公共 API 不使用 `needle2`。版本为 `0.2.0-experimental` / wire-v2 / package-v2 / ABI-2。v1 只读加载时能力固定为 degraded，不能写回或声明完整产品能力。
