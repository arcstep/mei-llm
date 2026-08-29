# Rust crates

| crate | 职责 |
|-------|------|
| `mei-sdk-core` | 协议、模型包、会话 API |
| `mei-sdk-ffi` | `libmei_sdk` C ABI |
| `mei-sdk-cli` | `mei-sdk` 同语义 CLI（parity 门） |
| `mei-sdk-wasm` | WASM tier-0 |

公共 API 不使用 `needle2`。推理 kernel 尚未移植；无 `candidate_text` 时返回 `engine_unavailable`。
