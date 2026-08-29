# Performance / ABI 门（experimental）

本轮不宣称产品 Runtime 性能。可移植推理未发布，禁止用协议路径的 `wall_ms` 当 QPS。

| 门 | 本轮 |
|----|------|
| 同题 golden parity（Python / Rust CLI / Node） | `tools/run_gates.py` |
| C ABI ctypes 冒烟 | `tools/run_gates.py`（`libmei_sdk`） |
| WASM tier-0 编译 | `cargo build -p mei-sdk-wasm --target wasm32-unknown-unknown` |
| 内存泄漏 / 取消 | 会话 `cancel` + `mei_sdk_string_free`；无长期跑批泄漏门 |
| 推理性能矩阵 | **默认关闭**。`MEI_SDK_PERF_GATE=1` 时 `tools/run_gates.py` 跑 `tools/bench_mlx_complete.py --backend mlx-fused --gate`（只走 `Engine.load` → `session.complete/embed`）。工程门（非产品 SLA）：空闲 M4 Max、未量化权重、warm raw 128-token decode p50 ≥ 300 tok/s；若提供 `MEI_SDK_PERF_BASELINE`，还须相对隔离的 86.7 tok/s reference 基线 ≥ 2×。 |

`CURRENT.runtime=null` 期间结果只用于 SDK 骨架与参考后端工程验收，不用于产品 release 对比。
