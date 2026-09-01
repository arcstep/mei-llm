# Python / Browser-WASM 门（experimental）

300M 机制验证周期只把 Python/MLX 服务端路径与 Browser-WASM 路径列为交付门。Rust core 仍是 Browser-WASM 的实现依赖；独立 Rust SDK、Node SDK、C/FFI 的完整 parity、性能与发布资格延后至更大语料 Base 验证模型效果之后。现有代码不删除，也不能据此宣称这些延后路径已经产品化。

本轮不宣称产品 Runtime 性能。协议路径的 `wall_ms` 不能当 QPS；资源或速度不达标时仍可得到 `process_complete / release_ineligible`。

| 门 | 本轮 |
|----|------|
| 默认作用域 | `tools/run_gates.py --scope python-browser-wasm` |
| Python/MLX | v2 package、安全校验、状态机、真实 CQ2/activation Q/DQ/int8-KV 数值前向 |
| Browser-WASM | release WASM 构建、同一 v2 包、frozen tool index、独立 heads、状态机、真实有界 int8-KV 前向 |
| Rust core | 只作为 Browser-WASM 数学与内存语义依赖参与单元/契约测试，不计为独立 Rust SDK 交付 |
| 延后矩阵 | `tools/run_gates.py --scope extended` 可诊断 Rust CLI、C/FFI；结果不阻塞本轮 |
| CQ2 data/scales/bitmap parity | `golden/cq2_v2.json`；Python/Browser core 共享布局 |
| package 安全 | hash、safe path、重复 tensor、offset 越界/重叠、manifest/container directory 一致性 |
| learned-head 身份 | ready receipt payload + contrastive/MW/confidence canonical tensor 名称、shape、dtype；MW 20 类及 0.70/0.15 fail-closed 阈值 |
| 状态机 | pending call、stale call ID、verified provenance、result size、max_steps、cancel |
| 资源资格 | package ≤18 MiB、Browser-WASM heap ≤96 MiB；Rust session 数字保留为 core 诊断但不代表独立 SDK 交付。缺 receipt 或超限即不 eligible，不阻塞 process completion |
| 推理性能矩阵 | **默认关闭**。`MEI_SDK_PERF_GATE=1 MEI_51M_PACKAGE_DIR=...` 时运行 `mlx-cq2` 真实 native-v2 profile。工程门（非产品 SLA）：空闲 Apple Silicon、warm raw 128-token decode p50 ≥300 tok/s |

`CURRENT.runtime=null` 期间结果只用于 SDK/候选工程验收，不用于产品 release 对比。v1 包永远是 `read_only/degraded`。

`mei-model-package-v2` 还必须通过 tensor identity 门：LM role tensor 的 `n_params` 合计精确为 51,463,797，且 tensor 名称的任一点分组件不得等于 `mtp` 或以 `mtp_` 开头。否则包可作为 diagnostic 被检查，但 `tensor_identity_complete=false`，不得声明 inference、product ready 或 release eligible。
