# Runtime and release gates

Runtime/release 阶段复用已评测 run，只执行：

1. CQ2 包与 tensor/hash identity；
2. Python SDK 注册、预算、head 边界与 MLX 吞吐；
3. Browser-WASM 数值 forward、KV cache、堆内存与吞吐；
4. 包体积、资源报告和 final audit。

模型质量 receipt 是只读输入。任何 input、recipe、源码 closure 或 run
fingerprint 漂移都会拒绝继续。
