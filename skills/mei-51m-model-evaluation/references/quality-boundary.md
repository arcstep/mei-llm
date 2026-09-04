# Model quality boundary

模型评测回答“权重能力是否达到门槛”，覆盖 generation、retrieval、full-call、
Agent、MW、confidence 与 narration。

Python SDK API、Browser-WASM、包体积、内存和吞吐属于 runtime/release，
不能用于抬高模型质量分，也不能因 runtime 通过而覆盖 locked-test 退化。
