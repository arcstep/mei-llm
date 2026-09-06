# Runtime and resource evaluation

这里测量模型包与当前 Platform 的数值一致性、内存和吞吐，不改变权重。

- `benchmark_mlx_runtime_51m.py`、`smoke_apple_q4_51m.py`：Python/MLX；
- `smoke_wasm_q4_51m.py`：Browser-WASM；
- `run_three_runtime_parity_51m.py`、`qat_q4_three_runtime_parity_51m.py`：跨实现 parity；
- `measure_resources_51m.py`、`resource_baseline_51m.py`：发布资源门禁；
- `eval_q4_parity_51m.py`：Q4 诊断。

硬件、浏览器、profile、模型包 SHA 与 warm/cold 定义缺一项时，吞吐数字不得用于跨 cycle
结论。
