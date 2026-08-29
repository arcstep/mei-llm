# Python SDK

`PYTHONPATH=python` 后：

```python
from mei_sdk import Engine
engine = Engine.load("../fixtures/packages/tiny-protocol-v1")
```

绑定优先级：

1. 协议实现（本包，始终可用）
2. `mei_sdk.ffi.NativeEngine` — 包装同一 `libmei_sdk`
3. `mei_sdk.reference` — 显式加载 Python+MLX golden oracle（历史目录 `runtime/mei-1.0-58m-needle2-v2/`，不进入公共类型名）

禁止在 Python 里重写 retrieval / grammar / provenance 语义；数值对照以 MLX 参考实现为准，协议对照以 `spec/golden/` 为准。
