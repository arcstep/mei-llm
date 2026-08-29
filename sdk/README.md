# MEI 嵌入式 SDK

本地嵌入式 **`mei-1.0-58m Runtime`**（亦称 **MEI Runtime**）的实验性 SDK。

应用在进程内加载 `mei-1.0-58m` 模型包并执行工具调用协议；不是远程网关，也不是 Needle / libneedle 兼容层。

## 状态

| 项 | 值 |
|----|----|
| SDK | `0.1.0-experimental` |
| `CURRENT.runtime` | `null`（禁止宣称产品 Runtime release） |
| 可移植推理 | 未发布。Rust/C 目前提供协议、模型包校验与稳定 ABI 骨架 |
| Golden oracle | `runtime/mei-1.0-58m-needle2-v2/` 的 Python+MLX 实现（历史目录名，不是产品名） |

## 产品名

- 使用：`mei-1.0-58m Runtime`、`MEI Runtime`、`mei-sdk`
- 禁止进入公共 API / ABI / 兼容承诺：`Needle 2`、`needle2`、`.cact`、`libneedle`

## 目录

```text
sdk/
  spec/                 语言无关合同
  rust/mei-sdk-core/    Rust 类型与协议核心
  rust/mei-sdk-ffi/     C ABI (cdylib / staticlib)
  rust/mei-sdk-cli/     同语义命令行（parity 门）
  rust/mei-sdk-wasm/    WASM 分级：先协议
  c/mei_sdk.h           稳定 C 头文件
  python/               协议实现 + FFI 包装 + MLX 参考入口
  js/                   Node 协议 SDK；浏览器 API 单独声明
  fixtures/             可验证的微型模型包
  tools/run_gates.py    跨语言门
```

## 最短路径

```bash
cd sdk
python3 tools/run_gates.py
```

Python 协议：

```python
from mei_sdk import Engine

engine = Engine.load("fixtures/packages/tiny-protocol-v1")
session = engine.create_session()
result = session.complete({
    "query": "开灯",
    "oracle_tools": [{"name": "light.set", "parameters": {"type": "object", "properties": {}}}],
    "candidate_text": "[]",
})
assert result["refuse"] is True
```

## 拆仓

达到 `spec/split-criteria.md` 所列门之后，可将本目录原样迁为独立 `mei-sdk` 仓。现在不要拆。
