# MEI 嵌入式 SDK

本地嵌入式 **`mei-1.0-51m Runtime`**（亦称 **MEI Runtime**）的实验性 SDK。

应用在进程内加载 `mei-1.0-51m` 模型包并执行工具调用协议；不是远程网关，也不是 Needle / libneedle 兼容层。

## 状态

| 项 | 值 |
|----|----|
| SDK | `0.2.0-experimental` |
| `CURRENT.runtime` | `null`（禁止宣称产品 Runtime release） |
| 本轮产品路径 | Python/MLX 服务端 + Browser-WASM；均保持 experimental |
| 延后路径 | 独立 Rust SDK、Node SDK、C/FFI 的产品资格待更大语料 Base 验证后再扩展 |
| 数值 oracle | Python+MLX；语言无关语义位于仓内 `platform/_shared/runtime/` |

## 产品名

- 使用：`mei-1.0-51m Runtime`、`MEI Runtime`、`mei-sdk`
- 禁止进入公共 API / ABI / 兼容承诺：`Needle 2`、`needle2`、`.cact`、`libneedle`

## 目录

```text
platform/
  python-sdk/           Python API + MLX 服务端数值 runtime
  browser-sdk/          浏览器包装；Node 仅作 WASM 测试宿主
  _shared/spec/         语言无关合同
  _shared/rust/         Browser-WASM core 与延期的 CLI/FFI
  _shared/fixtures/     可验证的微型模型包
  _shared/tools/        跨运行时门禁与 benchmark
  _experimental/c/      延后资格的 C 头文件
```

## 最短路径

```bash
python3 platform/_shared/tools/run_gates.py
```

默认命令等价于 `--scope python-browser-wasm`。需要检查保留的非阻塞实现时可显式运行 `--scope extended`；这不会把 Rust/Node/C 自动升格成本轮产品交付。

Python 协议：

```python
from mei_sdk import Engine

engine = Engine.load("platform/_shared/fixtures/packages/tiny-protocol-v1")
session = engine.create_session()
result = session.complete({
    "query": "开灯",
    "oracle_tools": [{"name": "light.set", "parameters": {"type": "object", "properties": {}}}],
    "candidate_text": "[]",
})
assert result["refuse"] is True
```

## 拆仓

达到 `spec/split-criteria.md` 所列门之后，再评估独立 SDK 仓；当前不拆仓。
