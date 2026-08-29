# WASM SDK（分级）

产品：`mei-1.0-58m Runtime`。本目录不实现 Needle / libneedle。

| 级别 | 内容 | 本轮 |
|------|------|------|
| tier-0 | 协议解析、`complete(candidate_text)` 校验、版本查询 | 已提供 `mei-sdk-wasm` cdylib |
| tier-1 | 浏览器内完整 58M 推理 | **未发布**。须通过量化包、峰值内存、浏览器算子门 |

构建（不启动推理）：

```bash
cargo build -p mei-sdk-wasm --target wasm32-unknown-unknown
```

浏览器公共 API 与 Node 分开声明，见 `../js/browser.mjs`。
