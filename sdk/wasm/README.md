# WASM SDK（分级）

产品：`mei-1.0-51m Runtime`。本目录不实现 Needle / libneedle。

| 级别 | 内容 | 本轮 |
|------|------|------|
| tier-0 | 协议解析、v2 状态机、package 校验、版本查询 | 已提供 `mei-sdk-wasm` cdylib |
| tier-1 | 浏览器内 51M CQ2/legacy-q4 诊断推理 | CQ2 包、frozen tool index、独立 heads、状态机与 bounded int8-KV 已实跑；当前 CPU-WASM 资源/速度不合格，仍 `release_eligible=false` |

WASM 与 C ABI 一样使用 `complete` → 宿主执行 → `submit_tool_result`，不把 JS 回调传进 Rust。`create_session` 返回独立 `session_id`，多会话状态存放在 Rust handle map 中；`cancel` / `close_session` 只作用于指定会话，重新 load/unload 才清空全部会话。浏览器包装层提供 `version/load_model/register_tools/create_session/complete/submit_tool_result/run/cancel/close_session/close_engine` 的同语义 façade。v1 q4 仅为 read-only/degraded 兼容，不能声明 v2 产品能力。

构建（不启动推理）：

```bash
cargo build -p mei-sdk-wasm --target wasm32-unknown-unknown
```

浏览器公共 API 见仓内 `../js/browser.mjs`。Node 在本轮只作为自动化 Browser-WASM 宿主，不作为独立产品 runtime 验收。
