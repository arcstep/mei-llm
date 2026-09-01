# 拆成独立 `mei-sdk` 仓的条件

全部满足后才评估拆仓。本文件是清单，不是授权。

## 必须

1. Rust `mei-sdk-core`、C ABI、Python 绑定三语言稳定，且公共 API 不出现 `needle2`。
2. 模型包可重复加载：同一 `mei-model.json` + 文件哈希在两台机器上得到同一能力清单。
3. 同题 golden parity：Rust CLI、C ABI、Python 对 `spec/golden/` 向量逐字段一致。
4. ABI compatibility 门：`mei_sdk.h` 的 `runtime_abi_version` 与 `spec/versions.json` 一致；符号无 Rust 类型泄漏。
5. 取消 / 会话释放路径有测试；C ABI 字符串可 `mei_sdk_string_free`。
6. `CURRENT.runtime` 不再为 null，或明确维持 experimental 并拒绝产品 release 宣称——拆仓不等于产品发布。

## 仍需产品候选证明

- Node 已绑定同一 Rust/WASM 数值 core；仍需正式 300M 候选 receipt 证明全链
- 完整 51M CQ2 v2 单容器在 Rust/WASM 的最终数值与资源 receipt
- SFT / contrastive / MW / confidence 四类 tensor 与训练 receipt 全部入同一 v2 包
- 浏览器 ≤96 MiB heap、Rust session ≤64 MiB 的实测 receipt

## 拆仓时

- 保持 `wire_version` 与 `runtime_abi_version` 不变
- `mei-llm` 用固定 revision + manifest 依赖 `mei-sdk`
- 禁止把 `sdk/` 复制一份到新仓后再在 `mei-llm` 里继续改协议
