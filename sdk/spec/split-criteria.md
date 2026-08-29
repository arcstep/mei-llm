# 拆成独立 `mei-sdk` 仓的条件

全部满足后才评估拆仓。本文件是清单，不是授权。

## 必须

1. Rust `mei-sdk-core`、C ABI、Python 绑定三语言稳定，且公共 API 不出现 `needle2`。
2. 模型包可重复加载：同一 `mei-model.json` + 文件哈希在两台机器上得到同一能力清单。
3. 同题 golden parity：Rust CLI、C ABI、Python 对 `spec/golden/` 向量逐字段一致。
4. ABI compatibility 门：`mei_sdk.h` 的 `runtime_abi_version` 与 `spec/versions.json` 一致；符号无 Rust 类型泄漏。
5. 取消 / 会话释放路径有测试；C ABI 字符串可 `mei_sdk_string_free`。
6. `CURRENT.runtime` 不再为 null，或明确维持 experimental 并拒绝产品 release 宣称——拆仓不等于产品发布。

## 明确未达标（本轮）

- 可移植 Rust 推理 kernel 与量化执行
- Node 本地 native 绑定（当前为协议层，待链同一 cdylib）
- WASM tier-1 完整 58M 推理
- SFT / contrastive / MW / confidence head 过门

## 拆仓时

- 保持 `wire_version` 与 `runtime_abi_version` 不变
- `mei-llm` 用固定 revision + manifest 依赖 `mei-sdk`
- 禁止把 `sdk/` 复制一份到新仓后再在 `mei-llm` 里继续改协议
