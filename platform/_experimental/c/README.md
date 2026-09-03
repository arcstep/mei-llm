# C ABI

头文件 [`mei_sdk.h`](mei_sdk.h) 冻结为 `runtime_abi_version = mei-runtime-abi-2`。

- opaque `MeiSdkEngine*` / `MeiSdkSession*`
- 返回值是 `int32_t` 错误码（`spec/errors.json`）
- JSON 为 UTF-8；`char**` 输出必须 `mei_sdk_string_free`
- 禁止跨边界泄漏 Rust 类型
- 工具执行采用 `session_complete` → 宿主执行 → `session_submit_tool_result` 的 stepwise ABI；不把语言回调穿过 C 边界
- 活跃 handle 的调用由实现内部串行化；宿主不得让 `close` 与同一 handle 上的其他调用并发
- 所有 `char **` / handle 输出在入口先清为 `NULL`，panic 被拦截为稳定错误码而不会跨越 C ABI
- ABI-1 只作为旧二进制的只读识别版本，不由本头文件继续导出

实现 crate：`../rust/mei-sdk-ffi`（`cdylib` 名 `mei_sdk`）。

编译表面门：`cc -std=c11 -Wall -Wextra -Werror -fsyntax-only abi_smoke.c`。
