# C ABI

头文件 [`mei_sdk.h`](mei_sdk.h) 冻结为 `runtime_abi_version = mei-runtime-abi-1`。

- opaque `MeiSdkEngine*` / `MeiSdkSession*`
- 返回值是 `int32_t` 错误码（`spec/errors.json`）
- JSON 为 UTF-8；`char**` 输出必须 `mei_sdk_string_free`
- 禁止跨边界泄漏 Rust 类型

实现 crate：`../rust/mei-sdk-ffi`（`cdylib` 名 `mei_sdk`）。
