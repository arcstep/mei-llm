# MEI Runtime 公共 API（冻结）

产品名：`mei-1.0-51m Runtime` / `MEI Runtime`。公共符号、头文件、包名不得包含 `needle2`。

语言无关入口（各绑定必须同语义）：

| 方法 | 作用 |
|------|------|
| `version()` | 返回 `sdk_semver`、`wire_version`、`model_package_version`、`runtime_abi_version` |
| `load_model(package_dir)` | 加载并校验模型包；返回能力清单（含缺失 head） |
| `register_tools(engine, tools)` | 校验并注册完整工具 catalog；不支持的 schema 立即 fail closed |
| `create_session(engine, options?)` | 创建推理/协议会话 |
| `complete(session, request)` | 执行一个模型步；至多返回一个带 `call_id` 的 call，或 respond/refuse/error |
| `submit_tool_result(session, result)` | 校验 pending `call_id`，写入有 provenance 的工具结果并推进会话 |
| `run(session, request, executors)` | 宿主执行 call 并回填结果，直到 respond/refuse/error/step limit → `LoopResult`；只有 respond 代表成功闭环 |
| `narrate(session, options?)` | 仅在 respond 后消费该 Session 已验收的成功 ToolResult；adapter 未就绪时确定性降级 |
| `cancel(session)` | 协作式取消进行中的 `complete`/`run` |
| `close_session` / `close_engine` | 显式释放 |

每次 call 的执行门顺序固定为 `retrieval → grammar → JSON Schema → provenance → permission → state → MW disposition → confidence`。`MW` 正式值来自 `mw_disposition` head 或确定性策略，confidence 正式值来自 package 内 confidence head；请求中的 `mw` / `confidence` 只供锁定评测和 protocol fixture 注入，并须标记来源。v2 的 `mw` 只接受 `source=protocol-test`，`mw_disposition` 只接受 `source=mw-head|deterministic-policy`；缺省 MW 等价于 `continue/deterministic-policy`。请求 confidence 必须为带 `source=protocol-test` 的对象，禁止无来源裸数值；生产 head 输出不从 request 读取。任何 learned head 均不能恢复被前置确定性 gate 拒绝的调用。

参数 provenance 只接受 schema `const`、显式绑定且 verified 的 evidence，或**同一 Session 内已经通过 `submit_tool_result` 验收**的 ToolResultV2 payload 同名字段。请求自行携带的 `tool_results` 不构成信任；Session 必须保存验收结果的不可替换副本，并以 call ID 定位，防止复用 call ID 替换 payload。

工具 catalog 的确定性策略字段是 `required_permissions` 与 `required_state`；v2 迁移期只读接受 `x-mei-permissions` / `x-mei-state` 别名，生成的新 catalog 不再写别名。catalog fingerprint 必须覆盖这些字段，并按 UTF-8 tool ID 排序，不能受注册顺序影响。

`pattern` 使用三运行时安全子集：采用 Unicode 模式，只允许共同的字面量、字符类、普通捕获组、锚点、分支和量词；拒绝 `(?...)` 扩展组、反向引用、所有字母数字反斜线转义（含 `\\d` / `\\w` / `\\p` / `\\1`）、嵌套/交并字符类。无法在 Python、Rust regex、ECMAScript `/u` 下保持同一语义的 pattern 在注册时返回 `unsupported_schema`。

C ABI 2 只暴露 opaque handle、定长 `int32_t` 错误码、调用方提供或 `mei_sdk_string_free` 释放的 UTF-8 JSON。工具执行由宿主逐步调用 `complete` / `submit_tool_result`，ABI 不接收不稳定的语言回调。禁止泄漏 Rust 类型。

Python+MLX 是数值 golden oracle，**不是**跨平台 ABI。Rust core 是 package/状态机/C/WASM 的 portable 语义主实现。

`mei-runtime-wire-v1` 和 `mei-model-package-v1` 仅通过只读 degraded adapter 接受：可以检查、迁移和做旧协议验证，但缺少三契约哈希或完整 heads 时不得声明 v2 产品能力，也不得写回 v1 包。

`mei-model-package-v2` 的 `tensor_prefixes` 使用点分组件前缀：仅当 tensor 名称等于 prefix，或以 `prefix + "."` 开头时匹配；同时 tensor 的 `role` 必须等于对应 head。尾随点、跨 role 借用 tensor 和普通字符串前缀均非法。

v2 `files` 是完整 payload 清单：每项固定 `path/sha256/nbytes/role`，tokenizer、可选 vocab、CQ2 container、tool index 与任何辅助/receipt 文件都必须列入。`mei-model.json` 因无法自哈希是唯一例外；filesystem loader 拒绝未声明文件、缺失文件、symlink、哈希/字节数不一致，并要求 `resources.package_bytes` 等于 manifest 加全部 payload 的实测总字节数。

v2 的每个 `status=ready` head 都必须携带 `training_receipt_sha256`；该哈希必须同时出现在顶层 `training_receipts`，并匹配 `files` 中一个 `role=training_receipt` 的真实 payload。三个 portable 独立 head 的 ready tensor 契约固定如下，均为 `f16 / transform=none / codebook=none`，同 role 不允许夹带额外 tensor：

| head | canonical tensor（shape） |
|------|--------------------------|
| contrastive | `heads.contrastive.tok_probes` `[4,512]`；`lay_probes` `[4,512]`；`proj.weight` `[128,2048]` |
| mw_disposition | `heads.mw_disposition.proj.weight` `[20,512]`；`proj.bias` `[20]` |
| confidence | `heads.confidence.cell_probes` `[8,512]`；`proj.weight` `[1,4096]`；`proj.bias` `[1]` |

MW disposition 使用 20 类 reason-code head，聚合冻结为各层 last-token 表示的均值。label codebook 固定为 `mw-disposition-codebook-v1.json` 原始字节 SHA-256 `913d2c4bca8a796c9baddb9a79539cc703aa6420af4260be2c5ca0e0d5a68d40`；ready head 必须在 `label_codebook_sha256` 声明该值，并在 `files` 以 `role=head_codebook` 携带对应 payload。仅当 class 0 (`continue`) 的概率 `>=0.70` 且相对第二名 margin `>=0.15` 时可继续，否则确定性 `stop`；请求注入不能冒充该 head。portable loader 即使尚未执行 learned head，也必须先验证上述 tensor、codebook 和 receipt，验证失败时不得声明该能力 ready。

Schema 路由由 `schemas.json` 固定。无后缀的旧 schema 与现有 v1 golden 保留作兼容证据；所有新输出必须满足 `*-v2.schema.json`。
