# QAT binding

正式 pipeline：`mei-51m-qat-cq2-v1`。

每次 cycle 使用新 binding。即使 Base ID 相同，只要 weights、corpus、recipe、
源码 revision、dirty patch 或训练参数改变，也必须创建新 binding ID 和 run dir。

绑定文件中的每个 input 都必须给出 `ref` 与 `sha256`。`ref` 可以是
`mei-artifact://` URI、仓内相对路径或明确的绝对路径；目录必须用 `target`
点名需要校验的 manifest/receipt，并设置 `pass_ref: true`，这样 hash 校验
manifest，但传给阶段入口的仍是目录。源码必须同时绑定 source manifest 与
`mei-51m-stage-source-closure-v1`；closure 中列出的每个源文件会在执行前复算
SHA。
