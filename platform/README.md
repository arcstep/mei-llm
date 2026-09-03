# Platform

这里的浅层只展示两种当前产品接入方式：

- [`python-sdk/`](python-sdk/)：服务器端 / Apple Silicon 的 Python + MLX SDK。
- [`browser-sdk/`](browser-sdk/)：浏览器端 JavaScript + WASM SDK。

共享协议、Runtime 语义、Rust/WASM 内核、schema、golden 和打包工具统一收在
[`_shared/`](_shared/)；延期的独立 C 接口收在 [`_experimental/`](_experimental/)。
使用者无需在 `runtime / sdk / packaging` 三套相互交叉的目录中寻找入口。

Platform 只保留当前实现。历史版本由 Git revision 和 cycle receipt 的 source manifest
定位，不复制 `300m/600m/v1-old` 分支。模型权重不在 Platform；它们是
[`../models/mei-1.0-51m/releases/`](../models/mei-1.0-51m/releases/) 下的一等资产。

当前 Platform 只接受文本请求并返回结构化工具结果或终态文字解说，不管理麦克风、PCM、
ASR、TTS 或音频播放。上层 `mei-agent` / `mei-avatar` 可通过 provider 组合语音链路；这些
provider 的权重和资源不进入 `mei-1.0-51m` package。
