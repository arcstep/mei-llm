# Mei 浏览器 CPU / GPU 实验

在浏览器实际 WebGPU 适配器上跑既有 v1.2 链 900M-SFT CQ2 权重，并与 CPU-WASM 做交替数值性能测量。900M 是训练曝光量，模型族为 mei-51m。该目录是实验入口，不是新的模型版本或正式发布 SDK。

从产品仓根目录运行：

```sh
.venv/bin/python src/platform/browser-sdk/build_speed.py --id my-numeric-bench --numeric-bench
MEI_BENCH_WASM=.local/cache/wasm-speed/my-numeric-bench/runtime.wasm \
MEI_BENCH_PACKAGE=.local/cache/data-check/data-check-d919985609f9 \
node src/demos/runtime-bench/server.mjs
```

打开 `http://127.0.0.1:8786/`，依次运行 GPU 检测、模型数值探针、CPU/GPU 交替实验。默认直接使用本机已冻结的 `mei-51m-wasm-cpu-gpu-bench-20260914-v3` 二进制。只从本机获取模型，不调用云模型；报告保存在 `.local/cache/runtime-bench/`。服务只监听 loopback，文件读取使用白名单；结果只回传本机。

## 测量范围

- GPU 检测不仅检查 `navigator.gpu`，还申请实际适配器、执行 shader、校验回读结果；数值后端拒绝已知的软件回退适配器。
- CPU 是真实 WASM 数值内核；GPU 是 JavaScript 调度 WebGPU/WGSL，GPU 计算并非 WASM 指令本身执行。两者统一在浏览器 Worker 中测量。
- 两端都执行 27 层 LM、mHC、Engram、激活 Q/DQ、因果注意力、KV 续接、绑定嵌入输出矩阵。
- 这是 raw 数值推理对照：两端都不执行检索/MW/confidence/narration sidecar、grammar、工具和完整 Agent 协议。GPU 后端尚未接入数据体检 Agent，不得将本报告当作完整 Agent 验收。
- CPU 保持 packed CQ2 + int8 KV，采用先前的近似速度内核。GPU 首版从相同 CQ2 权重在内存解码为 dense f32，KV 是 Q/DQ 后的 f32 存储；不宣称逐位数值一致，也不宣称满足原低内存目标。
- 默认输入长度 1/128/512，固定不同 token ID 的序列，输出 32 token。每种长度先预热，再按 CPU/GPU、GPU/CPU、CPU/GPU 顺序各跑三轮。不将合成序列当作自然语言能力测试。
- prefill 计时包括第一个 logits 回读与首 token 选择之前的工作；decode 直接累计后续 31 个前向步骤时间，不再用两个包含 prefill 的总耗时相减。输出 token ID、每步时间和总耗时均记录。
- 每轮清空逻辑 KV；模型权重和编译后的 shader 常驻。没有跨请求前缀/答案缓存。GPU 首次使用可能复用浏览器/驱动缓存，不宣称彻底冷编译；初始化时间单独记录。
- GPU 数字包含完整 24,000 维 logits 回读，CPU 也复制相同大小的 logits。GPU 记录显式分配 buffer 总量，CPU 记录 WASM linear heap；这两个值都不是浏览器进程完整 RSS。
- 单次 GPU prefill 最多 512 tokens，总上下文硬拒绝超过 2048；尚未验证完整 2048 资源与 KV 滚动淘汰契约。GPU reset 用于新请求，不能并发共享一个实例。

## 数值与工程验证

`gpu_oracle.rs` 通过独立 native Rust CQ2 数值路径生成参考。GPU 数值探针包含 `[2]`、增量 `[37]`、批量 `[2,37]`、128-token 不同输入及其下一步；另在插入其他请求并 reset 后重复128-token输入三次，检查 logits 稳定性。

`numeric-bench` 是默认关闭的编译特性，公开实验 raw-forward ABI 以直接计时；它强制启用实验标记，不能作为正式 Agent 执行入口。模型加载/卸载清空实验 KV 状态；检查空输入、批长、词表、总上下文、输出容量。

最终本机证据见 `validation/2026-09-14-browser-cpu-gpu.json`。运行时构建工具保存可恢复源码包、逐文件哈希、编译命令、日志和二进制哈希；浏览器实验检查加载的源码在测量期间未变化。

## 后续工作

GPU 的完整 Agent 接入、packed/f16 权重内核、真实 int8 KV、长上下文、自然语言/工具任务准确率和跨浏览器/低端设备测量仍需单独完成。保持完整 heads 的产品设计；本实验没有把这些能力降为可忽略项。

## CPU 多 Worker 与 GPU 并行粒度实验

```sh
.venv/bin/python src/platform/browser-sdk/build_speed.py --id my-parallel --parallel-kernels
PORT=8787 MEI_PARALLEL_WASM=.local/cache/wasm-speed/my-parallel/runtime.wasm \
node src/demos/runtime-bench/server.mjs
```

打开 `http://127.0.0.1:8787/parallel.html`。默认使用已冻结的 `mei-51m-wasm-parallel-20260914-v2`。先分别运行 CPU/GPU 配置扫描，再用两个“复核”按钮进行五轮 128/512-token 对测。复核 CPU 使用 prefill 8 helpers、decode 4 helpers，与 0/4 helpers 对照；复核 GPU 使用批量不超过 256 时 tile 8，否则 tile 16，与原 tile 16 对照。这个启发式来自本机探索，不能外推到所有设备。

`parallel-kernels` 默认关闭，包含 raw numeric ABI，不是 Agent API。CPU 将同一个矩阵的输出行切成互不重叠、按 4 行对齐的区间，helper 调用 `mei_sdk_wasm_parallel_rows`，协调器等待全部完成后继续。两端使用相同 CQ2/SIMD 运算次序；prefill 和增量 logits 对照必须通过。每个 helper 在初始化时验证无效批长、行范围、对齐和输出容量被拒绝。helper 出错/超时终止本轮，不把部分输出当作完成结果。

SharedArrayBuffer 要求跨源隔离，实验服务发送 COOP/COEP。协调器和 helper 都是 Dedicated Worker，页面主线程可响应和停止实验；这不是 Service Worker/浏览器后台常驻保障，也没有线程绑核。8 个 helper 常驻、未参与的休眠；线程数不等于性能核心数。当前 helper 各自加载私有模型和 heap，是优先验证并行收益的原型，**不是低内存共享权重的最终实现**。报告分别给出每个 heap 和 SAB，不把所有复制占用归到正式单实例。

GPU 实验子类 `webgpu-parallel.mjs` 只改变矩阵向量乘法的 workgroup 大小和批量矩阵乘法的 tile，注意力与其他 kernel 沿用 `webgpu-51m.mjs`。GPU 本来就并行执行；32/64/128/256 是 invocation/workgroup，不能等同物理 GPU 核数。工作组变化会改变浮点归约次序，报告记录数值差异；不以速度证明质量。

测量的复制/等待时间用于定位 CPU 协同成本。`wait_ms` 同时包含 helper 计算和调度等待，**不能解释为纯同步浪费**。所有时间包含原型插桩开销，不将估算的扣除值作为实测速率。最终结果存入 `validation/2026-09-14-browser-parallel.json`。

## 解码时间戳剖析与融合候选

```sh
PORT=8788 node src/demos/runtime-bench/server.mjs
```

打开 `http://127.0.0.1:8788/decode.html`，运行解码剖析。需要硬件 WebGPU 与 `timestamp-query`；所有模型与报告仍只在本机。页面运行 1/128/512-token 输入的基线、融合回读 logits、融合回读 token 各五轮，另测整段 GPU 时间戳。每次生成 32 token，清空逻辑 KV，无跨请求前缀/结果复用。

`experimental/webgpu-decode.mjs` 是默认未接入业务 Agent 的实验子类。融合保留 27 层、20 轮 log-domain Sinkhorn、Engram 和权重，减少单 token 阶段的后处理、Q/gate 与 K/V 投影、非 Engram 层 routing/混合/归一化的 dispatch，498 次变为 286 次；GPU greedy argmax 再加两次。argmax 并列时选最低 ID，NaN/Infinity 返回无效 ID，由 `forwardToken` 拒绝。该接口未提供随机采样或 grammar 约束。

最终报告 `validation/2026-09-14-webgpu-decode.json`：M4 Max 上五轮中位数，128-token 上下文解码约 135.94 → 165.10 tok/s，尚未达到 300 tok/s。所测数值与 32-token 序列一致，GPU 边界检查通过。当前 GPU buffer 约 336.02 MiB，不是低内存发布版。没有更换 8765 数据体检页的 CPU 后端，也不将 raw LM 速度当成完整 Agent 速度。

诊断时间戳与正式吞吐分开。初步逐 kernel 剖析改变了 pass 数量，且观测到 65,536 ns 时间戳量化；小 kernel 的分类耗时只供定位，不作精确占比。GPU 概率域 Sinkhorn 和单工作组注意力的未采用实验连同源码已留档，最终后端不包含这两条路径。
