# mei-llm

MEI 的本地模型工程仓。当前唯一主产品是参数固定为 51,463,797 的
[`mei-1.0-51m`](models/mei-1.0-51m/README.md)；300M、600M、900M 等表示累计训练
token exposure，不是模型参数规模。

`mei-1.0-51m` 的输入边界是文本：它负责工具检索、受约束调用、多步结果闭环和终态文字
解说。ASR、TTS、麦克风及实时交互由 `mei-agent` / `mei-avatar` 等上层产品组合，不进入
51M 权重或模型包；语音转写形成的口语文本仍可作为带来源的 SFT/Eval 输入。

## 从哪里开始

1. 看长期周期和当前成果：[`cycles/mei-1.0-51m/INDEX.md`](cycles/mei-1.0-51m/INDEX.md)
2. 看某一轮语料：进入该 cycle 的 `corpus/CORPUS.md`
3. 看模型定义：[`models/mei-1.0-51m/`](models/mei-1.0-51m/)
4. 看训练、评估和发布方法：[`model-factory/`](model-factory/)
5. 看当前运行实现：[`platform/`](platform/)

```text
models/          模型身份、架构、tokenizer，以及不可替代的正式模型资产
corpus-factory/  当前语料生产和审计系统；历史由 Git 追踪
model-factory/   当前训练、评估、编排和发布系统；每轮由 pipeline lock 固化
cycles/          每个 exposure rung 的永久成果入口
platform/        当前最新 Python 与 Browser SDK/Runtime
```

这五个目录是唯一的人类主导航。正式 Base、QAT、SFT/heads 和可加载 package 直接进入
[`models/mei-1.0-51m/releases/`](models/mei-1.0-51m/releases/)；它们是不可替代的模型
资产，不是可由源码重新编译的缓存。训练与评估源码同样不是临时脚本：正式实现可见于
`model-factory/`，轻量 CLI/registry 代码在 `src/`。`.internal/` 只保存派生机器索引、
迁移映射和非主线说明；语料大文件、训练 run、日志、构建缓存和历史证据收在 Gitignored
`.local/`。日常理解成果无需翻找隐藏源码或大型 run 目录。

## 当前结论

- 300M Base 已冻结并仍由未改字节的 `CURRENT.json` 选中。
- 300M/600M adaptive-v5 产品化均已完成机制流程，但都不具备发布资格。
- 600M checkpoint 数值上可以继续 CPT；其 hybrid recovery 与语料多样性问题使其不能
  自动晋升父版，且旧增量语料不能原样复用。
- 300M/600M 指标、语料和决策分别见各 cycle；不要到 `.local/artifacts/` 人工翻找结论。

## 统一只读导航

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm model status
PYTHONPATH=src .venv/bin/python -m mei_llm cycle list
PYTHONPATH=src .venv/bin/python -m mei_llm cycle show exp-000600m
PYTHONPATH=src .venv/bin/python -m mei_llm corpus show exp-000600m
PYTHONPATH=src .venv/bin/python -m mei_llm cycle compare exp-000300m exp-000600m
PYTHONPATH=src .venv/bin/python -m mei_llm artifact resolve '<URI-or-legacy-path>'
```

同一门面还提供 `cycle plan|run`、`corpus plan|build|audit|release|diff`、
`platform test|benchmark` 与 `release prepare|verify`。写入型命令仍受不可变 artifact、
fingerprint、离线和 `CURRENT.json` 只读规则约束；`release prepare` 只输出候选模板，
不会发布。

`CURRENT.json` 在本次迁移中保持原字节和 SHA。其历史路径由 `.internal/registry/migrations/` 解析；
这不是 symlink，也不会改写历史 receipt。

本仓是可独立 clone 的公开仓边界，文档必须自包含，不能链接私有 monorepo 文档。权重、
大语料、真实安全审查和备案实例不进入公开 Git。
