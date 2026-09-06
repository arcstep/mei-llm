# mei-1.2-51m

`mei-1.2-51m` 是中文工具 Agent 模型的现役产品版本，部署 LM 参数固定为 51,463,797。
`300M`、`600M`、`900M` 等均表示累计训练 token exposure，不是参数规模。
旧链归档为 `mei-1.1-51m`（历史证据，只读）。

现行模型只接收文本，不包含 ASR/TTS 或声学编码器；语音产品由上层把转写文本送入本模型，
不会改变 51M 的参数身份。

- 架构源码：`../../src/architecture/mei-1.2-51m/`
- 词表（唯一现役 + 冻结指针）：[`tokenizer/`](tokenizer/)（`TOKENIZER.json` 为唯一权威）
- runtime 成果（wasm 构建产物）：[`runtime/`](runtime/)
- 正式权重与可加载包：[`exp-00300m/`](exp-00300m/)（每轮 cycle 一个子目录，成果主入口）
- 训练过程证据：`../../cycles/mei-1.2-51m/`（旧链在 `../../cycles/mei-1.1-51m/`）
- 语料：`../../corpus/`；工厂源码：`../../src/`

`exp-XXXm/` 保存不可变的 Base、最终 SFT 主干、独立 heads 和可加载 CQ2 package。
它们不是可随时由源码重建的缓存；训练 run 只保存产生这些资产的过程证据。
大二进制不进入 Git，但必须由每轮 `ASSETS.json` 固定哈希，并建立仓外备份。
