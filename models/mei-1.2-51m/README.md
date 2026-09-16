# mei-1.2-51m

`mei-1.2-51m` 是中文工具 Agent 模型的现役产品版本，部署 LM 参数固定为 51,463,797。
`300M`、`600M`、`900M` 等均表示累计训练 token exposure，不是参数规模。
旧链归档为 `mei-1.1-51m`（历史证据，只读）。

现行模型只接收文本，不包含 ASR/TTS 或声学编码器；语音产品由上层把转写文本送入本模型，
不会改变 51M 的参数身份。

- 架构源码：`../../src/architecture/mei-1.2-51m/`
- 词表（唯一现役 + 冻结指针）：[`tokenizer/`](tokenizer/)（`TOKENIZER.json` 为唯一权威）
- runtime 成果（wasm 构建产物）：[`runtime/`](runtime/)
- 货架导航（用哪个跑、归档了什么）：[`SHELF.md`](SHELF.md)
- 正式权重与可加载包（活跃基线）：[`releases/`](releases/)（每轮 cycle 一个子目录，只放可作基线的成果）
- 历史流程产物（非基线，只读）：[`archive/`](archive/)
- 训练过程证据：`../../cycles/mei-1.2-51m/`（旧链在 `../../cycles/mei-1.1-51m/`）
- 语料：`../../corpus/`；工厂源码：`../../src/`

`releases/exp-XXXm/` 保存不可变的 Base 与最终 SFT 主干。它们不是可随时由源码重建的缓存；
训练 run 只保存产生这些资产的过程证据。**只有「可作对照基线」的产物留在 releases/ 货架，
老流程（如 narration 未训练的 cq2-v1）归档到 `archive/`，不在货架上混淆。**
大二进制不进入 Git，但必须由每轮 `ASSETS.json` 固定哈希，并建立仓外备份。
