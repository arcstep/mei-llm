# mei-1.0-51m

`mei-1.0-51m` 是中文工具 Agent 模型家族，部署 LM 参数固定为 51,463,797。
`300M`、`600M`、`900M` 等均表示累计训练 token exposure，不是参数规模。

现行模型只接收文本，不包含 ASR/TTS 或声学编码器；语音产品由上层把转写文本送入本模型，
不会改变 51M 的参数身份。

- 架构与权重合同：[`MODEL.json`](MODEL.json)
- 长期周期：首页见 [`../../cycles/mei-1.0-51m/INDEX.md`](../../cycles/mei-1.0-51m/INDEX.md)
- 冻结 tokenizer：`tokenizer/`
- 当前架构实现：`architecture/`
- 正式权重与可加载包：[`releases/`](releases/)。这是模型成果的唯一主入口。

`releases/exp-*/` 保存不可变的 Base、QAT 主权重、最终 SFT 主干、独立 heads 和可加载
CQ2 package。它们不是可随时由源码重建的缓存；训练 run 只保存产生这些资产的过程证据。
大二进制不进入 Git，但必须由每轮 `ASSETS.json` 固定哈希，并建立仓外备份。
