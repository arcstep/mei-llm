# mei-1.3-51m

`mei-1.3-51m` 是 2026-09-14 开始从零训练的新一代。部署 LM 参数仍为 51,463,797；
`800M`、`1B` 等表示累计 CPT token exposure，不是模型参数规模。

当前状态：A10 正在连续执行 2,500,001,792 token 的 CPT。三个观察阶段为
800,000,000／800,000,000／900,001,792 token；阶段评估不阻塞服务器续训。

- 模型合同：[`MODEL.json`](MODEL.json)
- 架构沿用记录：[`ARCHITECTURE_HISTORY.md`](ARCHITECTURE_HISTORY.md)
- 当前训练词表：[`tokenizer/`](tokenizer/)，包含实际 `.model` 和 `.vocab` 文件
- CPT 冻结输入：[`../../corpus/pools/frozen/2026-09-14-mei-51m-v1.3-cpt-25b-r01/`](../../corpus/pools/frozen/2026-09-14-mei-51m-v1.3-cpt-25b-r01/)
- v1.3 采用登记：[`../../corpus/adoptions/mei-51m-v1.3/`](../../corpus/adoptions/mei-51m-v1.3/)
- 当前 A10 过程证据：[`../../cycles/mei-1.3-51m/exp-00800m/runs/2026-09-14-a10-cpt-campaign/`](../../cycles/mei-1.3-51m/exp-00800m/runs/2026-09-14-a10-cpt-campaign/)

本目录当前只登记已经真实采用的 tokenizer，不预建尚未产生的 Base、QAT、SFT 或 package。
阶段权重复制并核验后，再按实际累计曝光和 run ID 增加不可变成果。
