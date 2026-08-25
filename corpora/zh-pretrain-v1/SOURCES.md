# zh-pretrain-v1 sources

v1 是 1B+ unique train 的 mix 槽。维基 epoch 仍以 v0 为准；本槽只追加互补 unique token。

| 源 | 本机路径 | 包装许可 | 档 | 用途 |
|----|----------|----------|----|------|
| 中文维基 v0 train/valid | `../zh-pretrain-v0/tokens/{train,valid}-*.bin` | CC BY-SA 3.0/4.0 + GFDL | A | 冻结主池；valid 只用这一份 |
| FineWeb2-HQ `cmn_Hani` | `raw/fineweb2-hq/cmn_Hani/*.parquet` | ODC-By 1.0；上游 Common Crawl ToU | C | 填 1B/3B 缺口；按 parquet 序号增量，不下全量 |
| SchemaStore JSON | `raw/schemastore/src/schemas/json/*.json` | Apache-2.0（仓内文件）；catalog 外链不继承 | B | 中性 JSON Schema；丢 &gt;32k 字符生成物 |
| OAI OpenAPI 示例 | `raw/oai-examples/*.yaml` | Apache-2.0；取 tag `3.0.3`（`main` 已无 examples/） | B | 结构种子，token 可忽略 |

FineWeb2-HQ 发布：https://huggingface.co/datasets/epfml/FineWeb2-HQ  
采用 C 档 = 接受包装许可 + CC ToU，**不是**无条件商用。再分发须保留 ODC-By 署名。维基衍生文本再分发须保留署名与 ShareAlike。

**不用**：FineWeb2 test、HPLT 主池、CCI3/CCI4（含与 FineWeb2 双计）、SkyPile、MAP-CC、WuDao、对话集（CrossWOZ/KdConv 走 SFT 账）、重复 epoch。

大文件不入库。cursor / mix / manifest 记录消费过的 parquet 名，3B 续跑不要改过滤门。
