# zh-pretrain-v0 sources

本槽只使用**已经在本机**的中文维基文本，不在本阶段下载或引入 SkyPile / CCI3 / MAP-CC。

| 源 | 本机路径 | 许可 | 用途 |
|----|----------|------|------|
| 完整中文维基 dump | `../zh-vocab-v0/dumps/zhwiki-latest-pages-articles.xml.bz2` | CC BY-SA 3.0/4.0 + GFDL | 全文预训抽取 |
| 全文抽取 raw shards | `raw/pages-*.jsonl`（gitignored） | 同上 | 未截断正文 + page_id/chunk_id |
| 预 token 二进制 | `tokens/{train,valid}-*.bin`（gitignored） | 同上 | uint16 mmap 训练 |
| 词表抽样短行（仅 `--smoke`） | `../zh-vocab-v0/raw/spm-sample.txt` | 同上 | smoke 小 shard |
| 合成模板（仅 `--smoke`） | `scripts/build_zh_pretrain_v0.py` | 本仓 | 探针风格短句，不进入 wiki 主 shard |

冻结 dump：`zhwiki-latest-pages-articles.xml.bz2`，sha256 `4699d4dc8342b21beaca36b3d699ad28844dd3837e319e84b9640759b6c62455`，3,379,207,113 bytes。清洗器 id：`zhwiki-pretrain-v1-paragraph`（无每页 256 截断、无总字符早停；段落切块 4096）。

维基衍生文本再分发须保留署名与 ShareAlike 义务。大 shard / dump **不入库**；本目录提交 README、SOURCES、manifest。

**不用**：SkyPile、CCI3、MAP-CC（BY-NC-ND）；未完成许可审查的外部预训源。禁止用重复 epoch 把 650M unique train token 改写成 1B。
