# zh-vocab-v0 sources

P0 词表草稿用到的来源。大 dump 只留本机 `dumps/` / `raw/`（gitignore）。

| 源 | URL | 许可 | 用途 | 入 SP 训练 |
|----|-----|------|------|------------|
| 通用规范汉字表（一/二级数字化） | https://github.com/shengdoushi/common-standard-chinese-characters-table | 公开规范转写（脚本生成 seed） | `seeds/hanzi-level1.txt` 作 user-defined；level2 补字种 | 一级：是（reserved）；二级：覆盖统计 |
| THUOCL 地名 | https://github.com/thunlp/THUOCL `data/THUOCL_diming.txt` | MIT | 抽「市」作 `seeds/cities-zh.txt` | 高频市名：是 |
| 行政区划（市） | https://github.com/modood/Administrative-divisions-of-China `dist/cities.json` | MIT | 地级市中文名 | 地级市：是 |
| 中文维基 dump | https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 | CC BY-SA 3.0/4.0 + GFDL | 频率样本 `raw/spm-sample.txt` | 是（抽样明文） |
| 工具 reserved | 本仓自造 | 与 mei-llm 相同 | `get_weather` 等整词 + JSON 标点 | 是 |

**不用**：MAP-CC（CC BY-NC-ND）。SkyPile / CCI3 仅作 P1 预训候选，本轮不下载。

维基衍生文本再分发须保留 BY-SA 署名。本仓只提交 seed、`vocab.txt`、覆盖率，不提交 dump。
