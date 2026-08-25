# zh-vocab-v0

中文端侧词表槽。P0 交付：seed 表 + 本机维基抽样 + SentencePiece 草稿（提交 `vocab.txt`，`.model` 不入库）。

```text
seeds/           # 可提交：汉字 / 城市 / 房间 / reserved
dumps/           # 本机：zhwiki xml.bz2（gitignore）
raw/             # 本机：wiki-text jsonl、spm-sample.txt
vocab.txt        # 提交：SP 词表
coverage-v0.json # 提交：一级汉字与探针覆盖率
SOURCES.md       # 许可与 URL
```

P0 草稿 `zh-24k.model` 已废止，不得与新 checkpoint 混用。现行冻结件：`zh-24k-v1.model`（PAD=0 EOS=1 BOS=2 UNK=3，含协议特符与 VRM 整词）。

```bash
python3 scripts/train_zh_vocab_spm.py --freeze-v1
python3 scripts/report_zh_vocab_coverage.py --v1
```

约 5 亿汉字是字种覆盖上限，不是本轮抽样门槛。本轮 SP 用约 5e7–2e8 汉字即可。
