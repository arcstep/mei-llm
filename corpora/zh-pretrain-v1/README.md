# zh-pretrain-v1

Needle-zh **1B unique train 版图**（清单式 mix）。大 parquet / tokens / hashes **不入库**。公开仓只提交本 README、`SOURCES.md`、`mix.json`、`manifest.json`、`ingest-cursor.json`。

本槽**引用**冻结的 `../zh-pretrain-v0` 维基 train/valid shards，不复制、不改写 v0。补充源：FineWeb2-HQ `cmn_Hani` + SchemaStore JSON + OAI OpenAPI 示例（tag `3.0.3`）。

valid 继续用 v0 维基 valid（34,610,229 token），以便 100M/300M 曲线可比。HQ / Schema **不**进 valid。

精确 token 数以同目录 `mix.json` / `manifest.json` 为准（构建器写入）。

## 构建

在 `mei-llm/` 下：

```bash
.venv/bin/python scripts/build_zh_pretrain_v1.py --rung 1b
# 探针仍改善后再升吨位：
.venv/bin/python scripts/build_zh_pretrain_v1.py --rung 3b
```

过滤：wiki-norm SHA256 去重、eval leak、短文、HQ 须 CJK、UNK 门 0.5%。禁止重复 epoch 把 v0 的 650M 改写成 1B。

可续跑：已消费 parquet 记在 `ingest-cursor.json`。目标达到后仍会把**当前** parquet 吃完，避免 3B 从文件中段续。

## 训练

默认 trainer 仍读 v0。1B 档必须显式：

```bash
.venv/bin/python scripts/train_needle_zh_pretrain.py --rung 1b \
  --corpus-dir corpora/zh-pretrain-v1 --batch-size 8 --grad-accum 1
```

来源与许可见 `SOURCES.md`。规划正文在 monorepo draft（本 README 不链 `docs/`）。
