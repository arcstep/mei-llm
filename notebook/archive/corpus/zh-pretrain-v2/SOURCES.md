# zh-pretrain-v2 sources

Band policy: A/B/C admitted; D/E, NC, unauthorized gated, and benchmark gold stay out.

| Source | Path | License tag | Band | Role |
|--------|------|-------------|------|------|
| Chinese Wikipedia v0 train/valid | `../zh-pretrain-v0/tokens/{train,valid}-*.bin` | CC BY-SA 3.0/4.0 + GFDL | A | Frozen encyclopedia; wiki valid is the only longitudinal valid |
| FineWeb2-HQ `cmn_Hani` (v1 snapshot) | v1 parquet, hash-split into `tokens/hq-*.bin` | ODC-By-1.0 + Common Crawl ToU | C | High-quality web; wrapper license ≠ page copyright |
| ChineseWebText2.0 `part-0001.jsonl.gz` | `raw/cwt2/` | Apache-2.0 dataset tag; webpage copyright not cleared | C | Colloquial web (forum/blog/QA/life), 30–100M unique |
| SchemaStore JSON + OAI examples 3.0.3 | v1 snapshot, hash-split + large-schema chunks | Apache-2.0 | B | Neutral schemas / OpenAPI |
| `codeparrot/github-code` parquet shards | `raw/github-code/` | Per-file detected license; ingest permissive only | C | JSON/YAML/TOML/config and short signatures; no full function bodies |

Blocked on this host (not ingested): `bigcode/the-stack`, `bigcode/starcoderdata`.

Never in this CPT unique pool: FineWeb2 test, JSON Schema Test Suite, CrossWOZ, KdConv, ToolACE, API-Bank, product eval banks.

See `download-manifest.json` for URL/revision/bytes/SHA256. See `unique-ledger.json` for frozen unique counts.
