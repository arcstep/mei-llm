---
name: mei-51m-corpus-sourcing
description: >-
  Manages natural CPT source pools for mei-1.0-51m: inventory, 300M mix
  planning, explicitly authorized FineWeb2-HQ downloads, provenance and
  license capture, frozen-tokenizer admission, document dedup, unseen-first
  accounting, and immutable pool releases.
---

# mei-1.0-51m 天然语料来源

## 不变量

- 天然池是可耗尽资源；配额不得超过未消费余额。
- 默认 `allow_repeat=false`、`selection=unseen_first`。
- 下载、入池、质量通过、cycle 绑定是四个独立状态。
- FineWeb2-HQ 固定为 `epfml/FineWeb2-HQ` / `cmn_Hani`；每批下载必须列出文件并显式授权。
- license 未审查、hash 漂移、document duplicate 或 tokenizer 漂移时 fail closed。
- 本 Skill 不生成合成语料、不训练模型。

## sourcing v2（从零重建，2026-09-04 起）

自 v2 起，来源治理改为**注册表驱动**（`corpus-factory/sources/source_registry.json`，
git 版本化）：能下载什么、以什么 band/license 入池、用什么 admit 模式，全部来自
注册表，代码不硬编码。旧 `wiki/fineweb2_hq` 双 role 保留为 `SOURCE_ROLES_LEGACY`
（仅读 v1 历史产物）；新 role 六类：

| role | band 政策 | admit | 备注 |
|---|---|---|---|
| fineweb2_hq | B | text | ODC-By + CC ToU |
| wiki_zh / wiki_en | B（wiki_en 配额克制） | text | CC BY-SA + GFDL |
| dialogue | C | text | 字幕/对话；每批须 clearance receipt |
| structured | A | **structured**（记录级去重） | SQL/JSON/XML/YAML 等 |
| code | C | structured | GitHub 代码；逐文件 license |

band A = 许可宽松；B = 署名/共享/ToU 约束（`distribution_clearance_asserted`
恒 false）；C = 权利未清算，除 license 双字段外必须附 clearance receipt，
否则 quality 政策门（`quality/policy/source-policy-v1.json`）直接 blocked。

## 入口

```bash
python scripts/inventory.py --help
python scripts/plan_mix.py --help
python scripts/download_hq.py --help     # 兼容别名，新流程用 download
python scripts/download.py --help         # generic collector，两段式授权
python scripts/admit.py --help
python scripts/freeze_pool.py --help
```

入口调用 `corpus-factory/sources/source_manager.py`。下载前读
`references/download-and-admission.md`；配额和余额读
`references/pool-mix-planning.md`。

### 两段式授权下载（所有新来源）

```bash
# 段一：离线解析清单，打印授权令牌（无任何网络调用）
python -m mei_llm corpus source download \
  --source-id fineweb2-hq-cmn-hani \
  --subset "data/cmn_Hani/train-00*.parquet" --dry-run

# 段二：令牌逐字节匹配才发起网络
python -m mei_llm corpus source download \
  --source-id fineweb2-hq-cmn-hani \
  --subset "data/cmn_Hani/train-00*.parquet" \
  --authorize-download "fineweb2-hq-cmn-hani:<令牌>" \
  --out DIR
```

令牌 = `source_id:sha256(排序后文件清单)[:16]`；清单任何漂移都会使令牌失配。
expected_sha256 不符 → 抛错且不落盘（先临时目录，全批校验通过才 rename）。
未钉扎批次 receipt 标 `hash_pinned:false`，治理回填钉扎 = supersede 事件。

### v2 入池（provenance v2）

```bash
python -m mei_llm corpus source admit \
  --role dialogue --input raw.jsonl \
  --license-id subtitle-rights-uncleared --license-reviewed \
  --source-id opensubtitles-zh --source-url https://... \
  --clearance-receipt clearance.json \
  --expected-tokenizer-id zh-24k-v1 \
  --out admitted/dialogue
```

- 带 `--source-id` = provenance v2（manifest `...-v2`，记录 source_id/source_url/
  subset/dataset_version/acquired_at + registry_entry_sha256 + tokenizer_id）；
  不带 = 旧 v1 兼容路径。
- `--expected-tokenizer-id` 与冻结指针（`models/mei-1.2-51m/architecture/
  tokenizer/TOKENIZER.json`）双绑；指针缺失/status≠frozen/hash 漂移一律拒绝。
- 结构化源加 `--mode structured`：记录级 sha256 去重、tokenize 保留结构、
  坏结构比例超阈值（注册表 admit.max_invalid_ratio，默认 0.001，只可调严）
  fail-closed 且无产物。XML 源必须注册表有 `record_element`。
- v1 admitted 目录只读不改；v2 重建 = 对原始下载用新词表重新 admit
  （新 release ID + supersede 链），seen-ledger 基于文本哈希可跨词表代际复用。

### 质量政策门

```bash
python -m mei_llm corpus evaluate audit-source --manifest ... --out receipt.json
python -m mei_llm corpus evaluate audit-structured --manifest ... --out receipt.json
```

band 违例、band C 缺 clearance receipt、structure_check 超阈值、tokenizer_id
缺失一律 blocked；audit-structured 另做回源重哈希 + token layout 校验。
