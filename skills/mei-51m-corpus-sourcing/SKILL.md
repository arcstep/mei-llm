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

## 入口

```bash
python scripts/inventory.py --help
python scripts/plan_mix.py --help
python scripts/download_hq.py --help
python scripts/admit.py --help
python scripts/freeze_pool.py --help
```

入口调用 `corpus-factory/sources/source_manager.py`。下载前读
`references/download-and-admission.md`；配额和余额读
`references/pool-mix-planning.md`。
