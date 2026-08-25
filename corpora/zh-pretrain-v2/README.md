# zh-pretrain-v2

Needle-zh **1.0–1.5B unique CPT** mix. Large parquet / tokens / hashes **are not committed**. This public repo keeps README, `SOURCES.md`, `mix.json`, `manifest.json`, `schedule.json`, `RELEASE.json`, `download-manifest.json`, `source-role.json`, `unique-ledger.json`, and ingest cursors.

This slot **references** frozen `../zh-pretrain-v0` wiki shards and the v1 HQ/schema snapshot. It does not rewrite v0 or v1 binaries.

- Wiki valid stays the longitudinal baseline.
- HQ / colloquial / structure each have a ~1% hash-split valid set.
- `schedule.json` is the 300M → v2 CPT contract: wiki `skip_tokens=300_000_000`, window-level mix, structure `max_epochs=1.2` as exposure only.

Exact frozen unique counts (2026-08-25):

| 源 | unique train | valid |
|----|----------------|-------|
| wiki v0 | 649,904,474 | 34,610,229 |
| HQ | 383,617,452 | 3,778,466 |
| colloquial CWT2 part-0001 | 79,215,403 | 784,817 |
| structure | 39,681,707 | 408,065 |
| **total unique** | **1,152,419,036** | — |

UNK ≈ 0.0437%. The Stack / StarCoderData remain blocked. See `RELEASE.json` / `unique-ledger.json`.

## Build

From `mei-llm/`:

```bash
.venv/bin/python scripts/fetch_zh_pretrain_v2_sources.py --github-shards 16
.venv/bin/python scripts/build_zh_pretrain_v2.py
```

The builder is resumable via `ingest-cursor.json`. Freeze fails if colloquial unique < 30M, structure unique < 20M, unique outside 1.0–1.5B, or UNK > 0.5%.

## Train

Default trainer still reads v0. After 300M promotion:

```bash
.venv/bin/python scripts/train_needle_zh_pretrain.py --rung 1b \
  --corpus-dir corpora/zh-pretrain-v2 \
  --init-weights tasks/needle-zh/checkpoints/pretrain-300m-state.npz \
  --lr 1e-4 --batch-size 8 --grad-accum 1
```

`--init-weights` loads parameters only and resets Adam. Same-stage resume stays `--resume` with strict corpus/schedule hashes.

Sources and licenses: `SOURCES.md`. Design notes live in the private monorepo; this README does not link `docs/`.
