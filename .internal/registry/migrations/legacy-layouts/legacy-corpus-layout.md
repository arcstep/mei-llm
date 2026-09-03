# corpus/

Frozen serving packs for training. Canonical consumption is here; notebook is the workbench.

Current pointer: `CURRENT.json` → `corpus/lm-v1`.

Each pack keeps only README / RELEASE / manifest / SOURCES or source-license / hashes / `tokens/*.bin`.
Status may be `active`, `inactive`, or `not_admitted`. From-scratch mix follows `schedule-scratch.json` (`quota_plan`, 300M exposure). Archived CPT plans must not be restored onto the serving root. Processing ledgers stay in `notebook/`.

- `lm-v1/language/zh-pretrain-v0/` — active wiki
- `lm-v1/language/hq/` — active FineWeb2-HQ (hardlinked with archive v2)
- `lm-v1/language/zh-pretrain-v1/` — inactive 1B sidecar
- `lm-v1/structure/zh-pretrain-v3/` — active structure
- `lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1/` — active 30.1M mixed pack; internal scratch only

Process extract/reviews/indexes live under `notebook/corpus/lm-v1/`. Assemble work is a receipt directory, not a writable alias of this tree.
