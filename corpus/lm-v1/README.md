# corpus/lm-v1

Serving atlas for `mei-1.0-58m`. `CURRENT.corpus` points here.

```text
mix.json / schedule-scratch.json / RELEASE.json / manifest.json
language/zh-pretrain-v0/   # active wiki
language/hq/               # active FineWeb2-HQ
language/zh-pretrain-v1/   # inactive 1B sidecar
structure/zh-pretrain-v3/  # active structure
colloquial/...pooled-v1/   # active mixed-fleet colloquial, internal scratch only
```

Trainer consumes this tree via `mix.json` + `tokens/*.bin`.
Scratch mix uses `schedule-scratch.json` (`sampler=quota_plan`, `parent_tokens_seen=0`, 300M exposure, 512→1024→2048).
CPT `schedule.json` is archived and must not reappear on the serving root.
Build, extract, reviews, unique ledgers, and indexes live under `notebook/corpus/lm-v1/`.
The mixed-fleet 30.1M pack is admitted for internal unlabeled scratch pretraining only (`public_distribution_clearance_asserted=false`).
