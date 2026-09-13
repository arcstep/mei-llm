# Source survey (preparation only)

Survey sources before selecting or reusing a training pool. Existing tokens and
downloaded shards are inventory facts, not proof of representative or suitable data.

```sh
PYTHONPATH=src .venv/bin/python -m mei_llm corpus source profile \
  --config src/corpus-factory/recipes/source-survey-20260913-v1.json \
  --out cycles/mei-1.2-51m/exp-corpus-survey/runs/NEW-ID
```

This defaults to local, read-only source access. The only writes are new survey
artifacts in `--out`. Use `--allow-network` for explicit public metadata and HTTP
range sampling. `--metadata-only` records the frame without downloading content.
The recipe captures intended source patterns, seed, per-shard sample size,
network budget and disk reserve. Local frames are explicitly not full upstream
populations. GitHub tree counts include code/docs/tests and are not data counts.

The first wave selects one file from each of up to 64 strata across the entire
ordered frame; a second wave selects a different file from each non-singleton
stratum. Waves execute in this order. The second draw is disjoint, not statistically
independent. Each sample binds its file, row group/row index, raw text and metadata,
inclusion probability, source revision and payload hash. JSONL uses a full-stream
reservoir, preserving whole dialogue arrays. No tokenization or text rewriting is
performed. Unsupported relation formats remain explicit gaps.

Remote reads require exact HTTP 206 ranges; servers that ignore ranges are refused.
Each file has a bounded 16 MiB range cache; concurrent shard workers are capped at
two. Oversized individual reads fail with evidence rather than downloading an
entire file silently. Network reservations include failed/uncertain reads and are
conservative upper bounds, not billed byte measurements. Transport URLs with CDN
signatures are kept only in memory.

Runs are write-once. For interruption recovery, pass `--resume-from OLD-RUN` with
the identical frozen config and a NEW `--out`. Complete shard samples are adopted
only if upstream revision, frame and sampling design still match. Old runs and
their samples remain untouched; failed/missing shards are retried. An interrupted
run without a final manifest is not an integrity-verified survey release.

```sh
PYTHONPATH=src .venv/bin/python -m mei_llm corpus evaluate audit-coverage \
  --surveys cycles/mei-1.2-51m/exp-corpus-survey/runs/LOCAL-ID \
            cycles/mei-1.2-51m/exp-corpus-survey/runs/GLOBAL-ID \
  --out cycles/mei-1.2-51m/exp-corpus-survey/NEW-AUDIT.json
```

The audit verifies all manifest-bound artifact hashes before comparing descriptive
distributions and observed exact sample overlaps. Incomplete samples do not support
whole-population extrapolation. UTF-8 bytes are not model tokens. Zero sample
overlap does not certify whole-corpus isolation. Unknown topic and split metadata
remain unknown. There is no automatic M1/M2 approval or training admission.

Optional `--local-teacher-review` on `audit-coverage` instead creates a new review
directory at `--out`. It uses only the configured local `qwen3.6:35b-mlx`, captures
its digest and exact request/response bytes, and annotates up to 32 snippets per
source. No cloud provider is called, no training text is generated, and machine
annotations are never human signoff or gold. The local worker is sequential.

Additional recipe fields:

- `ledger_root`: census all existing manifests and source-path ledgers beneath
  the specified root, keeping declared tokens and review claims separate.
- `local_diagnostics`: compare a DBLP head with its full local original and bind
  historical preparation/clearance receipts. XML opening-tag counts are not
  parser validity or semantic coverage.

Output conventions: `frame.json`, `sampling-plan.json`, immutable per-shard sample
or failure JSON, `profile.json`, implementation snapshots and final `survey.json`.
Source review, clearance, M1 selection decisions and M2 recipe decisions are still
required before this project's new production preparation. Existing release,
tokenizer and training command contracts are unchanged.
