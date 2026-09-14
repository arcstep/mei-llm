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
performed. Explicit relation views support CrossWOZ dialogue/state, JSON schema/test groups, schema documents, table bundles and document/config/test bundles. Unsupported relation formats remain explicit gaps. These views keep upstream labels separate from visible inputs and never turn expected results into executed gold.

Remote reads require exact HTTP 206 ranges; servers that ignore ranges are refused.
Each file has a bounded 16 MiB range cache; concurrent shard workers are capped at
two. Oversized individual reads fail with evidence rather than downloading an
entire file silently. Network reservations include failed/uncertain reads and are
conservative upper bounds, not billed byte measurements. Transport URLs with CDN
signatures are kept only in memory.

Runs are write-once. For interruption recovery, pass `--resume-from OLD-RUN` with
the identical frozen design and a NEW `--out` (a subset of the same source entries is allowed). Complete shard samples are adopted
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

The separately authorized `--flash-compare-from LOCAL-REVIEW` mode replays only
the existing 24-record quality probe's request messages, blind to prior judgments.
It requires saved provider `--teacher-pricing` and `--teacher-provider-status`
snapshots. `--cloud-teacher-model` selects `deepseek-v4-flash` (1 CNY cap) or
`deepseek-v4-pro` (5 CNY cap); selecting Pro requires its own task authorization.
`--teacher-continue-from` adopts a known-billed prefix into a new output ID and
never retries earlier requests. The worker reserves conservative peak-price
costs before each call, preserves raw responses and usage, and stops on unknown
billing. Reported costs are conservative estimates, not invoices. Credentials
are read in memory from the user-authorized settings file, never copied into
receipts. This probe does not authorize future cloud batches or corpus admission.

The user's subsequent Pro budget decision supersedes its historical price cap:
`--pro-calibrate-from LOCAL-REVIEW --teacher-system-prompt RUBRIC
--teacher-offsets ...` runs bounded rubric checks against the original 24 records.
Pro usage is capped at 20,000,000 cumulative input+output tokens, including earlier
Pro calls. The worker serializes requests under a lock and reconstructs usage
from actual response receipts in sibling `pro-*` runs; unresolved requests block
new spending. Each request reserves UTF-8 input bytes plus framing and the output
limit before calling. Calibration outcomes remain machine judgments, not gold.
The historical price-based Pro CLI path is disabled; Flash remains unchanged.

## Fast raw-candidate preparation

The user's simplified acquisition plan uses a fixed random shard count instead
of population-wide quality scoring. `frozen_frame` plus its SHA256 reuses an
existing complete listing; `random_shard_count: 40` samples uniformly without
replacement. `purpose: candidate_preparation` allows up to 10,000 requested
records per shard. The existing reader selects at most eight row groups, so the
actual output may be smaller; receipts report actual counts. Body access to
remote frozen listings still requires `--allow-network`.

`python -m mei_llm corpus source materialize --config RECIPE --out NEW-POOL`
exports acquired texts as JSONL with `text`, provenance, hashes and available
group IDs. It supports whole-dialogue JSONL reservoirs and frozen survey records,
checks referenced sample hashes, excludes configured texts and non-train source
splits, and deduplicates exact text within the batch. Prior candidate files can
be listed as exclusions for subsequent batches. This does not perform teacher
grading or generate training text; raw candidates remain separate from frozen
training releases. Project Gutenberg body export removes only explicit wrapper
boundaries, retains raw source text and records the selected character range.

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

GitHub relation sources can adopt a hash-bound `catalog_frame`, filter its file
list, then download selected known-size files and verify their Git blob IDs.
They preserve original bytes and archive-member hashes. Broader patterns than a
partial parent catalog do not inherit complete-frame status. Local schema and
metadata censuses are available through `local_diagnostics.schema_inventory`
and `local_diagnostics.parquet_metadata_inventory`.

The coverage audit emits six-question `source_dossiers`, single-wave marginal
statistics, and distinct observed text hashes. No count of sample positions is
a claim of independent usable capacity. Literature and scripts are recorded as
`authored-dialogue`, separate from natural conversation; unknown identity stays
unknown.

`--frozen-evidence-only --resume-from OLD` seals already complete sample files
into a new manifest without network access. Missing samples stay failed/pending;
this is useful for interrupted multi-source jobs and does not upgrade coverage.
Large CrossWOZ archive members use a streaming JSON mapping reservoir, preserve
whole dialogues and member hashes, and never load the entire expanded member.

For a blinded human entry packet, run `audit-coverage --review-bundle` with a
new output directory. It includes a bounded set of complete sampled records,
links to originals and empty review fields. It is an entry point, not a substitute
for the planned stratified human review and not an M1 approval.


Extended survey adapters preserve whole GitHub source documents, ToolACE and
NaturalConv dialogues, streamed RiSAWOZ dialogues, versioned Wikisource pages,
and complete ZIP table/subtitle members. A downloaded table is a survey unit,
not automatically a CPT unit. Bare tables remain check-fixture candidates until
a task/constraint/interpretation relationship justifies training use.

`download_workers` is 1 or 2 (default 2); choose 1 when another single-download
survey is running so the task-wide limit remains 2. `whole-bounded` ZIP transport
streams a known or probed archive of at most 2 GiB, reserves its bytes first, and
keeps the original archive and checksums. Range transport retains selected raw
members and does not claim a full archive hash. Selected oversized members fail
explicitly and are never replaced to improve apparent yield.

The `local_diagnostics` hooks include table project/header concentration,
subtitle directory/version and upstream-label diagnostics, baseline tokenizer
measurements, literary overlap, and ToolACE schema/call preflight. Optional
ToolACE `declared_name_parser` reads names containing spaces without evaluating
calls or rewriting quoted values; `omit_null_outer_required` removes only null
outer metadata and preserves parameter constraints. Wire/schema success does
not certify request semantics, call dependencies, execution, budget fit, or gold.

Literary originals, templates, translations, source-code tests and upstream
synthetic tool answers retain their identities. None of these adapters runs
source code, calls APIs described by source text, trains a model, or grants M1/M2.

### CPT long-conversation windows

`python -m mei_llm corpus source materialize --config <recipe> --out <new-id>` supports
`mode: cpt_windows` with a hash-pinned candidate JSONL and parent manifest. See
`recipes/nemotron-windows-full-20260913-v1.json`. Output is a derived CPT view with
lossless primary source spans and separately recorded repeated context. It is not
additional independent corpus capacity or executable SFT gold. Actual encoded windows,
including context, continuation labels and BOS/EOS, must fit `limit` (2048 by default).

### v1.2 node-task preparation

`mode: node_task_v12_preparation` is an offline-only audit and preparation run.
It binds one CPT input release, one explicit tokenizer model, the legacy SFT
manifest and hash-pinned local task sources. The output contains:

- 120 non-independent engineering fixtures replayed through the real data-check
  JavaScript tools and the bounded plan runtime;
- a per-binding legacy SFT audit and at most 300 CPT evidence candidates;
- 100M/20M/1M train references plus a 500K dev reference set for future QAT,
  with every file hash and token range verified;
- bounded 200-row ToolACE and Nemotron adapter pilots; and
- a task-domain tokenizer preservation and length audit bound to the frozen real
  Browser-WASM receipt.

Engineering fixtures and structurally compiled public calls remain
`training_eligible=false` until semantic and runtime admission. Sources already
selected for CPT cannot become independent locked Eval without a successor CPT
release that excludes their association families. This mode does not download,
call a teacher, train, mutate CURRENT, or promote the tokenizer.
