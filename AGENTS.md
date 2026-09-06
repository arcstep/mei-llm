# mei-llm agent contract

This repository has one active model product: `mei-1.0-51m` with exactly
51,463,797 deployed language-model parameters. Exposure labels such as 300M,
600M, 900M, and 1B are cumulative training tokens, not model sizes.

## Five-domain navigation

- `models/` defines model identity and architecture; `corpus-factory/` defines the current data-production system.
- `model-factory/` is the visible, first-class source of current training, evaluation, orchestration, and release behavior. Never hide a unique implementation under `.internal/`.
- `cycles/` is the only human-facing history of each cumulative exposure rung. Every executed rung binds corpus, model, evaluation, and decision evidence.
- `platform/` exposes `python-sdk/` and `browser-sdk/` directly; shared runtime/spec/Rust internals live under `_shared/`. Historical implementations come from Git, never copied exposure directories.
- Canonical Base/QAT/SFT/head/package assets live visibly under `models/mei-1.0-51m/releases/<cycle>/` and are addressed through `mei-artifact://` URIs. Corpora, runs, intermediate checkpoints, and legacy notebooks remain under Gitignored `.local/artifacts/`.
- Resolve legacy paths from immutable receipts and `CURRENT.json` through `.internal/registry/migrations/`; never rewrite an old receipt merely because files moved.
- `src/mei_llm/` is only the thin CLI/registry facade. `.internal/` contains derived indexes and migration evidence, never active training or evaluation algorithms.

## Model-factory source control

- Select formal entrypoints only from `model-factory/contracts/PIPELINES.json`; a `v3`, `v4`, or `v5` filename does not select the newest pipeline.
- Keep one current implementation in `model-factory/`. Git preserves implementation history; each executed cycle preserves the exact run-specific binding in `pipeline/PIPELINE.lock.json`.
- Before a future formal run starts, freeze the Git revision, dirty-tree patch or immutable source bundle, complete source manifest, environment, and per-stage source closure. A hash list without recoverable bytes is not exact reproducibility.
- Classify utilities through `model-factory/contracts/CODE_CATALOG.json`. Diagnostic and compatibility code may not silently enter a formal lineage.
- Historical reconstructed locks must state missing source bytes honestly. Never replace an old hash with current source or claim byte-exact reproduction from a semantic equivalent.

## Before changing anything

- Inspect `CURRENT.json`, the selected base `RELEASE.json`, and any live
  lifecycle run before editing training code or starting a job.
- A stale ledger is not proof that a run stopped. Reconcile PID/lock,
  heartbeat, checkpoint progress, and artifact hashes.
- Preserve the existing dirty worktree. Do not reset, checkout, delete, or
  overwrite unrelated user changes.
- Use the repository `.venv` and offline mode. Do not download models or data,
  call providers, publish, deploy, push, or assert distribution clearance.
- A live CPT defers heavy productization by default. An explicit user decision
  to run an already-frozen base independently may enable the fingerprinted
  `--allow-live-cpt` path; keep separate run directories, never signal or write
  the CPT run, and do not make that CPT part of the productization lineage.

## Immutable boundaries

- Treat every canonical asset under `models/<model>/releases/<cycle>/` as immutable and hash-locked. Model binaries are irreplaceable product assets, not rebuild caches.
- Promotion from `.local/artifacts/.../runs` uses an independent copy/clone plus byte/hash verification. Preserve the original run evidence; on-machine duplication does not replace off-device backup.
- Never train in place or overwrite a checkpoint/package/run. Create a new ID.
- Treat `CURRENT.json` as read-only. Only the explicit, compare-and-swap
  `finalize_current` operation may change it after separate user authorization.
- Weight compatibility is decided by the canonical weight contract, not by a
  raw `model.json` byte hash. Runtime profile and training-only auxiliary
  contracts are versioned separately.
- Needle 2 is a mechanism reference. Never claim `.cact`, `libneedle`, API, or
  ABI compatibility.

## Required productization order

1. Freeze serializer, grammar, schema subset, scorer, data splits, and eval.
2. Implement and validate runtime behavior before task SFT.
3. Record Float Base-LM Anchor and same-data Float Task Control before QAT.
4. Run Q4 only as a diagnostic, then the primary CQ2 QAT path.
5. Train retrieval, oracle-top5 full-call, learned-top5 E2E, MW disposition,
   and confidence in that order.
6. Rebuild the tool index after every final LM or retrieval-head change.
7. Repack all heads with the final LM and rerun cross-runtime and resource
   gates. Narration consumes verified results only and cannot execute tools.
8. Treat narration as a terminal-only optional sidecar: freeze the LM, train
   only the rank-16 residual, run the entire frozen generation eval split on
   the final CQ2 package, and fall back deterministically when grounding fails.

## Corpus factory and cycle invariants

- Corpus production is a first-class peer of architecture work, not a one-time release folder.
- Every cycle binds its own CPT delta, cumulative lineage, SFT suite, eval lock, factory revision, quality gates, and reuse decisions.
- Never inherit a corpus slice implicitly. Mark it `reuse`, `replace`, or `retire`; diversity-degraded slices default to `retire` for the next exposure.
- Keep CPT, SFT, and Eval releases separate. Retrieval/no-match, full-call, Agent, MW disposition, runtime outcome/confidence, and narration require separately auditable SFT bindings.
- Hash/dedup/leakage success proves integrity and isolation, not naturalness or semantic diversity.

## Tool-context and candidate-scan invariants

- Retrieval processes candidates in rank-stable batches of at most five. The
  first batch is not reduced to one or two merely because one score is high.
- `retrieval_discard_threshold` removes unusable candidates. A separately
  calibrated `retrieval_expand_threshold >= discard` controls which rank>5
  candidates may enter later batches. Locked test data never tunes either.
- Only an empty call with MW disposition class 10
  `capability_insufficient` may scan the next batch. Calls, class 0 empty
  output, safety/permission/state/evidence dispositions, exhaustion, and the
  configured batch limit are distinct terminal outcomes.
- Candidate scans are internal inference passes, not external tool steps.
  After a verified ToolResult, retrieval restarts from the first batch.
- Fixed prompt, projected schemas, query/context/evidence, history/results,
  and output reserve share the 2048-token contract. Compact/standard stable
  caps are 1024/1536 soft layout budgets; the ordinary KV region consumes the
  dynamic remainder, not a fixed 256-token input allowance.
- The model may see a shortened schema projection, but names, parameter
  structure, validation constraints, and the complete execution schema are
  never weakened. Context overflow is a planning branch, not an exception.
- MW disposition training consumes immutable per-batch visible views. It must
  not perform live retrieval inside the optimizer loop.

Productization must accept any real frozen 51M base through explicit release
and weights paths. Exposure labels must change fingerprints but never select a
different hard-coded stage graph; schema fixtures are not evidence that a
future 600M/900M artifact already exists.

## Runtime acceptance profiles

- Every run must freeze `validation_scope` in its fingerprint and receipts; a
  resumed run may not silently add or remove runtime targets.
- Before model quality is proven, a mechanism-validation run may use the
  `python-browser-wasm` profile. Python/MLX and real Browser-WASM execution are
  mandatory in that profile; the Rust core may remain an implementation
  dependency of WASM, while independent Rust SDK, Node SDK, and C/FFI delivery
  stay explicitly open and deferred.
- A release that intends to ship any deferred binding must select an expanded
  profile and make that binding's API, parity, integration, and resource gates
  mandatory. Never report a deferred binding as validated.

## MW terminology boundary

- **MW deviation** is an evaluation/governance finding about evidence-boundary
  drift. It is recorded by gates and receipts; it is not a model head, tensor,
  training label, package capability, or substitute for task evaluation.
- **MW disposition** is the independent closed-set reason-code sidecar required
  by the product flow. It may constrain execution but cannot satisfy or replace
  the MW deviation gate.
- Retrieval and confidence are separate heads with separate data, checkpoints,
  calibration, receipts, and runtime duties. Never group them under “MW”.
- Confidence calibration is validated only when the frozen outcome set contains
  both positive and negative classes. Single-class calibration is implemented
  but degraded and remains a release gap.

## Completion and failure semantics

- Treat the cycle contract in `cycles/mei-1.0-51m/LIFECYCLE.md` as the stable
  completion boundary. Add sub-gaps to the affected cycle capability instead
  of replacing the lifecycle or calling a finished training process a product.
- `process_complete` means every planned stage ran and has a terminal receipt.
- `release_eligible` additionally requires all integrity, safety, quality,
  parity, and resource hard gates. Do not tune thresholds after seeing scores.
- A quality miss is an ineligible candidate, not permission for unbounded
  optimization. Preserve evidence and continue independent diagnostic stages.
- Hash drift, data contamination, checkpoint corruption, or changed CURRENT
  fails closed. Never use the affected artifact as a parent.
- Keep package, Rust-session, and WASM-heap targets at 18 MiB, 64 MiB, and
  96 MiB respectively unless a later explicit contract supersedes them.
- When only runtime gates, reports, or control-plane code changes after a
  package is receipt-complete, use a new-fingerprint packaged adoption and
  rerun only downstream gates/audit. A one-token functional probe must not
  leak `max_new=1` into the 128-token-reserve joint-budget gate.

## Workflow entrypoint

Start at [cycles/mei-1.1-51m/INDEX.md](cycles/mei-1.1-51m/INDEX.md) for the
longitudinal status, then read the selected cycle's corpus, scorecard, and
decision pages. Read that cycle's `pipeline/PIPELINE.md` for the exact training
and evaluation chain. Use [cycles/mei-1.1-51m/LIFECYCLE.md](cycles/mei-1.1-51m/LIFECYCLE.md)
for the detailed productization contract.
After each immutable Base registration, productization, or frozen reevaluation,
append a hash-bound record to its longitudinal history; never rewrite an older
record or carry an unmeasured value forward.

Training is driven through the stable `python -m mei_llm` control plane and the
src/ pipeline modules; there is no Skills layer. Do not reconstruct a pipeline
from filenames. New product phases require `mei-51m-phase-binding-v1`;
exposure-specific script defaults are compatibility only and must not select
inputs for a new cycle.
