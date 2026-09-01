# mei-llm agent contract

This repository has one active model product: `mei-1.0-51m` with exactly
51,463,797 deployed language-model parameters. Exposure labels such as 300M,
600M, 900M, and 1B are cumulative training tokens, not model sizes.

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

- Treat every directory under `base/` with a release receipt as immutable.
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

- `process_complete` means every planned stage ran and has a terminal receipt.
- `release_eligible` additionally requires all integrity, safety, quality,
  parity, and resource hard gates. Do not tune thresholds after seeing scores.
- A quality miss is an ineligible candidate, not permission for unbounded
  optimization. Preserve evidence and continue independent diagnostic stages.
- Hash drift, data contamination, checkpoint corruption, or changed CURRENT
  fails closed. Never use the affected artifact as a parent.
- Keep package, Rust-session, and WASM-heap targets at 18 MiB, 64 MiB, and
  96 MiB respectively unless a later explicit contract supersedes them.

## Workflow entrypoint

Read [docs/mei-1.0-51m-work-guide.md](docs/mei-1.0-51m-work-guide.md)
for the product goals, capability boundaries, current evidence baseline,
longitudinal evaluation contract, work checklist, and Definition of Done.
After each immutable Base registration, productization, or frozen reevaluation,
append a hash-bound record to its longitudinal history; never rewrite an older
record or carry an unmeasured value forward.
Use the workspace skill `mei-51m-cpt-lifecycle` for status, continuation,
productization, receipts, gates, and freeze proposals. Keep this file concise;
put run-specific commands and recovery details in the skill references.
