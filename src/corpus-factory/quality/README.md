# Corpus quality

Formal entrypoint:

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm corpus evaluate audit-source --help
```

Source-population surveys and coverage diagnostics are documented in
[`../sources/PROFILING.md`](../sources/PROFILING.md). They precede selection of new
training inputs and do not replace source clearance or semantic review.

Quality receipts are write-once and separate:

- natural-source integrity, provenance and license review;
- synthetic template diversity, eval leakage and named human semantic review;
- SFT release/governance integrity;
- cross-receipt `reuse` or fail-closed `retire` decisions.

Hash, dedup and leakage gates establish integrity and isolation only. They do
not establish naturalness, semantic consistency, learnability or model quality.

`audit-source` supports explicit diagnostic/preparation modes through mutually exclusive config arguments:
`--dialogue-config` audits MOSS turn overlap and dangling tool evidence;
`--lccc-config` prepares a pinned LCCC candidate;
`--dialogue-split-config` removes heldout turn and near-document overlap;
`--dialogue-content-config` applies frozen lexical filters without reassigning splits.
These modes produce independent candidates and retain pending review status until named review.

`revocations.py` checks hash-bound `corpus/pools/*/quality/REVOKED*.json` decisions during source audit
and pool freezing. Renaming a revoked manifest or rewrapping its token hash cannot clear the revocation.
Clearance audits require a matching source ID, passed status, reviewer and review date.
