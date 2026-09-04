# Corpus quality

Formal entrypoint:

```bash
.venv/bin/python corpus-factory/quality/audit.py --help
```

Quality receipts are write-once and separate:

- natural-source integrity, provenance and license review;
- synthetic template diversity, eval leakage and named human semantic review;
- SFT release/governance integrity;
- cross-receipt `reuse` or fail-closed `retire` decisions.

Hash, dedup and leakage gates establish integrity and isolation only. They do
not establish naturalness, semantic consistency, learnability or model quality.
