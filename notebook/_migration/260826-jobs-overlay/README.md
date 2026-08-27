# jobs

Process surface for mei-llm. **Does not replace** `corpora/`, `tasks/`, or `eval/`.

```text
jobs/<topic>/
  inbox/                         # contracts, fleet mix, input refs
  jobs/<YYMMDD-NN-slug>/         # one executable job
    job.json
    work/                        # mutable produce/audit output
    _/                           # logs
  outbox/{draft,accepted,archive}/<job-id>/receipt.json
```

Rules:

1. `raw/accepted.jsonl` means line-level filter pass, not formal `outbox/accepted`.
2. Training jobs subscribe only to `corpora/<id>` or `tasks/<id>/train/packs` listed in a registry as accepted.
3. Eval banks stay in `notebook/evaluation/banks/` and never promote into train.
4. Existing large corpora stay in place; new synthesis writes `jobs/.../work` then `promote`.
