# notebook/corpus/lm-v3

Conditional 1B→2B unique expansion workdir. **Blocked** until `base/mei-1.0-51m-base-cpt1b-v1` is promoted and the 1B benefit gate passes (wiki/HQ valid improve; no catastrophic role regression).

Do not append shards into `corpus/lm-v2`. After the gate:

1. Harvest unused `epfml/FineWeb2-HQ` `cmn_Hani` parquet (~733.35M general-language tokens).
2. Synthesize ~16.204M fresh structure and ~100.362M fresh colloquial.
3. Collect ≥900M unique train tokens including packing/valid/drop slack.
4. Global exact/near dedup vs all historical train/eval; license + CWT2 + UNK/PII isolation.
5. Publish a new serving atlas `corpus/lm-v3` and `schedule-cpt-2b.json`.

Recipe stub: `training/mei-1.0-51m-train-v1/recipes/cpt-2b-conditional.json`.
