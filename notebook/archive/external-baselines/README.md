# External baselines

Optional local checkpoints for sft-v2 capacity peers.

Expected layout (not downloaded by default):

```
minimind-25m/
minimind-45m/
```

Set `MEI_MINIMIND_25M` / `MEI_MINIMIND_45M` to override. Same-data comparison is only valid after attaching the identical retrieval and MW heads and training on the clean 10k packs. Missing checkpoints are skipped, not invented.
