# zh-pretrain-colloquial-synth-v1 (engineering contrast only)

Offline `offline-frame-renderer` pack. **Not** a formal CPT spoken source.

- Claimed `n_unique_train_tokens` ≈ 38.1M is **exposure**, including duplicate `frame_id`.
- Corrected unique-by-first-frame is recorded in `ACCOUNTING.json` (≈23.2M) and is still below a quality bar.
- Do not mix or resume qwen production into `raw/accepted.jsonl`.
- Formal spoken role is `notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus/` after quality/isolation/blind gates.
