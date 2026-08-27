# needle-vrm-agent-en-v0

English **contrast** bank parallel to `needle-vrm-agent-v0`. **Not** a needle-zh product KPI.

- 48 items, `EVAL-NVA-EN-001`…`048`, `review_status=translated` (parallel of reviewed zh items, not a second human review)
- Toolset: `notebook/evaluation/shared/toolsets/needle-vrm-agent-en-v0.json` (same 16 skills; food enums in English)
- Prefix: `Scene: {scene}. User: {query}`
- Gold contract is still needle-zh phase-1: missing / offtopic / scene conflict / illegal pair → `[]` (no clarify)
- Official Needle 2 historically **asks to fill slots**; exact-match will punish that. Report execute-family separately.
- Isolation scans `EVAL-NVA-EN-*` and English `query` text. Do not copy into any train seed.
- Do **not** subscribe this bank in `tasks/index.json` for needle-zh KPI.

```bash
PYTHONPATH=scripts python3 notebook/_tooling/scripts/eval_needle_toolcall_v0.py \
  --bank notebook/evaluation/banks/needle-vrm-agent-en-v0/eval-bank-v0.jsonl
# Official Needle 2 (hello-needle venv)
workspaces/ws-needles/notebook/hello-needle/.venv/bin/python \
  scripts/run_eval_needle2_official_v0.py
# Qwen on the same English bank
PYTHONPATH=scripts python3 notebook/_tooling/scripts/run_eval_needle_qwen_v0.py \
  --backend ollama --model qwen3.5:4b-mlx \
  --bank notebook/evaluation/banks/needle-vrm-agent-en-v0/eval-bank-v0.jsonl
```
