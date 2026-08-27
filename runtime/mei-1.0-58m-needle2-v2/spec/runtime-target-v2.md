# mei-1.0-58m runtime target v2

> **status**: `target_contract`  
> **implementation_state**: `implemented_in_code`  
> **blocking_for_publish**: `false`  
> **legacy_peer**: `runtime.md` / `mei-route-protocol-v1`  
> **warning**: Route-ID v1 runners do not consume this file. Product retrieval/SFT/confidence thresholds are still null.

## Request lifecycle

```text
registered schemas
→ encode/cache tool embeddings

per turn:
  encode query
  → top-5
  → render canonical prompt
  → compile byte grammar over selected schemas
  → constrained full-call generation
  → post-generation provenance validator
  → confidence gate
  → execute/escalate/stop
```

Five or fewer tools render directly.

## Prompt

Visible fields:

- locked task contract;
- optional system facts;
- selected top-5 canonical schemas;
- short conversation/query;
- optional prior tool results.

Forbidden:

- `<routes>` or compiled call candidates;
- gold tool/route ID;
- gold provenance/entity link;
- full catalog when retrieval is required;
- side-specific prompts for baselines.

## Output

Phase 1:

```json
[
  {
    "name": "registered_tool",
    "arguments": {}
  }
]
```

or:

```json
[]
```

At most one call in phase 1. Frozen: missing required arguments emit `[]`; missing optional arguments are omitted. MTP is off. Retrieval first version uses `stop_gradient` on the backbone. Ordinary KV window is 256 plus explicit system/tool sinks. First platform is Apple Silicon / MLX. Quantization starts with 4-bit PTQ.

## Grammar

- Compile from the selected schema set.
- Apply token-byte transitions inside the decode loop.
- Mask invalid token logits to negative infinity.
- Enforce JSON, registered tool names, arguments, required/optional, supported types and declared value constraints.
- Collect call-token decode probabilities.
- Grammar validity does not imply semantic/provenance validity.

## KV memory

```text
visible KV =
  fixed system/tool sink KV
  + latest approximately 256 ordinary token KV
```

Requirements:

- fixed-capacity ordinary ring buffer;
- selected tool/system sink region;
- correct RoPE offsets;
- full-sequence vs incremental logits parity;
- RAM bounded across turns;
- tools remain callable after ordinary history slides.

`max_seq_len=2048` is the position limit, not the ordinary memory target.

## Confidence

```text
confidence = min(
  calibrated full-call correctness head,
  call-token decoding probability
)
```

Low confidence must block execution or escalate. Recalibrate after SFT, QAT or quantization.

Pre-registered auto-execute gate: 95% upper confidence bound on error rate at most 1%, with coverage at least 50%. Report coverage also at 0.1% and 0.5% risk.

## Loop

- `complete`: one retrieval/generation turn.
- `run(max_steps)`: execute calls, feed results back, repeat until response/empty/limit.
- Preserve ordered `function_calls`; parallel execution is an upper-layer decision only for proven independent calls.
- No unbounded autonomous loop or free-text chat fallback.

## Implementation gate

Runtime code and layered tests are in:

- `model/prompt_v2.py`, `byte_grammar.py`, `kv_manager.py`, `runtime_v2.py`, `provenance_validator_v2.py`, `tool_index.py`, `contrastive_head.py`, `confidence_v2.py`
- `scripts/test_mei_v2_runtime.py`, `eval_mei_retrieval_v2.py`, `run_eval_mei_toolcall_v2_oracle.py`

This proves the stack is trainable and testable. It is not a product-quality claim.

Still waiting after independent CPT + full-call SFT:

- retrieval product thresholds;
- confidence calibration / auto-execute coverage;
- frozen target eval protocol and bank.
