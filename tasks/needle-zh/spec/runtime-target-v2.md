# mei-1.0-58m runtime target v2

> **status**: `target_contract`  
> **implementation_state**: `not_implemented`  
> **blocking_for_publish**: `false`  
> **legacy_peer**: `runtime.md` / `mei-route-protocol-v1`  
> **warning**: This file is not consumed by current Route-ID code.

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

At most one call in phase 1. Missing-required behavior must be selected before SFT.

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

## Loop

- `complete`: one retrieval/generation turn.
- `run(max_steps)`: execute calls, feed results back, repeat until response/empty/limit.
- Preserve ordered `function_calls`; parallel execution is an upper-layer decision only for proven independent calls.
- No unbounded autonomous loop or free-text chat fallback.

## Implementation gate

Do not mark implemented until:

- retrieval/index tests pass;
- byte grammar fuzz passes;
- KV/sink parity and bounded-memory tests pass;
- full-call scorer and provenance validator pass;
- confidence is calibrated and connected to execution;
- target eval protocol is frozen.
