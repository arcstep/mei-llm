"""Greedy decode with phase-1 JSON-array prefix filter."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from .grammar import is_legal_prefix, parse_phase1_text
except ImportError:
    from grammar import is_legal_prefix, parse_phase1_text


def greedy_constrained(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new: int = 96,
) -> dict[str, Any]:
    ids = list(prompt_ids)
    pieces: list[int] = []
    for _ in range(max_new):
        arr = mx.array([ids], dtype=mx.int32)
        logits = model(arr)["logits"][:, -1, :]
        logp = nn.log_softmax(logits, axis=-1)
        order = mx.argsort(logp[0])[::-1]
        chosen = None
        prefix = tokenizer.decode(pieces)
        complete = bool(prefix.strip()) and bool(parse_phase1_text(prefix).get("ok"))
        if complete:
            break
        for idx in order.tolist():
            tok = int(idx)
            if tok == tokenizer.pad_id:
                continue
            if tok == tokenizer.eos_id:
                if is_legal_prefix(prefix) and (
                    not prefix.strip() or parse_phase1_text(prefix).get("ok")
                ):
                    chosen = tok
                    break
                continue
            trial = tokenizer.decode(pieces + [tok])
            if is_legal_prefix(trial):
                chosen = tok
                break
        if chosen is None or chosen == tokenizer.eos_id:
            break
        ids.append(chosen)
        pieces.append(chosen)
        text_now = tokenizer.decode(pieces)
        if text_now.endswith("]") and parse_phase1_text(text_now).get("ok"):
            break
    text = tokenizer.decode(pieces)
    parsed = parse_phase1_text(text)
    return {"text": text, "parsed": parsed, "ids": pieces}
