"""Greedy decode: legacy JSON-array filter, plus route-ID trie with KV cache."""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from .grammar import is_legal_prefix, parse_phase1_text
    from .route_protocol import allowed_internal_texts, is_legal_internal_prefix, materialize_internal
except ImportError:
    from grammar import is_legal_prefix, parse_phase1_text
    from route_protocol import allowed_internal_texts, is_legal_internal_prefix, materialize_internal


def greedy_constrained(
    model,
    tokenizer,
    prompt_ids: list[int],
    toolset: Any,
    max_new: int = 96,
) -> dict[str, Any]:
    if toolset is None:
        raise ValueError("toolset is required; no default VRM")
    ids = list(prompt_ids)
    pieces: list[int] = []
    for _ in range(max_new):
        arr = mx.array([ids], dtype=mx.int32)
        logits = model(arr)["logits"][:, -1, :]
        logp = nn.log_softmax(logits, axis=-1)
        order = mx.argsort(logp[0])[::-1]
        ranked = order.tolist()
        chosen = None
        prefix = tokenizer.decode(pieces)
        complete = bool(prefix.strip()) and bool(parse_phase1_text(prefix, toolset).get("ok"))
        if complete:
            break
        for idx in ranked[:256] + ranked[256:]:
            tok = int(idx)
            if tok == tokenizer.pad_id:
                continue
            if tok == tokenizer.eos_id:
                if is_legal_prefix(prefix, toolset) and (
                    not prefix.strip() or parse_phase1_text(prefix, toolset).get("ok")
                ):
                    chosen = tok
                    break
                continue
            trial = tokenizer.decode(pieces + [tok])
            if is_legal_prefix(trial, toolset):
                chosen = tok
                break
        if chosen is None or chosen == tokenizer.eos_id:
            break
        ids.append(chosen)
        pieces.append(chosen)
        text_now = tokenizer.decode(pieces)
        if text_now.endswith("]") and parse_phase1_text(text_now, toolset).get("ok"):
            break
    text = tokenizer.decode(pieces)
    parsed = parse_phase1_text(text, toolset)
    return {"text": text, "parsed": parsed, "ids": pieces}


def _allowed_seqs(tokenizer, n_routes: int) -> list[list[int]]:
    seqs = []
    for text in allowed_internal_texts(n_routes):
        seqs.append(tokenizer.encode(text, add_bos=False, add_eos=False))
    return seqs


def greedy_route_id(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    manifest,
    toolset: Any,
    max_new: int = 24,
) -> dict[str, Any]:
    if toolset is None:
        raise ValueError("toolset is required; no default VRM")
    n_routes = len(getattr(manifest, "routes", []) or [])
    seqs = _allowed_seqs(tokenizer, n_routes)
    pieces: list[int] = []
    timings = {}
    t_prefill = time.perf_counter()
    out = model(mx.array([list(prompt_ids)], dtype=mx.int32))
    mx.eval(out["logits"])
    cache = out.get("cache")
    logits = out["logits"][:, -1, :]
    timings["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000
    t_dec = time.perf_counter()
    complete_text = None
    for _ in range(max_new):
        allowed: set[int] = set()
        done = False
        for seq in seqs:
            if pieces == seq:
                done = True
                break
            if len(pieces) < len(seq) and seq[: len(pieces)] == pieces:
                allowed.add(seq[len(pieces)])
        if done:
            complete_text = tokenizer.decode(pieces)
            break
        if tokenizer.eos_id is not None:
            prefix = tokenizer.decode(pieces) if pieces else ""
            if prefix in allowed_internal_texts(n_routes):
                complete_text = prefix
                break
        if not allowed:
            break
        logp = nn.log_softmax(logits, axis=-1)[0]
        best = None
        best_s = None
        for tok in allowed:
            score = float(logp[tok])
            if best_s is None or score > best_s:
                best, best_s = int(tok), score
        if best is None:
            break
        pieces.append(best)
        trial = tokenizer.decode(pieces)
        if not is_legal_internal_prefix(trial, n_routes):
            pieces.pop()
            break
        if trial in allowed_internal_texts(n_routes):
            complete_text = trial
            break
        step = model(mx.array([[best]], dtype=mx.int32), cache=cache)
        mx.eval(step["logits"])
        cache = step.get("cache")
        logits = step["logits"][:, -1, :]
    timings["route_decode_ms"] = (time.perf_counter() - t_dec) * 1000
    text = complete_text if complete_text is not None else tokenizer.decode(pieces)
    t_val = time.perf_counter()
    materialized = materialize_internal(text, manifest, toolset)
    timings["validate_render_ms"] = (time.perf_counter() - t_val) * 1000
    parsed = {
        "ok": bool(materialized.get("ok")),
        "function_calls": materialized.get("function_calls") or [],
        "error": materialized.get("error"),
        "route_id": materialized.get("route_id"),
    }
    return {
        "text": text,
        "parsed": parsed,
        "ids": pieces,
        "external": materialized.get("external"),
        "route_id": materialized.get("route_id"),
        "timings": timings,
    }


def greedy_byte_grammar(
    model,
    tokenizer,
    prompt_ids: list[int],
    tools: list,
    *,
    max_new: int = 96,
    cache=None,
    position_ids=None,
    cache_position_ids=None,
    engram_prefix_ids=None,
    kv=None,
) -> dict[str, Any]:
    try:
        from .byte_grammar import (
            attach_token_bytes,
            compile_byte_grammar,
            is_accept_bytes,
            mask_illegal_logits,
            token_to_bytes,
        )
    except ImportError:
        from byte_grammar import (
            attach_token_bytes,
            compile_byte_grammar,
            is_accept_bytes,
            mask_illegal_logits,
            token_to_bytes,
        )

    grammar = attach_token_bytes(compile_byte_grammar(tools), tokenizer)
    prefix_bytes = b""
    pieces: list[int] = []
    logps: list[float] = []
    if kv is not None:
        out = model(
            mx.array([kv.visible_ids], dtype=mx.int32),
            position_ids=mx.array(kv.visible_positions, dtype=mx.int32),
        )
        mx.eval(out["logits"])
        kv.absorb_packed_cache(out["cache"])
        logits = out["logits"][:, -1, :]
    else:
        ids = list(prompt_ids)
        kwargs = {}
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if cache_position_ids is not None:
            kwargs["cache_position_ids"] = cache_position_ids
        if engram_prefix_ids is not None:
            kwargs["engram_prefix_ids"] = engram_prefix_ids
        out = model(mx.array([ids], dtype=mx.int32), cache=cache, **kwargs)
        mx.eval(out["logits"])
        cache = out.get("cache")
        logits = out["logits"][:, -1, :]
    for _ in range(max_new):
        if is_accept_bytes(prefix_bytes, grammar):
            break
        logp = nn.log_softmax(logits, axis=-1)[0]
        ranked = mx.argsort(logp)[::-1].tolist()
        masked, chosen = mask_illegal_logits(logits, tokenizer, grammar, prefix_bytes, ranked)
        if chosen is None:
            break
        masked_logp = nn.log_softmax(masked, axis=-1)[0]
        chosen = int(mx.argmax(masked_logp).item())
        if float(masked_logp[chosen]) < -1e8:
            break
        if chosen == tokenizer.eos_id:
            break
        extra = token_to_bytes(tokenizer, chosen)
        prefix_bytes += extra
        pieces.append(chosen)
        logps.append(float(masked_logp[chosen]))
        if kv is not None:
            step = kv.decode_step(model, chosen)
        else:
            step = model(mx.array([[chosen]], dtype=mx.int32), cache=cache)
            mx.eval(step["logits"])
            cache = step.get("cache")
        logits = step["logits"][:, -1, :]
    text = tokenizer.decode(pieces)
    call_lp = float(sum(logps)) if logps else 0.0
    return {
        "text": text,
        "ids": pieces,
        "prefix_bytes": prefix_bytes,
        "call_logprob": call_lp,
        "mean_token_logprob": (call_lp / max(1, len(logps))),
        "cache": None if kv is not None else cache,
        "kv": kv,
    }


def greedy_unconstrained(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    max_new: int = 96,
    kv=None,
) -> dict[str, Any]:
    """Product-unconstrained greedy decode for the 58M raw baseline column."""
    pieces: list[int] = []
    logps: list[float] = []
    if kv is not None:
        logits = model(
            mx.array([kv.visible_ids[-1:]], dtype=mx.int32),
            position_ids=mx.array(kv.visible_positions[-1:], dtype=mx.int32),
        )["logits"][:, -1, :]
    else:
        out = model(mx.array([list(prompt_ids)], dtype=mx.int32))
        mx.eval(out["logits"])
        logits = out["logits"][:, -1, :]
    for _ in range(max_new):
        logp = nn.log_softmax(logits, axis=-1)[0]
        chosen = int(mx.argmax(logp).item())
        if chosen == tokenizer.eos_id or chosen == tokenizer.pad_id:
            break
        pieces.append(chosen)
        logps.append(float(logp[chosen]))
        if kv is not None:
            step = kv.decode_step(model, chosen)
            logits = step["logits"][:, -1, :]
        else:
            step = model(mx.array([[chosen]], dtype=mx.int32))
            mx.eval(step["logits"])
            logits = step["logits"][:, -1, :]
        text_now = tokenizer.decode(pieces)
        if text_now.rstrip().endswith("]") and "[" in text_now:
            break
    call_lp = float(sum(logps)) if logps else 0.0
    return {
        "text": tokenizer.decode(pieces),
        "ids": pieces,
        "call_logprob": call_lp,
        "mean_token_logprob": (call_lp / max(1, len(logps))),
        "kv": kv,
    }
