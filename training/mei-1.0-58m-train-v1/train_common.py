"""Shared CE, clipping, logging, and step loop for Needle-zh MLX training."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as xu


def flatten_tree(tree) -> dict[str, mx.array]:
    leaves = xu.tree_flatten(tree)
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    out: dict[str, mx.array] = {}
    for i, item in enumerate(leaves):
        if isinstance(item, tuple) and len(item) == 2:
            key, val = item
            out[str(key).replace("/", ".")] = val
        else:
            out[f"p{i}"] = item
    return out


def masked_lm_loss(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    logp = nn.log_softmax(logits, axis=-1)
    b, t, v = logp.shape
    idx = targets.reshape(b * t)
    lp = logp.reshape(b * t, v)[mx.arange(b * t), idx]
    w = mask.reshape(b * t)
    return -(lp * w).sum() / mx.maximum(w.sum(), 1)


def cosine_lr(step: int, total: int, base: float, final: float) -> float:
    if total <= 1:
        return final
    s = min(max(step, 0), total - 1)
    frac = 0.5 * (1.0 + math.cos(math.pi * s / (total - 1)))
    return final + (base - final) * frac


def cosine_lr_tokens(tokens_seen: int, horizon_tokens: int, base: float, final: float) -> float:
    if horizon_tokens <= 0:
        return final
    t = min(max(int(tokens_seen), 0), int(horizon_tokens))
    frac = 0.5 * (1.0 + math.cos(math.pi * t / int(horizon_tokens)))
    return final + (base - final) * frac


def clip_grads(grads, max_norm: float = 1.0):
    leaves = flatten_tree(grads)
    sq = mx.array(0.0)
    for v in leaves.values():
        sq = sq + mx.sum(v.astype(mx.float32) ** 2)
    nrm = mx.sqrt(sq)
    scale = mx.minimum(1.0, max_norm / (nrm + 1e-6))

    def _scale(x):
        if isinstance(x, mx.array):
            return (x * scale).astype(x.dtype)
        return x

    return xu.tree_map(_scale, grads), nrm


def tokens_from_mask(mask: mx.array) -> int:
    return int(mask.sum())


def stack_windows(windows: list[dict[str, Any]], *, conf: bool = False):
    x = mx.array([w["x"] for w in windows], dtype=mx.int32)
    y = mx.array([w["y"] for w in windows], dtype=mx.int32)
    mask = mx.array([w["mask"] for w in windows], dtype=mx.float32)
    out = {"x": x, "y": y, "mask": mask}
    if conf:
        out["confidence"] = mx.array([float(w.get("confidence") or 0.0) for w in windows], dtype=mx.float32)
    return out


def peak_bytes() -> int | None:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "get_peak_memory"):
        return int(metal.get_peak_memory())
    return None


def segment_throughput(tokens_seen: int, start_tokens: int, elapsed_s: float) -> float:
    return max(0.0, float(tokens_seen) - float(start_tokens)) / max(float(elapsed_s), 1e-6)


def stratified_window_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    k = min(int(k), int(n))
    if k == 1:
        return [0]
    return [i * (int(n) - 1) // (k - 1) for i in range(k)]


def eval_lm_loss(
    model,
    batches,
    *,
    conf_weight: float = 0.0,
    max_windows: int | None = 128,
    batch_size: int = 1,
    indices: list[int] | None = None,
    token_weighted: bool = False,
) -> float:
    n_src = len(batches)
    if n_src == 0:
        return float("nan")
    if indices is None:
        limit = n_src if max_windows is None else min(n_src, int(max_windows))
        idxs = list(range(limit))
    else:
        idxs = [int(i) for i in indices]
    if not idxs:
        return float("nan")
    sample0 = batches[idxs[0]]
    want_conf = conf_weight > 0 and "confidence" in sample0
    total = 0.0
    denom = 0.0
    bs = max(1, batch_size)
    for i in range(0, len(idxs), bs):
        chunk = [batches[j] for j in idxs[i : i + bs]]
        stacked = stack_windows(chunk, conf=want_conf)
        out = model(stacked["x"], return_confidence=want_conf)
        nll = masked_lm_loss(out["logits"], stacked["y"], stacked["mask"])
        if want_conf:
            c_loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(out["confidence_logit"]),
                stacked["confidence"],
            )
            nll = nll + conf_weight * c_loss.mean()
        if token_weighted:
            w = float(stacked["mask"].sum())
            total += float(nll) * w
            denom += w
        else:
            total += float(nll)
            denom += 1.0
    return total / max(denom, 1.0)


def _add_trees(a, b):
    return xu.tree_map(lambda x, y: x + y, a, b)


def _scale_tree(tree, scale: float):
    return xu.tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree)


def train_lm_steps(
    model,
    batches: list[dict[str, Any]],
    *,
    steps: int | None = None,
    lr: float,
    seed: int = 0,
    start_step: int = 0,
    start_tokens_seen: int = 0,
    total_steps: int | None = None,
    optimizer=None,
    conf_weight: float = 0.0,
    max_grad_norm: float = 1.0,
    reseed: bool = True,
    on_step: Callable[[int, dict], None] | None = None,
    batch_size: int = 1,
    grad_accum: int = 1,
    target_tokens: int | None = None,
    start_window: int = 0,
    allow_repeat: bool = True,
    compile_train: bool = False,
    precision: str = "fp32",
    horizon_tokens: int | None = None,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("no batches")
    if reseed:
        mx.random.seed(seed)
    opt = optimizer or optim.Adam(learning_rate=lr)
    last = None
    seen = int(start_tokens_seen)
    bs = max(1, batch_size)
    accum = max(1, grad_accum)
    window_i = int(start_window)
    opt_steps = 0
    micro_in_accum = 0
    acc_grads = None
    acc_loss = 0.0
    acc_tok = 0
    n_windows = len(batches)
    sample0 = batches[0]
    want_conf = conf_weight > 0 and "confidence" in sample0
    take_windows = getattr(batches, "take_windows", None)
    toks_per_update = max(1, bs * accum * max(1, len(sample0["x"])))
    precision = str(precision or "fp32").lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision {precision}")
    if target_tokens is not None:
        remain = max(0, int(target_tokens) - seen)
        estimated = max(1, (remain + toks_per_update - 1) // toks_per_update)
        horizon = max(total_steps or (start_step + estimated), 1)
        max_updates = estimated + 2
    else:
        n_steps = int(steps or 0)
        horizon = max(total_steps or (n_steps + start_step), 1)
        max_updates = max(n_steps, 1)

    def loss_fn(m, stacked):
        out = m(stacked["x"], return_confidence=want_conf)
        logits = out["logits"]
        if precision != "fp32":
            logits = logits.astype(mx.float32)
        nll = masked_lm_loss(logits, stacked["y"], stacked["mask"])
        if want_conf:
            c_loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(out["confidence_logit"]).astype(mx.float32),
                stacked["confidence"].astype(mx.float32),
            )
            return nll + conf_weight * c_loss.mean()
        return nll

    value_and_grad = nn.value_and_grad(model, loss_fn)
    compiled_vjp = None
    if compile_train:
        compile_state = [model.state]

        def _vjp(x, y, mask):
            return value_and_grad(model, {"x": x, "y": y, "mask": mask})

        compiled_vjp = partial(mx.compile, inputs=compile_state, outputs=compile_state)(_vjp)

    def take_micro():
        nonlocal window_i
        if take_windows is not None:
            chunk = take_windows(bs)
            if not chunk:
                return None
            window_i = int(getattr(batches, "consumed_windows", window_i + len(chunk)))
            return stack_windows(chunk, conf=want_conf)
        chunk = []
        for _ in range(bs):
            if not allow_repeat and window_i >= n_windows:
                break
            chunk.append(batches[window_i % n_windows])
            window_i += 1
        if not chunk:
            return None
        return stack_windows(chunk, conf=want_conf)

    def apply_update() -> None:
        nonlocal last, seen, acc_grads, acc_loss, acc_tok, micro_in_accum, opt_steps
        if acc_grads is None or acc_tok <= 0:
            acc_grads = None
            acc_loss = 0.0
            acc_tok = 0
            micro_in_accum = 0
            return
        acc_grads = _scale_tree(acc_grads, 1.0 / float(acc_tok))
        acc_grads, gn = clip_grads(acc_grads, max_grad_norm)
        opt.update(model, acc_grads)
        mx.eval(model.parameters(), opt.state)
        last = acc_loss / float(acc_tok)
        seen += acc_tok
        if on_step:
            info = {
                "loss": last,
                "grad_norm": float(gn),
                "tokens_seen_step": acc_tok,
                "tokens_seen": seen,
                "lr": float(opt.learning_rate),
                "window_index": window_i,
                "peak_bytes": peak_bytes(),
            }
            cursors = getattr(batches, "cursor", None)
            counts = getattr(batches, "window_counts", None)
            if isinstance(cursors, dict):
                info["source_cursors"] = dict(cursors)
            if isinstance(counts, dict):
                info["source_window_counts"] = dict(counts)
            drawn = getattr(batches, "source_tokens_drawn", None)
            remaining = getattr(batches, "quota_remaining", None)
            stage_drawn = getattr(batches, "stage_tokens_drawn", None)
            if isinstance(drawn, dict):
                info["source_tokens_drawn"] = dict(drawn)
            if isinstance(remaining, dict):
                info["quota_remaining"] = dict(remaining)
            if isinstance(stage_drawn, dict):
                info["stage_tokens_drawn"] = dict(stage_drawn)
            slack = getattr(batches, "alignment_slack", None)
            if isinstance(slack, dict):
                info["alignment_slack"] = dict(slack)
            overshoot = getattr(batches, "alignment_overshoot", None)
            if isinstance(overshoot, dict):
                info["alignment_overshoot"] = dict(overshoot)
            on_step(start_step + opt_steps, info)
        opt_steps += 1
        micro_in_accum = 0
        acc_grads = None
        acc_loss = 0.0
        acc_tok = 0

    exhausted = False
    while opt_steps < max_updates:
        if target_tokens is not None and seen >= int(target_tokens):
            break
        stacked = take_micro()
        if stacked is None:
            exhausted = True
            break
        if horizon_tokens:
            opt.learning_rate = cosine_lr_tokens(seen, int(horizon_tokens), lr, lr * 0.1)
        else:
            opt.learning_rate = cosine_lr(start_step + opt_steps, horizon, lr, lr * 0.1)
        if compiled_vjp is not None:
            loss, grads = compiled_vjp(stacked["x"], stacked["y"], stacked["mask"])
        else:
            loss, grads = value_and_grad(model, stacked)
        n_tok = tokens_from_mask(stacked["mask"])
        grads = _scale_tree(grads, float(max(n_tok, 1)))
        acc_grads = grads if acc_grads is None else _add_trees(acc_grads, grads)
        acc_loss += float(loss) * max(n_tok, 1)
        acc_tok += n_tok
        micro_in_accum += 1
        crossed = target_tokens is not None and (seen + acc_tok) >= int(target_tokens)
        if micro_in_accum < accum and not crossed:
            continue
        apply_update()
        if target_tokens is None and opt_steps >= max_updates:
            break
    if acc_grads is not None and acc_tok > 0:
        apply_update()
    return {
        "last_loss": last,
        "steps": opt_steps,
        "start_step": start_step,
        "tokens_seen": seen,
        "optimizer": opt,
        "window_index": window_i,
        "total_steps": horizon,
        "exhausted": exhausted,
        "allow_repeat": allow_repeat,
        "precision": precision,
        "compile_train": compile_train,
    }
