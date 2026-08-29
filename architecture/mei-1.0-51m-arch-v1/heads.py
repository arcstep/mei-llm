"""51M ContrastiveHead and ConfidenceV2. Not part of the frozen 51,463,797 backbone."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


def l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(mx.float32)
    return (xf / mx.maximum(mx.linalg.norm(xf, axis=-1, keepdims=True), eps)).astype(x.dtype)


class ContrastiveHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int, dim: int = 128, probes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dim = dim
        self.probes = probes
        self.tok_probes = mx.random.normal((probes, d_model)) * 0.02
        self.lay_probes = mx.random.normal((probes, d_model)) * 0.02
        self.proj = nn.Linear(probes * d_model, dim, bias=False)

    def __call__(self, cells: Sequence[mx.array] | mx.array, *, stop_gradient: bool = True) -> mx.array:
        if isinstance(cells, (list, tuple)):
            stacked = mx.stack(list(cells), axis=2)
        else:
            stacked = cells
        if stop_gradient:
            stacked = mx.stop_gradient(stacked)
        x = stacked.astype(mx.float32)
        summaries = []
        layer_mean = mx.mean(x, axis=2)
        scale = math.sqrt(self.d_model)
        for p in range(self.probes):
            tok_logits = (layer_mean @ self.tok_probes[p]) / scale
            tok_w = mx.softmax(tok_logits, axis=-1)
            pooled_t = mx.einsum("bt,btld->bld", tok_w, x)
            lay_logits = (pooled_t @ self.lay_probes[p]) / scale
            lay_w = mx.softmax(lay_logits, axis=-1)
            summaries.append(mx.einsum("bl,bld->bd", lay_w, pooled_t))
        z = mx.concatenate(summaries, axis=-1)
        return l2_normalize(self.proj(z))


class ConfidenceV2Head(nn.Module):
    def __init__(self, d_model: int, probes: int = 8):
        super().__init__()
        self.probes = probes
        self.d_model = d_model
        self.cell_probes = mx.random.normal((probes, d_model)) * 0.02
        self.proj = nn.Linear(probes * d_model, 1, bias=True)

    def __call__(self, cells: Sequence[mx.array] | mx.array) -> mx.array:
        if isinstance(cells, (list, tuple)):
            stacked = mx.stack(list(cells), axis=2)
        else:
            stacked = cells
        x = stacked.astype(mx.float32)
        last = mx.mean(x[:, -1, :, :], axis=1)
        scale = math.sqrt(self.d_model)
        scores = mx.softmax((last @ self.cell_probes.T) / scale, axis=-1)
        pooled = mx.concatenate(
            [scores[:, j : j + 1] * last for j in range(self.probes)], axis=-1
        )
        return self.proj(pooled).squeeze(-1)


def combine_confidence(head_logit: float, decode_logprob: float) -> float:
    import math as _m

    p_head = 1.0 / (1.0 + _m.exp(-float(head_logit)))
    p_dec = float(_m.exp(min(0.0, decode_logprob)))
    return min(p_head, p_dec)
