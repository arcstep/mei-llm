"""Full-call correctness head. Combine with decode probability at the gate."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


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
