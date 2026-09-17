from __future__ import annotations

"""Product heads outside the frozen 51,463,797-parameter LM contract.
（头定义：只含结构与前向，不含训练循环；训练见 model-factory/training/head_training/。）

These heads are trained only after the final LM is frozen and live in the
portable v2 tensor container.  They therefore never change the canonical
backbone parameter count.
"""


import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


MW_DISPOSITION_CLASSES = 20
NARRATION_ADAPTER_RANK = 16


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


class MWDispositionHead(nn.Module):
    """Deterministic-contract 20-way MW disposition classifier.

    The input is the model's per-cell hidden-state sequence.  We select the
    final prompt token and take an f32 mean across cells, producing exactly one
    512-dimensional feature vector per batch item.  The only trained tensors
    are ``proj.weight`` [20, 512] and ``proj.bias`` [20].

    Runtime policy remains deterministic: only class 0 (``ready_to_execute``)
    can permit execution, and even that cannot override schema, provenance,
    permission, state, or confidence gates.
    """

    def __init__(self, d_model: int, n_classes: int = MW_DISPOSITION_CLASSES):
        super().__init__()
        if int(n_classes) != MW_DISPOSITION_CLASSES:
            raise ValueError(f"MW disposition requires exactly {MW_DISPOSITION_CLASSES} classes")
        self.d_model = int(d_model)
        self.n_classes = int(n_classes)
        self.proj = nn.Linear(self.d_model, self.n_classes, bias=True)

    def __call__(
        self,
        cells: Sequence[mx.array] | mx.array,
        *,
        stop_gradient: bool = True,
    ) -> mx.array:
        if isinstance(cells, (list, tuple)):
            if not cells:
                raise ValueError("MW disposition requires at least one cell")
            stacked = mx.stack(list(cells), axis=2)
        else:
            stacked = cells
        if stop_gradient:
            stacked = mx.stop_gradient(stacked)
        if stacked.ndim != 4 or int(stacked.shape[-1]) != self.d_model:
            raise ValueError("MW disposition cells must have shape [B,T,C,d_model]")
        pooled = mx.mean(stacked[:, -1, :, :].astype(mx.float32), axis=1)
        return self.proj(pooled)


class NarrationAdapterHead(nn.Module):
    """Frozen-backbone low-rank logit residual for grounded Chinese narration.

    This sidecar is deliberately outside the canonical 51,463,797 LM weight
    contract.  It never receives an executor or Session handle; the runtime
    supplies only a sanitized verified-result prompt and accepts generated text
    only when the deterministic grounding verifier approves it.  ``rank`` defaults
    to ``NARRATION_ADAPTER_RANK`` (16) but is a train-time dial for capacity
    ablation; runtime loading always uses the canonical 16.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        rank: int = NARRATION_ADAPTER_RANK,
    ):
        super().__init__()
        if int(rank) < 1:
            raise ValueError("narration adapter rank must be positive")
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.vocab_size, bias=False)
        self.down.weight = mx.random.normal((self.rank, self.d_model)) * 0.02
        self.up.weight = mx.zeros((self.vocab_size, self.rank))

    def __call__(self, hidden: mx.array, *, stop_gradient: bool = True) -> mx.array:
        value = mx.stop_gradient(hidden) if stop_gradient else hidden
        low_rank = self.down(value.astype(mx.float32))
        return self.up(low_rank) / math.sqrt(self.rank)


def combine_confidence(head_logit: float, decode_logprob: float) -> float:
    import math as _m

    p_head = 1.0 / (1.0 + _m.exp(-float(head_logit)))
    p_dec = float(_m.exp(min(0.0, decode_logprob)))
    return min(p_head, p_dec)
