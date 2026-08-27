"""MLX Needle-zh student.

Follows cactus-needle `architecture.py` (Apache-2.0) as a read-only oracle:
ZCRMSNorm, RoPE GQA, HadamardMLP, engram, 4-lane mHC, tied LM head, confidence
head. Engineering baseline — not a claim of pure-paper SAN.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from .config import NeedleZhConfig
except ImportError:
    from config import NeedleZhConfig

ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def walsh_matrix(n: int) -> mx.array:
    h = mx.array([[1.0]], dtype=mx.float32)
    while int(h.shape[0]) < n:
        h = mx.concatenate(
            [mx.concatenate([h, h], axis=1), mx.concatenate([h, -h], axis=1)],
            axis=0,
        )
    return h / math.sqrt(n)


def rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)


def shift_right(x: mx.array, offset: int) -> mx.array:
    t = int(x.shape[-2])
    if offset <= 0:
        return x
    if offset >= t:
        return mx.zeros_like(x)
    pad = mx.zeros((*x.shape[:-2], offset, x.shape[-1]), dtype=x.dtype)
    return mx.concatenate([pad, x[..., : t - offset, :]], axis=-2)


def precompute_rope(head_dim: int, seq_len: int, theta: float) -> tuple[mx.array, mx.array]:
    dims = mx.arange(0, head_dim, 2).astype(mx.float32)
    freqs = 1.0 / (theta ** (dims / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    angles = mx.outer(t, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array, offset: int = 0, position_ids: mx.array | None = None) -> mx.array:
    half = x.shape[-1] // 2
    if position_ids is not None:
        pos = position_ids.astype(mx.int32)
        if int(pos.ndim) == 1:
            c = cos[pos][None, None, :, :]
            s = sin[pos][None, None, :, :]
        else:
            c = cos[pos][:, None, :, :]
            s = sin[pos][:, None, :, :]
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
    t = x.shape[2]
    cos = cos[offset : offset + t][None, None, :, :]
    sin = sin[offset : offset + t][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def engram_geometry(cfg: NeedleZhConfig) -> tuple[tuple[int, ...], int, int]:
    orders = tuple(cfg.engram_orders)
    heads = max(1, cfg.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = max(1, cfg.d_model // (len(orders) * heads))
    return orders, heads, sub_dim


def engram_indices(tokens: mx.array, orders: tuple[int, ...], heads: int, slots: int) -> mx.array:
    u = tokens.astype(mx.uint32)
    cols = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = mx.full(u.shape, seed, dtype=mx.uint32)
            for j in range(order):
                shifted = shift_right(u[..., None], j)[..., 0]
                acc = (acc ^ shifted) * mx.array(_ENGRAM_PRIME, dtype=mx.uint32)
            acc = acc ^ (acc >> mx.array(15, dtype=mx.uint32))
            cols.append((acc % mx.array(slots, dtype=mx.uint32)).astype(mx.int32))
    return mx.stack(cols, axis=-1)


def sinkhorn(logits: mx.array, iters: int = 8) -> mx.array:
    log_k = logits.astype(mx.float32)
    for _ in range(iters):
        log_k = log_k - mx.logsumexp(log_k, axis=-1, keepdims=True)
        log_k = log_k - mx.logsumexp(log_k, axis=-2, keepdims=True)
    return mx.exp(log_k)


class ZCRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return ((1.0 + self.scale) * xf / rms).astype(x.dtype)


class HadamardMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n = 1 << (d_model - 1).bit_length()
        self.H = walsh_matrix(self.n)
        self.d1 = mx.ones((self.n,))
        self.d2 = mx.ones((self.n,))
        self.d3 = mx.full((self.n,), 0.02)

    def __call__(self, x: mx.array) -> mx.array:
        pad = self.n - self.d_model
        z = mx.pad(x, [(0, 0), (0, 0), (0, pad)]) if pad else x
        h = self.H.astype(z.dtype)
        z = (self.d1.astype(z.dtype) * z) @ h
        z = nn.silu(self.d2.astype(z.dtype) * z) @ h
        return (self.d3.astype(z.dtype) * z)[..., : self.d_model]


class GroupedAttention(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)
        self.q_norm = ZCRMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = ZCRMSNorm(self.head_dim, cfg.rms_eps)

    def __call__(self, x, rope, mask=None, cache=None, rope_offset: int = 0, position_ids=None):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, *rope, offset=rope_offset, position_ids=position_ids)
        k = apply_rope(k, *rope, offset=rope_offset, position_ids=position_ids)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        repeats = self.n_heads // self.n_kv_heads
        k_use = mx.repeat(k, repeats, axis=1) if repeats > 1 else k
        v_use = mx.repeat(v, repeats, axis=1) if repeats > 1 else v
        attn = (q * self.scale) @ mx.swapaxes(k_use, -1, -2)
        if mask is not None:
            attn = mx.where(mask, attn, mx.array(-1e9, dtype=attn.dtype))
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v_use).transpose(0, 2, 1, 3).reshape(b, t, -1)
        out = self.o_proj(out * mx.sigmoid(self.gate_proj(x)))
        return out, (k, v)


class Engram(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        orders, heads, sub_dim = engram_geometry(cfg)
        self.orders = orders
        self.heads = heads
        self.n_tables = len(orders) * heads
        self.slots = cfg.engram_slots
        self.tables = mx.random.normal((self.n_tables, self.slots, sub_dim)) * 0.02
        self.key_proj = nn.Linear(self.n_tables * sub_dim, cfg.d_model, bias=False)
        self.value_proj = nn.Linear(self.n_tables * sub_dim, cfg.d_model, bias=False)
        taps = mx.zeros((ENGRAM_CONV_TAPS, cfg.d_model))
        taps = mx.concatenate([mx.ones((1, cfg.d_model)), mx.zeros((ENGRAM_CONV_TAPS - 1, cfg.d_model))], axis=0)
        self.taps = taps
        self.conv_dilation = max(orders)

    def __call__(self, tokens: mx.array) -> tuple[mx.array, mx.array]:
        idx = engram_indices(tokens, self.orders, self.heads, self.slots)
        fetched = [self.tables[i][idx[:, :, i]] for i in range(self.n_tables)]
        e = mx.concatenate(fetched, axis=-1)
        v = self.value_proj(e)
        mixed = mx.zeros_like(v)
        for j in range(ENGRAM_CONV_TAPS):
            mixed = mixed + self.taps[j] * shift_right(v, j * self.conv_dilation)
        return self.key_proj(e), mixed


class Block(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.attn_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GroupedAttention(cfg)
        self.post_attn_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn_gate = mx.array(0.0)
        self.mlp_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = HadamardMLP(cfg.d_model)

    def __call__(self, x, rope, mask=None, cache=None, engram_kv=None, site_flags=None, rope_offset: int = 0, position_ids=None):
        if engram_kv is not None and site_flags is not None:
            ek, ev = engram_kv
            alpha = mx.sigmoid(
                mx.einsum("btd,sbtd->sbt", rms_unit(x), rms_unit(ek)) / math.sqrt(self.d_model)
            )
            x = x + mx.einsum(
                "s,sbt,sbtd->btd",
                site_flags.astype(mx.float32),
                alpha,
                ev.astype(mx.float32),
            ).astype(x.dtype)
        skip = x
        x, new_cache = self.attn(
            self.attn_norm(x), rope, mask=mask, cache=cache, rope_offset=rope_offset, position_ids=position_ids
        )
        x = skip + mx.sigmoid(self.attn_gate).astype(x.dtype) * self.post_attn_norm(x)
        skip = x
        x = skip + self.mlp(self.mlp_norm(x))
        return x, new_cache


class NeedleZh(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.engrams = [Engram(cfg) for _ in cfg.engram_layers]
        n, d, L = cfg.mhc_lanes, cfg.d_model, cfg.n_layers
        nc = n * d
        self.mhc_phi_pre = mx.random.normal((L, nc, n)) * 0.02
        self.mhc_phi_post = mx.random.normal((L, nc, n)) * 0.02
        self.mhc_phi_res = mx.random.normal((L, nc, n * n)) * 0.02
        self.mhc_b_pre = mx.zeros((L, n))
        self.mhc_b_post = mx.zeros((L, n))
        self.mhc_b_res = mx.broadcast_to(4.0 * mx.eye(n), (L, n, n))
        self.mhc_a_pre = mx.full((L,), 0.01)
        self.mhc_a_post = mx.full((L,), 0.01)
        self.mhc_a_res = mx.full((L,), 0.01)
        lane = mx.eye(n)[mx.arange(L) % n]
        self.mhc_pre_off = 8 * lane - 4
        self.mhc_post_off = -4 * (1 - lane)
        self.conf_probes = mx.random.normal((cfg.conf_probes, cfg.d_model)) * 0.02
        self.conf_proj = nn.Linear(cfg.conf_probes * cfg.d_model, 1, bias=True)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.contrastive = None
        self.conf_v2 = None
        if getattr(cfg, "contrastive_head_v2", False):
            try:
                from .contrastive_head import ContrastiveHead
            except ImportError:
                from contrastive_head import ContrastiveHead

            self.contrastive = ContrastiveHead(
                cfg.d_model, cfg.n_layers, dim=cfg.contrastive_dim, probes=cfg.contrastive_probes
            )
        if getattr(cfg, "confidence_v2", False):
            try:
                from .confidence_v2 import ConfidenceV2Head
            except ImportError:
                from confidence_v2 import ConfidenceV2Head

            self.conf_v2 = ConfidenceV2Head(cfg.d_model, probes=cfg.conf_probes)

    def _rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        return precompute_rope(self.cfg.head_dim, seq_len, self.cfg.rope_theta)

    def _engram_stack(self, tokens: mx.array, prefix_ids: mx.array | None = None):
        if not self.engrams:
            return None
        if prefix_ids is not None and int(prefix_ids.size) > 0:
            tokens = mx.concatenate([prefix_ids, tokens], axis=-1)
        pairs = [e(tokens) for e in self.engrams]
        keys = mx.stack([k for k, _ in pairs])
        vals = mx.stack([v for _, v in pairs])
        if prefix_ids is not None and int(prefix_ids.size) > 0:
            keep = int(tokens.shape[-1] - prefix_ids.shape[-1])
            keys = keys[:, :, -keep:, :]
            vals = vals[:, :, -keep:, :]
        return keys, vals

    def __call__(
        self,
        tokens: mx.array,
        return_confidence: bool = False,
        cache=None,
        *,
        position_ids=None,
        cache_position_ids=None,
        return_cells: bool = False,
        return_contrastive: bool = False,
        engram_prefix_ids=None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        b, t = tokens.shape
        x = self.embed(tokens) * self.embed_scale
        cells = [x] if (return_cells or return_contrastive or (self.conf_v2 is not None and return_confidence)) else None
        cache_len = 0
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            cache_len = int(cache[0][0].shape[2])
        if position_ids is not None:
            pos = position_ids if isinstance(position_ids, mx.array) else mx.array(position_ids, dtype=mx.int32)
            max_pos = int(mx.max(pos).item()) + 1
            if cache_position_ids is not None:
                cpos = cache_position_ids if isinstance(cache_position_ids, mx.array) else mx.array(cache_position_ids, dtype=mx.int32)
                max_pos = max(max_pos, int(mx.max(cpos).item()) + 1)
            rope = self._rope(max_pos)
            q_pos = pos
            k_pos = mx.concatenate([cpos, pos]) if cache_position_ids is not None else pos
            mask = (q_pos[:, None] >= k_pos[None, :])[None, None, :, :]
            rope_offset = 0
            attn_pos = pos
        else:
            rope = self._rope(cache_len + t)
            q_pos = mx.arange(t) + cache_len
            k_pos = mx.arange(cache_len + t)
            mask = (q_pos[:, None] >= k_pos[None, :])[None, None, :, :]
            rope_offset = cache_len
            attn_pos = None
        prefix = None
        if engram_prefix_ids is not None:
            prefix = engram_prefix_ids if isinstance(engram_prefix_ids, mx.array) else mx.array([engram_prefix_ids], dtype=mx.int32)
            if prefix.ndim == 1:
                prefix = prefix[None, :]
        engram_kv = self._engram_stack(tokens, prefix)
        n = cfg.mhc_lanes
        lanes = mx.broadcast_to(x[:, :, None, :], (b, t, n, cfg.d_model))
        new_cache = []
        for i, block in enumerate(self.blocks):
            xf = lanes.astype(mx.float32)
            nx = rms_unit(lanes.reshape(b, t, n * cfg.d_model))
            hpre = mx.sigmoid(
                self.mhc_a_pre[i] * (nx @ self.mhc_phi_pre[i].astype(mx.float32))
                + self.mhc_b_pre[i]
                + self.mhc_pre_off[i]
            )
            u = mx.einsum("btn,btnc->btc", hpre, xf).astype(x.dtype)
            site = e_kv = None
            if engram_kv is not None:
                site = mx.array([1.0 if layer == i else 0.0 for layer in cfg.engram_layers])
                e_kv = engram_kv
            layer_cache = None if cache is None else cache[i]
            y, lc = block(
                u,
                rope,
                mask=mask,
                cache=layer_cache,
                engram_kv=e_kv,
                site_flags=site,
                rope_offset=rope_offset,
                position_ids=attn_pos,
            )
            y = y - u
            hpost = 2 * mx.sigmoid(
                self.mhc_a_post[i] * (nx @ self.mhc_phi_post[i].astype(mx.float32))
                + self.mhc_b_post[i]
                + self.mhc_post_off[i]
            )
            res = (nx @ self.mhc_phi_res[i].astype(mx.float32)).reshape(b, t, n, n)
            hres = sinkhorn(self.mhc_a_res[i] * res + self.mhc_b_res[i])
            lanes = (
                mx.einsum("btij,btjc->btic", hres, xf)
                + hpost[..., None] * y.astype(mx.float32)[:, :, None, :]
            ).astype(x.dtype)
            new_cache.append(lc)
            if cells is not None:
                cells.append(mx.mean(lanes, axis=2))
        x = self.final_norm(mx.mean(lanes, axis=2))
        wte = self.embed.weight if self.cfg.tie_embeddings else self.lm_head.weight
        logits = x.astype(mx.float32) @ wte.T
        out: dict[str, Any] = {"logits": logits, "hidden": x, "cache": new_cache}
        if cells is not None:
            out["cells"] = cells
        if return_contrastive and self.contrastive is not None:
            out["contrastive"] = self.contrastive(
                cells, stop_gradient=bool(getattr(cfg, "contrastive_stop_gradient", True))
            )
        if return_confidence:
            last = x[:, -1, :]
            scores = mx.softmax(
                (last.astype(mx.float32) @ self.conf_probes.T) / math.sqrt(cfg.d_model),
                axis=-1,
            )
            pooled = mx.concatenate(
                [scores[:, j : j + 1] * last for j in range(cfg.conf_probes)], axis=-1
            )
            out["confidence_logit"] = self.conf_proj(pooled).squeeze(-1).astype(mx.float32)
            if self.conf_v2 is not None and cells is not None:
                out["confidence_v2_logit"] = self.conf_v2(cells)
        return out


def count_params(model: nn.Module) -> int:
    import mlx.utils as xu

    n = 0
    leaves = xu.tree_flatten(model.parameters())
    # mlx.utils.tree_flatten returns (list[(k,v)], treedef) or list
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    for item in leaves:
        v = item[1] if isinstance(item, tuple) else item
        n += int(v.size)
    return n
