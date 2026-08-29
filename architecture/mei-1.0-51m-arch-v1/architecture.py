"""Corrected MLX SAN backbone for MEI 1.0 51M training.

Walsh-Hadamard transforms and mHC routing offsets are fixed operations, not
trainable arrays. Engram boundary masks and the 20-iteration Sinkhorn
contract match the reference SAN semantics.
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
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193
_ROPE_TABLES: dict[tuple, tuple[mx.array, mx.array]] = {}
# Weight-only packed inference does not fake-quant activations. QAT replay may
# turn this on so KV/activation STE matches quant_ops_51m int8-per-tensor.
QAT_ACTIVATION_STE = False
_ACT_STE_LEVELS = 127.0


def _ste_activation_int8(x: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    abs_max = mx.maximum(mx.max(mx.abs(xf)), 1e-8)
    q = mx.clip(mx.round(xf / abs_max * _ACT_STE_LEVELS), -_ACT_STE_LEVELS - 1, _ACT_STE_LEVELS)
    recon = q / _ACT_STE_LEVELS * abs_max
    return (recon + mx.stop_gradient(xf) - mx.stop_gradient(recon)).astype(x.dtype)


def _normal(shape: tuple[int, ...], std: float) -> mx.array:
    return mx.random.normal(shape).astype(mx.float32) * float(std)


def _init_linear(layer: nn.Linear, std: float) -> None:
    layer.weight = _normal(tuple(layer.weight.shape), std)


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


def mask_diag(mask: mx.array, offset: int) -> mx.array:
    """Return whether each position may read the requested history offset."""
    m = mask[:, 0]
    t = int(m.shape[-1])
    if offset >= t:
        return mx.zeros((int(m.shape[0]), t), dtype=m.dtype)
    diagonal = mx.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    if offset <= 0:
        return diagonal
    pad = mx.zeros((int(diagonal.shape[0]), offset), dtype=diagonal.dtype)
    return mx.concatenate([pad, diagonal], axis=-1)


def causal_mask(batch: int, tokens: int) -> mx.array:
    pos = mx.arange(tokens)
    base = (pos[:, None] >= pos[None, :])[None, None, :, :]
    return mx.broadcast_to(base, (batch, 1, tokens, tokens))


def precompute_rope(head_dim: int, seq_len: int, theta: float) -> tuple[mx.array, mx.array]:
    dims = mx.arange(0, head_dim, 2).astype(mx.float32)
    freqs = 1.0 / (theta ** (dims / head_dim))
    angles = mx.outer(mx.arange(seq_len).astype(mx.float32), freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
    position_ids: mx.array | None = None,
) -> mx.array:
    half = int(x.shape[-1]) // 2
    if position_ids is not None:
        pos = position_ids.astype(mx.int32)
        if int(pos.ndim) == 1:
            c = cos[pos][None, None, :, :]
            s = sin[pos][None, None, :, :]
        else:
            c = cos[pos][:, None, :, :]
            s = sin[pos][:, None, :, :]
    else:
        t = int(x.shape[2])
        c = cos[offset : offset + t][None, None, :, :]
        s = sin[offset : offset + t][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)


def walsh_matrix(n: int) -> mx.array:
    """Dense reference matrix for tests only; never attach it to a Module."""
    h = mx.array([[1.0]], dtype=mx.float32)
    while int(h.shape[0]) < n:
        h = mx.concatenate(
            [mx.concatenate([h, h], axis=1), mx.concatenate([h, -h], axis=1)],
            axis=0,
        )
    return h / math.sqrt(n)


def _walsh_hadamard_graph(x: mx.array) -> mx.array:
    n = int(x.shape[-1])
    if n <= 0 or n & (n - 1):
        raise ValueError("Walsh-Hadamard width must be a positive power of two")
    y = x
    width = 1
    while width < n:
        shape = (*y.shape[:-1], n // (2 * width), 2, width)
        pairs = y.reshape(shape)
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        y = mx.concatenate([left + right, left - right], axis=-1).reshape(y.shape)
        width *= 2
    return y * (1.0 / math.sqrt(n))


_WHT_512_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint index = row * 512 + tid;
    threadgroup float values[512];
    values[tid] = float(x[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 512; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[index] = T(values[tid] * 0.04419417382415922f);
"""
_wht_512_kernel = None


def _walsh_hadamard_512_metal(x: mx.array) -> mx.array:
    global _wht_512_kernel
    if _wht_512_kernel is None:
        _wht_512_kernel = mx.fast.metal_kernel(
            name="mei_51m_arch_v1_walsh_hadamard_512_f32",
            input_names=["x"],
            output_names=["out"],
            source=_WHT_512_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _wht_512_kernel(
        inputs=[x],
        template=[("T", mx.float32)],
        grid=(int(x.size), 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[tuple(x.shape)],
        output_dtypes=[x.dtype],
    )[0]


@mx.custom_function
def walsh_hadamard(x: mx.array) -> mx.array:
    """Fixed orthonormal WHT with a differentiable Metal fast path at width 512."""
    if int(x.shape[-1]) == 512 and x.dtype == mx.float32:
        return _walsh_hadamard_512_metal(x)
    return _walsh_hadamard_graph(x)


@walsh_hadamard.vjp
def _walsh_hadamard_vjp(x, cotangent, output):
    del x
    del output
    return walsh_hadamard(cotangent)


def engram_geometry(cfg: NeedleZhConfig) -> tuple[tuple[int, ...], int, int]:
    orders = tuple(cfg.engram_orders)
    heads = max(1, cfg.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = max(1, cfg.d_model // (len(orders) * heads))
    return orders, heads, sub_dim


def engram_indices(tokens: mx.array, orders: tuple[int, ...], heads: int, slots: int) -> mx.array:
    u = tokens.astype(mx.uint32)
    cols = []
    for oi, order in enumerate(orders):
        for head in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + head + 1)) & 0xFFFFFFFF
            acc = mx.full(u.shape, seed, dtype=mx.uint32)
            for j in range(order):
                shifted = shift_right(u[..., None], j)[..., 0]
                acc = (acc ^ shifted) * mx.array(_ENGRAM_PRIME, dtype=mx.uint32)
            acc = acc ^ (acc >> mx.array(15, dtype=mx.uint32))
            cols.append((acc % mx.array(slots, dtype=mx.uint32)).astype(mx.int32))
    return mx.stack(cols, axis=-1)


def sinkhorn(logits: mx.array, iters: int = 20) -> mx.array:
    log_k = logits.astype(mx.float32)
    for _ in range(int(iters)):
        log_k = log_k - mx.logsumexp(log_k, axis=-1, keepdims=True)
        log_k = log_k - mx.logsumexp(log_k, axis=-2, keepdims=True)
    return mx.exp(log_k)


def mhc_offsets(n_layers: int, lanes: int) -> tuple[mx.array, mx.array]:
    """Fixed reference routing offsets; returned arrays are not model fields."""
    lane = mx.eye(lanes)[mx.arange(n_layers) % lanes]
    return 8 * lane - 4, -4 * (1 - lane)


class ZCRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return ((1.0 + self.scale) * xf / rms).astype(x.dtype)


class FixedWalshHadamardMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n = 1 << (d_model - 1).bit_length()
        self.d1 = mx.ones((self.n,))
        self.d2 = mx.ones((self.n,))
        self.d3 = mx.full((self.n,), 0.02)

    def __call__(self, x: mx.array) -> mx.array:
        pad = self.n - self.d_model
        z = mx.pad(x, [(0, 0), (0, 0), (0, pad)]) if pad else x
        z = walsh_hadamard(self.d1.astype(z.dtype) * z)
        z = walsh_hadamard(nn.silu(self.d2.astype(z.dtype) * z))
        return (self.d3.astype(z.dtype) * z)[..., : self.d_model]


HadamardMLP = FixedWalshHadamardMLP


class GroupedAttention(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.gate_proj):
            _init_linear(layer, 0.02)
        _init_linear(self.o_proj, 0.02 / math.sqrt(2 * cfg.n_layers))
        self.q_norm = ZCRMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = ZCRMSNorm(self.head_dim, cfg.rms_eps)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        cache=None,
        rope_offset: int = 0,
        position_ids: mx.array | None = None,
        cache_write_index=None,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        gate = self.gate_proj(x)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, *rope, offset=rope_offset, position_ids=position_ids)
        k = apply_rope(k, *rope, offset=rope_offset, position_ids=position_ids)
        if cache is not None:
            if cache_write_index is not None:
                index = (
                    cache_write_index
                    if isinstance(cache_write_index, mx.array)
                    else mx.array([int(cache_write_index)])
                )
                k = mx.slice_update(cache[0], k, index, [2])
                v = mx.slice_update(cache[1], v, index, [2])
            else:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
        if QAT_ACTIVATION_STE:
            q = _ste_activation_int8(q)
            k = _ste_activation_int8(k)
            v = _ste_activation_int8(v)
        repeats = self.n_heads // self.n_kv_heads
        k_use = mx.repeat(k, repeats, axis=1) if repeats > 1 else k
        v_use = mx.repeat(v, repeats, axis=1) if repeats > 1 else v
        scores = (q * self.scale) @ mx.swapaxes(k_use, -1, -2)
        if mask is not None:
            scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (probs @ v_use).transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.o_proj(out * mx.sigmoid(gate)), (k, v)


class Engram(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        orders, heads, sub_dim = engram_geometry(cfg)
        self.orders = orders
        self.heads = heads
        self.n_tables = len(orders) * heads
        self.slots = cfg.engram_slots
        self.conv_taps = cfg.engram_conv_taps
        self.conv_dilation = max(orders)
        self.tables = _normal((self.n_tables, self.slots, sub_dim), 0.02)
        self.key_proj = nn.Linear(self.n_tables * sub_dim, cfg.d_model, bias=False)
        self.value_proj = nn.Linear(self.n_tables * sub_dim, cfg.d_model, bias=False)
        _init_linear(self.key_proj, 0.02)
        _init_linear(self.value_proj, 0.02 / math.sqrt(2 * cfg.n_layers))
        self.taps = mx.concatenate(
            [
                mx.ones((1, cfg.d_model)),
                mx.zeros((self.conv_taps - 1, cfg.d_model)),
            ],
            axis=0,
        )

    def __call__(self, tokens: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
        indices = engram_indices(tokens, self.orders, self.heads, self.slots)
        ngram_ok = mx.stack(
            [mask_diag(mask, order - 1) for order in self.orders for _ in range(self.heads)],
            axis=-1,
        )
        fetched = [
            self.tables[i][indices[:, :, i]] * ngram_ok[:, :, i, None].astype(self.tables.dtype)
            for i in range(self.n_tables)
        ]
        e = mx.concatenate(fetched, axis=-1)
        value = self.value_proj(e)
        tap_ok = [
            mask_diag(mask, j * self.conv_dilation)[..., None].astype(value.dtype)
            for j in range(self.conv_taps)
        ]
        mixed = mx.zeros_like(value)
        for j in range(self.conv_taps):
            mixed = mixed + self.taps[j] * shift_right(value, j * self.conv_dilation) * tap_ok[j]
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
        self.mlp = FixedWalshHadamardMLP(cfg.d_model)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        cache=None,
        engram_kv=None,
        site_index: int | None = None,
        rope_offset: int = 0,
        position_ids: mx.array | None = None,
        cache_write_index=None,
    ):
        if engram_kv is not None and site_index is not None:
            ek, ev = engram_kv
            key = ek[site_index]
            value = ev[site_index]
            alpha = mx.sigmoid(
                mx.sum(rms_unit(x) * rms_unit(key), axis=-1) / math.sqrt(self.d_model)
            )
            x = x + (alpha[..., None] * value.astype(mx.float32)).astype(x.dtype)
        skip = x
        attn, new_cache = self.attn(
            self.attn_norm(x),
            rope,
            mask=mask,
            cache=cache,
            rope_offset=rope_offset,
            position_ids=position_ids,
            cache_write_index=cache_write_index,
        )
        x = skip + mx.sigmoid(self.attn_gate).astype(x.dtype) * self.post_attn_norm(attn)
        return x + self.mlp(self.mlp_norm(x)), new_cache


class NeedleZh(nn.Module):
    def __init__(self, cfg: NeedleZhConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed.weight = _normal((cfg.vocab_size, cfg.d_model), 0.02)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.engrams = [Engram(cfg) for _ in cfg.engram_layers]
        n, d, layers = cfg.mhc_lanes, cfg.d_model, cfg.n_layers
        nc = n * d
        self.mhc_phi_pre = _normal((layers, nc, n), 0.02)
        self.mhc_phi_post = _normal((layers, nc, n), 0.02)
        self.mhc_phi_res = _normal((layers, nc, n * n), 0.02)
        self.mhc_b_pre = mx.zeros((layers, n))
        self.mhc_b_post = mx.zeros((layers, n))
        self.mhc_b_res = mx.broadcast_to(4.0 * mx.eye(n), (layers, n, n))
        self.mhc_a_pre = mx.full((layers,), 0.01)
        self.mhc_a_post = mx.full((layers,), 0.01)
        self.mhc_a_res = mx.full((layers,), 0.01)
        self.conf_probes = _normal((cfg.conf_probes, cfg.d_model), 0.02)
        self.conf_proj = nn.Linear(cfg.conf_probes * cfg.d_model, 1, bias=True)
        _init_linear(self.conf_proj, 0.02)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(
            cfg.d_model, cfg.vocab_size, bias=False
        )
        if self.lm_head is not None:
            _init_linear(self.lm_head, 0.02)
        self.set_inference_backend("mlx-reference")

    def set_inference_backend(self, backend: str) -> None:
        if backend != "mlx-reference":
            raise ValueError("51m fused kernels are not enabled until fixed-WHT parity passes")
        object.__setattr__(self, "_inference_backend", backend)

    def _rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        need = max(int(seq_len), 1)
        cap = max(need, int(self.cfg.max_seq_len))
        key = (int(self.cfg.head_dim), cap, float(self.cfg.rope_theta))
        table = _ROPE_TABLES.get(key)
        if table is None:
            table = precompute_rope(self.cfg.head_dim, cap, self.cfg.rope_theta)
            mx.eval(*table)
            _ROPE_TABLES[key] = table
        return table

    def _engram_stack(self, tokens: mx.array, prefix_ids: mx.array | None = None):
        if not self.engrams:
            return None
        if prefix_ids is not None and int(prefix_ids.size) > 0:
            tokens = mx.concatenate([prefix_ids, tokens], axis=-1)
        mask = causal_mask(int(tokens.shape[0]), int(tokens.shape[1]))
        pairs = [engram(tokens, mask) for engram in self.engrams]
        keys = mx.stack([key for key, _ in pairs])
        values = mx.stack([value for _, value in pairs])
        if prefix_ids is not None and int(prefix_ids.size) > 0:
            keep = int(tokens.shape[-1] - prefix_ids.shape[-1])
            keys = keys[:, :, -keep:, :]
            values = values[:, :, -keep:, :]
        return keys, values

    def __call__(
        self,
        tokens: mx.array,
        return_confidence: bool = False,
        cache=None,
        *,
        position_ids=None,
        cache_position_ids=None,
        return_cells: bool = False,
        engram_prefix_ids=None,
        cache_write_index=None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        b, t = tokens.shape
        x = self.embed(tokens) * self.embed_scale
        cells = [x] if return_cells else None
        cache_len = 0
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            cache_len = int(cache[0][0].shape[2])
        rope = self._rope(max(cache_len + t, int(cfg.max_seq_len)))
        if position_ids is not None:
            pos = (
                position_ids
                if isinstance(position_ids, mx.array)
                else mx.array(position_ids, dtype=mx.int32)
            )
            q_pos = pos
            if cache_position_ids is not None:
                cpos = (
                    cache_position_ids
                    if isinstance(cache_position_ids, mx.array)
                    else mx.array(cache_position_ids, dtype=mx.int32)
                )
                k_pos = cpos if cache_write_index is not None else mx.concatenate([cpos, pos])
            else:
                k_pos = pos
            mask = (q_pos[:, None] >= k_pos[None, :])[None, None, :, :]
            rope_offset = 0
            attn_pos = pos
        else:
            q_pos = mx.arange(t) + cache_len
            k_pos = mx.arange(cache_len + t)
            mask = (q_pos[:, None] >= k_pos[None, :])[None, None, :, :]
            rope_offset = cache_len
            attn_pos = None
        prefix = None
        if engram_prefix_ids is not None:
            prefix = (
                engram_prefix_ids
                if isinstance(engram_prefix_ids, mx.array)
                else mx.array(engram_prefix_ids, dtype=mx.int32)
            )
            if int(prefix.ndim) == 1:
                prefix = prefix[None, :]
        engram_kv = self._engram_stack(tokens, prefix)
        site_by_layer = {layer: site for site, layer in enumerate(cfg.engram_layers)}
        pre_off, post_off = mhc_offsets(cfg.n_layers, cfg.mhc_lanes)
        n = cfg.mhc_lanes
        lanes = mx.broadcast_to(x[:, :, None, :], (b, t, n, cfg.d_model))
        new_cache = []
        for i, block in enumerate(self.blocks):
            xf = lanes.astype(mx.float32)
            nx = rms_unit(lanes.reshape(b, t, n * cfg.d_model))
            hpre = mx.sigmoid(
                self.mhc_a_pre[i] * (nx @ self.mhc_phi_pre[i].astype(mx.float32))
                + self.mhc_b_pre[i]
                + pre_off[i]
            )
            u = mx.einsum("btn,btnc->btc", hpre, xf).astype(x.dtype)
            layer_cache = None if cache is None else cache[i]
            block_out, layer_next = block(
                u,
                rope,
                mask=mask,
                cache=layer_cache,
                engram_kv=engram_kv,
                site_index=site_by_layer.get(i),
                rope_offset=rope_offset,
                position_ids=attn_pos,
                cache_write_index=cache_write_index,
            )
            delta = block_out - u
            hpost = 2 * mx.sigmoid(
                self.mhc_a_post[i] * (nx @ self.mhc_phi_post[i].astype(mx.float32))
                + self.mhc_b_post[i]
                + post_off[i]
            )
            residual_logits = (
                nx @ self.mhc_phi_res[i].astype(mx.float32)
            ).reshape(b, t, n, n)
            hres = sinkhorn(
                self.mhc_a_res[i] * residual_logits + self.mhc_b_res[i],
                cfg.sinkhorn_iters,
            )
            lanes = (
                mx.einsum("btij,btjc->btic", hres, xf)
                + hpost[..., None] * delta.astype(mx.float32)[:, :, None, :]
            ).astype(x.dtype)
            new_cache.append(layer_next)
            if cells is not None:
                cells.append(mx.mean(lanes, axis=2))
        hidden = self.final_norm(mx.mean(lanes, axis=2))
        output_weight = self.embed.weight if cfg.tie_embeddings else self.lm_head.weight
        logits = hidden.astype(mx.float32) @ output_weight.T
        out: dict[str, Any] = {"logits": logits, "hidden": hidden, "cache": new_cache}
        if cells is not None:
            out["cells"] = cells
        if return_confidence:
            last = hidden[:, -1, :]
            scores = mx.softmax(
                (last.astype(mx.float32) @ self.conf_probes.T) / math.sqrt(cfg.d_model),
                axis=-1,
            )
            pooled = mx.concatenate(
                [scores[:, j : j + 1] * last for j in range(cfg.conf_probes)],
                axis=-1,
            )
            out["confidence_logit"] = self.conf_proj(pooled).squeeze(-1).astype(mx.float32)
        return out


def count_params(model: nn.Module) -> int:
    import mlx.utils as xu

    leaves = xu.tree_flatten(model.parameters())
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    total = 0
    for item in leaves:
        value = item[1] if isinstance(item, tuple) else item
        total += int(value.size)
    return total
