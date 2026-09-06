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
    from .fused_ops import (
        fused_block_mhc_post,
        fused_engram_two_full_taps,
        fused_fixed_wht_mlp,
        fused_gqa,
        fused_mhc_pre,
        fused_qk_norm_rope,
        fused_zcrms_norm,
        supports_decode_fusion,
    )
except ImportError:
    from config import NeedleZhConfig
    from fused_ops import (
        fused_block_mhc_post,
        fused_engram_two_full_taps,
        fused_fixed_wht_mlp,
        fused_gqa,
        fused_mhc_pre,
        fused_qk_norm_rope,
        fused_zcrms_norm,
        supports_decode_fusion,
    )

ENGRAM_SUB_DIM = 128
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193
_ROPE_TABLES: dict[tuple, tuple[mx.array, mx.array]] = {}
# Weight-only packed inference does not fake-quant activations. QAT replay may
# turn this on to match Needle's per-vector activation Q/DQ and MEI's explicit
# int8-KV extension.  Leading rows must never share a scale: doing so makes a
# token's quantized value depend on unrelated batch/sequence positions.
QAT_ACTIVATION_STE = False
_ACT_STE_LEVELS = 127.0
ACTIVATION_STE_SEMANTICS = "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2"
ACTIVATION_STE_POINTS = (
    "engram_input",
    "attention_input",
    "attention_output",
    "lm_head_input",
)
KV_STE_SEMANTICS = "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1"
KV_STE_POINTS = ("attention_key_after_rope", "attention_value")


def _ste_activation_int8(x: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    abs_max = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = mx.where(abs_max > 0, abs_max / _ACT_STE_LEVELS, 1.0)
    q = mx.clip(mx.round(xf / scale), -_ACT_STE_LEVELS - 1, _ACT_STE_LEVELS)
    recon = q * scale
    # Forward uses the actual int8 quantize/dequantize value; backward is the
    # identity STE.  The inverse ordering would silently train on float
    # activations while only differentiating through the quantizer.
    return (xf + mx.stop_gradient(recon - xf)).astype(x.dtype)


def _quantize_int8_vectors(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Return int8 codes, per-last-axis scales and float32 reconstruction."""

    xf = x.astype(mx.float32)
    abs_max = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = mx.where(abs_max > 0, abs_max / _ACT_STE_LEVELS, 1.0).astype(mx.float32)
    codes = mx.clip(
        mx.round(xf / scale),
        -_ACT_STE_LEVELS - 1,
        _ACT_STE_LEVELS,
    ).astype(mx.int8)
    reconstructed = codes.astype(mx.float32) * scale
    return codes, scale, reconstructed


def _activation_qdq_enabled(module: Any) -> bool:
    return bool(QAT_ACTIVATION_STE or getattr(module, "_mei_activation_int8", False))


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
        kwargs = {
            "name": "mei_51m_arch_v1_walsh_hadamard_512_f32",
            "input_names": ["x"],
            "output_names": ["out"],
            "source": _WHT_512_SOURCE,
        }
        try:
            _wht_512_kernel = mx.fast.metal_kernel(
                **kwargs,
                compile_options={"math_mode": "safe"},
            )
        except TypeError:
            # MLX 0.31 has no compile_options argument; the default is the
            # required safe-math mode.  Keep 0.32+ explicit for receipts.
            _wht_512_kernel = mx.fast.metal_kernel(**kwargs)
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
        if getattr(self, "_mei_fused", False):
            return fused_zcrms_norm(x, self.scale, self.eps)
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
        if (
            getattr(self, "_mei_fused", False)
            and self.d_model == 512
            and tuple(x.shape) == (1, 1, 512)
            and x.dtype == mx.float32
        ):
            return fused_fixed_wht_mlp(x, self.d1, self.d2, self.d3)
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
        zcrms_norm: ZCRMSNorm | None = None,
    ):
        b, t, _ = x.shape
        cq_qkvg = getattr(self, "_mei_cq_qkvg", None)
        cq_o_proj = getattr(self, "_mei_cq_o_proj", None)
        activation_qdq = _activation_qdq_enabled(self)
        kv_int8_storage = bool(getattr(self, "_mei_kv_int8", False))
        if zcrms_norm is not None:
            if cq_qkvg is not None and not activation_qdq:
                projected = x
                prepared = cq_qkvg.prepare_zcrms(x, zcrms_norm.scale)
            else:
                projected = zcrms_norm(x)
                projected = _ste_activation_int8(projected) if activation_qdq else projected
                prepared = None
        else:
            projected = _ste_activation_int8(x) if activation_qdq else x
            prepared = None
        if cq_qkvg is not None:
            projected_all = (
                cq_qkvg.linear_prepared(prepared)
                if prepared is not None
                else cq_qkvg.linear(projected)
            )
            q_raw, k_raw, v_raw, gate = mx.split(
                projected_all,
                [512, 768, 1024],
                axis=-1,
            )
            if tuple(projected.shape) == (1, 1, 512):
                position = (
                    position_ids
                    if position_ids is not None
                    else mx.array([rope_offset], dtype=mx.int32)
                )
                q, k = fused_qk_norm_rope(
                    q_raw,
                    k_raw,
                    self.q_norm.scale,
                    self.k_norm.scale,
                    rope,
                    position,
                )
            else:
                q = q_raw.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
                k = k_raw.reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
                q, k = self.q_norm(q), self.k_norm(k)
                q = apply_rope(q, *rope, offset=rope_offset, position_ids=position_ids)
                k = apply_rope(k, *rope, offset=rope_offset, position_ids=position_ids)
            v = v_raw.reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        packed_projection = (
            cq_qkvg is None
            and getattr(self, "_mei_fused", False)
            and tuple(projected.shape) == (1, 1, 512)
            and self.n_heads == 8
            and self.n_kv_heads == 4
            and self.head_dim == 64
            and projected.dtype == mx.float32
            and getattr(self, "_mei_packed_qkvg", None) is not None
        )
        if cq_qkvg is not None:
            pass
        elif packed_projection:
            q_raw, k_raw, v_raw, gate = mx.split(
                projected @ self._mei_packed_qkvg.T,
                [512, 768, 1024],
                axis=-1,
            )
            position = (
                position_ids
                if position_ids is not None
                else mx.array([rope_offset], dtype=mx.int32)
            )
            q, k = fused_qk_norm_rope(
                q_raw,
                k_raw,
                self.q_norm.scale,
                self.k_norm.scale,
                rope,
                position,
            )
            v = v_raw.reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        else:
            q = self.q_proj(projected).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(projected).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(projected).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            gate = self.gate_proj(projected)
            q, k = self.q_norm(q), self.k_norm(k)
            q = apply_rope(q, *rope, offset=rope_offset, position_ids=position_ids)
            k = apply_rope(k, *rope, offset=rope_offset, position_ids=position_ids)
        index = None
        if cache_write_index is not None:
            index = (
                cache_write_index
                if isinstance(cache_write_index, mx.array)
                else mx.array([int(cache_write_index)])
            )
        if QAT_ACTIVATION_STE and not kv_int8_storage:
            # Training keeps the cache in floating storage, but its forward
            # values must still be the exact per-vector Q/DQ reconstruction.
            # Quantize the newly produced rows before merging them so the
            # returned cache cannot accidentally preserve pre-Q/DQ k/v.
            k = _ste_activation_int8(k)
            v = _ste_activation_int8(v)
        if kv_int8_storage:
            k_codes, k_scales, k = _quantize_int8_vectors(k)
            v_codes, v_scales, v = _quantize_int8_vectors(v)
            if cache is not None:
                if len(cache) != 4:
                    raise ValueError("int8 KV runtime requires code/scale cache tuples")
                old_k_codes, old_k_scales, old_v_codes, old_v_scales = cache
                if index is not None:
                    k_codes = mx.slice_update(old_k_codes, k_codes, index, [2])
                    k_scales = mx.slice_update(old_k_scales, k_scales, index, [2])
                    v_codes = mx.slice_update(old_v_codes, v_codes, index, [2])
                    v_scales = mx.slice_update(old_v_scales, v_scales, index, [2])
                else:
                    k_codes = mx.concatenate([old_k_codes, k_codes], axis=2)
                    k_scales = mx.concatenate([old_k_scales, k_scales], axis=2)
                    v_codes = mx.concatenate([old_v_codes, v_codes], axis=2)
                    v_scales = mx.concatenate([old_v_scales, v_scales], axis=2)
                k = k_codes.astype(mx.float32) * k_scales
                v = v_codes.astype(mx.float32) * v_scales
            next_cache = (k_codes, k_scales, v_codes, v_scales)
        else:
            if cache is not None:
                if len(cache) != 2:
                    raise ValueError("float KV runtime requires key/value cache tuples")
                if index is not None:
                    k = mx.slice_update(cache[0], k, index, [2])
                    v = mx.slice_update(cache[1], v, index, [2])
                else:
                    k = mx.concatenate([cache[0], k], axis=2)
                    v = mx.concatenate([cache[1], v], axis=2)
            next_cache = (k, v)
        if getattr(self, "_mei_fused", False):
            out = fused_gqa(q, k, v, scale=self.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        else:
            repeats = self.n_heads // self.n_kv_heads
            k_use = mx.repeat(k, repeats, axis=1) if repeats > 1 else k
            v_use = mx.repeat(v, repeats, axis=1) if repeats > 1 else v
            scores = (q * self.scale) @ mx.swapaxes(k_use, -1, -2)
            if mask is not None:
                scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
            out = (probs @ v_use).transpose(0, 2, 1, 3).reshape(b, t, -1)
        if cq_o_proj is not None and not activation_qdq:
            projected_out = cq_o_proj.linear_prepared(cq_o_proj.prepare_gated(out, gate))
        else:
            out = out * mx.sigmoid(gate)
            if activation_qdq:
                out = _ste_activation_int8(out)
            projected_out = (
                cq_o_proj.linear(out)
                if cq_o_proj is not None
                else self.o_proj(out)
            )
        return projected_out, next_cache


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
        cq_tables = getattr(self, "_mei_cq_tables", None)
        activation_qdq = _activation_qdq_enabled(self)
        if cq_tables is not None:
            offsets = getattr(self, "_mei_cq_table_offsets")
            rows = indices + offsets
            if activation_qdq:
                fetched = cq_tables.gather_rows(rows)
                fetched = fetched * ngram_ok[..., None].astype(fetched.dtype)
                e = fetched.reshape(
                    (*fetched.shape[:-2], self.n_tables * int(fetched.shape[-1]))
                )
                e = _ste_activation_int8(e)
                transformed_e = None
            else:
                transformed = cq_tables.gather_transformed_rows(rows)
                transformed = transformed * ngram_ok[..., None].astype(transformed.dtype)
                transformed_e = transformed.reshape(
                    (*transformed.shape[:-2], self.n_tables * int(transformed.shape[-1]))
                )
                e = None
        else:
            fetched = [
                self.tables[i][indices[:, :, i]]
                * ngram_ok[:, :, i, None].astype(self.tables.dtype)
                for i in range(self.n_tables)
            ]
            e = mx.concatenate(fetched, axis=-1)
            transformed_e = None
        if activation_qdq and e is not None and cq_tables is None:
            e = _ste_activation_int8(e)
        cq_key_value = getattr(self, "_mei_cq_key_value", None)
        if cq_key_value is not None:
            key, value = mx.split(
                (
                    cq_key_value.linear(e)
                    if e is not None
                    else cq_key_value.linear_transformed(transformed_e)
                ),
                [512],
                axis=-1,
            )
        else:
            key = self.key_proj(e)
            value = self.value_proj(e)
        tap_ok = [
            mask_diag(mask, j * self.conv_dilation)[..., None].astype(value.dtype)
            for j in range(self.conv_taps)
        ]
        mixed = mx.zeros_like(value)
        for j in range(self.conv_taps):
            mixed = mixed + self.taps[j] * shift_right(value, j * self.conv_dilation) * tap_ok[j]
        return key, mixed

    def packed_decode_full_taps(
        self,
        tokens: mx.array,
        selected_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Evaluate only the four history positions consumed by decode.

        The deployed decode contract supplies 12 history tokens plus the
        current token.  Dilation three therefore consumes positions
        12, 9, 6 and 3; every order-2/3 ngram is valid at those positions.
        Keeping the values in stored WHT space avoids reconstructing table
        rows and avoids evaluating the nine positions that are sliced away.
        """

        cq_tables = getattr(self, "_mei_cq_tables", None)
        cq_key_value = getattr(self, "_mei_cq_key_value", None)
        if (
            cq_tables is None
            or cq_key_value is None
            or tuple(tokens.shape) != (1, 13)
            or self.conv_taps != 4
            or self.conv_dilation != 3
        ):
            raise ValueError("packed Engram decode requires the exact 4-tap history contract")
        if selected_indices is None:
            indices = engram_indices(tokens, self.orders, self.heads, self.slots)
            positions = mx.array([12, 9, 6, 3], dtype=mx.int32)
            selected_indices = mx.take(indices, positions, axis=1)
        if tuple(selected_indices.shape) != (1, 4, self.n_tables):
            raise ValueError("packed Engram selected-index shape mismatch")
        rows = selected_indices + getattr(self, "_mei_cq_table_offsets")
        if _activation_qdq_enabled(self):
            fetched = cq_tables.gather_rows(rows)
            e = fetched.reshape(1, self.conv_taps, self.n_tables * ENGRAM_SUB_DIM)
            e = _ste_activation_int8(e)
            projected = cq_key_value.linear(e)
        else:
            transformed = cq_tables.gather_transformed_rows(rows)
            transformed_e = transformed.reshape(
                1, self.conv_taps, self.n_tables * ENGRAM_SUB_DIM
            )
            projected = cq_key_value.linear_transformed(transformed_e)
        key_all, value_all = mx.split(
            projected,
            [512],
            axis=-1,
        )
        key = key_all[:, :1, :]
        mixed = mx.sum(
            value_all * self.taps[None, :, :].astype(value_all.dtype),
            axis=1,
            keepdims=True,
        )
        return key, mixed


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
        mhc_post_context=None,
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
        cq_attention = getattr(self.attn, "_mei_cq_qkvg", None) is not None
        attn, new_cache = self.attn(
            x if cq_attention else self.attn_norm(x),
            rope,
            mask=mask,
            cache=cache,
            rope_offset=rope_offset,
            position_ids=position_ids,
            cache_write_index=cache_write_index,
            zcrms_norm=self.attn_norm if cq_attention else None,
        )
        if (
            mhc_post_context is not None
            and tuple(skip.shape) == (1, 1, 512)
            and skip.dtype == mx.float32
        ):
            lanes, hpost, hres, original_u = mhc_post_context
            return (
                fused_block_mhc_post(
                    skip,
                    attn,
                    self.attn_gate,
                    self.post_attn_norm.scale,
                    self.mlp_norm.scale,
                    self.mlp.d1,
                    self.mlp.d2,
                    self.mlp.d3,
                    original_u,
                    lanes,
                    hpost,
                    hres,
                ),
                new_cache,
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
        object.__setattr__(self, "_mei_cq_embed", None)
        self.set_inference_backend("mlx-reference")

    def install_packed_cq2(self, packed: dict[str, Any]) -> None:
        """Install the resident packed subset without changing logical identity."""

        expected = {"embed.weight"}
        for index in range(int(self.cfg.n_layers)):
            expected.update(
                {
                    f"blocks.{index}.attn.{name}.weight"
                    for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
                }
            )
        for index in range(len(self.engrams)):
            expected.update(
                {
                    f"engrams.{index}.tables",
                    f"engrams.{index}.key_proj.weight",
                    f"engrams.{index}.value_proj.weight",
                }
            )
        missing = sorted(expected - set(packed))
        unexpected = sorted(set(packed) - expected)
        if missing or unexpected:
            detail = missing[:1] or unexpected[:1]
            raise ValueError(f"packed CQ2 topology mismatch: {detail[0]}")
        object.__setattr__(self, "_mei_cq_embed", packed["embed.weight"])
        resident = [packed["embed.weight"]]
        self.embed.weight = mx.zeros((1,), dtype=mx.float16)
        for index, block in enumerate(self.blocks):
            weights = {
                name: packed[f"blocks.{index}.attn.{name}.weight"]
                for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
            }
            qkvg = type(weights["q_proj"]).concatenate_rows(
                f"blocks.{index}.attn.qkvg.weight",
                [
                    weights["q_proj"],
                    weights["k_proj"],
                    weights["v_proj"],
                    weights["gate_proj"],
                ],
            )
            qkvg.enable_q2_bitplane_linear(drop_codes=True)
            weights["o_proj"].enable_q2_bitplane_linear(drop_codes=True)
            object.__setattr__(block.attn, "_mei_cq_qkvg", qkvg)
            object.__setattr__(block.attn, "_mei_cq_o_proj", weights["o_proj"])
            resident.extend([qkvg, weights["o_proj"]])
            for name in weights:
                getattr(block.attn, name).weight = mx.zeros((1,), dtype=mx.float16)
        for index, engram in enumerate(self.engrams):
            table = packed[f"engrams.{index}.tables"]
            object.__setattr__(engram, "_mei_cq_tables", table)
            key_value = type(packed[f"engrams.{index}.key_proj.weight"]).concatenate_rows(
                f"engrams.{index}.key_value.weight",
                [
                    packed[f"engrams.{index}.key_proj.weight"],
                    packed[f"engrams.{index}.value_proj.weight"],
                ],
            )
            key_value.enable_q2_bitplane_linear(drop_codes=True)
            object.__setattr__(engram, "_mei_cq_key_value", key_value)
            resident.extend([table, key_value])
            object.__setattr__(
                engram,
                "_mei_cq_table_offsets",
                mx.arange(engram.n_tables, dtype=mx.int32) * int(engram.slots),
            )
            engram.tables = mx.zeros((1,), dtype=mx.float16)
            engram.key_proj.weight = mx.zeros((1,), dtype=mx.float16)
            engram.value_proj.weight = mx.zeros((1,), dtype=mx.float16)
        object.__setattr__(self, "_logical_param_count", 51_463_797)
        object.__setattr__(
            self,
            "_mei_cq_resident_bytes",
            sum(int(value.resident_bytes) for value in resident),
        )
        mx.eval(self.parameters())

    def _set_native_v2_quantization(self, enabled: bool) -> None:
        """Apply the deployment activation/KV profile independently of weights.

        Keeping this separate from packed CQ2 installation lets unit tests and
        float-oracle diagnostics exercise the exact native-v2 cache contract.
        Product loading still selects it only through ``mlx-cq2``.
        """

        active = bool(enabled)
        object.__setattr__(self, "_mei_activation_int8", active)
        object.__setattr__(self, "_mei_kv_int8", active)
        for block in self.blocks:
            object.__setattr__(block.attn, "_mei_activation_int8", active)
            object.__setattr__(block.attn, "_mei_kv_int8", active)
        for engram in self.engrams:
            object.__setattr__(engram, "_mei_activation_int8", active)

    def set_inference_backend(self, backend: str) -> None:
        if backend not in {"mlx-reference", "mlx-fused", "mlx-cq2"}:
            raise ValueError(f"unknown 51m inference backend: {backend}")
        cq2 = backend == "mlx-cq2"
        if cq2 and self._mei_cq_embed is None:
            raise ValueError("mlx-cq2 requires resident packed weights")
        fused = backend in {"mlx-fused", "mlx-cq2"}
        object.__setattr__(self, "_inference_backend", backend)
        self._set_native_v2_quantization(cq2)
        object.__setattr__(self.final_norm, "_mei_fused", fused)
        for block in self.blocks:
            object.__setattr__(block.attn_norm, "_mei_fused", fused)
            object.__setattr__(block.post_attn_norm, "_mei_fused", fused)
            object.__setattr__(block.mlp_norm, "_mei_fused", fused)
            object.__setattr__(block.mlp, "_mei_fused", fused)
            object.__setattr__(block.attn, "_mei_fused", fused)
            packed_qkvg = None
            if backend == "mlx-fused":
                packed_qkvg = mx.concatenate(
                    [
                        block.attn.q_proj.weight,
                        block.attn.k_proj.weight,
                        block.attn.v_proj.weight,
                        block.attn.gate_proj.weight,
                    ],
                    axis=0,
                )
                mx.eval(packed_qkvg)
            object.__setattr__(block.attn, "_mei_packed_qkvg", packed_qkvg)
        packed_mhc = None
        if fused:
            packed_mhc = [
                mx.concatenate(
                    [
                        self.mhc_phi_pre[i].T,
                        self.mhc_phi_post[i].T,
                        self.mhc_phi_res[i].T,
                    ],
                    axis=0,
                )
                for i in range(int(self.cfg.n_layers))
            ]
            mx.eval(*packed_mhc)
        object.__setattr__(self, "_mei_mhc_phi_packed", packed_mhc)
        pre_off, post_off = mhc_offsets(self.cfg.n_layers, self.cfg.mhc_lanes)
        mx.eval(pre_off, post_off)
        object.__setattr__(self, "_mei_mhc_pre_off", pre_off)
        object.__setattr__(self, "_mei_mhc_post_off", post_off)

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
        if (
            getattr(self, "_inference_backend", "mlx-reference") == "mlx-cq2"
            and len(self.engrams) == 2
            and tuple(self.cfg.engram_orders) == (2, 3)
            and int(self.cfg.engram_slots) == 8192
            and int(self.cfg.engram_conv_taps) == 4
            and tuple(tokens.shape) == (1, 13)
            and tokens.dtype == mx.int32
            and all(getattr(engram, "_mei_cq_tables", None) is not None for engram in self.engrams)
        ):
            selected = self.engrams[0]._mei_cq_tables.engram_indices_4_taps(tokens)
            pairs = [
                engram.packed_decode_full_taps(tokens, selected)
                for engram in self.engrams
            ]
            return (
                mx.stack([key for key, _ in pairs]),
                mx.stack([value for _, value in pairs]),
            )
        if (
            getattr(self, "_inference_backend", "mlx-reference") == "mlx-fused"
            and len(self.engrams) == 2
            and tuple(self.cfg.engram_orders) == (2, 3)
            and int(self.cfg.engram_slots) == 8192
            and int(self.cfg.engram_conv_taps) == 4
            and tuple(tokens.shape) == (1, 13)
            and tokens.dtype == mx.int32
            and getattr(self.engrams[0], "_mei_cq_tables", None) is None
        ):
            return fused_engram_two_full_taps(tokens, self.engrams[0], self.engrams[1])
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
        packed_embed = self._mei_cq_embed
        x = (
            packed_embed.gather_rows(tokens)
            if packed_embed is not None
            else self.embed(tokens)
        ) * self.embed_scale
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
        pre_off = self._mei_mhc_pre_off
        post_off = self._mei_mhc_post_off
        n = cfg.mhc_lanes
        lanes = mx.broadcast_to(x[:, :, None, :], (b, t, n, cfg.d_model))
        fused_decode = (
            getattr(self, "_inference_backend", "mlx-reference")
            in {"mlx-fused", "mlx-cq2"}
            and supports_decode_fusion(
                batch=b,
                tokens=t,
                lanes=n,
                d_model=cfg.d_model,
                dtype=lanes.dtype,
            )
        )
        new_cache = []
        for i, block in enumerate(self.blocks):
            xf = lanes.astype(mx.float32)
            if fused_decode:
                u, hpost, hres = fused_mhc_pre(
                    lanes,
                    self._mei_mhc_phi_packed[i],
                    self.mhc_a_pre[i],
                    self.mhc_a_post[i],
                    self.mhc_a_res[i],
                    self.mhc_b_pre[i],
                    self.mhc_b_post[i],
                    self.mhc_b_res[i],
                    pre_off[i],
                    post_off[i],
                )
                nx = None
            else:
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
                mhc_post_context=(lanes, hpost, hres, u) if fused_decode else None,
            )
            if fused_decode:
                lanes = block_out
            else:
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
        lm_input = _ste_activation_int8(hidden) if _activation_qdq_enabled(self) else hidden
        if packed_embed is not None and cfg.tie_embeddings:
            logits = packed_embed.linear(lm_input)
        else:
            output_weight = self.embed.weight if cfg.tie_embeddings else self.lm_head.weight
            logits = lm_input.astype(mx.float32) @ output_weight.T
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
    logical = getattr(model, "_logical_param_count", None)
    if logical is not None:
        return int(logical)
    import mlx.utils as xu

    leaves = xu.tree_flatten(model.parameters())
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    total = 0
    for item in leaves:
        value = item[1] if isinstance(item, tuple) else item
        total += int(value.size)
    return total
