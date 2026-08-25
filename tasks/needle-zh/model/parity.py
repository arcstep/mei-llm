"""NumPy reference ops matching cactus-needle formulas for MLX parity checks."""

from __future__ import annotations

import math

import numpy as np

_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def rms_unit_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    return xf * (1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps))


def zc_rms_np(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return ((1.0 + scale.astype(np.float32)) * xf / rms).astype(x.dtype, copy=False)


def walsh_np(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def silu_np(z: np.ndarray) -> np.ndarray:
    return z * (1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))))


def hadamard_forward_np(x: np.ndarray, h: np.ndarray, d1, d2, d3, d_model: int) -> np.ndarray:
    n = h.shape[0]
    pad = n - d_model
    z = np.pad(x.astype(np.float32), ((0, 0), (0, 0), (0, pad))) if pad else x.astype(np.float32)
    z = (d1.astype(np.float32) * z) @ h
    z = silu_np(d2.astype(np.float32) * z) @ h
    return (d3.astype(np.float32) * z)[..., :d_model]


def apply_rope_np(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int = 0) -> np.ndarray:
    t = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[offset : offset + t][None, None, :, :]
    sin = sin[offset : offset + t][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype, copy=False)


def precompute_rope_np(head_dim: int, seq_len: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    dims = np.arange(0, head_dim, 2).astype(np.float32)
    freqs = 1.0 / (theta ** (dims / head_dim))
    t = np.arange(seq_len).astype(np.float32)
    angles = np.outer(t, freqs)
    return np.cos(angles), np.sin(angles)


def shift_right_np(x: np.ndarray, offset: int) -> np.ndarray:
    t = int(x.shape[-2])
    if offset <= 0:
        return x
    if offset >= t:
        return np.zeros_like(x)
    pad = np.zeros((*x.shape[:-2], offset, x.shape[-1]), dtype=x.dtype)
    return np.concatenate([pad, x[..., : t - offset, :]], axis=-2)


def engram_indices_np(tokens: np.ndarray, orders: tuple[int, ...], heads: int, slots: int) -> np.ndarray:
    u = tokens.astype(np.uint32)
    cols = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = np.full(u.shape, seed, dtype=np.uint32)
            for j in range(order):
                shifted = shift_right_np(u[..., None], j)[..., 0]
                acc = (acc ^ shifted) * np.uint32(_ENGRAM_PRIME)
            acc = acc ^ (acc >> np.uint32(15))
            cols.append((acc % np.uint32(slots)).astype(np.int32))
    return np.stack(cols, axis=-1)


def sinkhorn_np(logits: np.ndarray, iters: int = 8) -> np.ndarray:
    log_k = logits.astype(np.float32)
    for _ in range(iters):
        log_k = log_k - np.logaddexp.reduce(log_k, axis=-1, keepdims=True)
        log_k = log_k - np.logaddexp.reduce(log_k, axis=-2, keepdims=True)
    return np.exp(log_k)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x.astype(np.float32)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -20, 20))
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def gqa_attn_np(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, mask=None) -> np.ndarray:
    repeats = q.shape[1] // k.shape[1]
    k_use = np.repeat(k, repeats, axis=1) if repeats > 1 else k
    v_use = np.repeat(v, repeats, axis=1) if repeats > 1 else v
    attn = (q.astype(np.float32) * scale) @ np.swapaxes(k_use.astype(np.float32), -1, -2)
    if mask is not None:
        attn = np.where(mask, attn, -1e9)
    attn = softmax_np(attn, axis=-1)
    return attn @ v_use.astype(np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
