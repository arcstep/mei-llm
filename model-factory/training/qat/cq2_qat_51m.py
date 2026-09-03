"""True CQ2 v2 fake-quant and STE for MLX training.

This module mirrors the portable ``mei-cq-v2-g128-wht-codebook`` encoder:
flattened groups of 128, orthonormal Walsh-Hadamard transform, binary16 RMS
scale, and the fixed Gaussian/Lloyd-Max Q2/Q4 codebooks.  It deliberately does
not import or reuse the historical block64 uniform quantizer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx
import mlx.utils as xu

from training.qat.cq2_policy_51m import GROUP_SIZE, QUANT_MATH_ID, lm_storage_dtype, uniform_group_bits


Q2_CODEBOOK = (-1.5104176, -0.45278, 0.45278, 1.5104176)
Q4_CODEBOOK = (
    -2.732589,
    -2.069018,
    -1.618046,
    -1.256231,
    -0.94234,
    -0.656759,
    -0.388055,
    -0.128396,
    0.128396,
    0.388055,
    0.656759,
    0.94234,
    1.256231,
    1.618046,
    2.069018,
    2.732589,
)
F16_MIN_SUBNORMAL = 2.0**-24


def _flatten_tree(tree: Any) -> dict[str, mx.array]:
    leaves = xu.tree_flatten(tree)
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    return {str(name): value for name, value in leaves}


def wht_orthonormal(blocks: mx.array) -> mx.array:
    """Apply the self-inverse orthonormal WHT on the final 128 axis."""

    if int(blocks.shape[-1]) != GROUP_SIZE:
        raise ValueError(f"CQ2 WHT requires a final axis of {GROUP_SIZE}")
    values = blocks.astype(mx.float32)
    prefix = tuple(int(value) for value in values.shape[:-1])
    stride = 1
    while stride < GROUP_SIZE:
        paired = values.reshape(*prefix, GROUP_SIZE // (2 * stride), 2, stride)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        values = mx.stack((left + right, left - right), axis=-2).reshape(
            *prefix, GROUP_SIZE
        )
        stride *= 2
    return values * (1.0 / math.sqrt(GROUP_SIZE))


def _nearest_reconstruction(normalized: mx.array, codebook: Sequence[float]) -> mx.array:
    """Nearest sorted codebook without materializing values×codebook distances."""

    book = mx.array(tuple(float(value) for value in codebook), dtype=mx.float32)
    indices = mx.zeros(normalized.shape, dtype=mx.int32)
    for left, right in zip(codebook, codebook[1:]):
        # Strictly greater preserves the lower-index tie rule used by the
        # portable Python/Rust/JS encoders.
        midpoint = (float(left) + float(right)) * 0.5
        indices = indices + (normalized > midpoint).astype(mx.int32)
    return mx.take(book, indices)


def fake_quant_weight(
    weight: mx.array,
    group_bits: Sequence[int],
    *,
    ste: bool,
) -> mx.array:
    """Reconstruct one tensor using exact CQ2 group semantics."""

    original_shape = tuple(int(value) for value in weight.shape)
    n_values = int(weight.size)
    if n_values <= 0:
        raise ValueError("CQ2 tensor is empty")
    groups = (n_values + GROUP_SIZE - 1) // GROUP_SIZE
    widths = tuple(int(value) for value in group_bits)
    if len(widths) != groups or any(value not in (2, 4) for value in widths):
        raise ValueError("CQ2 requires one 2/4-bit selector per group")

    source = weight.astype(mx.float32).reshape(-1)
    padded_values = groups * GROUP_SIZE
    if padded_values != n_values:
        source_padded = mx.concatenate(
            (source, mx.zeros((padded_values - n_values,), dtype=mx.float32))
        )
    else:
        source_padded = source
    transformed = wht_orthonormal(source_padded.reshape(groups, GROUP_SIZE))
    scale = mx.sqrt(mx.sum(transformed * transformed, axis=-1, keepdims=True) / GROUP_SIZE)
    # Package encoders select codes against the exact serialized binary16
    # scale.  Clamp after conversion so zero/subnormal groups match 0x0001.
    scale = mx.maximum(scale, mx.array(1.17549435e-38, dtype=mx.float32))
    quant_scale = scale.astype(mx.float16).astype(mx.float32)
    quant_scale = mx.maximum(
        quant_scale, mx.array(F16_MIN_SUBNORMAL, dtype=mx.float32)
    )
    normalized = transformed / quant_scale

    unique_widths = set(widths)
    if unique_widths == {2}:
        reconstructed = _nearest_reconstruction(normalized, Q2_CODEBOOK) * quant_scale
    elif unique_widths == {4}:
        reconstructed = _nearest_reconstruction(normalized, Q4_CODEBOOK) * quant_scale
    else:
        q2 = _nearest_reconstruction(normalized, Q2_CODEBOOK) * quant_scale
        q4 = _nearest_reconstruction(normalized, Q4_CODEBOOK) * quant_scale
        use_q4 = mx.array([value == 4 for value in widths], dtype=mx.bool_)[:, None]
        reconstructed = mx.where(use_q4, q4, q2)

    dequantized = wht_orthonormal(reconstructed).reshape(-1)[:n_values]
    dequantized = dequantized.reshape(original_shape).astype(weight.dtype)
    if not ste:
        return dequantized
    return weight + mx.stop_gradient(dequantized - weight)


def fake_quant_safe_f16(weight: mx.array, *, ste: bool) -> mx.array:
    dequantized = weight.astype(mx.float16).astype(weight.dtype)
    return weight + mx.stop_gradient(dequantized - weight) if ste else dequantized


def explicit_group_map(tree: Any) -> dict[str, tuple[int, ...]]:
    """Build the canonical explicit group map for an MLX parameter tree."""

    result: dict[str, tuple[int, ...]] = {}
    for name, value in _flatten_tree(tree).items():
        bits = uniform_group_bits(name, tuple(int(part) for part in value.shape))
        if bits is not None:
            result[name] = bits
    return result


def quantize_tree(
    tree: Any,
    group_map: Mapping[str, Sequence[int]] | None = None,
    *,
    ste: bool,
) -> Any:
    """Quantize all LM leaves with the same policy as package v2 export."""

    flat = _flatten_tree(tree)
    resolved = dict(group_map or explicit_group_map(tree))
    unknown = sorted(set(resolved) - set(flat))
    if unknown:
        raise ValueError(f"CQ2 group map contains unknown tensors: {unknown[:3]}")
    output: dict[str, mx.array] = {}
    for name, value in flat.items():
        dtype = lm_storage_dtype(name, tuple(int(part) for part in value.shape))
        if dtype == "f16":
            if name in resolved:
                raise ValueError(f"safe f16 tensor must not have a CQ2 group map: {name}")
            output[name] = fake_quant_safe_f16(value, ste=ste)
            continue
        widths = resolved.get(name)
        if widths is None:
            raise ValueError(f"missing CQ2 group map for tensor: {name}")
        expected_width = 4 if dtype == "cq4" else 2
        if dtype == "cq4" and any(int(width) != expected_width for width in widths):
            raise ValueError(f"canonical Q4 tensor has non-Q4 group: {name}")
        output[name] = fake_quant_weight(value, widths, ste=ste)
    return xu.tree_unflatten(list(output.items()))


def group_map_receipt(tree: Any) -> dict[str, Any]:
    flat = _flatten_tree(tree)
    groups = explicit_group_map(tree)
    q2_groups = sum(sum(int(width) == 2 for width in widths) for widths in groups.values())
    q4_groups = sum(sum(int(width) == 4 for width in widths) for widths in groups.values())
    return {
        "quant_math_id": QUANT_MATH_ID,
        "group_size": GROUP_SIZE,
        "tensor_count": len(flat),
        "quantized_tensor_count": len(groups),
        "safe_f16_tensor_count": len(flat) - len(groups),
        "q2_groups": q2_groups,
        "q4_groups": q4_groups,
        "per_group_map": {name: list(widths) for name, widths in sorted(groups.items())},
    }
