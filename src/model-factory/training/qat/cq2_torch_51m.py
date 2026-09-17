"""CUDA CQ2 Q/DQ. Same portable codebooks, grouping and storage policy as MLX."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from training.qat.cq2_policy_51m import GROUP_SIZE, QUANT_MATH_ID, lm_storage_dtype
from mei_sdk.cq2 import Q2_CODEBOOK, Q4_CODEBOOK


def wht_orthonormal(blocks):
    if blocks.shape[-1] != GROUP_SIZE:
        raise ValueError("CQ2 groups must have 128 values")
    x = blocks.float()
    stride = 1
    while stride < GROUP_SIZE:
        pairs = x.reshape(*x.shape[:-1], GROUP_SIZE // (2 * stride), 2, stride)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        stride *= 2
    return x * (1.0 / math.sqrt(GROUP_SIZE))


def _nearest(normalized, codebook):
    book = torch.tensor(codebook, device=normalized.device, dtype=torch.float32)
    boundaries = torch.tensor([(float(a) + float(b)) * .5 for a, b in zip(codebook, codebook[1:])], device=normalized.device, dtype=torch.float32)
    # right=False selects the lower code on a midpoint tie, matching MLX.
    return book[torch.bucketize(normalized.contiguous(), boundaries, right=False)]


def fake_quant_weight(weight, group_bits, *, ste):
    widths = tuple(int(b) for b in group_bits)
    groups = (weight.numel() + GROUP_SIZE - 1) // GROUP_SIZE
    if not groups or len(widths) != groups or any(b not in (2, 4) for b in widths):
        raise ValueError("CQ2 requires one 2/4-bit selector per group")
    with torch.autocast(device_type=weight.device.type, enabled=False):
        # Quantization is forward evidence; STE intentionally has no derivative
        # through WHT/code selection/scale rounding.
        x = weight.detach().float().flatten()
        x = F.pad(x, (0, groups * GROUP_SIZE - x.numel())).reshape(groups, GROUP_SIZE)
        transformed = wht_orthonormal(x)
        scale = transformed.square().mean(-1, keepdim=True).sqrt().clamp_min(1.17549435e-38)
        scale = scale.half().float().clamp_min(2.0 ** -24)
        normalized = transformed / scale
        if set(widths) == {2}:
            q = _nearest(normalized, Q2_CODEBOOK)
        elif set(widths) == {4}:
            q = _nearest(normalized, Q4_CODEBOOK)
        else:
            use4 = torch.tensor([b == 4 for b in widths], device=weight.device)[:, None]
            q = torch.where(use4, _nearest(normalized, Q4_CODEBOOK), _nearest(normalized, Q2_CODEBOOK))
        recon = wht_orthonormal(q * scale).flatten()[:weight.numel()].reshape(weight.shape).to(weight.dtype)
    return weight + (recon - weight).detach() if ste else recon


def fake_quant_safe_f16(weight, *, ste):
    recon = weight.detach().half().to(weight.dtype)
    return weight + (recon - weight).detach() if ste else recon


def activation_int8(x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        maximum = xf.detach().abs().amax(-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
        recon = (xf.detach() / scale).round().clamp(-128, 127) * scale
        return (xf + (recon - xf).detach()).to(x.dtype)


def quantized_parameters(model, *, ste=True):
    result = {}
    for name, value in model.named_parameters():
        storage = lm_storage_dtype(name, value.shape)
        if storage == "f16":
            result[name] = fake_quant_safe_f16(value, ste=ste)
        else:
            width = 4 if storage == "cq4" else 2
            result[name] = fake_quant_weight(value, (width,) * ((value.numel() + 127) // 128), ste=ste)
    return result


def forward_qat(model, tokens, **kwargs):
    if not model.qat_activation_ste:
        raise ValueError("QAT requires explicit activation/KV STE activation")
    return torch.func.functional_call(model, quantized_parameters(model), (tokens,), kwargs, strict=False)
