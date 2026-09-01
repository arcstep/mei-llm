"""Diagnostic 4-bit fake-quant ops for 51M. STE matches packed kernel math."""

from __future__ import annotations

from quant_pack_51m import (
    ACTIVATION_KV_BITS,
    KERNEL_FEASIBILITY_RECORDED,
    Q4_LEVELS,
    STE_IMPLEMENTED,
    component_of,
    mlx_fake_quant_activation,
    mlx_fake_quant_weight,
    mlx_ste_activation,
    mlx_ste_quantize,
)

FAKE_QUANT_BITS = 4
FAKE_QUANT_LEVELS = Q4_LEVELS


def fake_quant_4bit(arr):
    return mlx_fake_quant_weight(arr, bits=4)


def fake_quant_2bit(arr):
    return mlx_fake_quant_weight(arr, bits=2)


def ste_quantize(arr, bits: int = 4):
    return mlx_ste_quantize(arr, bits=bits)


def ste_activation(arr, bits: int = ACTIVATION_KV_BITS):
    return mlx_ste_activation(arr, bits=bits)


def fake_quant_activation(arr, bits: int = ACTIVATION_KV_BITS):
    return mlx_fake_quant_activation(arr, bits=bits)


def fake_quant_kv(arr, bits: int = ACTIVATION_KV_BITS):
    return mlx_fake_quant_activation(arr, bits=bits)
