from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.identity_51m import refuse_base_write
from release.quant_pack_51m import (
    BLOCK_SIZE,
    QUANT_MATH_ID,
    build_pack_bytes,
    fake_quant_block,
    parse_pack_bytes,
    propose_mixed_bit_map,
    q4_baseline_map,
    reconstruction_mse,
)


def test_q4_roundtrip_small() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(17, 64)).astype(np.float32)
    packed, header = build_pack_bytes({"w": x}, {"w": 4})
    assert packed[:8] == b"MEIQPK01"
    _h, arrays = parse_pack_bytes(packed)
    recon = arrays["w"]
    assert recon.shape == x.shape
    mse = float(np.mean((x - recon) ** 2))
    assert float(np.max(np.abs(recon - fake_quant_block(x, 4)))) < 1e-6
    assert mse < 5e-2
    assert abs(mse - reconstruction_mse(x, 4)) < 1e-6
    assert header.quant_math_id == QUANT_MATH_ID
    assert header.block_size == BLOCK_SIZE


def test_q2_roundtrip_small() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(128,)).astype(np.float32)
    packed, _ = build_pack_bytes({"w": x}, {"w": 2})
    _h, arrays = parse_pack_bytes(packed)
    recon = fake_quant_block(x, 2)
    assert float(np.max(np.abs(arrays["w"] - recon))) < 1e-5


def test_q4_baseline_under_30mb_math() -> None:
    tensors = {
        "embed.weight": np.zeros((24000, 512), dtype=np.float32),
        "blocks.0.attn.q_proj.weight": np.zeros((512, 512), dtype=np.float32),
        "final_norm.scale": np.zeros((512,), dtype=np.float32),
        "tiny": np.zeros((8,), dtype=np.float32),
    }
    baseline = q4_baseline_map(tensors)
    assert baseline["tensor_bits"]["embed.weight"] == 4
    assert baseline["tensor_bits"]["tiny"] == 32
    assert baseline["raw_payload_mb"] < 30


def test_mixed_map_hits_byte_budget_and_stays_blocked() -> None:
    rng = np.random.default_rng(3)
    tensors = {
        "embed.weight": rng.normal(size=(256, 64)).astype(np.float32),
        "blocks.0.attn.q_proj.weight": rng.normal(size=(512, 512)).astype(np.float32),
        "blocks.0.attn.k_proj.weight": rng.normal(size=(256, 512)).astype(np.float32),
        "mhc_phi_pre": rng.normal(size=(4, 256, 4)).astype(np.float32),
        "engrams.0.tables": rng.normal(size=(4, 128, 32)).astype(np.float32),
        "final_norm.scale": rng.normal(size=(64,)).astype(np.float32),
    }
    mixed = propose_mixed_bit_map(tensors, target_avg_bits=3.0)
    assert mixed["product_final"] is False
    assert mixed["quality_blocked"] is True
    assert mixed["qat_mandatory"] is True
    assert mixed["embed_bits"] == 4
    assert mixed["meets_size_budget"] is True
    assert mixed["avg_bits"] <= 3.0 + 1e-6


def test_refuse_pack_write_under_base() -> None:
    from common.identity_51m import ROOT, MODEL_ID

    blocked = refuse_base_write(ROOT / "base" / MODEL_ID / "weights.q4")
    assert blocked is not None


if __name__ == "__main__":
    test_q4_roundtrip_small()
    test_q2_roundtrip_small()
    test_q4_baseline_under_30mb_math()
    test_mixed_map_hits_byte_budget_and_stays_blocked()
    test_refuse_pack_write_under_base()
    print("test_quant_pack_51m: ok")
