#!/usr/bin/env python3
"""Per-tensor / per-block pack-budget scan for the immutable 51M base.

Ranking is quality_loss / bytes_saved, not component top-half. Embedding stays
Q4. Attention / engram / mHC may mix Q2/Q4 by block. Writes diagnostic maps
only; does not pack weights or claim a product mixed-bit map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from identity_51m import (
    ARCHITECTURE_ID,
    EXPECTED_PARAMS,
    JOBS_DIR,
    MODEL_ID,
    PACK_BUDGET_NAME,
    PRODUCT_MIXED_MAP_NAME,
    Q4_BASELINE_MAP_NAME,
    QAT_MANDATORY_NOTE,
    RELEASE_PATH,
    ROOT,
    WEIGHTS_PATH,
    fail,
    load_json,
    sha256_file,
    validate_release,
    write_json,
)
from quant_pack_51m import (
    BLOCK_SIZE,
    CQ2_AVG_BITS,
    CQ2_RAW_PAYLOAD_BUDGET,
    Q4_PACKAGE_BUDGET_BYTES,
    QUANT_MATH_ID,
    component_of,
    payload_bytes,
    propose_mixed_bit_map,
    q4_baseline_map,
    reconstruction_mse,
)


def _params_to_numpy(model) -> dict[str, np.ndarray]:
    from checkpoint import flatten_params

    out: dict[str, np.ndarray] = {}
    for name, val in flatten_params(model).items():
        out[name] = np.array(val, dtype=np.float32)
    return out


def load_51m_numpy() -> dict[str, np.ndarray]:
    import mlx.core as mx

    from architecture import NeedleZh, count_params
    from checkpoint import load_params
    from config import NeedleZhConfig
    from identity_51m import ARCHITECTURE_SPEC

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError("loaded param count mismatch")
    return _params_to_numpy(model)


def scan_tensors(tensors: dict[str, np.ndarray]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for name, arr in tensors.items():
        n = int(arr.size)
        mse4 = reconstruction_mse(arr, 4)
        mse2 = reconstruction_mse(arr, 2) if n >= BLOCK_SIZE else mse4
        bytes4 = payload_bytes(n, 4 if n >= BLOCK_SIZE else 32)
        bytes2 = payload_bytes(n, 2 if n >= BLOCK_SIZE else 32)
        saved = max(bytes4 - bytes2, 0)
        extra = max(mse2 - mse4, 0.0)
        rows[name] = {
            "component": component_of(name),
            "n_params": n,
            "shape": [int(x) for x in arr.shape],
            "mse_q4": mse4,
            "mse_q2": mse2,
            "quality_loss_q2_vs_q4": extra,
            "bytes_q4": bytes4,
            "bytes_q2": bytes2,
            "bytes_saved_if_q2": saved,
            "loss_per_byte": extra / saved if saved else None,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    release = load_json(RELEASE_PATH)
    error = validate_release(release, WEIGHTS_PATH)
    if error:
        return fail(error)

    tensors = load_51m_numpy()
    tensor_rows = scan_tensors(tensors)
    baseline = q4_baseline_map(tensors)
    mixed = propose_mixed_bit_map(tensors, target_avg_bits=CQ2_AVG_BITS)

    ranked = sorted(
        (
            (name, row)
            for name, row in tensor_rows.items()
            if row.get("loss_per_byte") is not None and row["bytes_saved_if_q2"] > 0
        ),
        key=lambda item: float(item[1]["loss_per_byte"]),
    )
    by_component: dict[str, dict] = {}
    for name, row in tensor_rows.items():
        comp = row["component"]
        slot = by_component.setdefault(
            comp,
            {"n_tensors": 0, "n_params": 0, "mse_q4_sum": 0.0, "mse_q2_sum": 0.0},
        )
        slot["n_tensors"] += 1
        slot["n_params"] += int(row["n_params"])
        slot["mse_q4_sum"] += float(row["mse_q4"]) * int(row["n_params"])
        slot["mse_q2_sum"] += float(row["mse_q2"]) * int(row["n_params"])
    components = {}
    for comp, slot in by_component.items():
        n = max(int(slot["n_params"]), 1)
        components[comp] = {
            "n_tensors": slot["n_tensors"],
            "n_params": slot["n_params"],
            "mse_q4": slot["mse_q4_sum"] / n,
            "mse_q2": slot["mse_q2_sum"] / n,
        }

    scan = {
        "stage": "P0",
        "kind": "pack-budget-scan",
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "params": EXPECTED_PARAMS,
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "quant_math_id": QUANT_MATH_ID,
        "block_size": BLOCK_SIZE,
        "qat_mandatory": True,
        "product_final": False,
        "source": str(WEIGHTS_PATH.relative_to(ROOT)),
        "fp32_mb": EXPECTED_PARAMS * 4 / (1024 * 1024),
        "fp16_mb": EXPECTED_PARAMS * 2 / (1024 * 1024),
        "uniform_q4_raw_mb": EXPECTED_PARAMS * 4 / 8 / (1024 * 1024),
        "q4_package_budget_mb": Q4_PACKAGE_BUDGET_BYTES / (1024 * 1024),
        "cq2_raw_payload_budget_mb": CQ2_RAW_PAYLOAD_BUDGET / (1024 * 1024),
        "cq2_avg_bits_target": CQ2_AVG_BITS,
        "components": components,
        "lowest_loss_per_byte_tensors": [name for name, _ in ranked[:20]],
        "highest_loss_per_byte_tensors": [name for name, _ in ranked[-20:]],
        "n_tensors": len(tensor_rows),
        "note": QAT_MANDATORY_NOTE,
        "not_a_claim": "Reconstruction MSE ranking is not LM quality, kernel parity, or tool-calling ability.",
        "coarse_component_map_rejected": (
            "The earlier component top-half candidate only moved ~91k params to Q2 "
            "and cannot be a product mixed-bit map."
        ),
    }
    mixed_out = {
        **mixed,
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "note": QAT_MANDATORY_NOTE,
    }
    baseline_out = {
        **baseline,
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "note": QAT_MANDATORY_NOTE,
    }
    jobs = (
        (args.out_dir / PACK_BUDGET_NAME, scan),
        (args.out_dir / Q4_BASELINE_MAP_NAME, baseline_out),
        (args.out_dir / PRODUCT_MIXED_MAP_NAME, mixed_out),
    )
    for path, payload in jobs:
        blocked = write_json(path, payload)
        if blocked:
            return fail(blocked)
    print(
        json.dumps(
            {
                "ok": True,
                "pack_budget": str(args.out_dir / PACK_BUDGET_NAME),
                "q4_baseline": str(args.out_dir / Q4_BASELINE_MAP_NAME),
                "cq2_mixed": str(args.out_dir / PRODUCT_MIXED_MAP_NAME),
                "q4_raw_mb": baseline_out["raw_payload_mb"],
                "cq2_raw_mb": mixed_out["raw_payload_mb"],
                "cq2_avg_bits": mixed_out["avg_bits"],
                "cq2_meets_size": mixed_out["meets_size_budget"],
                "cq2_quality_blocked": mixed_out["quality_blocked"],
                "qat_mandatory": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
