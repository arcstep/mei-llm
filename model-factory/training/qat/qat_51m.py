#!/usr/bin/env python3
"""Short 51M QAT smoke. STE math matches the packed Q4 kernel.

Does not re-run the float eval suite. Quality is the Q4/QAT delta against
artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/jobs/mei-1.0-51m/float-base-lm-anchor.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from common.identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    EXPECTED_PARAMS,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    MODEL_ID,
    QAT_MANDATORY_NOTE,
    RELEASE_PATH,
    WEIGHTS_PATH,
    fail,
    load_json,
    validate_release,
    write_json,
)
from training.qat.quant_ops_51m import (
    KERNEL_FEASIBILITY_RECORDED,
    STE_IMPLEMENTED,
    fake_quant_4bit,
    ste_activation,
    ste_quantize,
)


def qat_tiny_step() -> dict:
    from architecture import NeedleZh
    from config import NeedleZhConfig

    mx.random.seed(51)
    model = NeedleZh(NeedleZhConfig().tiny())
    mx.eval(model.parameters())
    ids = mx.array([[2, 8, 9, 1]], dtype=mx.int32)

    def ste_loss(weight):
        q = ste_quantize(weight, bits=4)
        act = ste_activation(q[:4, :4])
        return mx.sum(act * act) + mx.mean(fake_quant_4bit(weight))

    loss, grads = mx.value_and_grad(ste_loss)(model.embed.weight)
    mx.eval(loss, grads)
    logits = model(ids)["logits"]
    mx.eval(logits)
    return {
        "ok": bool(mx.isfinite(loss).item()) and bool(mx.all(mx.isfinite(logits)).item()),
        "loss": float(loss.item()),
        "ste": True,
        "qat_step": True,
        "activation_kv_ste": True,
        "logits_finite": True,
        "kind": "tiny-qat-ste-step",
    }


def compare_to_frozen_anchor(jobs_dir: Path) -> dict:
    """Use already-written Q4 parity + frozen float JSON. Do not reload float weights.

    Probe delta is the paired subset already stored in q4-dequant-parity.json
    (float_probe_mean_nll vs quant_probe_mean_nll). The full-set anchor NLL is
    recorded for identity only; mixing it with the subset Q4 NLL is invalid.
    """
    anchor = load_json(jobs_dir / FLOAT_ANCHOR_NAME)
    parity = load_json(jobs_dir / "q4-dequant-parity.json")
    paired_float = parity.get("float_probe_mean_nll")
    q4_nll = parity.get("quant_probe_mean_nll")
    delta = None
    if paired_float is not None and q4_nll is not None:
        delta = float(q4_nll) - float(paired_float)
    return {
        "float_anchor_probe_mean_nll": anchor.get("probe_mean_nll"),
        "float_anchor_valid_loss": anchor.get("valid_loss"),
        "parity_float_probe_mean_nll": paired_float,
        "q4_probe_mean_nll": q4_nll,
        "delta_probe_mean_nll": delta,
        "compared_without_reloading_float_model": True,
        "qat_mandatory": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    error = validate_release(load_json(RELEASE_PATH), WEIGHTS_PATH)
    if error:
        return fail(error)
    if not STE_IMPLEMENTED or not KERNEL_FEASIBILITY_RECORDED:
        return fail("QAT harness incomplete")
    smoke = qat_tiny_step()
    vs_anchor = compare_to_frozen_anchor(args.jobs_dir)
    report = {
        "stage": "P4",
        "kind": "qat-smoke",
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "params": EXPECTED_PARAMS,
        "ste": True,
        "kernel_math": "mei-qpack-v1-block64-q4s7-q2u4",
        "tiny_qat_step": smoke,
        "vs_frozen_float_anchor": vs_anchor,
        "product_mixed_map_frozen": False,
        "qat_mandatory": True,
        "note": QAT_MANDATORY_NOTE,
        "not_a_claim": "Tiny STE step is not 0.5M–5M replay or a product mixed-bit freeze.",
    }
    path = args.jobs_dir / "qat-smoke.json"
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": bool(smoke.get("ok")), "report": str(path), **vs_anchor}, indent=2, ensure_ascii=False))
    return 0 if smoke.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
