#!/usr/bin/env python3
"""Fail-closed QAT pilot readiness for 51M. Does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.identity_51m import (
    ARCHITECTURE_ID,
    CANDIDATE_MAP_NAME,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    MODEL_ID,
    fail,
    load_json,
)
from common.paths import RECIPES_DIR
from training.qat.quant_ops_51m import KERNEL_FEASIBILITY_RECORDED, STE_IMPLEMENTED, fake_quant_4bit

CONTRACT_PATH = RECIPES_DIR / "qat-pilot-51m-contract.json"


def collect_blockers(*, jobs_dir: Path, contract_path: Path) -> list[str]:
    blockers: list[str] = []
    contract = load_json(contract_path)
    if not contract:
        blockers.append(f"missing QAT pilot contract: {contract_path}")
    else:
        if contract.get("parent") != MODEL_ID:
            blockers.append(f"contract parent mismatch: {contract.get('parent')!r}")
        if contract.get("architecture_id") != ARCHITECTURE_ID:
            blockers.append(f"contract architecture_id mismatch: {contract.get('architecture_id')!r}")
        if contract.get("train_authorized") is True and not STE_IMPLEMENTED:
            blockers.append("contract train_authorized is true but STE is not implemented")
        if contract.get("fake_quant", {}).get("ste_required") and not STE_IMPLEMENTED:
            blockers.append("STE is required but not implemented")
        act = str(contract.get("fake_quant", {}).get("activation_kv") or "")
        if act == "blocked":
            blockers.append("activation/KV fake-quant is blocked")
        if not contract.get("kernel", {}).get("feasibility_recorded") and not KERNEL_FEASIBILITY_RECORDED:
            blockers.append("kernel feasibility record missing")
        stop = contract.get("stop_gates") or {}
        if stop.get("valid_loss_delta_vs_float_anchor") == "unregistered":
            blockers.append("valid_loss stop threshold is unregistered")
        if stop.get("probe_mean_nll_delta_vs_float_anchor") == "unregistered":
            blockers.append("probe stop threshold is unregistered")

    anchor = load_json(jobs_dir / FLOAT_ANCHOR_NAME)
    if not anchor:
        blockers.append(f"missing float Base-LM Anchor: {jobs_dir / FLOAT_ANCHOR_NAME}")
    elif not anchor.get("qat_mandatory"):
        blockers.append("float anchor must set qat_mandatory=true")

    candidate = load_json(jobs_dir / CANDIDATE_MAP_NAME)
    if not candidate:
        blockers.append(f"missing Q2/Q4 candidate map: {jobs_dir / CANDIDATE_MAP_NAME}")
    else:
        if candidate.get("product_final"):
            blockers.append("candidate map must not be marked product_final")
        if not candidate.get("qat_mandatory"):
            blockers.append("candidate map must set qat_mandatory=true")
        if "only_if_ptq_misses" in json.dumps(candidate):
            blockers.append("candidate map uses forbidden only_if_ptq_misses language")

    if not STE_IMPLEMENTED:
        blockers.append("quant_ops_51m.STE_IMPLEMENTED is false")
    if not KERNEL_FEASIBILITY_RECORDED:
        blockers.append("quant_ops_51m.KERNEL_FEASIBILITY_RECORDED is false")
    return blockers


def tiny_fake_quant_smoke() -> dict:
    import mlx.core as mx

    from architecture import NeedleZh
    from config import NeedleZhConfig

    mx.random.seed(51)
    model = NeedleZh(NeedleZhConfig().tiny())
    mx.eval(model.parameters())
    original = model.embed.weight
    model.embed.weight = fake_quant_4bit(original)
    mx.eval(model.embed.weight)
    ids = mx.array([[2, 4, 5, 1]], dtype=mx.int32)
    logits = model(ids)["logits"]
    mx.eval(logits)
    model.embed.weight = original
    return {
        "ok": True,
        "kind": "tiny-random-weight-fake-quant-forward",
        "ste": STE_IMPLEMENTED,
        "qat_step": False,
        "logits_finite": bool(mx.all(mx.isfinite(logits)).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--skip-tiny-smoke", action="store_true")
    args = parser.parse_args()
    blockers = collect_blockers(jobs_dir=args.jobs_dir, contract_path=args.contract)
    smoke = None if args.skip_tiny_smoke else tiny_fake_quant_smoke()
    contract = load_json(args.contract)
    report = {
        "ok": not bool(blockers),
        "train_authorized": bool(contract.get("train_authorized")),
        "qat_mandatory": True,
        "message": "QAT harness ready" if not blockers else "QAT pilot readiness failed",
        "blockers": blockers,
        "tiny_fake_quant_smoke": smoke,
        "eval_policy": "quantized_vs_frozen_float_anchor_only",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if blockers:
        return fail("QAT pilot readiness failed: " + "; ".join(blockers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
