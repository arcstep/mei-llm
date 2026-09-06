#!/usr/bin/env python3
"""Offline admission and CQ2 SFT-gap A/B gate for ``mei-1.0-51m``.

This driver deliberately has no CPT integration.  It creates only a
hash-bound run plan or a terminal quality/resource receipt.  Actual training
is delegated to the existing offline stage runners after this admission has
passed; this prevents a scale campaign from silently changing package,
confidence or narration behaviour.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PRODUCT = "mei-1.0-51m"
CAMPAIGN_ID = "mei-1.0-51m-sft-gap-scale-v1"
LINEAGE_KEYS = (
    "base_weights_sha256", "qat_binding_sha256", "recipe_sha256", "seed",
    "eval_lock_sha256", "serializer", "grammar", "scorer", "runtime_scope",
)


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise GateError(f"immutable output already exists with different content: {path}")
        return
    path.write_bytes(data)


def _metric(payload: Mapping[str, Any], *path: str) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise GateError(f"missing metric: {'.'.join(path)}")
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"metric must be numeric: {'.'.join(path)}") from error


def _lineage_equal(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> list[str]:
    left, right = control.get("lineage"), treatment.get("lineage")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return ["missing lineage"]
    return [key for key in LINEAGE_KEYS if left.get(key) != right.get(key)]


def _global_gate(control: Mapping[str, Any], treatment: Mapping[str, Any], failures: list[str]) -> None:
    for bank in ("structural", "natural", "schema"):
        if _metric(treatment, "retrieval_recall_at_5", bank) < 0.98:
            failures.append(f"{bank} Retrieval Recall@5 < 0.98")
    if _metric(treatment, "unsupported_accepted") != 0 or _metric(treatment, "unprovenanced_accepted") != 0:
        failures.append("unsupported or unprovenanced accepted is nonzero")
    left = control.get("non_target_core_metrics")
    right = treatment.get("non_target_core_metrics")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
        failures.append("non-target core metric set mismatch")
    else:
        for name in left:
            if float(right[name]) < float(left[name]) - 0.02:
                failures.append(f"non-target metric fell by more than 2pp: {name}")


def evaluate_gate(cell_id: str, from_total: int, control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    if cell_id not in {
        "sft.full_call.boundary", "sft.schema.generalization", "sft.multi_step.3_4", "sft.mw.disposition",
    }:
        raise GateError("unknown campaign cell")
    if from_total not in {40, 160, 640}:
        raise GateError("invalid semantic task scale")
    failures = _lineage_equal(control, treatment)
    if cell_id == "sft.full_call.boundary":
        reasons = ("missing_slot", "offtopic", "unknown_slot_value", "negation_cancels")
        if from_total >= 160:
            reasons += ("ambiguous_scope",)
        before = [_metric(control, "full_call", "refusal_by_reason", reason) for reason in reasons]
        after = [_metric(treatment, "full_call", "refusal_by_reason", reason) for reason in reasons]
        if sum(after) / len(after) < sum(before) / len(before) + 0.10:
            failures.append("refusal macro gain is below 10pp")
        failures.extend(f"target reason regressed: {reason}" for reason, left, right in zip(reasons, before, after) if right < left)
    elif cell_id == "sft.schema.generalization":
        if _metric(treatment, "schema", "balanced_accuracy") < _metric(control, "schema", "balanced_accuracy") + 0.05:
            failures.append("schema balanced accuracy gain is below 5pp")
        if _metric(treatment, "schema", "argument_exact") < _metric(control, "schema", "argument_exact") + 0.03:
            failures.append("schema argument exact gain is below 3pp")
    elif cell_id == "sft.multi_step.3_4":
        if _metric(treatment, "multi_step", "three_step_success") < 0.20:
            failures.append("3-step success is below 0.20")
        if _metric(treatment, "multi_step", "four_step_success") < 0.10:
            failures.append("4-step success is below 0.10")
        if _metric(treatment, "multi_step", "two_step_success") < _metric(control, "multi_step", "two_step_success") - 0.03:
            failures.append("2-step success fell by more than 3pp")
        if _metric(treatment, "multi_step", "call_id_integrity") != 1.0:
            failures.append("call_id_integrity is not 1.0")
    else:
        if _metric(treatment, "mw", "macro_f1") < _metric(control, "mw", "macro_f1") + 0.05:
            failures.append("MW macro-F1 gain is below 5pp")
        false_continue = _metric(treatment, "mw", "class_0_false_continue")
        if false_continue > 0.10 or false_continue > _metric(control, "mw", "class_0_false_continue") + 0.02:
            failures.append("MW class-0 false-continue gate failed")
    _global_gate(control, treatment, failures)
    return {
        "schema": "mei-51m-sft-gap-ab-receipt-v1", "status": "passed" if not failures else "quality_miss",
        "campaign_id": CAMPAIGN_ID, "cell_id": cell_id, "from_total": from_total,
        "semantic_task_total": from_total, "control_metrics_sha256": hashlib.sha256(canonical(control)).hexdigest(),
        "treatment_metrics_sha256": hashlib.sha256(canonical(treatment)).hexdigest(), "failures": failures,
        "process_complete": True, "scale_authorized": not failures, "corpus_reuse_eligible": not failures,
        "model_release_eligible": False, "training_started": True, "provider_calls": 0, "paid_cny": 0,
        "current_mutated": False,
    }


def admit(campaign: Path, treatment: Path, run_root: Path, cell_id: str, from_total: int, out: Path) -> dict[str, Any]:
    campaign_payload, treatment_manifest = load(campaign), load(treatment / "manifest.json")
    if campaign_payload.get("campaign_id") != CAMPAIGN_ID:
        raise GateError("unknown scale campaign")
    if treatment_manifest.get("scale_campaign_sha256") != sha_file(campaign):
        raise GateError("treatment is not bound to the selected campaign")
    if treatment_manifest.get("treatment_cell_id") != cell_id:
        raise GateError("treatment cell mismatch")
    lock_path = run_root / ".mlx-metal-scale.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt = {"schema": "mei-51m-sft-gap-resource-receipt-v1", "status": "resource_deferred", "campaign_id": CAMPAIGN_ID, "cell_id": cell_id, "from_total": from_total, "process_complete": True, "training_started": False, "provider_calls": 0, "current_mutated": False}
            write_once(out, receipt)
            return receipt
        plan = {"schema": "mei-51m-sft-gap-ab-run-plan-v1", "status": "admitted", "campaign_id": CAMPAIGN_ID, "cell_id": cell_id, "from_total": from_total, "run_root": str(run_root), "treatment_manifest_sha256": sha_file(treatment / "manifest.json"), "execution_contract": {"serial_only": True, "fresh_cq2_master_per_treatment": True, "control_recomputed": True, "confidence_training": False, "narration_training": False, "package_or_publish": False}, "process_complete": True, "training_started": False, "provider_calls": 0, "current_mutated": False}
        write_once(out, plan)
        return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("admit")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--treatment", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--cell-id", required=True)
    p.add_argument("--from-total", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate-gate")
    p.add_argument("--cell-id", required=True)
    p.add_argument("--from-total", type=int, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--treatment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "admit":
            result = admit(args.campaign, args.treatment, args.run_root, args.cell_id, args.from_total, args.out)
        else:
            result = evaluate_gate(args.cell_id, args.from_total, load(args.control), load(args.treatment))
            write_once(args.out, result)
    except GateError as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {"admitted", "passed", "resource_deferred"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
