#!/usr/bin/env python3
"""Verify and adopt immutable upstream productization stages after a downstream-only fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file())
CURRENT = ROOT / "CURRENT.json"
UPSTREAM_STAGES = (
    "float_base_lm_anchor",
    "float_task_control",
    "q4_diagnostic",
    "cq2_qat_v2",
    "retrieval_r0",
    "oracle_top5_fullcall",
    "retrieval_r1",
    "tool_index_final",
    "learned_top5_e2e",
    "mw_disposition_20class",
    "confidence_calibration",
    "narration_adapter",
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def adopt(source_run: Path, out: Path) -> dict:
    source_run = source_run.resolve()
    plan_path = source_run / "plan.json"
    plan = load_json(plan_path)
    run_fingerprint = str(plan.get("run_fingerprint_sha256") or "")
    if len(run_fingerprint) != 64:
        raise RuntimeError("source run has no valid run fingerprint")
    baseline = str((plan.get("immutable") or {}).get("current_baseline_sha256") or "")
    current_sha = sha_file(CURRENT)
    if not baseline or current_sha != baseline:
        raise RuntimeError("CURRENT.json drifted from the source productization baseline")

    adopted = []
    for stage_id in UPSTREAM_STAGES:
        receipt_path = source_run / "stages" / stage_id / "receipt.json"
        receipt = load_json(receipt_path)
        terminal = str(receipt.get("terminal_status") or "")
        if terminal not in {"passed", "degraded"}:
            raise RuntimeError(f"upstream stage is not adoptable: {stage_id}={terminal}")
        if receipt.get("run_fingerprint_sha256") != run_fingerprint:
            raise RuntimeError(f"upstream run fingerprint drifted: {stage_id}")
        verified_outputs = {}
        for raw_path, expected_sha in sorted((receipt.get("output_hashes") or {}).items()):
            artifact = Path(raw_path).resolve()
            if not artifact.is_relative_to(source_run):
                raise RuntimeError(f"upstream output escaped source run: {artifact}")
            if not artifact.is_file() or sha_file(artifact) != expected_sha:
                raise RuntimeError(f"upstream output hash mismatch: {artifact}")
            verified_outputs[str(artifact)] = expected_sha
        if not verified_outputs:
            raise RuntimeError(f"upstream stage has no verified outputs: {stage_id}")
        adopted.append(
            {
                "stage_id": stage_id,
                "terminal_status": terminal,
                "stage_fingerprint_sha256": receipt.get("stage_fingerprint_sha256"),
                "receipt": str(receipt_path.resolve()),
                "receipt_sha256": sha_file(receipt_path),
                "verified_outputs": verified_outputs,
            }
        )

    blocked_package = source_run / "stages/package_v2_cq2/receipt.json"
    blocked = load_json(blocked_package)
    if blocked.get("terminal_status") != "blocked":
        raise RuntimeError("source package stage is not the expected blocked downstream boundary")

    result = {
        "schema": "mei-productization-upstream-adoption-receipt-v1",
        "product": "mei-1.0-51m",
        "status": "passed",
        "source_run": str(source_run),
        "source_run_fingerprint_sha256": run_fingerprint,
        "source_plan_sha256": sha_file(plan_path),
        "adopted_through": "narration_adapter",
        "adopted_stages": adopted,
        "blocked_boundary": {
            "stage_id": "package_v2_cq2",
            "terminal_status": "blocked",
            "receipt_sha256": sha_file(blocked_package),
            "error": blocked.get("error"),
        },
        "current_sha256": current_sha,
        "current_unchanged": True,
        "adoption_script_sha256": sha_file(Path(__file__).resolve()),
        "note": (
            "Only immutable stages through narration_adapter are adopted. "
            "Package and all downstream gates must run again under current source hashes."
        ),
    }
    atomic_json(out.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adopt(args.source_run, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
