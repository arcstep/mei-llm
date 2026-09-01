#!/usr/bin/env python3
"""Close the 17-stage local productization run without mutating CURRENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _repo import CURRENT_PATH, ROOT, architecture_contracts


PREVIOUS_STAGES = (
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
    "package_v2_cq2",
    "mtp_ablation",
    "portable_runtime_gates",
    "resource_measurement",
    "needle2_alignment_report",
)

EXPECTED_SEMANTIC_BOUNDARIES = {
    "retrieval": "independent-contrastive-head",
    "mw_disposition": "independent-20class-sidecar",
    "confidence": "independent-calibrated-binary-head",
    "narration_adapter": "independent-frozen-backbone-rank16-generation-sidecar",
    "mw_deviation": "deterministic-governance-gate-no-tensors",
}


def semantic_boundaries_complete(value: Any) -> bool:
    return isinstance(value, dict) and all(
        value.get(key) == expected
        for key, expected in EXPECTED_SEMANTIC_BOUNDARIES.items()
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def validate_receipts(run_dir: Path, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = {str(row["stage_id"]): row for row in plan.get("stages") or []}
    receipts: dict[str, dict[str, Any]] = {}
    for stage in PREVIOUS_STAGES:
        path = run_dir / "stages" / stage / "receipt.json"
        row = load_json(path)
        if row.get("stage_id") != stage or row.get("terminal_status") not in {
            "passed",
            "degraded",
            "blocked",
        }:
            raise RuntimeError(f"stage receipt is not terminal: {stage}")
        if row.get("stage_fingerprint_sha256") != expected.get(stage, {}).get(
            "stage_fingerprint_sha256"
        ):
            raise RuntimeError(f"stage fingerprint mismatch: {stage}")
        for output, digest in (row.get("output_hashes") or {}).items():
            target = Path(output)
            if not target.is_file() or sha_file(target) != digest:
                raise RuntimeError(f"stage output hash mismatch: {stage}: {output}")
        receipts[stage] = row
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_json(args.run_dir / "plan.json")
    receipts = validate_receipts(args.run_dir, plan)
    portable = load_json(args.portable_gates)
    resources = load_json(args.resource_report)
    alignment = load_json(args.alignment_report)

    sys.path.insert(0, str(ROOT / "sdk/python"))
    from mei_sdk.package import load_package

    package = load_package(args.package)
    manifest = package.manifest
    entries = (manifest.get("tensor_container") or {}).get("directory") or []
    lm_entries = [row for row in entries if row.get("role") == "lm"]
    mw_deviation_tensors = [
        row.get("name") for row in entries if "mw_deviation" in str(row.get("name") or "")
    ]
    mtp_tensors = [row.get("name") for row in entries if "mtp" in str(row.get("name") or "").lower()]
    contracts = architecture_contracts()
    expected_contracts = {
        key: contracts[key]
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    e2e_metrics = receipts["learned_top5_e2e"].get("metrics") or {}
    quality = e2e_metrics.get("quality_gates") or {}
    quality_complete = bool(quality) and all(
        isinstance(row, dict) and isinstance(row.get("ok"), bool)
        for row in quality.values()
    )
    quality_ok = quality_complete and all(bool(row["ok"]) for row in quality.values())
    stage_terminal = all(
        row.get("terminal_status") in {"passed", "degraded", "blocked"}
        for row in receipts.values()
    )
    release_gates = {
        "current_unchanged": sha_file(CURRENT_PATH)
        == plan["immutable"]["current_baseline_sha256"],
        "all_prerequisite_stages_terminal": stage_terminal,
        "no_blocked_prerequisite_stage": all(
            row.get("terminal_status") != "blocked" for row in receipts.values()
        ),
        "package_hash_and_structure": (
            package.verified_hashes
            and package.tensor_identity_complete()
            and package.portable_quantization_policy_complete()
            and package.runtime_quantization_complete()
            and package.tool_index_payload_complete()
        ),
        "contracts_match": manifest.get("contracts") == expected_contracts,
        "exact_deployed_lm_identity": (
            len(lm_entries) == 400
            and sum(int(row.get("n_params") or 0) for row in lm_entries) == 51_463_797
        ),
        "all_independent_heads_ready": (
            not package.heads.missing()
            and package.heads.narration_adapter.present
            and package.heads.narration_adapter.trained
            and package.heads.narration_adapter.status == "ready"
        ),
        "mw_semantic_boundary": (
            semantic_boundaries_complete(
                plan["immutable"].get("semantic_boundaries")
            )
            and not mw_deviation_tensors
        ),
        "mtp_not_deployed": not mtp_tensors,
        "portable_runtime_parity": portable.get("all_gates_passed") is True,
        "bounded_int8_runtime": all(
            (
                (portable.get("bounded_int8_runtime") or {}).get("kv_storage_dtype") == "int8",
                (portable.get("bounded_int8_runtime") or {}).get("activation_quantization")
                == "int8-qdq",
                (portable.get("bounded_int8_runtime") or {}).get("cache_growth_bounded") is True,
            )
        ),
        "resource_measurement_verified": package.resource_measurement_verified,
        "resource_limits": resources.get("resource_eligible") is True,
        "quality_evaluation_complete": quality_complete,
        "quality_thresholds": quality_ok,
        "needle2_critical_mechanisms": alignment.get("mechanism_alignment_validated") is True,
    }
    process_complete = stage_terminal
    release_eligible = process_complete and all(release_gates.values())
    failed = [name for name, ok in release_gates.items() if not ok]
    audit = {
        "schema": "mei-51m-productization-final-audit-v2",
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "base_id": plan["immutable"]["base"]["base_id"],
        "base_exposure_tokens": plan["immutable"]["base"]["tokens_seen_exposure"],
        "package_id": package.package_id,
        "package_path": str(args.package.resolve()),
        "package_manifest_sha256": sha_file(args.package / "mei-model.json"),
        "contracts": expected_contracts,
        "stage_count": 17,
        "terminal_prerequisite_stage_count": len(receipts),
        "final_audit_stage_self_terminal_on_write": True,
        "stage_statuses": {
            name: row["terminal_status"] for name, row in receipts.items()
        },
        "release_gates": release_gates,
        "failed_release_gates": failed,
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "current_baseline_sha256": plan["immutable"]["current_baseline_sha256"],
        "current_after_sha256": sha_file(CURRENT_PATH),
        "current_mutated": False,
        "score_parity_claimed": False,
        "cactus_compatibility_claimed": False,
    }
    proposal = {
        "schema": "mei-51m-freeze-proposal-v2",
        "proposal_only": True,
        "package_id": package.package_id,
        "candidate_path": str(args.package.resolve()),
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "failed_release_gates": failed,
        "current_baseline_sha256": audit["current_baseline_sha256"],
        "current_after_sha256": audit["current_after_sha256"],
        "current_mutated": False,
        "finalize_current_authorized": False,
    }
    write_json(args.out, audit)
    write_json(args.freeze_proposal, proposal)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--portable-gates", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-proposal", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
