#!/usr/bin/env python3
"""Close the adopted+downstream 51M productization chain without freezing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from common.paths import CURRENT_PATH, ROOT, architecture_contracts
from orchestration.run_downstream_needle2_alignment_51m import (
    UPSTREAM_REQUIRED,
    load_json,
    require_report_output,
    sha_file,
    verify_adoption,
    verify_downstream_receipt,
    verify_package_receipt,
)


SOURCE_FILES = (
    "src/model-factory/orchestration/run_downstream_final_audit_51m.py",
    "src/model-factory/orchestration/run_downstream_needle2_alignment_51m.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _terminal(receipt: dict[str, Any]) -> bool:
    return receipt.get("terminal_status") in {"passed", "degraded"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    adoption, source_plan, upstream = verify_adoption(args.adoption_receipt.resolve())
    package_receipt = verify_package_receipt(args.package_receipt.resolve())
    package_id = str(package_receipt["package_id"])
    downstream_paths = {
        "portable_runtime_gates": args.portable_receipt.resolve(),
        "narration_generation_eval": args.narration_receipt.resolve(),
        "mtp_ablation": args.mtp_receipt.resolve(),
        "resource_measurement": args.resource_receipt.resolve(),
        "arbitrary_base_entry": args.base_entry_receipt.resolve(),
        "needle2_alignment_report": args.alignment_receipt.resolve(),
    }
    downstream = {
        stage: verify_downstream_receipt(
            path,
            stage_id=stage,
            package_id=None if stage == "arbitrary_base_entry" else package_id,
        )
        for stage, path in downstream_paths.items()
    }
    portable = require_report_output(
        downstream["portable_runtime_gates"],
        args.portable_report.resolve(),
        label="portable_runtime_gates",
    )
    narration = require_report_output(
        downstream["narration_generation_eval"],
        args.narration_report.resolve(),
        label="narration_generation_eval",
    )
    resource = require_report_output(
        downstream["resource_measurement"],
        args.resource_report.resolve(),
        label="resource_measurement",
    )
    base_entry = require_report_output(
        downstream["arbitrary_base_entry"],
        args.base_entry_report.resolve(),
        label="arbitrary_base_entry",
    )
    alignment = require_report_output(
        downstream["needle2_alignment_report"],
        args.alignment_report.resolve(),
        label="needle2_alignment_report",
    )

    sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
    from mei_sdk.package import load_package

    package = load_package(args.package.resolve())
    if package.package_id != package_id or not package.resource_measurement_verified:
        raise RuntimeError("final audit requires the measured v2 package")
    manifest = package.manifest
    entries = (manifest.get("tensor_container") or {}).get("directory") or []
    lm_entries = [row for row in entries if row.get("role") == "lm"]
    mtp_tensors = [
        row
        for row in entries
        if str(row.get("role") or "").lower() == "mtp"
        or "mtp" in str(row.get("name") or "").lower()
    ]
    deviation_tensors = [
        row for row in entries if "deviation" in str(row.get("name") or "").lower()
    ]
    contracts = architecture_contracts()
    expected_contracts = {
        key: contracts[key]
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    base = source_plan["immutable"]["base"]
    base_release = Path(base["release"])
    base_weights = Path(base["weights"])
    base_immutable = (
        base_release.is_file()
        and base_weights.is_file()
        and sha_file(base_release) == base["release_sha256"]
        and sha_file(base_weights) == base["weights_sha256"]
    )
    upstream_terminal = all(_terminal(row["receipt"]) for row in upstream.values())
    downstream_terminal = all(_terminal(row) for row in downstream.values())
    learned = upstream["learned_top5_e2e"]["receipt"].get("metrics") or {}
    quality_gates = learned.get("quality_gates") or {}
    locked_quality_complete = bool(quality_gates) and all(
        isinstance(row, dict) and isinstance(row.get("ok"), bool)
        for row in quality_gates.values()
    )
    locked_quality_pass = locked_quality_complete and all(
        bool(row["ok"]) for row in quality_gates.values()
    )
    confidence = upstream["confidence_calibration"]["receipt"].get("metrics") or {}
    confidence_non_degenerate = (
        int(confidence.get("n") or 0) > 0
        and int(confidence.get("n_positive") or 0) > 0
        and int(confidence.get("n_negative") or 0) > 0
    )
    narration_metrics = narration.get("metrics") or {}
    narration_mechanism = (
        narration.get("packed_cq2_evaluated") is True
        and int(narration_metrics.get("rows") or 0) == 600
        and narration_metrics.get("mechanism_status") == "passed"
        and float(narration_metrics.get("deterministic_fallback_exact_rate") or 0.0)
        == 1.0
    )
    narration_adapter_quality = (
        narration_metrics.get("learned_adapter_quality") == "validated"
    )
    exact_lm = len(lm_entries) == 400 and sum(
        int(row.get("n_params") or 0) for row in lm_entries
    ) == 51_463_797
    semantic_boundaries = source_plan["immutable"].get("semantic_boundaries") or {}
    mw_boundary = (
        semantic_boundaries.get("mw_disposition")
        == "independent-20class-sidecar"
        and semantic_boundaries.get("mw_deviation")
        == "deterministic-governance-gate-no-tensors"
        and not deviation_tensors
    )
    bounded = portable.get("bounded_int8_runtime") or {}
    release_gates = {
        "current_unchanged": sha_file(CURRENT_PATH)
        == source_plan["immutable"]["current_baseline_sha256"],
        "base_release_and_weights_immutable": base_immutable,
        "all_upstream_adopted_stages_terminal": upstream_terminal,
        "all_downstream_stages_terminal": downstream_terminal,
        "superseded_blocked_package_not_reused": (
            (adoption.get("blocked_boundary") or {}).get("stage_id") == "package_v2_cq2"
            and package_receipt.get("stage_id") == "package_v2_cq2_downstream"
        ),
        "package_hash_and_structure": (
            package.verified_hashes
            and package.tensor_identity_complete()
            and package.portable_quantization_policy_complete()
            and package.tool_index_payload_complete()
        ),
        "contracts_match": manifest.get("contracts") == expected_contracts,
        "exact_deployed_lm_identity": exact_lm,
        "all_independent_heads_ready": not package.heads.missing(),
        "mw_semantic_boundary": mw_boundary,
        "mtp_training_only_and_not_deployed": (
            downstream["mtp_ablation"].get("training_only") is True
            and downstream["mtp_ablation"].get("temporary_heads_exported") is False
            and not mtp_tensors
        ),
        "portable_runtime_parity": portable.get("all_gates_passed") is True,
        "bounded_int8_runtime": (
            bounded.get("kv_storage_dtype") == "int8"
            and bounded.get("activation_quantization") == "int8-qdq"
            and bounded.get("cache_growth_bounded") is True
        ),
        "narration_grounded_fallback": narration_mechanism,
        "narration_learned_adapter_quality": narration_adapter_quality,
        "arbitrary_frozen_base_entry": (
            (base_entry.get("claims") or {}).get("productizer_is_frozen_base_agnostic")
            is True
        ),
        "resource_measurement_verified": package.resource_measurement_verified,
        "resource_limits": resource.get("resource_eligible") is True,
        "locked_quality_evaluation_complete": locked_quality_complete,
        "locked_quality_thresholds": locked_quality_pass,
        "confidence_calibration_non_degenerate": confidence_non_degenerate,
        "needle2_critical_mechanisms": alignment.get("mechanism_alignment_validated")
        is True,
        "needle2_release_quality": alignment.get("release_quality_validated") is True,
    }
    process_complete = upstream_terminal and downstream_terminal and all(
        (
            release_gates["current_unchanged"],
            release_gates["base_release_and_weights_immutable"],
            release_gates["superseded_blocked_package_not_reused"],
            release_gates["package_hash_and_structure"],
            release_gates["contracts_match"],
            release_gates["exact_deployed_lm_identity"],
            release_gates["all_independent_heads_ready"],
            release_gates["mw_semantic_boundary"],
            release_gates["mtp_training_only_and_not_deployed"],
            release_gates["portable_runtime_parity"],
            release_gates["bounded_int8_runtime"],
            release_gates["narration_grounded_fallback"],
            release_gates["resource_measurement_verified"],
            release_gates["locked_quality_evaluation_complete"],
            release_gates["arbitrary_frozen_base_entry"],
            release_gates["needle2_critical_mechanisms"],
        )
    )
    release_eligible = process_complete and all(release_gates.values())
    failed = [name for name, passed in release_gates.items() if not passed]
    stage_statuses = {
        **{
            stage: upstream[stage]["receipt"]["terminal_status"]
            for stage in UPSTREAM_REQUIRED
        },
        "package_v2_cq2_downstream": package_receipt["terminal_status"],
        **{stage: row["terminal_status"] for stage, row in downstream.items()},
        "final_audit": "passed",
    }
    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    input_paths = [
        args.adoption_receipt,
        args.package_receipt,
        *downstream_paths.values(),
        args.portable_report,
        args.narration_report,
        args.resource_report,
        args.base_entry_report,
        args.alignment_report,
    ]
    input_hashes = {
        str(Path(path).resolve()): sha_file(Path(path).resolve()) for path in input_paths
    }
    fingerprint = hashlib.sha256(
        canonical_bytes({"inputs": input_hashes, "sources": source_hashes})
    ).hexdigest()
    audit = {
        "schema": "mei-51m-productization-final-audit-v3",
        "terminal_status": "passed",
        "stage_fingerprint_sha256": fingerprint,
        "source_run_fingerprint_sha256": source_plan["run_fingerprint_sha256"],
        "base_id": base["base_id"],
        "base_exposure_tokens": base["tokens_seen_exposure"],
        "package_id": package_id,
        "package_path": str(args.package.resolve()),
        "package_manifest_sha256": sha_file(args.package.resolve() / "mei-model.json"),
        "contracts": expected_contracts,
        "stage_count_including_final_audit": len(stage_statuses),
        "stage_statuses": stage_statuses,
        "release_gates": release_gates,
        "failed_release_gates": failed,
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "quality_diagnostics": {
            "learned_top5": learned,
            "confidence": confidence,
            "narration": narration_metrics,
            "mw_disposition_last_loss": (
                upstream["mw_disposition_20class"]["receipt"].get("metrics") or {}
            ).get("last_loss"),
            "mw_random_cross_entropy_reference": math.log(20),
        },
        "upstream_blocked_boundary_preserved": adoption.get("blocked_boundary"),
        "upstream_blocked_boundary_superseded_by": str(args.package_receipt.resolve()),
        "current_baseline_sha256": source_plan["immutable"][
            "current_baseline_sha256"
        ],
        "current_after_sha256": sha_file(CURRENT_PATH),
        "current_mutated": False,
        "score_parity_claimed": False,
        "cactus_compatibility_claimed": False,
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
    }
    proposal = {
        "schema": "mei-51m-freeze-proposal-v3",
        "proposal_only": True,
        "package_id": package_id,
        "candidate_path": str(args.package.resolve()),
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "failed_release_gates": failed,
        "current_baseline_sha256": audit["current_baseline_sha256"],
        "current_after_sha256": audit["current_after_sha256"],
        "current_mutated": False,
        "finalize_current_authorized": False,
        "public_release_authorized": False,
    }
    out_dir = args.out_dir.resolve()
    audit_path = out_dir / "final-audit.json"
    proposal_path = out_dir / "freeze-proposal.json"
    receipt_path = out_dir / "receipt.json"
    write_once(audit_path, audit)
    write_once(proposal_path, proposal)
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "final_audit",
        "terminal_status": "passed",
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "product": "mei-1.0-51m",
        "package_id": package_id,
        "stage_fingerprint_sha256": fingerprint,
        "failed_release_gates": failed,
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
        "output_hashes": {
            str(audit_path): sha_file(audit_path),
            str(proposal_path): sha_file(proposal_path),
        },
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adoption-receipt", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--portable-receipt", type=Path, required=True)
    parser.add_argument("--portable-report", type=Path, required=True)
    parser.add_argument("--narration-receipt", type=Path, required=True)
    parser.add_argument("--narration-report", type=Path, required=True)
    parser.add_argument("--mtp-receipt", type=Path, required=True)
    parser.add_argument("--resource-receipt", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--base-entry-receipt", type=Path, required=True)
    parser.add_argument("--base-entry-report", type=Path, required=True)
    parser.add_argument("--alignment-receipt", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
