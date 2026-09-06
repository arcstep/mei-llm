#!/usr/bin/env python3
"""Verify a recovered native-v2 package without rewriting upstream receipts.

This is intentionally a read-only verifier for the package directory.  It is
used when immutable training stages were adopted after a downstream packaging
bug was fixed, so the original blocked package receipt remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT


PACKAGE_LIMIT = 18 * 1024 * 1024
EXPECTED_LM_PARAMETERS = 51_463_797
EXPECTED_LM_TENSORS = 400
EXPECTED_TOTAL_TENSORS = 410
EXPECTED_ROLE_COUNTS = {
    "lm": 400,
    "contrastive": 3,
    "mw_disposition": 2,
    "confidence": 3,
    "narration_adapter": 2,
}
REQUIRED_CAPABILITIES = (
    "retrieval",
    "full_call",
    "mw_disposition",
    "confidence",
    "multi_step",
    "narration",
)
BOUND_SOURCES = (
    "src/model-factory/release/pack_cq2_v2_51m.py",
    "src/model-factory/orchestration/adopt_productization_upstream_51m.py",
    "platform/_shared/spec/model-package-v2.schema.json",
    "platform/python-sdk/mei_sdk/package.py",
    "platform/python-sdk/mei_sdk/cq2.py",
    "platform/_shared/rust/mei-sdk-core/src/package.rs",
    "platform/_shared/rust/mei-sdk-core/src/packed.rs",
    "platform/browser-sdk/package.mjs",
    "platform/browser-sdk/cq2.mjs",
    "src/model-factory/orchestration/productize_sft_v3_300m.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise RuntimeError(f"refusing to overwrite a different package receipt: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _verify_adoption(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    adoption = load_json(path)
    if (
        adoption.get("schema")
        != "mei-productization-upstream-adoption-receipt-v1"
        or adoption.get("status") != "passed"
        or adoption.get("current_unchanged") is not True
        or adoption.get("adopted_through") != "narration_adapter"
    ):
        raise RuntimeError("upstream adoption receipt is not complete")
    if sha_file(CURRENT_PATH) != adoption.get("current_sha256"):
        raise RuntimeError("CURRENT.json drifted after upstream adoption")

    source_run = Path(str(adoption.get("source_run") or "")).resolve()
    plan_path = source_run / "plan.json"
    plan = load_json(plan_path)
    if (
        sha_file(plan_path) != adoption.get("source_plan_sha256")
        or plan.get("run_fingerprint_sha256")
        != adoption.get("source_run_fingerprint_sha256")
    ):
        raise RuntimeError("adopted source plan identity drifted")

    stages = list(adoption.get("adopted_stages") or [])
    if len(stages) != 12:
        raise RuntimeError("upstream adoption stage inventory is incomplete")
    for stage in stages:
        receipt_path = Path(str(stage.get("receipt") or "")).resolve()
        if not receipt_path.is_relative_to(source_run):
            raise RuntimeError(f"adopted receipt escaped source run: {receipt_path}")
        if sha_file(receipt_path) != stage.get("receipt_sha256"):
            raise RuntimeError(f"adopted receipt drifted: {receipt_path}")
        for raw, digest in (stage.get("verified_outputs") or {}).items():
            artifact = Path(raw).resolve()
            if not artifact.is_relative_to(source_run):
                raise RuntimeError(f"adopted output escaped source run: {artifact}")
            if not artifact.is_file() or sha_file(artifact) != digest:
                raise RuntimeError(f"adopted output drifted: {artifact}")
    return adoption, plan


def _current_source_drift(source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    drift: list[dict[str, Any]] = []
    for relative, expected in sorted(source_manifest.items()):
        source = (ROOT / str(relative)).resolve()
        if not source.is_relative_to(ROOT):
            raise RuntimeError(f"SFT-v4 source manifest escaped repository: {relative}")
        actual = sha_file(source) if source.is_file() else None
        if actual != expected:
            drift.append(
                {
                    "path": str(relative),
                    "expected_sha256": expected,
                    "current_sha256": actual,
                }
            )
    return drift


def _verify_v4_productization_run(
    run_dir: Path,
    package_dir: Path,
    *,
    current_source_reevaluation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import orchestration.productize_51m as lifecycle

    run_dir = run_dir.resolve()
    plan_path = run_dir / "plan.json"
    plan = load_json(plan_path)
    immutable = plan.get("immutable") or {}
    if (
        plan.get("schema") != "mei-51m-sft-v4-productization-plan-v1"
        or plan.get("run_fingerprint_sha256") != sha_bytes(canonical_bytes(immutable))
        or immutable.get("stage_graph_mode")
        != "verified-productization-prefix-continuation-v1"
        or immutable.get("current_baseline_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("SFT-v4 productization plan identity is invalid")
    source_drift = _current_source_drift(immutable.get("source_manifest") or {})
    if source_drift and not current_source_reevaluation:
        raise RuntimeError(
            "SFT-v4 productization source drifted: " + source_drift[0]["path"]
        )

    expected_stages = (
        "adopt_productization_prefix_v4",
        "sidecar_eval_v4",
        "locked_test_eval_v4",
        "package_v2_cq2_v4",
    )
    if tuple(row.get("stage_id") for row in plan.get("stages") or []) != expected_stages:
        raise RuntimeError("SFT-v4 finalization stage graph is invalid")
    stage_rows = lifecycle.stage_map(plan)
    stages: dict[str, Any] = {}
    for stage_id in expected_stages:
        receipt_path = run_dir / "stages" / stage_id / "receipt.json"
        receipt = load_json(receipt_path)
        expected_terminal = (
            {"passed", "degraded"}
            if stage_id in {"sidecar_eval_v4", "locked_test_eval_v4"}
            else {"passed"}
        )
        input_fingerprint, _input_evidence = lifecycle._stage_input_fingerprint(
            run_dir, plan, stage_id
        )
        outputs = receipt.get("output_hashes") or {}
        if (
            receipt.get("terminal_status") not in expected_terminal
            or receipt.get("run_fingerprint_sha256")
            != plan.get("run_fingerprint_sha256")
            or receipt.get("stage_fingerprint_sha256")
            != stage_rows[stage_id].get("stage_fingerprint_sha256")
            or receipt.get("stage_input_fingerprint_sha256") != input_fingerprint
            or not outputs
            or any(
                not Path(raw).is_file() or sha_file(Path(raw)) != digest
                for raw, digest in outputs.items()
            )
        ):
            raise RuntimeError(f"SFT-v4 finalization stage is not reusable: {stage_id}")
        stages[stage_id] = {
            "terminal_status": receipt["terminal_status"],
            "receipt": str(receipt_path),
            "receipt_sha256": sha_file(receipt_path),
            "stage_fingerprint_sha256": receipt["stage_fingerprint_sha256"],
        }

    package_outputs = load_json(
        run_dir / "stages/package_v2_cq2_v4/receipt.json"
    ).get("output_hashes") or {}
    package_files = sorted(path for path in package_dir.iterdir() if path.is_file())
    for artifact in package_files:
        resolved = str(artifact.resolve())
        if package_outputs.get(resolved) != sha_file(artifact):
            raise RuntimeError(f"SFT-v4 package file is not stage-bound: {artifact}")
    binding = {
        "source_kind": "sft-v4-productization-run",
        "source_run": str(run_dir),
        "source_run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "source_plan": str(plan_path),
        "source_plan_sha256": sha_file(plan_path),
        "historical_training_source_current_match": not source_drift,
        "current_source_reevaluation": bool(current_source_reevaluation),
        "source_drift": source_drift,
        "source_drift_disposition": (
            "recorded_confound_for_current_runtime_reevaluation"
            if source_drift
            else "none"
        ),
        "stages": stages,
    }
    return binding, plan


def verify(
    package_dir: Path,
    adoption_path: Path | None,
    out: Path,
    *,
    productization_run: Path | None = None,
    current_source_reevaluation: bool = False,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    if (adoption_path is None) == (productization_run is None):
        raise RuntimeError(
            "exactly one source binding is required: adoption receipt or productization run"
        )
    if productization_run is not None:
        source_binding, source_plan = _verify_v4_productization_run(
            productization_run,
            package_dir,
            current_source_reevaluation=current_source_reevaluation,
        )
        source_binding_path = Path(str(source_binding["source_plan"]))
    else:
        if current_source_reevaluation:
            raise RuntimeError(
                "--current-source-reevaluation requires --productization-run"
            )
        assert adoption_path is not None
        adoption_path = adoption_path.resolve()
        adoption, source_plan = _verify_adoption(adoption_path)
        source_binding = {
            "source_kind": "upstream-adoption-receipt",
            "source_run_fingerprint_sha256": adoption[
                "source_run_fingerprint_sha256"
            ],
            "source_adoption_receipt": str(adoption_path),
            "source_adoption_receipt_sha256": sha_file(adoption_path),
        }
        source_binding_path = adoption_path

    sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
    from mei_sdk.package import load_package

    package = load_package(package_dir, verify_hashes=True)
    manifest = package.manifest
    if manifest.get("package_format") != "mei-model-package-v2":
        raise RuntimeError("recovered package is not native v2")
    if manifest.get("contracts") != (source_plan.get("immutable") or {}).get("contracts"):
        raise RuntimeError("package contracts disagree with the adopted source plan")

    entries = list((manifest.get("tensor_container") or {}).get("directory") or [])
    role_counts = {
        role: sum(1 for row in entries if row.get("role") == role)
        for role in EXPECTED_ROLE_COUNTS
    }
    if len(entries) != EXPECTED_TOTAL_TENSORS or role_counts != EXPECTED_ROLE_COUNTS:
        raise RuntimeError(
            f"portable tensor inventory mismatch: total={len(entries)} roles={role_counts}"
        )
    lm_entries = [row for row in entries if row.get("role") == "lm"]
    lm_parameters = sum(int(row.get("n_params") or 0) for row in lm_entries)
    if len(lm_entries) != EXPECTED_LM_TENSORS or lm_parameters != EXPECTED_LM_PARAMETERS:
        raise RuntimeError("deployed LM tensor identity is not the canonical 51M identity")

    mtp_tensors = [
        str(row.get("name") or "")
        for row in entries
        if "mtp" in str(row.get("name") or "").lower()
    ]
    mw_deviation_tensors = [
        str(row.get("name") or "")
        for row in entries
        if "mw_deviation" in str(row.get("name") or "").lower()
    ]
    if mtp_tensors or mw_deviation_tensors:
        raise RuntimeError(
            "forbidden deployment tensors: "
            + json.dumps(
                {"mtp": mtp_tensors, "mw_deviation": mw_deviation_tensors},
                ensure_ascii=False,
            )
        )

    head_report = package.heads.as_dict()
    not_ready = [
        name
        for name in EXPECTED_ROLE_COUNTS
        if name != "lm"
        and (
            (head_report.get(name) or {}).get("status") != "ready"
            or (head_report.get(name) or {}).get("present") is not True
            or (head_report.get(name) or {}).get("trained") is not True
        )
    ]
    capabilities = package.capabilities()
    false_capabilities = [
        name for name in REQUIRED_CAPABILITIES if capabilities.get(name) is not True
    ]
    if not_ready or false_capabilities:
        raise RuntimeError(
            "package heads/capabilities are incomplete: "
            + json.dumps(
                {"heads": not_ready, "capabilities": false_capabilities},
                ensure_ascii=False,
            )
        )

    structural_checks = {
        "verified_hashes": package.verified_hashes,
        "tensor_identity_complete": package.tensor_identity_complete(),
        "portable_quantization_policy_complete": (
            package.portable_quantization_policy_complete()
        ),
        "tool_index_payload_complete": package.tool_index_payload_complete(),
        "packed_structure_ready": package.packed_structure_ready(),
    }
    if not all(structural_checks.values()):
        raise RuntimeError(f"package structure validation failed: {structural_checks}")

    package_files = sorted(path for path in package_dir.iterdir() if path.is_file())
    package_bytes = sum(path.stat().st_size for path in package_files)
    if package_bytes > PACKAGE_LIMIT:
        raise RuntimeError(
            f"unmeasured package exceeds 18 MiB: {package_bytes} > {PACKAGE_LIMIT}"
        )
    output_hashes = {str(path.resolve()): sha_file(path) for path in package_files}
    source_hashes = {
        relative: sha_file(ROOT / relative)
        for relative in BOUND_SOURCES
        if (ROOT / relative).is_file()
    }
    if set(source_hashes) != set(BOUND_SOURCES):
        missing = sorted(set(BOUND_SOURCES) - set(source_hashes))
        raise RuntimeError(f"package verifier source inventory is incomplete: {missing}")

    container = manifest.get("tensor_container") or {}
    fingerprint_input = {
        "source_binding_sha256": sha_file(source_binding_path),
        "source_run_fingerprint_sha256": source_binding[
            "source_run_fingerprint_sha256"
        ],
        "package_id": package.package_id,
        "output_hashes": output_hashes,
        "source_hashes": source_hashes,
    }
    result = {
        "schema": "mei-productization-downstream-package-receipt-v1",
        "stage_id": "package_v2_cq2_downstream",
        "terminal_status": "passed",
        "product": "mei-1.0-51m",
        "package_id": package.package_id,
        "package_path": str(package_dir),
        "stage_fingerprint_sha256": sha_bytes(canonical_bytes(fingerprint_input)),
        "source_run_fingerprint_sha256": source_binding[
            "source_run_fingerprint_sha256"
        ],
        "source_binding": source_binding,
        "source_binding_sha256": sha_file(source_binding_path),
        "contracts": manifest["contracts"],
        "quant_math_id": container.get("quant_math_id"),
        "tensor_container_sha256": container.get("sha256"),
        "tensor_count": len(entries),
        "lm_tensor_count": len(lm_entries),
        "lm_parameter_count": lm_parameters,
        "role_counts": role_counts,
        "mtp_tensor_count": 0,
        "mw_deviation_tensor_count": 0,
        "heads": head_report,
        "capabilities": {
            name: capabilities[name] for name in REQUIRED_CAPABILITIES
        },
        "structural_checks": structural_checks,
        "package_bytes": package_bytes,
        "package_limit_bytes": PACKAGE_LIMIT,
        "package_within_limit": True,
        "output_hashes": output_hashes,
        "source_hashes": source_hashes,
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
        "current_source_reevaluation": bool(current_source_reevaluation),
        "release_claim_from_reevaluation": False,
    }
    atomic_json(out.resolve(), result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adoption-receipt", type=Path)
    source.add_argument("--productization-run", type=Path)
    parser.add_argument(
        "--current-source-reevaluation",
        action="store_true",
        help=(
            "re-evaluate an immutable historical SFT-v4 package with current runtime "
            "source while recording, rather than hiding, training-source drift"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            verify(
                args.package,
                args.adoption_receipt,
                args.out,
                productization_run=args.productization_run,
                current_source_reevaluation=args.current_source_reevaluation,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
