#!/usr/bin/env python3
"""Prove the productizer is bound to a frozen contract, not a 300M label.

The real execution evidence in this run remains the immutable 300M base.  To
prove future 600M/900M bases use the same entrypoint without pretending those
artifacts already exist, this verifier also builds schema-only dry-run plans
whose exposure and model id vary while the canonical architecture/tokenizer
contract stays fixed.  Those fixtures are explicitly not training artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from _repo import CURRENT_PATH, ROOT, architecture_contracts
from productize_51m import build_plan, parse_args as parse_productize_args, validate_base


DEFAULT_BASE = ROOT / "base/mei-1.0-51m-base-scratch300m-v1"
SOURCE_FILES = (
    "training/mei-1.0-51m-train-v1/verify_arbitrary_base_entry_51m.py",
    "training/mei-1.0-51m-train-v1/productize_51m.py",
    "architecture/mei-1.0-51m-arch-v1/architecture_contract.py",
)
FIXTURE_EXPOSURES = (600_000_000, 900_000_000)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _plan_args(
    *, release: Path, weights: Path, run_dir: Path, package_id: str
) -> argparse.Namespace:
    args = parse_productize_args([])
    args.base_release = release
    args.base_weights = weights
    args.run_dir = run_dir
    args.package_id = package_id
    args.allow_live_cpt = True
    args.dry_run = True
    return args


def _plan_view(plan: dict[str, Any]) -> dict[str, Any]:
    base = plan["immutable"]["base"]
    return {
        "base_id": base["base_id"],
        "tokens_seen_exposure": base["tokens_seen_exposure"],
        "weights_sha256": base["weights_sha256"],
        "release_sha256": base["release_sha256"],
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "stage_count": len(plan.get("stages") or []),
        "first_stage": (plan.get("stages") or [{}])[0].get("stage_id"),
        "last_stage": (plan.get("stages") or [{}])[-1].get("stage_id"),
        "contracts": plan["immutable"]["contracts"],
        "semantic_boundaries": plan["immutable"]["semantic_boundaries"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    release_path = args.base_release.resolve()
    weights_path = args.base_weights.resolve()
    actual_base = validate_base(release_path, weights_path)
    actual_plan = build_plan(
        _plan_args(
            release=release_path,
            weights=weights_path,
            run_dir=args.fixture_root / "actual-300m",
            package_id="mei-1.0-51m-base-entry-actual300m-dryrun",
        )
    )
    source_release = load_json(release_path)
    fixture_views = []
    with tempfile.TemporaryDirectory(prefix="mei-51m-base-entry-") as temporary:
        temporary_root = Path(temporary)
        for exposure in FIXTURE_EXPOSURES:
            fixture = dict(source_release)
            fixture["model_id"] = f"mei-1.0-51m-base-schema-fixture-{exposure}"
            fixture["tokens_seen_exposure"] = exposure
            if "tokens_seen" in fixture:
                fixture["tokens_seen"] = exposure
            fixture_path = temporary_root / f"RELEASE-{exposure}.json"
            fixture_path.write_bytes(canonical_bytes(fixture) + b"\n")
            plan = build_plan(
                _plan_args(
                    release=fixture_path,
                    weights=weights_path,
                    run_dir=args.fixture_root / f"schema-{exposure}",
                    package_id=f"mei-1.0-51m-base-entry-schema-{exposure}-dryrun",
                )
            )
            view = _plan_view(plan)
            view.update(
                {
                    "schema_fixture_only": True,
                    "actual_artifact_available": False,
                    "reuses_300m_weights_only_to_test_control_plane": True,
                    "training_or_quality_claimed": False,
                }
            )
            fixture_views.append(view)
    actual_view = _plan_view(actual_plan)
    fingerprints = [actual_view["run_fingerprint_sha256"], *[
        row["run_fingerprint_sha256"] for row in fixture_views
    ]]
    contracts = architecture_contracts()
    invariant_contracts = {
        key: contracts[key]
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    checks = {
        "actual_300m_release_and_weights_verified": (
            actual_base["weights_sha256"] == sha_file(weights_path)
            and actual_base["tokens_seen_exposure"] == actual_view["tokens_seen_exposure"]
        ),
        "all_plans_use_same_stage_graph": all(
            row["stage_count"] == actual_view["stage_count"]
            and row["first_stage"] == actual_view["first_stage"]
            and row["last_stage"] == actual_view["last_stage"]
            for row in fixture_views
        ),
        "all_plans_use_canonical_contracts": all(
            row["contracts"] == invariant_contracts
            for row in [actual_view, *fixture_views]
        ),
        "exposure_changes_run_fingerprint": len(set(fingerprints)) == len(fingerprints),
        "entrypoint_accepts_explicit_release_weights_package_and_run": True,
        "current_unchanged": sha_file(CURRENT_PATH) == args.current_sha256,
    }
    return {
        "schema": "mei-arbitrary-frozen-base-entry-report-v1",
        "product": "mei-1.0-51m",
        "terminal_status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "actual_execution_base": {
            **actual_view,
            "release": str(release_path),
            "release_sha256": sha_file(release_path),
            "actual_artifact_available": True,
            "schema_fixture_only": False,
        },
        "future_exposure_schema_fixtures": fixture_views,
        "entrypoint": {
            "script": "training/mei-1.0-51m-train-v1/productize_51m.py",
            "required_overrides": [
                "--base-release",
                "--base-weights",
                "--package-id",
                "--run-dir",
            ],
            "exposure_allowlist": None,
            "compatibility_basis": "weight-contract+tokenizer+exact-tensor-identity",
        },
        "claims": {
            "productizer_is_frozen_base_agnostic": all(checks.values()),
            "actual_full_chain_validated_at_exposure": actual_view[
                "tokens_seen_exposure"
            ],
            "600m_or_900m_artifact_availability_claimed": False,
            "600m_or_900m_quality_claimed": False,
            "repeat_full_training_for_schema_proof_required": False,
        },
        "current_sha256": sha_file(CURRENT_PATH),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    fingerprint_inputs = {
        "base_release_sha256": sha_file(args.base_release),
        "base_weights_sha256": sha_file(args.base_weights),
        "current_sha256": args.current_sha256,
        "fixture_exposures": list(FIXTURE_EXPOSURES),
        "source_hashes": source_hashes,
    }
    fingerprint = hashlib.sha256(canonical_bytes(fingerprint_inputs)).hexdigest()
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "stage_fingerprint_sha256": fingerprint,
            "fixture_exposures": list(FIXTURE_EXPOSURES),
        }
    report = build_report(args)
    if report["terminal_status"] != "passed":
        raise RuntimeError(f"arbitrary-base entry checks failed: {report['checks']}")
    report.update(
        {
            "stage_fingerprint_sha256": fingerprint,
            "source_hashes": source_hashes,
        }
    )
    report_path = args.out_dir / "arbitrary-base-entry.json"
    receipt_path = args.out_dir / "receipt.json"
    write_once(report_path, report)
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "arbitrary_base_entry",
        "terminal_status": "passed",
        "process_complete": True,
        "product": "mei-1.0-51m",
        "stage_fingerprint_sha256": fingerprint,
        "actual_full_chain_validated_at_exposure": report["claims"][
            "actual_full_chain_validated_at_exposure"
        ],
        "future_schema_exposures": list(FIXTURE_EXPOSURES),
        "output_hashes": {str(report_path.resolve()): sha_file(report_path)},
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-release", type=Path, default=DEFAULT_BASE / "RELEASE.json"
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=DEFAULT_BASE / "mei-1.0-51m-base-scratch300m-v1.npz",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.current_sha256 = sha_file(CURRENT_PATH)
    return args


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
