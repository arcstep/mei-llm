#!/usr/bin/env python3
"""Run the hash-bound runtime/resource gates needed by a paired SFT-v4 comparison.

This is deliberately a process driver, not another implementation of any gate.
Each child owns its write-once receipt and can be resumed safely after a terminal
failure.  The driver never polls training progress and never mutates CURRENT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT


HERE = Path(__file__).resolve().parent
DEFAULT_NARRATION_RELEASE = (
    ROOT
    / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-narration-sft-agent300m-v3"
)
DEFAULT_REPLAY_CORPUS = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/cpt-delta/lm-v1"
CHILD_SCRIPTS = {
    "package": HERE / "verify_downstream_package_51m.py",
    "portable": HERE / "run_downstream_portable_gates_51m.py",
    "narration": ROOT / "model-factory/evaluation/heads/evaluate_narration_adapter_51m.py",
    "mtp": HERE / "run_downstream_mtp_ablation_51m.py",
    "resources": HERE / "run_downstream_resource_gates_51m.py",
    "base_entry": HERE / "verify_arbitrary_base_entry_51m.py",
}


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


def write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def command_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    out = args.out_root.resolve()
    downstream = out / "downstream"
    package_receipt = downstream / "package-verification.json"
    portable_dir = downstream / "portable-gates"
    narration_dir = downstream / "narration-package-eval"
    mtp_dir = downstream / "mtp-ablation"
    resources_dir = downstream / "resources"
    measured_package = out / "candidate" / f"{args.package.resolve().name}-measured"
    base_entry_dir = downstream / "arbitrary-base-entry"
    fixture_root = out / "fixtures/arbitrary-base-entry"

    package_command = [
        sys.executable,
        str(CHILD_SCRIPTS["package"]),
        "--package",
        str(args.package.resolve()),
        "--productization-run",
        str(args.productization_run.resolve()),
        "--out",
        str(package_receipt),
    ]
    if args.current_source_reevaluation:
        package_command.append("--current-source-reevaluation")

    return [
        {
            "stage": "package_verification",
            "command": package_command,
            "receipt": package_receipt,
        },
        {
            "stage": "portable_runtime_gates",
            "command": [
                sys.executable,
                str(CHILD_SCRIPTS["portable"]),
                "--package",
                str(args.package.resolve()),
                "--package-receipt",
                str(package_receipt),
                "--out-dir",
                str(portable_dir),
            ],
            "receipt": portable_dir / "receipt.json",
        },
        {
            "stage": "narration_generation_eval",
            "command": [
                sys.executable,
                str(CHILD_SCRIPTS["narration"]),
                "--package",
                str(args.package.resolve()),
                "--package-receipt",
                str(package_receipt),
                "--release",
                str(args.narration_release.resolve()),
                "--out-dir",
                str(narration_dir),
            ],
            "receipt": narration_dir / "receipt.json",
        },
        {
            "stage": "mtp_ablation",
            "command": [
                sys.executable,
                str(CHILD_SCRIPTS["mtp"]),
                "--package",
                str(args.package.resolve()),
                "--package-receipt",
                str(package_receipt),
                "--master",
                str(args.master.resolve()),
                "--replay-corpus",
                str(args.replay_corpus.resolve()),
                "--steps",
                str(args.mtp_steps),
                "--out-dir",
                str(mtp_dir),
            ],
            "receipt": mtp_dir / "receipt.json",
        },
        {
            "stage": "resource_measurement",
            "command": [
                sys.executable,
                str(CHILD_SCRIPTS["resources"]),
                "--source-package",
                str(args.package.resolve()),
                "--package-receipt",
                str(package_receipt),
                "--portable-receipt",
                str(portable_dir / "receipt.json"),
                "--out-package",
                str(measured_package),
                "--out-dir",
                str(resources_dir),
            ],
            "receipt": resources_dir / "receipt.json",
        },
        {
            "stage": "arbitrary_base_entry",
            "command": [
                sys.executable,
                str(CHILD_SCRIPTS["base_entry"]),
                "--base-release",
                str(args.base_release.resolve()),
                "--base-weights",
                str(args.base_weights.resolve()),
                "--out-dir",
                str(base_entry_dir),
                "--fixture-root",
                str(fixture_root),
            ],
            "receipt": base_entry_dir / "receipt.json",
        },
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "productization_plan": args.productization_run.resolve() / "plan.json",
        "package_manifest": args.package.resolve() / "mei-model.json",
        "master": args.master.resolve(),
        "base_release": args.base_release.resolve(),
        "base_weights": args.base_weights.resolve(),
        "narration_manifest": args.narration_release.resolve() / "manifest.json",
        "replay_manifest": args.replay_corpus.resolve() / "manifest.json",
        "current": CURRENT_PATH,
    }
    input_hashes = {str(path): sha_file(path) for path in inputs.values()}
    source_hashes = {
        str(path.relative_to(ROOT)): sha_file(path)
        for path in [Path(__file__).resolve(), *CHILD_SCRIPTS.values()]
    }
    commands = [
        {
            "stage": row["stage"],
            "command": row["command"],
            "receipt": str(row["receipt"]),
        }
        for row in command_plan(args)
    ]
    immutable = {
        "schema": "mei-51m-paired-runtime-gates-plan-v1",
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
        "commands": commands,
        "current_source_reevaluation": bool(args.current_source_reevaluation),
        "current_baseline_sha256": sha_file(CURRENT_PATH),
        "mtp_steps": int(args.mtp_steps),
    }
    return {
        **immutable,
        "plan_fingerprint_sha256": hashlib.sha256(
            canonical_bytes(immutable)
        ).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    plan_path = args.out_root.resolve() / "runtime-gates-plan.json"
    write_once(plan_path, plan)
    receipts: dict[str, dict[str, Any]] = {}
    for row in command_plan(args):
        subprocess.run(row["command"], cwd=ROOT, check=True)
        receipt_path = Path(row["receipt"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("terminal_status") not in {"passed", "degraded"}:
            raise RuntimeError(f"stage did not reach a terminal receipt: {row['stage']}")
        receipts[row["stage"]] = {
            "path": str(receipt_path),
            "sha256": sha_file(receipt_path),
            "terminal_status": receipt["terminal_status"],
        }
    result = {
        "schema": "mei-51m-paired-runtime-gates-receipt-v1",
        "terminal_status": "passed",
        "process_complete": True,
        "plan": str(plan_path),
        "plan_sha256": sha_file(plan_path),
        "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
        "stages": receipts,
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": sha_file(CURRENT_PATH)
        == plan["current_baseline_sha256"],
        "scope": "package-python-browser-wasm-narration-mtp-resource-base-entry",
        "needle2_alignment_and_freeze_audit": "separate",
    }
    if not result["current_unchanged"]:
        raise RuntimeError("CURRENT.json drifted during runtime gates")
    write_once(args.out_root.resolve() / "runtime-gates-receipt.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--productization-run", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--narration-release", type=Path, default=DEFAULT_NARRATION_RELEASE
    )
    parser.add_argument("--replay-corpus", type=Path, default=DEFAULT_REPLAY_CORPUS)
    parser.add_argument("--mtp-steps", type=int, default=32)
    parser.add_argument("--current-source-reevaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mtp_steps <= 0:
        parser.error("--mtp-steps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
