#!/usr/bin/env python3
"""Wait locally for one SFT-v4 PID, then finish 300M/600M gates and compare.

The wait loop runs entirely on the host and does not wake Codex or read batch
logs.  It never signals the training process.  A missing/failed product receipt
stops the chain rather than restarting training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT


HERE = Path(__file__).resolve().parent
RUNTIME_DRIVER = HERE / "run_paired_product_runtime_gates_51m.py"
COMPARATOR = ROOT / "src/model-factory/evaluation/alignment/compare_longitudinal_products_51m.py"


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
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def training_product(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    plan_path = run_dir / "plan.json"
    progress_path = run_dir / "progress.json"
    plan = load_json(plan_path)
    progress = load_json(progress_path)
    immutable = plan.get("immutable") or {}
    package_id = str(immutable.get("package_id") or "")
    package = run_dir / "candidate" / package_id
    master = run_dir / "stages/oracle_top5_agent_v4/agent-master.npz"
    package_receipt = run_dir / "stages/package_v2_cq2_v4/receipt.json"
    receipt = load_json(package_receipt)
    if (
        plan.get("schema") != "mei-51m-sft-v4-productization-plan-v1"
        or progress.get("schema")
        != "mei-51m-sft-v4-productization-progress-v1"
        or receipt.get("terminal_status") != "passed"
        or not package.is_dir()
        or not master.is_file()
        or sha_file(CURRENT_PATH) != immutable.get("current_baseline_sha256")
    ):
        raise RuntimeError("candidate SFT-v4 product did not reach the package boundary")
    base = immutable.get("base") or {}
    return {
        "plan": plan,
        "plan_path": plan_path,
        "progress_path": progress_path,
        "package": package,
        "master": master,
        "base_release": Path(str(base.get("release") or "")),
        "base_weights": Path(str(base.get("weights") or "")),
        "package_receipt": package_receipt,
    }


def runtime_command(
    *,
    productization_run: Path,
    package: Path,
    master: Path,
    base_release: Path,
    base_weights: Path,
    downstream_run: Path,
) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME_DRIVER),
        "--productization-run",
        str(productization_run.resolve()),
        "--package",
        str(package.resolve()),
        "--master",
        str(master.resolve()),
        "--base-release",
        str(base_release.resolve()),
        "--base-weights",
        str(base_weights.resolve()),
        "--out-root",
        str(downstream_run.resolve()),
        "--current-source-reevaluation",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "baseline_plan": args.baseline_productization_run.resolve() / "plan.json",
        "baseline_package_manifest": args.baseline_package.resolve()
        / "mei-model.json",
        "baseline_master": args.baseline_master.resolve(),
        "baseline_base_release": args.baseline_base_release.resolve(),
        "baseline_base_weights": args.baseline_base_weights.resolve(),
        "candidate_plan": args.candidate_productization_run.resolve() / "plan.json",
        "current": CURRENT_PATH,
    }
    immutable = {
        "schema": "mei-51m-paired-sft-v4-supervisor-plan-v1",
        "sft_pid": int(args.sft_pid),
        "poll_seconds": int(args.poll_seconds),
        "inputs": {str(path): sha_file(path) for path in input_paths.values()},
        "sources": {
            str(path.relative_to(ROOT)): sha_file(path)
            for path in (Path(__file__).resolve(), RUNTIME_DRIVER, COMPARATOR)
        },
        "baseline_productization_run": str(
            args.baseline_productization_run.resolve()
        ),
        "baseline_downstream_run": str(args.baseline_downstream_run.resolve()),
        "candidate_productization_run": str(
            args.candidate_productization_run.resolve()
        ),
        "candidate_downstream_run": str(args.candidate_downstream_run.resolve()),
        "comparison_output": str(args.comparison_output.resolve()),
        "current_baseline_sha256": sha_file(CURRENT_PATH),
        "failure_policy": "fail-closed-no-training-restart",
    }
    return {
        **immutable,
        "plan_fingerprint_sha256": hashlib.sha256(
            canonical_bytes(immutable)
        ).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out_root.resolve()
    plan = build_plan(args)
    plan_path = out_root / "supervisor-plan.json"
    write_once(plan_path, plan)
    try:
        while pid_alive(args.sft_pid):
            time.sleep(args.poll_seconds)
        time.sleep(2)
        candidate = training_product(args.candidate_productization_run)
        if sha_file(CURRENT_PATH) != plan["current_baseline_sha256"]:
            raise RuntimeError("CURRENT.json drifted while waiting for SFT-v4")

        baseline_command = runtime_command(
            productization_run=args.baseline_productization_run,
            package=args.baseline_package,
            master=args.baseline_master,
            base_release=args.baseline_base_release,
            base_weights=args.baseline_base_weights,
            downstream_run=args.baseline_downstream_run,
        )
        candidate_command = runtime_command(
            productization_run=args.candidate_productization_run,
            package=candidate["package"],
            master=candidate["master"],
            base_release=candidate["base_release"],
            base_weights=candidate["base_weights"],
            downstream_run=args.candidate_downstream_run,
        )
        subprocess.run(baseline_command, cwd=ROOT, check=True)
        subprocess.run(candidate_command, cwd=ROOT, check=True)
        comparison_command = [
            sys.executable,
            str(COMPARATOR),
            "--baseline-run",
            str(args.baseline_productization_run.resolve()),
            "--baseline-downstream-run",
            str(args.baseline_downstream_run.resolve()),
            "--candidate-run",
            str(args.candidate_productization_run.resolve()),
            "--candidate-downstream-run",
            str(args.candidate_downstream_run.resolve()),
            "--output",
            str(args.comparison_output.resolve()),
        ]
        subprocess.run(comparison_command, cwd=ROOT, check=True)
        result = {
            "schema": "mei-51m-paired-sft-v4-supervisor-receipt-v1",
            "terminal_status": "passed",
            "process_complete": True,
            "plan": str(plan_path),
            "plan_sha256": sha_file(plan_path),
            "baseline_runtime_receipt": str(
                args.baseline_downstream_run.resolve() / "runtime-gates-receipt.json"
            ),
            "candidate_runtime_receipt": str(
                args.candidate_downstream_run.resolve() / "runtime-gates-receipt.json"
            ),
            "comparison": str(args.comparison_output.resolve()),
            "comparison_sha256": sha_file(args.comparison_output.resolve()),
            "current_sha256": sha_file(CURRENT_PATH),
            "current_unchanged": sha_file(CURRENT_PATH)
            == plan["current_baseline_sha256"],
        }
        write_once(out_root / "supervisor-receipt.json", result)
        return result
    except Exception as exc:
        blocker = {
            "schema": "mei-51m-paired-sft-v4-supervisor-blocker-v1",
            "terminal_status": "blocked",
            "plan": str(plan_path),
            "plan_sha256": sha_file(plan_path),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "training_restarted": False,
            "current_sha256": sha_file(CURRENT_PATH),
        }
        write_once(
            out_root
            / f"supervisor-blocker-{plan['plan_fingerprint_sha256'][:12]}.json",
            blocker,
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-pid", type=int, required=True)
    parser.add_argument("--candidate-productization-run", type=Path, required=True)
    parser.add_argument("--candidate-downstream-run", type=Path, required=True)
    parser.add_argument("--baseline-productization-run", type=Path, required=True)
    parser.add_argument("--baseline-downstream-run", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--baseline-master", type=Path, required=True)
    parser.add_argument("--baseline-base-release", type=Path, required=True)
    parser.add_argument("--baseline-base-weights", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.sft_pid <= 0 or args.poll_seconds <= 0:
        parser.error("PID and poll interval must be positive")
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
