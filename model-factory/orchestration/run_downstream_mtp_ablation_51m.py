#!/usr/bin/env python3
"""Run the fixed-budget MTP/no-MTP ablation against the adopted final LM.

MTP is training-only evidence.  This wrapper binds the ablation to the exact
``oracle_top5_fullcall`` master adopted by the downstream package, verifies
that the package exports zero MTP tensors before and after the run, and emits a
normal downstream stage receipt without modifying the package or CURRENT.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT
from training.heads.mtp_ablation_51m import build_plan, run as run_ablation


SOURCE_FILES = (
    "model-factory/orchestration/run_downstream_mtp_ablation_51m.py",
    "model-factory/training/heads/mtp_ablation_51m.py",
    "models/mei-1.0-51m/architecture/architecture.py",
    "models/mei-1.0-51m/architecture/architecture_contract.py",
    "model-factory/common/mlx_memory_policy_51m.py",
)


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


def verify_package_receipt(path: Path, package_dir: Path) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != "mei-productization-downstream-package-receipt-v1"
        or receipt.get("terminal_status") != "passed"
        or Path(str(receipt.get("package_path") or "")).resolve()
        != package_dir.resolve()
        or receipt.get("lm_parameter_count") != 51_463_797
        or receipt.get("lm_tensor_count") != 400
        or receipt.get("mtp_tensor_count") != 0
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("package receipt is not valid for MTP ablation")
    for raw, digest in (receipt.get("output_hashes") or {}).items():
        artifact = Path(raw).resolve()
        if not artifact.is_relative_to(package_dir.resolve()):
            raise RuntimeError(f"package receipt output escaped package: {artifact}")
        if not artifact.is_file() or sha_file(artifact) != digest:
            raise RuntimeError(f"package receipt output drifted: {artifact}")
    return receipt


def verify_zero_mtp_tensors(package_dir: Path) -> int:
    manifest = load_json(package_dir / "mei-model.json")
    directory = (manifest.get("tensor_container") or {}).get("directory") or []
    mtp = [
        row
        for row in directory
        if isinstance(row, dict)
        and (
            str(row.get("role") or "").lower() == "mtp"
            or "mtp" in str(row.get("name") or "").lower()
        )
    ]
    if mtp:
        raise RuntimeError(f"deployment package contains MTP tensors: {len(mtp)}")
    return 0


def verify_adopted_master(
    package_receipt: dict[str, Any], master: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    adoption_path = Path(str(package_receipt.get("upstream_adoption_receipt") or "")).resolve()
    if (
        not adoption_path.is_file()
        or sha_file(adoption_path)
        != package_receipt.get("upstream_adoption_receipt_sha256")
    ):
        raise RuntimeError("upstream adoption receipt drifted")
    adoption = load_json(adoption_path)
    if (
        adoption.get("schema") != "mei-productization-upstream-adoption-receipt-v1"
        or adoption.get("status") != "passed"
        or adoption.get("adopted_through") != "narration_adapter"
    ):
        raise RuntimeError("upstream adoption receipt is incomplete")
    oracle = next(
        (
            row
            for row in adoption.get("adopted_stages") or []
            if isinstance(row, dict) and row.get("stage_id") == "oracle_top5_fullcall"
        ),
        None,
    )
    if oracle is None or oracle.get("terminal_status") != "passed":
        raise RuntimeError("adoption does not contain the final full-call LM")
    outputs = oracle.get("verified_outputs") or {}
    master_digest = outputs.get(str(master))
    if master_digest is None:
        for raw, digest in outputs.items():
            if Path(raw).resolve() == master:
                master_digest = digest
                break
    if master_digest is None or not master.is_file() or sha_file(master) != master_digest:
        raise RuntimeError("MTP master is not the adopted final full-call LM")
    return adoption, oracle


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package.resolve()
    package_receipt_path = args.package_receipt.resolve()
    master = args.master.resolve()
    out_dir = args.out_dir.resolve()
    package_receipt = verify_package_receipt(package_receipt_path, package_dir)
    verify_zero_mtp_tensors(package_dir)
    adoption, oracle = verify_adopted_master(package_receipt, master)
    ablation_path = out_dir / "mtp-ablation.json"
    receipt_path = out_dir / "receipt.json"
    ablation_args = argparse.Namespace(
        master=master,
        replay_corpus=args.replay_corpus.resolve(),
        out=ablation_path,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
    )
    plan = build_plan(ablation_args)
    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    packages = {}
    for name in ("mlx", "numpy", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    fingerprint_inputs = {
        "ablation_stage_fingerprint_sha256": plan["stage_fingerprint_sha256"],
        "package_receipt_sha256": sha_file(package_receipt_path),
        "adoption_receipt_sha256": sha_file(
            Path(str(package_receipt["upstream_adoption_receipt"]))
        ),
        "oracle_receipt_sha256": oracle.get("receipt_sha256"),
        "master_sha256": sha_file(master),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_hashes": source_hashes,
    }
    fingerprint = hashlib.sha256(canonical_bytes(fingerprint_inputs)).hexdigest()
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "stage_fingerprint_sha256": fingerprint,
            "ablation_plan": plan,
            "package_mtp_tensor_count": 0,
        }
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if (
            receipt.get("terminal_status") == "passed"
            and receipt.get("stage_fingerprint_sha256") == fingerprint
        ):
            for raw, digest in (receipt.get("output_hashes") or {}).items():
                artifact = Path(raw)
                if not artifact.is_file() or sha_file(artifact) != digest:
                    raise RuntimeError(f"MTP ablation output drifted: {artifact}")
            return receipt
        raise RuntimeError("MTP ablation receipt exists with a different fingerprint")
    if ablation_path.exists():
        previous = load_json(ablation_path)
        if (
            previous.get("terminal_status") != "passed"
            or previous.get("stage_fingerprint_sha256")
            != plan["stage_fingerprint_sha256"]
        ):
            raise RuntimeError("existing MTP ablation artifact is not reusable")
        ablation = previous
        memory_policy = {"reused_existing_ablation": True}
    else:
        from common.mlx_memory_policy_51m import (
            configure_mlx_memory,
            mlx_memory_snapshot,
            release_mlx_memory,
        )

        mx, configured = configure_mlx_memory(reset_peak=True)
        ablation = run_ablation(ablation_args)
        memory_policy = {**configured, "final": mlx_memory_snapshot(mx)}
        release_mlx_memory(mx, collect_python=True)
    if (
        ablation.get("terminal_status") != "passed"
        or ablation.get("temporary_heads_persisted") is not False
        or ablation.get("temporary_heads_exported") is not False
        or ablation.get("deployment_mtp_tensor_count_required") != 0
    ):
        raise RuntimeError("MTP ablation did not preserve deployment identity")
    verify_zero_mtp_tensors(package_dir)
    if sha_file(CURRENT_PATH) != package_receipt.get("current_sha256"):
        raise RuntimeError("CURRENT changed during MTP ablation")
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "mtp_ablation",
        "terminal_status": "passed",
        "process_complete": True,
        "not_a_score_claim": True,
        "product": "mei-1.0-51m",
        "package_id": package_receipt["package_id"],
        "package_receipt": str(package_receipt_path),
        "package_receipt_sha256": sha_file(package_receipt_path),
        "upstream_adoption_receipt_sha256": sha_file(
            Path(str(package_receipt["upstream_adoption_receipt"]))
        ),
        "stage_fingerprint_sha256": fingerprint,
        "ablation_stage_fingerprint_sha256": plan["stage_fingerprint_sha256"],
        "metrics": ablation.get("metrics") or {},
        "mlx_memory_policy": memory_policy,
        "training_only": True,
        "future_recipe_evidence_only": True,
        "temporary_heads_exported": False,
        "deployment_mtp_tensor_count": 0,
        "lm_parameter_count": 51_463_797,
        "source_hashes": source_hashes,
        "packages": packages,
        "output_hashes": {str(ablation_path): sha_file(ablation_path)},
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replay-corpus", type=Path, default=ROOT / ".local/artifacts/mei-1.0-51m/exp-000300m/corpus/cpt-delta/lm-v1")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seq_len <= 2 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("invalid MTP ablation budget")
    return args


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
