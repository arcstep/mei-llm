#!/usr/bin/env python3
"""Run package-specific portable gates after verified upstream adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT
from orchestration.productize_51m import _portable_gate_report


SOURCE_FILES = (
    "model-factory/orchestration/run_downstream_portable_gates_51m.py",
    "model-factory/orchestration/productize_51m.py",
    "model-factory/evaluation/heads/compare_portable_heads_51m.py",
    "model-factory/orchestration/verify_downstream_package_51m.py",
    "platform/_shared/tools/run_gates.py",
    "platform/python-sdk/mei_sdk/ffi.py",
    "platform/python-sdk/mei_sdk/runtime_51m.py",
    "platform/_shared/rust/mei-sdk-core/src/infer.rs",
    "platform/_shared/rust/mei-sdk-core/src/model.rs",
    "platform/_shared/rust/mei-sdk-core/src/engine.rs",
    "platform/_shared/rust/mei-sdk-wasm/src/lib.rs",
    "platform/browser-sdk/wasm-abi.mjs",
    "platform/browser-sdk/node-wasm.mjs",
    "platform/browser-sdk/smoke_node_v2.mjs",
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


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _verify_package_receipt(path: Path, package_dir: Path) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema")
        != "mei-productization-downstream-package-receipt-v1"
        or receipt.get("terminal_status") != "passed"
        or Path(str(receipt.get("package_path") or "")).resolve()
        != package_dir.resolve()
        or receipt.get("current_unchanged") is not True
        or sha_file(CURRENT_PATH) != receipt.get("current_sha256")
    ):
        raise RuntimeError("downstream package receipt is not reusable")
    for raw, digest in (receipt.get("output_hashes") or {}).items():
        artifact = Path(raw).resolve()
        if not artifact.is_relative_to(package_dir.resolve()):
            raise RuntimeError(f"package receipt output escaped package: {artifact}")
        if not artifact.is_file() or sha_file(artifact) != digest:
            raise RuntimeError(f"package receipt output drifted: {artifact}")
    return receipt


def portable_report_passed(report: dict[str, Any]) -> bool:
    return bool(
        report.get("all_gates_passed") is True
        and (report.get("browser_wasm_build") or {}).get("profile") == "release"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package.resolve()
    package_receipt_path = args.package_receipt.resolve()
    package_receipt = _verify_package_receipt(package_receipt_path, package_dir)
    out_dir = args.out_dir.resolve()
    receipt_path = out_dir / "receipt.json"
    report_path = out_dir / "portable-runtime-gates.json"

    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    fingerprint_input = {
        "package_receipt_sha256": sha_file(package_receipt_path),
        "package_id": package_receipt["package_id"],
        "source_hashes": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
    }
    fingerprint = sha_bytes(canonical_bytes(fingerprint_input))
    if receipt_path.is_file():
        previous = load_json(receipt_path)
        if (
            previous.get("terminal_status") == "passed"
            and previous.get("stage_fingerprint_sha256") == fingerprint
        ):
            for raw, digest in (previous.get("output_hashes") or {}).items():
                artifact = Path(raw)
                if not artifact.is_file() or sha_file(artifact) != digest:
                    raise RuntimeError(f"portable gate output drifted: {artifact}")
            return previous
        raise RuntimeError("portable gate receipt exists with a different fingerprint")

    report = _portable_gate_report(package_dir, out_dir)
    if not portable_report_passed(report):
        raise RuntimeError("portable runtime report did not pass all gates")
    write_once(report_path, report)
    outputs = sorted(
        path for path in out_dir.iterdir() if path.is_file() and path != receipt_path
    )
    output_hashes = {str(path.resolve()): sha_file(path) for path in outputs}
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "portable_runtime_gates",
        "terminal_status": "passed",
        "product": "mei-1.0-51m",
        "package_id": package_receipt["package_id"],
        "package_receipt": str(package_receipt_path),
        "package_receipt_sha256": sha_file(package_receipt_path),
        "stage_fingerprint_sha256": fingerprint,
        "metrics": report,
        "source_hashes": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "output_hashes": output_hashes,
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
