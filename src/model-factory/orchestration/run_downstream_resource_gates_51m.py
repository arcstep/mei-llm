#!/usr/bin/env python3
"""Measure package/Rust/WASM resources after downstream portable parity.

The source candidate remains immutable.  Resource evidence is attached only
to a fingerprinted measured copy, and a limit miss is recorded as
``release_ineligible`` without turning a completed measurement into a blocked
stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.paths import CURRENT_PATH, ROOT
from evaluation.resources.measure_resources_51m import measure


SOURCE_FILES = (
    "src/model-factory/orchestration/run_downstream_resource_gates_51m.py",
    "src/model-factory/evaluation/resources/measure_resources_51m.py",
    "src/platform/_shared/rust/mei-sdk-core/examples/resource_51m.rs",
    "src/platform/browser-sdk/smoke_wasm_q4.mjs",
    "src/platform/browser-sdk/browser.mjs",
    "src/platform/browser-sdk/wasm-abi.mjs",
    "src/platform/_shared/.cargo/config.toml",
    "src/platform/_shared/rust/mei-sdk-core/src/cq2.rs",
    "src/platform/_shared/rust/mei-sdk-core/src/engine.rs",
    "src/platform/_shared/rust/mei-sdk-core/src/model.rs",
    "src/platform/_shared/rust/mei-sdk-core/src/packed.rs",
    "src/platform/_shared/rust/mei-sdk-wasm/src/lib.rs",
    "src/platform/_shared/Cargo.lock",
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
        or receipt.get("package_within_limit") is not True
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("source package receipt is not reusable")
    for raw, digest in (receipt.get("output_hashes") or {}).items():
        artifact = Path(raw).resolve()
        if not artifact.is_relative_to(package_dir.resolve()):
            raise RuntimeError(f"package receipt output escaped package: {artifact}")
        if not artifact.is_file() or sha_file(artifact) != digest:
            raise RuntimeError(f"source package output drifted: {artifact}")
    return receipt


def verify_portable_receipt(path: Path, package_id: str) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != "mei-productization-downstream-stage-receipt-v1"
        or receipt.get("stage_id") != "portable_runtime_gates"
        or receipt.get("terminal_status") != "passed"
        or receipt.get("package_id") != package_id
        or (receipt.get("metrics") or {}).get("all_gates_passed") is not True
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("portable gate receipt is not reusable")
    for raw, digest in (receipt.get("output_hashes") or {}).items():
        artifact = Path(raw)
        if not artifact.is_file() or sha_file(artifact) != digest:
            raise RuntimeError(f"portable gate output drifted: {artifact}")
    return receipt


def package_hashes(package_dir: Path) -> dict[str, str]:
    return {
        str(path.resolve()): sha_file(path)
        for path in sorted(package_dir.iterdir(), key=lambda value: value.name.encode("utf-8"))
        if path.is_file()
    }


def tool_version(command: list[str]) -> str:
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"cannot identify resource tool: {' '.join(command)}")
    return (process.stdout or process.stderr).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_package = args.source_package.resolve()
    package_receipt_path = args.package_receipt.resolve()
    portable_receipt_path = args.portable_receipt.resolve()
    out_package = args.out_package.resolve()
    out_dir = args.out_dir.resolve()
    work_dir = out_dir / "work"
    report_path = out_dir / "resource-measurement.json"
    receipt_path = out_dir / "receipt.json"
    package_receipt = verify_package_receipt(package_receipt_path, source_package)
    portable_receipt = verify_portable_receipt(
        portable_receipt_path, str(package_receipt["package_id"])
    )
    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    source_package_hashes = package_hashes(source_package)
    tool_versions = {
        "cargo": tool_version(["cargo", "--version"]),
        "rustc": tool_version(["rustc", "--version"]),
        "node": tool_version(["node", "--version"]),
    }
    fingerprint_inputs = {
        "package_receipt_sha256": sha_file(package_receipt_path),
        "portable_receipt_sha256": sha_file(portable_receipt_path),
        "source_package_hashes": source_package_hashes,
        "out_package": str(out_package),
        "python": sys.version,
        "platform": platform.platform(),
        "tool_versions": tool_versions,
        "source_hashes": source_hashes,
    }
    fingerprint = hashlib.sha256(canonical_bytes(fingerprint_inputs)).hexdigest()
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "package_id": package_receipt["package_id"],
            "stage_fingerprint_sha256": fingerprint,
            "out_package": str(out_package),
            "limits": {
                "package_bytes": 18 * 1024 * 1024,
                "rust_session_peak_bytes": 64 * 1024 * 1024,
                "wasm_heap_peak_bytes": 96 * 1024 * 1024,
                "wasm_steady_decode_tok_s": 10.0,
            },
            "qualification_scope": "python-browser-wasm",
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
                    raise RuntimeError(f"resource output drifted: {artifact}")
            return receipt
        raise RuntimeError("resource receipt exists with a different fingerprint")
    if report_path.exists() or out_package.exists():
        raise RuntimeError(
            "incomplete resource attempt exists; preserve it and choose a new fingerprinted out-package"
        )
    report = measure(
        argparse.Namespace(
            source_package=source_package,
            out_package=out_package,
            work_dir=work_dir,
            out=report_path,
        )
    )
    if report.get("schema") != "mei-resource-measurement-report-v1":
        raise RuntimeError("resource measurement report schema mismatch")
    if package_hashes(source_package) != source_package_hashes:
        raise RuntimeError("source package changed during resource measurement")
    if sha_file(CURRENT_PATH) != package_receipt.get("current_sha256"):
        raise RuntimeError("CURRENT changed during resource measurement")
    sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
    from mei_sdk.package import load_package

    measured = load_package(out_package)
    if not measured.resource_measurement_verified:
        raise RuntimeError("measured package does not verify its embedded resource receipt")
    measured_hashes = package_hashes(out_package)
    output_hashes = {
        str(report_path): sha_file(report_path),
        **measured_hashes,
    }
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "resource_measurement",
        "terminal_status": "passed",
        "process_complete": True,
        "resource_eligible": report.get("resource_eligible") is True,
        "release_eligible": report.get("resource_eligible") is True,
        "product": "mei-1.0-51m",
        "package_id": package_receipt["package_id"],
        "source_package": str(source_package),
        "measured_package": str(out_package),
        "package_receipt_sha256": sha_file(package_receipt_path),
        "portable_receipt_sha256": sha_file(portable_receipt_path),
        "stage_fingerprint_sha256": fingerprint,
        "measurements": report.get("measurements") or {},
        "limits": report.get("limits") or {},
        "gates": report.get("gates") or {},
        "diagnostic_gates": report.get("diagnostic_gates") or {},
        "qualification_scope": report.get("qualification_scope"),
        "source_hashes": source_hashes,
        "tool_versions": tool_versions,
        "output_hashes": output_hashes,
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--portable-receipt", type=Path, required=True)
    parser.add_argument("--out-package", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
