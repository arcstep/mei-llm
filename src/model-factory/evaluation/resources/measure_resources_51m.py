#!/usr/bin/env python3
"""Measure and bind real Rust/WASM resource evidence to a v2 package copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.paths import ROOT


CARGO_WORKSPACE = ROOT / "src/platform/_shared"
PYTHON_SDK = ROOT / "src/platform/python-sdk"
BROWSER_SDK = ROOT / "src/platform/browser-sdk"
TARGET_DIR = ROOT / ".local/cache/cargo-sdk-target"
RUST_RUNNER_SOURCE = CARGO_WORKSPACE / "rust/mei-sdk-core/examples/resource_51m.rs"
WASM_RUNNER_SOURCE = BROWSER_SDK / "smoke_wasm_q4.mjs"
PACKAGE_LIMIT = 18 * 1024 * 1024
RUST_LIMIT = 64 * 1024 * 1024
WASM_LIMIT = 96 * 1024 * 1024
WASM_STEADY_DECODE_FLOOR = 10.0


def qualification_gates(
    measurements: dict[str, Any], wasm: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, bool], float]:
    try:
        steady_decode_tok_s = float(wasm.get("steady_decode_tok_s") or 0.0)
    except (TypeError, ValueError):
        steady_decode_tok_s = 0.0
    if not math.isfinite(steady_decode_tok_s):
        steady_decode_tok_s = 0.0
    gates = {
        "package": int(measurements["package_bytes"]) <= PACKAGE_LIMIT,
        "wasm_heap": int(measurements["wasm_heap_peak_bytes"]) <= WASM_LIMIT,
        "wasm_steady_decode": steady_decode_tok_s >= WASM_STEADY_DECODE_FLOOR,
    }
    diagnostic_gates = {
        # Native Rust is retained as a numerical/build diagnostic while the
        # current product scope is Python/MLX + Browser-WASM. It does not block
        # Browser-WASM eligibility until the standalone Rust SDK is resumed.
        "rust_session": int(measurements["rust_session_peak_bytes"]) <= RUST_LIMIT,
    }
    return gates, diagnostic_gates, steady_decode_tok_s


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=True)


def build_runners() -> tuple[Path, Path]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    run(
        ["cargo", "build", "--release", "-p", "mei-sdk-core", "--example", "resource_51m"],
        cwd=CARGO_WORKSPACE,
        env=env,
    )
    run(
        ["cargo", "build", "--release", "-p", "mei-sdk-wasm", "--target", "wasm32-unknown-unknown"],
        cwd=CARGO_WORKSPACE,
        env=env,
    )
    rust = TARGET_DIR / "release/examples/resource_51m"
    wasm = TARGET_DIR / "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"
    if not rust.is_file() or not wasm.is_file():
        raise RuntimeError("release resource runners were not built")
    return rust, wasm


def measure_rust(binary: Path, package: Path) -> dict[str, Any]:
    time_binary = Path("/usr/bin/time")
    if not time_binary.is_file():
        raise RuntimeError("/usr/bin/time is required for peak RSS measurement")
    if platform.system() == "Darwin":
        command = [str(time_binary), "-l", str(binary), str(package)]
        pattern = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
        multiplier = 1
    else:
        command = [str(time_binary), "-v", str(binary), str(package)]
        pattern = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
        multiplier = 1024
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Rust resource runner failed: {proc.stderr[-4000:]}")
    match = pattern.search(proc.stderr)
    if match is None:
        raise RuntimeError("cannot parse Rust maximum resident set size")
    payload = json.loads(proc.stdout)
    if payload.get("numeric_forward_ran") is not True:
        raise RuntimeError("Rust runner did not prove a numeric forward")
    if payload.get("bounded_int8_cache_ran") is not True:
        raise RuntimeError("Rust runner did not prove bounded int8 KV storage")
    return {
        **payload,
        "peak_bytes": int(match.group(1)) * multiplier,
        "command": command,
        "command_sha256": sha_bytes(canonical_bytes(command)),
    }


def measure_wasm(package: Path, report_path: Path, wasm: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "CARGO_TARGET_DIR": str(TARGET_DIR),
            "MEI_51M_PACKAGE_DIR": str(package),
            "MEI_51M_JOBS_DIR": str(report_path.parent),
            "MEI_WASM_REPORT_PATH": str(report_path),
        }
    )
    command = ["node", str(WASM_RUNNER_SOURCE)]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"WASM resource runner failed: {(proc.stderr or proc.stdout)[-4000:]}")
    payload = load_json(report_path)
    if payload.get("numeric_forward_ran") is not True:
        raise RuntimeError("WASM runner did not prove a numeric forward")
    if payload.get("bounded_int8_cache_ran") is not True:
        raise RuntimeError("WASM runner did not prove bounded int8 KV storage")
    if payload.get("steady_decode_ran") is not True:
        raise RuntimeError("WASM runner did not separate and execute steady decode")
    payload["command"] = command
    payload["command_sha256"] = sha_bytes(canonical_bytes(command))
    payload["wasm_sha256"] = sha_file(wasm)
    return payload


def _combined_runner_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha_file(path)))
    return digest.hexdigest()


def attach_receipt(
    source: Path,
    destination: Path,
    *,
    rust_peak: int,
    wasm_peak: int,
    rust_runner: dict[str, Any],
    wasm_runner: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite measured package: {destination}")
    shutil.copytree(source, destination)
    manifest_path = destination / "mei-model.json"
    manifest = load_json(manifest_path)
    if manifest.get("package_format") != "mei-model-package-v2":
        raise RuntimeError("resource qualification requires a native v2 package")
    receipt_path = destination / "resource-measurement-receipt.json"
    files = [
        row
        for row in (manifest.get("files") or [])
        if row.get("path") != receipt_path.name and row.get("role") != "resource_receipt"
    ]
    package_bytes = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
    for _ in range(32):
        receipt = {
            "schema": "mei-resource-measurement-receipt-v1",
            "package_id": manifest["package_id"],
            "tensor_container_sha256": manifest["tensor_container"]["sha256"],
            "runtime_abi": "mei-runtime-abi-2",
            "measurements": {
                "package_bytes": package_bytes,
                "rust_session_peak_bytes": int(rust_peak),
                "wasm_heap_peak_bytes": int(wasm_peak),
            },
            "rust": rust_runner,
            "wasm": wasm_runner,
        }
        write_json(receipt_path, receipt)
        receipt_row = {
            "path": receipt_path.name,
            "sha256": sha_file(receipt_path),
            "nbytes": receipt_path.stat().st_size,
            "role": "resource_receipt",
        }
        manifest["files"] = sorted([*files, receipt_row], key=lambda row: row["path"])
        manifest["resources"] = {
            "package_bytes": package_bytes,
            "rust_session_peak_bytes": int(rust_peak),
            "wasm_heap_peak_bytes": int(wasm_peak),
            "measurement_receipt": {
                "file": receipt_path.name,
                "sha256": receipt_row["sha256"],
            },
        }
        write_json(manifest_path, manifest)
        measured = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
        if measured == package_bytes:
            return manifest, receipt
        package_bytes = measured
    raise RuntimeError("measured package byte count did not converge")


def measure(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.source_package / "mei-model.json")
    if manifest.get("package_format") != "mei-model-package-v2":
        raise RuntimeError("source package is not v2")
    rust_binary, wasm_binary = build_runners()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rust = measure_rust(rust_binary, args.source_package)
    wasm = measure_wasm(args.source_package, args.work_dir / "wasm-raw.json", wasm_binary)
    rust_runner = {
        "target": platform.machine() + "-" + platform.system().lower(),
        "build_profile": "release",
        "runner_id": "mei-rust-resource-51m-v1",
        "runner_sha256": sha_file(rust_binary),
        "command_sha256": rust["command_sha256"],
    }
    wasm_runner = {
        "target": "wasm32-unknown-unknown/node",
        "build_profile": "release",
        "runner_id": "mei-wasm-resource-51m-v1",
        "runner_sha256": _combined_runner_sha([WASM_RUNNER_SOURCE, wasm_binary]),
        "command_sha256": wasm["command_sha256"],
    }
    final_manifest, receipt = attach_receipt(
        args.source_package,
        args.out_package,
        rust_peak=int(rust["peak_bytes"]),
        wasm_peak=int(wasm["wasm_heap_peak_bytes"]),
        rust_runner=rust_runner,
        wasm_runner=wasm_runner,
    )
    sys.path.insert(0, str(PYTHON_SDK))
    from mei_sdk.package import load_package

    package = load_package(args.out_package)
    if not package.resource_measurement_verified:
        raise RuntimeError("final package did not verify its resource receipt")
    measurements = receipt["measurements"]
    gates, diagnostic_gates, steady_decode_tok_s = qualification_gates(measurements, wasm)
    report = {
        "schema": "mei-resource-measurement-report-v1",
        "package_id": final_manifest["package_id"],
        "source_package": str(args.source_package),
        "measured_package": str(args.out_package),
        "measurements": measurements,
        "limits": {
            "package_bytes": PACKAGE_LIMIT,
            "rust_session_peak_bytes": RUST_LIMIT,
            "wasm_heap_peak_bytes": WASM_LIMIT,
            "wasm_steady_decode_tok_s": WASM_STEADY_DECODE_FLOOR,
        },
        "gates": gates,
        "diagnostic_gates": diagnostic_gates,
        "qualification_scope": "python-browser-wasm",
        "wasm_steady_decode_tok_s": steady_decode_tok_s,
        "resource_eligible": all(gates.values()),
        "rust_raw": rust,
        "wasm_raw": wasm,
        "receipt_sha256": sha_file(args.out_package / "resource-measurement-receipt.json"),
    }
    write_json(args.out, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--out-package", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = measure(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
