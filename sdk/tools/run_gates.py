#!/usr/bin/env python3
"""Experimental SDK gates with an explicit product-scope boundary.

The default scope qualifies Python/MLX plus Browser-WASM.  Native Rust CLI,
Node-as-an-independent-runtime and C/FFI remain available under ``extended``
but do not block the current 300M mechanism-validation cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec"
TARGET = ROOT / "target"


def cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(TARGET)
    return env


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("python-browser-wasm", "extended"),
        default="python-browser-wasm",
    )
    args = parser.parse_args()
    extended = args.scope == "extended"
    failures: list[str] = []

    def native_library() -> Path:
        candidates = [
            TARGET / "debug" / "libmei_sdk.dylib",
            TARGET / "debug" / "libmei_sdk.so",
            TARGET / "debug" / "mei_sdk.dll",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(candidates[0])

    def step(name: str, fn) -> None:
        print(f"\n== {name} ==")
        try:
            fn()
            print(f"OK {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            failures.append(name)

    def python_tests() -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        if extended:
            # Pin extended FFI coverage to the immediately preceding build.
            env["MEI_SDK_LIB"] = str(native_library())
            command = [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT / "python" / "tests"),
                "-v",
            ]
        else:
            tests = [
                str(path)
                for path in sorted((ROOT / "python" / "tests").glob("test_*.py"))
                if path.name != "test_ffi_v2.py"
            ]
            command = [sys.executable, "-m", "unittest", "-v", *tests]
        run(command, env=env)

    def mlx_perf_gate() -> None:
        if os.environ.get("MEI_SDK_PERF_GATE") != "1":
            print("skip mlx perf gate (set MEI_SDK_PERF_GATE=1 to run on idle Apple Silicon)")
            return
        package = os.environ.get("MEI_51M_PACKAGE_DIR")
        if not package:
            raise RuntimeError("MEI_51M_PACKAGE_DIR is required for the CQ2 performance gate")
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "bench_mlx_complete.py"),
            "--package",
            str(Path(package).expanduser().resolve()),
            "--backend",
            "mlx-cq2",
            "--kernel-only",
            "--gate",
            "--warmup",
            "2",
            "--repeats",
            "3",
            "--modes",
            "raw,constrained",
            "--max-new",
            "32,128",
        ]
        baseline = Path(os.environ["MEI_SDK_PERF_BASELINE"]) if os.environ.get("MEI_SDK_PERF_BASELINE") else None
        if baseline:
            baseline = baseline.expanduser().resolve()
            cmd.extend(["--baseline", str(baseline)])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        run(cmd, env=env)

    def cargo_test() -> None:
        command = ["cargo", "test", "-p", "mei-sdk-core"]
        if extended:
            command.extend(["-p", "mei-sdk-cli"])
        run(command, env=cargo_env())

    def cargo_ffi() -> None:
        run(["cargo", "build", "-p", "mei-sdk-ffi"], env=cargo_env())

    def cargo_wasm() -> None:
        run(
            ["cargo", "build", "-p", "mei-sdk-wasm", "--target", "wasm32-unknown-unknown"],
            env=cargo_env(),
        )

    def node_tests() -> None:
        run(["node", "--test", "test.mjs"], cwd=ROOT / "js")

    def rust_cli_parity() -> None:
        bin_path = TARGET / "debug" / "mei-sdk"
        # Always ask Cargo to refresh the CLI.  Merely checking that a binary
        # exists can silently reuse a pre-51M executable against current
        # fixtures, turning source/binary drift into a misleading parity
        # failure.
        run(["cargo", "build", "-p", "mei-sdk-cli"], env=cargo_env())
        golden = json.loads((SPEC / "golden" / "turn_results.json").read_text(encoding="utf-8"))
        tiny = ROOT / "fixtures" / "packages" / "tiny-protocol-v1"
        light = {"name": "light.set", "parameters": {"type": "object", "properties": {}}}
        payload = {"query": "开灯", "oracle_tools": [light], "candidate_text": "[]"}
        proc = run(
            [str(bin_path), "complete", str(tiny), "--zero-wall"],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
        )
        got = json.loads(proc.stdout.decode("utf-8"))
        got["stats"]["wall_ms"] = 0
        if got != golden["turns"]["refuse"]:
            raise AssertionError("CLI complete refuse != python golden")

    def native_python_ffi() -> None:
        path = native_library()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        env["MEI_SDK_LIB"] = str(path)
        code = r"""
import json, os
from mei_sdk.ffi import NativeEngine
from mei_sdk.version import SDK_ROOT
eng = NativeEngine()
ver = eng.version()
assert "needle" not in json.dumps(ver).lower()
eng.open(str(SDK_ROOT / "fixtures/packages/tiny-protocol-v1"))
out = eng.complete_json({"query":"开灯","oracle_tools":[{"name":"light.set","parameters":{"type":"object","properties":{}}}],"candidate_text":"[]"})
assert out["refuse"] is True
eng.close()
print("ffi ok")
"""
        run([sys.executable, "-c", code], env=env)

    def split_checklist() -> None:
        text = (SPEC / "split-criteria.md").read_text(encoding="utf-8")
        print(text)
        print("EVALUATION: experimental SDK skeleton is in mei-llm/sdk/; split-to-mei-sdk is NOT authorized.")
        print("Blocking: final 300M CQ2/head package receipts, measured resource gates, CURRENT.runtime=null.")

    print(f"runtime scope: {args.scope}")
    if extended:
        # Build the current ABI before extended Python discovery imports it.
        step("cargo ffi", cargo_ffi)
    step("python unittest", python_tests)
    step("mlx perf gate (opt-in)", mlx_perf_gate)
    step("Rust core tests (Browser-WASM dependency)", cargo_test)
    step("Browser-WASM build", cargo_wasm)
    step("browser JS wrapper tests", node_tests)
    if extended:
        step("rust CLI parity", rust_cli_parity)
        step("python ctypes ABI", native_python_ffi)
    step("split criteria (informational)", split_checklist)

    print("\n==== summary ====")
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    print(f"all {args.scope} gates passed (experimental)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
