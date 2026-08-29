#!/usr/bin/env python3
"""Cross-language experimental SDK gates. Does not claim a product Runtime release."""

from __future__ import annotations

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
    failures: list[str] = []

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
        run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "python" / "tests"), "-v"],
            env=env,
        )

    def mlx_perf_gate() -> None:
        if os.environ.get("MEI_SDK_PERF_GATE") != "1":
            print("skip mlx perf gate (set MEI_SDK_PERF_GATE=1 to run on idle Apple Silicon)")
            return
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "bench_mlx_complete.py"),
            "--backend",
            "mlx-fused",
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
        run(["cargo", "test", "-p", "mei-sdk-core", "-p", "mei-sdk-cli"], env=cargo_env())

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
        if not bin_path.is_file():
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
        dylib = TARGET / "debug" / "libmei_sdk.dylib"
        so = TARGET / "debug" / "libmei_sdk.so"
        path = dylib if dylib.is_file() else so
        if not path.is_file():
            raise FileNotFoundError(path)
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
        print("Blocking: portable inference kernels, Node native cdylib wiring, WASM tier-1, CURRENT.runtime=null.")

    step("python unittest", python_tests)
    step("mlx perf gate (opt-in)", mlx_perf_gate)
    step("cargo test", cargo_test)
    step("cargo ffi", cargo_ffi)
    step("cargo wasm tier-0", cargo_wasm)
    step("node tests", node_tests)
    step("rust CLI parity", rust_cli_parity)
    step("python ctypes ABI", native_python_ffi)
    step("split criteria (informational)", split_checklist)

    print("\n==== summary ====")
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    print("all gates passed (experimental)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
