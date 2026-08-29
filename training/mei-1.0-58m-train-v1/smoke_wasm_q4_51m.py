#!/usr/bin/env python3
"""Build wasm32 mei-sdk-wasm and load the 51M Q4 package. Float npz is refused."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from identity_51m import JOBS_DIR, fail, write_json

SDK = Path(__file__).resolve().parents[2] / "sdk"
WASM_JS = SDK / "js" / "smoke_wasm_q4.mjs"


def main() -> int:
    cargo = shutil.which("cargo")
    node = shutil.which("node")
    if cargo is None or node is None:
        report = {
            "ok": False,
            "compiled": False,
            "quantized_only": True,
            "note": f"missing tools cargo={cargo} node={node}",
        }
        write_json(JOBS_DIR / "wasm-q4-smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    rustup = shutil.which("rustup")
    if rustup:
        subprocess.run(
            [rustup, "target", "add", "wasm32-unknown-unknown"],
            check=False,
            cwd=str(SDK),
        )
    build = subprocess.run(
        [
            cargo,
            "build",
            "-p",
            "mei-sdk-wasm",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ],
        cwd=str(SDK),
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        report = {
            "ok": False,
            "compiled": False,
            "quantized_only": True,
            "note": "cargo wasm32 build failed",
            "stderr_tail": (build.stderr or "")[-4000:],
        }
        write_json(JOBS_DIR / "wasm-q4-smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    smoke = subprocess.run([node, str(WASM_JS)], cwd=str(SDK / "js"))
    return 0 if smoke.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
