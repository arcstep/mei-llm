#!/usr/bin/env python3
"""Produce and gate Apple, Rust, and real browser-WASM parity evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from identity_51m import JOBS_DIR, QAT_Q4_PACKAGE_DIR, ROOT


HERE = Path(__file__).resolve().parent
SDK = ROOT / "sdk"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=QAT_Q4_PACKAGE_DIR)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    if not args.package_dir.is_absolute():
        args.package_dir = (ROOT / args.package_dir).resolve()
    if not args.jobs_dir.is_absolute():
        args.jobs_dir = (ROOT / args.jobs_dir).resolve()
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "MEI_OFFLINE": "1",
            "CARGO_NET_OFFLINE": "true",
            "MEI_51M_PACKAGE_DIR": str(args.package_dir),
            "MEI_51M_JOBS_DIR": str(args.jobs_dir),
        }
    )
    commands = [
        [
            sys.executable,
            str(HERE / "export_mlx_qat_golden_51m.py"),
            "--package-dir",
            str(args.package_dir),
            "--jobs-dir",
            str(args.jobs_dir),
        ],
        [
            sys.executable,
            str(HERE / "smoke_apple_q4_51m.py"),
            "--package-dir",
            str(args.package_dir),
            "--jobs-dir",
            str(args.jobs_dir),
        ],
        [
            "cargo",
            "test",
            "--offline",
            "-p",
            "mei-sdk-core",
            "--test",
            "q4_pack",
            "--",
            "--nocapture",
        ],
        [
            sys.executable,
            str(HERE / "smoke_wasm_q4_51m.py"),
            "--package-dir",
            str(args.package_dir),
            "--jobs-dir",
            str(args.jobs_dir),
        ],
        [
            sys.executable,
            str(HERE / "qat_q4_three_runtime_parity_51m.py"),
            "--package-dir",
            str(args.package_dir),
            "--jobs-dir",
            str(args.jobs_dir),
        ],
    ]
    for command in commands:
        cwd = SDK if command[0] == "cargo" else ROOT
        code = run(command, cwd=cwd, env=env)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
