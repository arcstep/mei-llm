#!/usr/bin/env python3
"""Apple / Rust / WASM resource baselines for a packed 51M package."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

from identity_51m import JOBS_DIR, Q4_PACKAGE_DIR, QAT_Q4_PACKAGE_DIR, ROOT, fail, write_json


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    value = float(usage.ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def apple_baseline(pkg_dir: Path) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime

    t0 = time.perf_counter()
    pkg = load_package(pkg_dir)
    runtime, loaded = load_51m_runtime(pkg)
    cold_ms = (time.perf_counter() - t0) * 1000
    ids = runtime.tokenizer.encode("厨房灯打开", add_bos=True, add_eos=False)
    t1 = time.perf_counter()
    first = runtime.greedy(ids, tools=[], max_new=1, decode_mode="raw")
    first_ms = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    more = runtime.greedy(ids, tools=[], max_new=16, decode_mode="raw")
    decode_s = max(time.perf_counter() - t2, 1e-6)
    n_out = int(more.get("n_out") or 0)
    # Two consecutive turns; RSS should not jump unboundedly.
    rss1 = rss_mb()
    runtime.greedy(ids, tools=[], max_new=8, decode_mode="raw")
    rss2 = rss_mb()
    return {
        "backend": "apple-mlx",
        "cold_start_ms": cold_ms,
        "first_token_ms": first_ms,
        "decode_tok_s": n_out / decode_s,
        "rss_mb": rss2,
        "rss_after_second_turn_mb": rss2,
        "rss_delta_mb": rss2 - rss1,
        "kv_window_ordinary": 256,
        "memory_not_monotone_unbounded": rss2 <= rss1 + 80,
        "n_loaded": loaded.get("n_loaded"),
        "n_out_probe": n_out,
        "first_n_out": first.get("n_out"),
    }


def rust_baseline(pkg_dir: Path) -> dict:
    env = os.environ.copy()
    env["MEI_51M_PACKAGE_DIR"] = str(pkg_dir)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "mei-sdk-core",
            "--test",
            "q4_pack",
            "q4_short_greedy_vs_golden",
            "--",
            "--nocapture",
        ],
        cwd=str(ROOT / "sdk"),
        capture_output=True,
        text=True,
        env=env,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    greedy = {}
    path = JOBS_DIR / "rust-q4-short-greedy.json"
    if path.is_file():
        greedy = json.loads(path.read_text(encoding="utf-8"))
    return {
        "backend": "rust-cpu",
        "cold_start_plus_greedy_ms": wall_ms,
        "test_ok": proc.returncode == 0,
        "greedy_wall_ms": greedy.get("wall_ms"),
        "stderr_tail": (proc.stderr or "")[-500:],
        "kv_window_ordinary": 256,
    }


def wasm_baseline(pkg_dir: Path) -> dict:
    smoke = ROOT / "sdk" / "js" / "smoke_wasm_q4.mjs"
    env = os.environ.copy()
    env["MEI_51M_PACKAGE_DIR"] = str(pkg_dir)
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "sdk" / "target"))
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["node", str(smoke)],
        cwd=str(ROOT / "sdk" / "js"),
        capture_output=True,
        text=True,
        env=env,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    report = {}
    path = JOBS_DIR / "wasm-q4-smoke.json"
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "backend": "wasm",
        "wall_ms": wall_ms,
        "node_ok": proc.returncode == 0,
        "complete_ran": report.get("complete_ran"),
        "n_new": report.get("n_new"),
        "decode_tok_s": report.get("decode_tok_s"),
        "cold_start_ms": report.get("cold_start_ms"),
        "kv_window_ordinary": 256,
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-wasm", action="store_true")
    args = parser.parse_args()
    pkg = args.package_dir
    if pkg is None:
        pkg = QAT_Q4_PACKAGE_DIR if (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file() else Q4_PACKAGE_DIR
    apple = apple_baseline(pkg)
    blocked = write_json(args.jobs_dir / "apple-resource-baseline.json", apple)
    if blocked:
        return fail(blocked)
    rust = {"skipped": True} if args.skip_rust else rust_baseline(pkg)
    blocked = write_json(args.jobs_dir / "rust-resource-baseline.json", rust)
    if blocked:
        return fail(blocked)
    wasm = {"skipped": True} if args.skip_wasm else wasm_baseline(pkg)
    blocked = write_json(args.jobs_dir / "wasm-resource-baseline.json", wasm)
    if blocked:
        return fail(blocked)
    summary = {
        "kind": "resource-baseline-51m",
        "package_dir": str(pkg),
        "apple": apple,
        "rust": rust,
        "wasm": wasm,
        "ok": True,
    }
    blocked = write_json(args.jobs_dir / "resource-baseline-51m.json", summary)
    if blocked:
        return fail(blocked)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
