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

from common.identity_51m import JOBS_DIR, Q4_PACKAGE_DIR, QAT_Q4_PACKAGE_DIR, ROOT, fail, load_json, write_json
from common.paths import RECIPES_DIR

RESOURCE_GATES = RECIPES_DIR / "resource-gates-51m.json"
CARGO_WORKSPACE = ROOT / "src/platform/_shared"
PYTHON_SDK = ROOT / "src/platform/python-sdk"
BROWSER_SDK = ROOT / "src/platform/browser-sdk"
TARGET_DIR = ROOT / ".local/cache/cargo-sdk-target"


def resolve_package_and_jobs(pkg: Path | None, jobs: Path) -> tuple[Path, Path]:
    """Rust/WASM workers chdir; relative package paths must be absolute first."""
    if pkg is None:
        pkg = QAT_Q4_PACKAGE_DIR if (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file() else Q4_PACKAGE_DIR
    return Path(pkg).resolve(), Path(jobs).resolve()


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    value = float(usage.ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def apple_baseline(pkg_dir: Path) -> dict:
    import sys

    sys.path.insert(0, str(PYTHON_SDK))
    import mlx.core as mx

    mx.set_memory_limit(6 * 1024**3)
    mx.set_cache_limit(int(0.5 * 1024**3))
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


def rust_baseline(pkg_dir: Path, jobs_dir: Path) -> dict:
    env = os.environ.copy()
    env["MEI_51M_PACKAGE_DIR"] = str(pkg_dir)
    env["MEI_51M_JOBS_DIR"] = str(jobs_dir)
    env["CARGO_NET_OFFLINE"] = "true"
    env.setdefault("CARGO_TARGET_DIR", str(TARGET_DIR))
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--offline",
            "--release",
            "-p",
            "mei-sdk-core",
            "--example",
            "resource_51m",
            "--",
            str(pkg_dir),
        ],
        cwd=str(CARGO_WORKSPACE),
        capture_output=True,
        text=True,
        env=env,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    report = {}
    try:
        report = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pass
    return {
        "backend": "rust-cpu",
        "cold_start_plus_greedy_ms": wall_ms,
        "test_ok": proc.returncode == 0 and bool(report.get("ok")),
        "load_ms": report.get("load_ms"),
        "greedy_wall_ms": report.get("greedy_ms"),
        "n_new": report.get("n_new"),
        "stderr_tail": (proc.stderr or "")[-500:],
        "kv_window_ordinary": 256,
    }


def wasm_baseline(pkg_dir: Path, jobs_dir: Path) -> dict:
    smoke = BROWSER_SDK / "smoke_wasm_q4.mjs"
    probe_dir = jobs_dir / "resource-wasm"
    probe_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MEI_51M_PACKAGE_DIR"] = str(pkg_dir)
    env["MEI_51M_JOBS_DIR"] = str(probe_dir)
    env["CARGO_NET_OFFLINE"] = "true"
    env.setdefault("CARGO_TARGET_DIR", str(TARGET_DIR))
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["node", str(smoke)],
        cwd=str(BROWSER_SDK),
        capture_output=True,
        text=True,
        env=env,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    report = {}
    path = probe_dir / "wasm-q4-smoke.json"
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
    pkg, args.jobs_dir = resolve_package_and_jobs(args.package_dir, args.jobs_dir)
    apple = apple_baseline(pkg)
    blocked = write_json(args.jobs_dir / "apple-resource-baseline.json", apple)
    if blocked:
        return fail(blocked)
    rust = {"skipped": True} if args.skip_rust else rust_baseline(pkg, args.jobs_dir)
    blocked = write_json(args.jobs_dir / "rust-resource-baseline.json", rust)
    if blocked:
        return fail(blocked)
    wasm = {"skipped": True} if args.skip_wasm else wasm_baseline(pkg, args.jobs_dir)
    blocked = write_json(args.jobs_dir / "wasm-resource-baseline.json", wasm)
    if blocked:
        return fail(blocked)
    gates = load_json(RESOURCE_GATES)
    package_bytes = sum(path.stat().st_size for path in pkg.iterdir() if path.is_file())
    checks = {
        "package_size": package_bytes <= int(gates["package_bytes_max"]),
        "apple_cold_start": float(apple.get("cold_start_ms") or float("inf")) <= float(gates["apple_cold_start_ms_max"]),
        "apple_first_token": float(apple.get("first_token_ms") or float("inf")) <= float(gates["apple_first_token_ms_max"]),
        "apple_decode": float(apple.get("decode_tok_s") or 0) >= float(gates["apple_decode_tok_s_min"]),
        "apple_rss": float(apple.get("rss_mb") or float("inf")) <= float(gates["apple_rss_mb_max"]),
        "apple_rss_delta": float(apple.get("rss_delta_mb") or float("inf")) <= float(gates["rss_delta_mb_max"]),
        "rust": bool(rust.get("test_ok")) if not args.skip_rust else False,
        "wasm": (
            bool(wasm.get("node_ok"))
            and int(wasm.get("n_new") or 0) >= int(gates["wasm_generated_tokens_min"])
            and float(wasm.get("decode_tok_s") or 0) >= float(gates["wasm_decode_tok_s_min"])
            and float(wasm.get("cold_start_ms") or float("inf")) <= float(gates["wasm_cold_start_ms_max"])
        ) if not args.skip_wasm else False,
    }
    summary = {
        "kind": "resource-baseline-51m",
        "package_dir": str(pkg),
        "package_bytes": package_bytes,
        "thresholds": str(RESOURCE_GATES.relative_to(ROOT)),
        "apple": apple,
        "rust": rust,
        "wasm": wasm,
        "gates": checks,
        "hard_ok": all(checks.values()),
    }
    blocked = write_json(args.jobs_dir / "resource-baseline-51m.json", summary)
    if blocked:
        return fail(blocked)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["hard_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
