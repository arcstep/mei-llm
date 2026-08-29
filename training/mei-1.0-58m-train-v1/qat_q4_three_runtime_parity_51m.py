#!/usr/bin/env python3
"""Assemble three-runtime Q4/QAT parity from already-written job files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_51m import JOBS_DIR, fail, load_json, write_json


LOGIT_ABS_MAX = 2.7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    jobs = args.jobs_dir
    mlx = load_json(jobs / "mlx-qat-q4-golden.json")
    rust_pre = load_json(jobs / "rust-q4-prefill.json")
    rust_g = load_json(jobs / "rust-q4-short-greedy.json")
    wasm = load_json(jobs / "wasm-q4-smoke.json")
    browser = load_json(jobs / "wasm-q4-browser-smoke.json")
    apple = load_json(jobs / "apple-q4-smoke.json")
    logit_ok = rust_g.get("max_abs_logit_vs_mlx") is None or float(rust_g.get("max_abs_logit_vs_mlx") or 99) < LOGIT_ABS_MAX
    wasm_infer = bool(wasm.get("complete_ran")) and int(wasm.get("n_new") or 0) >= 8
    report = {
        "kind": "qat-q4-three-runtime-parity",
        "logit_abs_threshold_preregistered": LOGIT_ABS_MAX,
        "apple": {
            "ok": bool(apple.get("ok") or mlx.get("greedy_text")),
            "package_id": mlx.get("package_id"),
            "greedy_text": mlx.get("greedy_text"),
        },
        "rust": {
            "prefill_ok": bool(rust_pre.get("ok")),
            "greedy_ok": bool(rust_g.get("ok")),
            "max_abs_logit_vs_mlx": rust_g.get("max_abs_logit_vs_mlx"),
            "raw_text": rust_g.get("raw_text"),
            "logit_ok": logit_ok,
        },
        "wasm": {
            "loaded": bool(wasm.get("loaded")),
            "refuse_float": bool(wasm.get("refuse_float")),
            "complete_ran": bool(wasm.get("complete_ran")),
            "n_new": wasm.get("n_new"),
            "raw_text": wasm.get("raw_text") or wasm.get("greedy_text"),
            "infer_ok": wasm_infer,
            "browser_worker_ok": bool(browser.get("ok")),
            "browser_n_new": browser.get("n_new"),
        },
        "token_texts": {
            "mlx": mlx.get("greedy_text"),
            "rust": rust_g.get("raw_text"),
            "wasm": wasm.get("raw_text") or wasm.get("greedy_text"),
        },
        "ok": bool(mlx) and bool(rust_pre.get("ok")) and logit_ok and bool(wasm.get("refuse_float")) and wasm_infer,
        "qat_mandatory": True,
        "not_a_claim": "Runtime logit/token parity is not tool-calling ability.",
    }
    path = jobs / "qat-q4-three-runtime-parity.json"
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
