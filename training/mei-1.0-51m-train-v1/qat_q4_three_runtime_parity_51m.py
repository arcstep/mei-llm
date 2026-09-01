#!/usr/bin/env python3
"""Assemble three-runtime Q4/QAT parity from already-written job files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_51m import JOBS_DIR, QAT_Q4_PACKAGE_DIR, fail, load_json, sha256_file, write_json


LOGIT_ABS_MAX = 2.7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--package-dir", type=Path, default=QAT_Q4_PACKAGE_DIR)
    args = parser.parse_args()
    jobs = args.jobs_dir
    mlx = load_json(jobs / "mlx-qat-q4-golden.json")
    rust_pre = load_json(jobs / "rust-q4-prefill.json")
    rust_g = load_json(jobs / "rust-q4-short-greedy.json")
    wasm = load_json(jobs / "wasm-q4-smoke.json")
    browser = load_json(jobs / "wasm-q4-browser-smoke.json")
    apple = load_json(jobs / "apple-q4-smoke.json")
    package_manifest = load_json(args.package_dir / "mei-model.json")
    package_id = package_manifest.get("package_id")
    weights = args.package_dir / "weights.q4"
    package_sha = sha256_file(weights) if weights.is_file() else None
    logit_delta = rust_g.get("max_abs_logit_vs_mlx")
    logit_ok = logit_delta is not None and float(logit_delta) < LOGIT_ABS_MAX
    wasm_infer = bool(wasm.get("complete_ran")) and int(wasm.get("n_new") or 0) >= 8
    golden_prompt = mlx.get("token_ids") or []
    golden_tokens = mlx.get("greedy_ids") or []
    golden_topk = mlx.get("prefill_topk_ids") or []
    runtime_rows = (rust_g, wasm, browser)
    prompt_ok = bool(golden_prompt) and all(row.get("prompt_token_ids") == golden_prompt for row in runtime_rows)
    token_ok = bool(golden_tokens) and all(row.get("generated_token_ids") == golden_tokens for row in runtime_rows)
    topk_ok = bool(golden_topk) and all(row.get("prefill_topk_ids") == golden_topk for row in runtime_rows)
    package_ok = bool(package_id and package_sha) and all(
        not row.get("package_id") or row.get("package_id") == package_id
        for row in (mlx, rust_pre, rust_g, wasm, browser)
    )
    report = {
        "kind": "qat-q4-three-runtime-parity",
        "logit_abs_threshold_preregistered": LOGIT_ABS_MAX,
        "apple": {
            "ok": bool(apple.get("ok") or mlx.get("greedy_text")),
            "package_id": package_id,
            "weights_sha256": package_sha,
            "greedy_text": mlx.get("greedy_text"),
        },
        "rust": {
            "prefill_ok": bool(rust_pre.get("ok")),
            "greedy_ok": bool(rust_g.get("ok")),
            "max_abs_logit_vs_mlx": rust_g.get("max_abs_logit_vs_mlx"),
            "raw_text": rust_g.get("raw_text"),
            "logit_ok": logit_ok,
            "prompt_token_ids": rust_g.get("prompt_token_ids"),
            "generated_token_ids": rust_g.get("generated_token_ids"),
            "prefill_topk_ids": rust_g.get("prefill_topk_ids"),
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
            "generated_token_ids": wasm.get("generated_token_ids"),
            "prefill_topk_ids": wasm.get("prefill_topk_ids"),
            "browser_generated_token_ids": browser.get("generated_token_ids"),
            "browser_prefill_topk_ids": browser.get("prefill_topk_ids"),
        },
        "token_texts": {
            "mlx": mlx.get("greedy_text"),
            "rust": rust_g.get("raw_text"),
            "wasm": wasm.get("raw_text") or wasm.get("greedy_text"),
        },
        "same_package_ok": package_ok,
        "same_prompt_tokens_ok": prompt_ok,
        "generated_token_ids_exact": token_ok,
        "prefill_topk_ids_exact": topk_ok,
        "ok": (
            bool(mlx)
            and bool(apple.get("ok"))
            and bool(rust_pre.get("ok"))
            and bool(rust_g.get("ok"))
            and logit_ok
            and bool(wasm.get("refuse_float"))
            and wasm_infer
            and bool(browser.get("ok"))
            and package_ok
            and prompt_ok
            and token_ok
            and topk_ok
        ),
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
