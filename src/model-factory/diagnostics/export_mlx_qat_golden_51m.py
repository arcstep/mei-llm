#!/usr/bin/env python3
"""Export a small MLX golden vector from a packed 51M package. Not a float reload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from evaluation.base.freeze_float_baseline_51m import greedy_continue
from common.identity_51m import (
    JOBS_DIR,
    Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_DIR,
    fail,
    write_json,
)
from tokenizer import ZhTokenizerV1


PROMPT = "厨房灯打开"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--max-new", type=int, default=8)
    args = parser.parse_args()
    pkg_dir = args.package_dir
    if pkg_dir is None:
        pkg_dir = QAT_Q4_PACKAGE_DIR if (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file() else Q4_PACKAGE_DIR
    import sys

    sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file()) / "src/platform/python-sdk"))
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime

    pkg = load_package(pkg_dir)
    runtime, report = load_51m_runtime(pkg)
    tok = runtime.tokenizer
    ids = tok.encode(PROMPT, add_bos=True, add_eos=False)[:32]
    arr = mx.array([ids], dtype=mx.int32)
    out = runtime.model(arr)
    mx.eval(out["logits"], out["hidden"])
    logits = out["logits"][0, -1]
    hidden = out["hidden"][0, -1]
    logits_list = [float(x) for x in logits.tolist()]
    prefill_topk_ids = sorted(range(len(logits_list)), key=lambda i: logits_list[i], reverse=True)[:5]
    gen = greedy_continue(runtime.model, list(ids), max_new=args.max_new, eos_id=tok.eos_id)
    golden = {
        "kind": "mlx-qat-q4-golden",
        "package_id": pkg.package_id,
        "package_dir": str(pkg_dir),
        "prompt": PROMPT,
        "token_ids": ids,
        "logits_head": logits_list[:64],
        "prefill_topk_ids": prefill_topk_ids,
        "hidden_head": [float(x) for x in hidden[:32].tolist()],
        "greedy_ids": gen[len(ids) :],
        "greedy_text": tok.decode(gen[len(ids) :]),
        "n_loaded": report.get("n_loaded"),
        "not_a_float_reload": True,
        "logit_abs_threshold": 2.7,
    }
    path = args.jobs_dir / "mlx-qat-q4-golden.json"
    blocked = write_json(path, golden)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": True, "report": str(path), "greedy_text": golden["greedy_text"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
