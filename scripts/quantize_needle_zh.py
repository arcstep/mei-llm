#!/usr/bin/env python3
"""4-bit (then lower) quantize after float correctness. Smoke packs tiny weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx
import mlx.nn as nn

from architecture import NeedleZh, count_params
from config import NeedleZhConfig


def pack_int4(arr: mx.array) -> bytes:
    """Naive even-odd nibble pack for a smoke artifact (not a production kernel)."""
    x = mx.clip(mx.round(arr * 7), -8, 7).astype(mx.int8)
    flat = x.reshape(-1)
    n = int(flat.size)
    if n % 2:
        flat = mx.concatenate([flat, mx.array([0], dtype=mx.int8)])
    even = flat[0::2] & 15
    odd = flat[1::2] & 15
    packed = (even | (odd << 4)).astype(mx.uint8)
    mx.eval(packed)
    return bytes(packed.tolist()[: min(4096, int(packed.size))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    blob = pack_int4(model.embed.weight)
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-quant-{args.bits}bit{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights-int4.head.bin").write_bytes(blob)
    report = {
        "bits": args.bits,
        "smoke": args.smoke,
        "params": count_params(model),
        "artifact_head_bytes": len(blob),
        "order": ["float_correctness", "4bit", "lower_bits"],
        "package_budget_bytes": 25 * 1024 * 1024,
        "note": "Do not ship a 14-25MB pack before float exact-match gates pass.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
