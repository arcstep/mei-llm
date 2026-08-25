#!/usr/bin/env python3
"""Accuracy + full-answer latency + peak RAM. Smoke measures tiny forward, not product KPI."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx

from architecture import NeedleZh, count_params
from config import NeedleZhConfig
from grammar import parse_phase1_text


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(usage if usage > 10_000_000 else usage * 1024)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    toks = mx.array([[cfg.bos_id, 7, 8, 9, cfg.eos_id]], dtype=mx.int32)
    t0 = time.perf_counter()
    out = model(toks, return_confidence=True)
    mx.eval(out["logits"])
    ms = (time.perf_counter() - t0) * 1000
    # Grammar gate on a known illegal string.
    illegal = parse_phase1_text("请补充城市")
    report = {
        "smoke": args.smoke,
        "metric": "full_answer_wall_clock_not_ttft",
        "forward_ms": round(ms, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "params": count_params(model),
        "grammar_rejects_nl": not illegal["ok"],
        "gates_file": "tasks/needle-zh/spec/gates.json",
        "note": "Product p50<=100ms / RAM<=64MB / exact-match>=90% require a real checkpoint on target silicon.",
    }
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-bench{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["grammar_rejects_nl"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
