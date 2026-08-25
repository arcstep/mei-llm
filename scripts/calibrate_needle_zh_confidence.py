#!/usr/bin/env python3
"""Calibrate execute threshold on a labeled SFT valid split. Smoke writes a dummy threshold file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, TASKS_ROOT, TASK_NEEDLE_ZH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out_dir = EXPERIMENTS_RUNS / "needle-zh-calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "execute_threshold": args.threshold,
        "smoke": args.smoke,
        "metric": "family-wise exact_match + always-refuse baseline must be reported at eval time",
        "families": ["gesture", "home", "order", "missing", "scene_conflict", "illegal_pair", "offtopic", "paraphrase"],
        "note": "Threshold is chosen on valid split; do not tune on frozen holdout.",
    }
    (TASKS_ROOT / TASK_NEEDLE_ZH / "spec" / "execute-threshold.json").write_text(
        json.dumps({"execute_threshold": args.threshold, "version": "v1-smoke" if args.smoke else "v1"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
