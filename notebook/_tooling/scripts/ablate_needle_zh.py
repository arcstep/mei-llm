#!/usr/bin/env python3
"""Single-variable architecture ablations. Blocked until product baseline passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, TASKS_ROOT, TASK_NEEDLE_ZH

ABLATIONS = [
    {"id": "no-hadamard", "replace": "HadamardMLP -> dense GeLU MLP"},
    {"id": "no-engram", "engram_layers": []},
    {"id": "no-mhc", "mhc_lanes": 1},
    {"id": "depth-18", "n_layers": 18},
    {"id": "width-384", "d_model": 384},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-baseline", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    flag = EXPERIMENTS_RUNS / "needle-zh-baseline" / "passed.json"
    out_dir = EXPERIMENTS_RUNS / "needle-zh-ablate"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_baseline and not flag.is_file():
        report = {
            "blocked": True,
            "reason": "product baseline has not passed gates.json",
            "ablations_queued": ABLATIONS,
            "phase2_blocked_until": "phase1 execute/[] + confidence calibration",
        }
        (out_dir / "blocked.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    report = {
        "blocked": False,
        "smoke": args.smoke,
        "ablations": ABLATIONS,
        "controls": "tokenizer, pretrain tokens, SFT pack, EVAL hash held fixed",
    }
    (out_dir / "plan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
