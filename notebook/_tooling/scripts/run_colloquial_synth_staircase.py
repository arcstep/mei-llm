#!/usr/bin/env python3
"""Qwen colloquial staircase: 0.5M → 5M → 20M → 35M raw, freeze ≥30M unique.

Each rung freezes an independent review. Does not write formal_cpt_eligible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1, ROOT, SCRIPTS_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colloquial_synth_lib import load_contract

PRODUCER = SCRIPTS_ROOT / "produce_colloquial_qwen.py"
AUDITOR = SCRIPTS_ROOT / "audit_zh_pretrain_colloquial.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    ap.add_argument("--max-rung", type=int, default=35_000_000)
    ap.add_argument("--max-spend-cny", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    contract = load_contract()
    rungs = [500_000, 5_000_000, 20_000_000, int(contract["formal_cpt"]["requires_unique_tokens"])]
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    reports = []
    for target in rungs:
        if target > args.max_rung:
            break
        cmd = [
            sys.executable,
            str(PRODUCER),
            "--out-dir",
            str(out),
            "--target-unique-tokens",
            str(target),
            "--resume",
            "--release-kind",
            f"scale-{target}",
            "--max-spend-cny",
            str(args.max_spend_cny),
            "--workers",
            str(args.workers),
            "--qps",
            str(args.qps),
            "--seed",
            str(args.seed),
        ]
        if target >= 30_000_000:
            cmd.extend(["--max-docs", "450000"])
        proc = subprocess.run(cmd, cwd=str(ROOT))
        audit = subprocess.run(
            [sys.executable, str(AUDITOR), "--corpus-dir", str(out), "--expected-generator", "qwen-plus"],
            cwd=str(ROOT),
        )
        reports.append(
            {
                "target": target,
                "produce_code": proc.returncode,
                "audit_code": audit.returncode,
            }
        )
        if proc.returncode != 0:
            print(json.dumps({"ok": False, "reports": reports}, indent=2))
            return proc.returncode
        if audit.returncode != 0 and target >= 5_000_000:
            print(json.dumps({"ok": False, "reason": "audit_failed_before_scaleup", "reports": reports}, indent=2))
            return audit.returncode
    print(json.dumps({"ok": True, "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
