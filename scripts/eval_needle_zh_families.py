#!/usr/bin/env python3
"""Family-wise exact-match report + always-refuse baseline on frozen VRM holdout.

Does not tune on holdout. Smoke mode reports gold-only baselines.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from repo_paths import BANK_NEEDLE_VRM_AGENT, EXPERIMENTS_RUNS, ROOT


REFUSE_FAMILIES = {"missing", "scene_conflict", "illegal_pair", "offtopic"}
EXECUTE_FAMILIES = {"gesture", "home", "order", "sequence", "paraphrase"}


def gold_calls(row: dict) -> list:
    g = row.get("gold") or {}
    return list(g.get("function_calls") or [])


def is_empty(calls: list) -> bool:
    return not calls


def family_bucket(fam: str) -> str:
    if fam in REFUSE_FAMILIES:
        return "refuse"
    if fam in EXECUTE_FAMILIES:
        return "execute"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--bank",
        type=Path,
        default=BANK_NEEDLE_VRM_AGENT,
        help="Explicit bank path. Default is frozen v1; pass eval-bank-v2.jsonl for current 2K KPI.",
    )
    args = ap.parse_args()
    bank = args.bank if args.bank.is_absolute() else (ROOT / args.bank)
    rows = [json.loads(l) for l in bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") == args.split]
    fam = defaultdict(lambda: {"n": 0, "gold_empty": 0, "always_refuse_correct": 0})
    buckets = defaultdict(lambda: {"n": 0, "always_refuse_correct": 0})
    for r in rows:
        f = str(r.get("family") or "unknown")
        empty = is_empty(gold_calls(r))
        fam[f]["n"] += 1
        fam[f]["gold_empty"] += int(empty)
        fam[f]["always_refuse_correct"] += int(empty)
        b = family_bucket(f)
        buckets[b]["n"] += 1
        buckets[b]["always_refuse_correct"] += int(empty)
    n = max(len(rows), 1)
    always_refuse = sum(1 for r in rows if is_empty(gold_calls(r))) / n
    report = {
        "bank": str(bank.resolve().relative_to(ROOT)),
        "split": args.split,
        "n": len(rows),
        "always_refuse_exact_match": round(always_refuse, 4),
        "buckets": {
            k: {
                "n": v["n"],
                "always_refuse_exact_match": round(v["always_refuse_correct"] / max(v["n"], 1), 4),
            }
            for k, v in buckets.items()
        },
        "families": {
            k: {
                "n": v["n"],
                "gold_empty": v["gold_empty"],
                "always_refuse_exact_match": round(v["always_refuse_correct"] / max(v["n"], 1), 4),
            }
            for k, v in sorted(fam.items())
        },
        "smoke": args.smoke,
        "note": "Always-refuse is a required baseline. Constant [] cannot pass execute_family>=90%.",
    }
    out = EXPERIMENTS_RUNS / f"needle-zh-family-baseline{'-smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
