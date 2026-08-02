#!/usr/bin/env python3
"""Fail if train seed intersects eval bank item_ids or embeds EVAL-* text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_RE = re.compile(r"\bEVAL-(?:DEV|EDGE)-\d+\b")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "data/eval/eval-bank-v0.pending.jsonl")
    ap.add_argument("--seed", type=Path, default=ROOT / "train/seed/sft-smoke-v0.jsonl")
    args = ap.parse_args()

    eval_ids = {r["item_id"] for r in load_jsonl(args.bank)}
    seed = load_jsonl(args.seed)
    seed_ids = {str(r.get("sample_id") or "") for r in seed}

    overlap = sorted(eval_ids & seed_ids)
    leaks: list[str] = []
    for r in seed:
        sid = r.get("sample_id")
        blob = json.dumps(r, ensure_ascii=False)
        if EVAL_RE.search(blob):
            leaks.append(str(sid))

    ok = not overlap and not leaks
    report = {
        "ok": ok,
        "eval_n": len(eval_ids),
        "seed_n": len(seed),
        "id_overlap": overlap,
        "eval_text_leaks_in_seed": leaks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
