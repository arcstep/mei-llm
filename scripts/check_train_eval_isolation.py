#!/usr/bin/env python3
"""Fail if train seed intersects eval bank item_ids, embeds EVAL-* text,
or lacks task/evidence release provenance when claimed as domain-train samples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_RE = re.compile(r"\bEVAL-(?:DEV|EDGE)-\d+\b")
GOLD_RE = re.compile(r"(acceptance/|gold/|holdout-answer)", re.I)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "data/eval/eval-bank-v0.pending.jsonl")
    ap.add_argument("--seed", type=Path, default=ROOT / "train/seed/sft-smoke-v0.jsonl")
    ap.add_argument(
        "--domain-train",
        type=Path,
        default=None,
        help="Optional Evidence-derived supervision JSONL (must carry provenance)",
    )
    ap.add_argument(
        "--heldout-task-ids",
        type=Path,
        default=None,
        help="Optional JSON list of heldout task_ids that must not appear in train text",
    )
    args = ap.parse_args()

    eval_ids = {r["item_id"] for r in load_jsonl(args.bank) if "item_id" in r}
    seed = load_jsonl(args.seed)
    seed_ids = {str(r.get("sample_id") or "") for r in seed}

    overlap = sorted(eval_ids & seed_ids)
    leaks: list[str] = []
    for r in seed:
        sid = r.get("sample_id")
        blob = json.dumps(r, ensure_ascii=False)
        if EVAL_RE.search(blob):
            leaks.append(str(sid))
        if GOLD_RE.search(blob):
            leaks.append(f"{sid}:gold_path")

    domain_errors: list[str] = []
    domain_rows = load_jsonl(args.domain_train) if args.domain_train else []
    for r in domain_rows:
        sid = r.get("sample_id")
        if not r.get("task_catalog_release"):
            domain_errors.append(f"{sid}:missing_task_catalog_release")
        if not r.get("evidence_release"):
            domain_errors.append(f"{sid}:missing_evidence_release")
        if not r.get("source_refs"):
            domain_errors.append(f"{sid}:missing_source_refs")
        if not r.get("transform_recipe"):
            domain_errors.append(f"{sid}:missing_transform_recipe")
        blob = json.dumps(r, ensure_ascii=False)
        if GOLD_RE.search(blob):
            domain_errors.append(f"{sid}:gold_leak")

    heldout_leaks: list[str] = []
    if args.heldout_task_ids and args.heldout_task_ids.is_file():
        heldout_ids = json.loads(args.heldout_task_ids.read_text(encoding="utf-8"))
        corpus = seed + domain_rows
        for r in corpus:
            blob = json.dumps(r, ensure_ascii=False)
            for hid in heldout_ids:
                if hid and hid in blob:
                    heldout_leaks.append(f"{r.get('sample_id')}:{hid}")

    ok = not overlap and not leaks and not domain_errors and not heldout_leaks
    report = {
        "ok": ok,
        "eval_n": len(eval_ids),
        "seed_n": len(seed),
        "domain_train_n": len(domain_rows),
        "id_overlap": overlap,
        "eval_text_leaks_in_seed": leaks,
        "domain_provenance_errors": domain_errors,
        "heldout_leaks": heldout_leaks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
