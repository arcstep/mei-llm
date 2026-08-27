#!/usr/bin/env python3
"""Validate retrieval-v2 packs: gold in catalog, family split, isolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import BANK_MEI_RETRIEVAL_V2, ROOT, PACK_MEI_RETRIEVAL_V2_SMOKE

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_canonical_lib import (  # noqa: E402
    freeze_family_splits,
    has_route_id_gold,
    load_jsonl,
    query_overlaps,
    stratified_retrieval_metrics,
)


def validate(rows: list[dict], eval_rows: list[dict]) -> dict:
    errors: list[str] = []
    if not rows:
        errors.append("empty pack")
    for row in rows:
        sid = str(row.get("sample_id") or "?")
        catalog = [str(t.get("name")) for t in row.get("catalog_tools") or []]
        gold = row.get("gold_tool")
        if gold and gold not in catalog:
            errors.append(f"{sid}: gold_not_in_catalog")
        if row.get("family") == "no_match" and gold is not None:
            errors.append(f"{sid}: no_match_has_gold")
        if row.get("family") != "no_match" and not gold:
            errors.append(f"{sid}: missing_gold")
        if has_route_id_gold(row):
            errors.append(f"{sid}: route_id")
        if not str(row.get("query") or "").strip():
            errors.append(f"{sid}: empty_query")
    leaks = query_overlaps(rows, eval_rows)
    errors.extend(f"eval_overlap:{h}" for h in leaks)
    catalog = []
    if rows:
        catalog = list(rows[0].get("catalog_tools") or [])
        # union catalogs for lexical floor
        names = {}
        for row in rows:
            for tool in row.get("catalog_tools") or []:
                names[str(tool.get("name"))] = tool
        catalog = list(names.values())
    splits = freeze_family_splits(rows)
    if not splits["ok"]:
        errors.append(f"cf_cross_split:{splits['cf_cross_split']}")
    return {
        "ok": not errors,
        "n": len(rows),
        "n_errors": len(errors),
        "errors": errors[:80],
        "splits": splits,
        "lexical": stratified_retrieval_metrics(rows, catalog) if catalog else {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=PACK_MEI_RETRIEVAL_V2_SMOKE)
    ap.add_argument("--eval-bank", type=Path, default=BANK_MEI_RETRIEVAL_V2)
    args = ap.parse_args()
    report = validate(load_jsonl(args.pack), load_jsonl(args.eval_bank))
    report["pack"] = str(args.pack.relative_to(ROOT)) if args.pack.is_relative_to(ROOT) else str(args.pack)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
