#!/usr/bin/env python3
"""Validate MW disposition packs against the frozen reason_code codebook and holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_MEI_MW_DISPOSITION_V2,
    BANK_MEI_MW_DISPOSITION_V2_DEV,
    BANK_MEI_MW_DISPOSITION_V2_TEST,
    BANK_NEEDLE_VRM_MW,
    PACK_MEI_MW_DISPOSITION_V2_SMOKE,
    ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_canonical_lib import (  # noqa: E402
    codebook_by_reason,
    freeze_family_splits,
    load_jsonl,
    project_cell,
    query_overlaps,
)


def validate(rows: list[dict]) -> dict:
    book = codebook_by_reason()
    errors: list[str] = []
    if not rows:
        errors.append("empty pack")
    for row in rows:
        sid = str(row.get("sample_id") or "?")
        reason = str(row.get("reason_code") or "")
        if reason not in book:
            errors.append(f"{sid}: unknown_reason")
            continue
        mapped = book[reason]
        if int(row.get("reason_class_id", -1)) != int(mapped["class_id"]):
            errors.append(f"{sid}: class_id")
        if row.get("act") != mapped["act"] or row.get("cell") != mapped["cell"]:
            errors.append(f"{sid}: act_cell_drift")
        if project_cell(mapped["act"], reason) != mapped["cell"]:
            errors.append(f"{sid}: codebook_cell")
        if row.get("head_target") != "reason_code":
            errors.append(f"{sid}: head_target")
        if mapped["act"] != "execute" and row.get("function_calls"):
            errors.append(f"{sid}: non_execute_calls")
        if mapped["act"] == "execute" and row.get("gaps"):
            errors.append(f"{sid}: execute_gaps")
        if mapped["act"] == "expand" and not row.get("gaps"):
            errors.append(f"{sid}: expand_no_gaps")
        if "<routes>" in json.dumps(row, ensure_ascii=False):
            errors.append(f"{sid}: routes")
    leaks = query_overlaps(
        rows,
        load_jsonl(BANK_NEEDLE_VRM_MW)
        + load_jsonl(BANK_MEI_MW_DISPOSITION_V2)
        + load_jsonl(BANK_MEI_MW_DISPOSITION_V2_DEV)
        + load_jsonl(BANK_MEI_MW_DISPOSITION_V2_TEST),
    )
    errors.extend(f"holdout_or_eval:{h}" for h in leaks)
    splits = freeze_family_splits(rows)
    if not splits["ok"]:
        errors.append(f"cf_cross_split:{splits['cf_cross_split']}")
    return {
        "ok": not errors,
        "n": len(rows),
        "n_errors": len(errors),
        "errors": errors[:80],
        "n_reason_codes": len({str(r.get("reason_code")) for r in rows}),
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=PACK_MEI_MW_DISPOSITION_V2_SMOKE)
    args = ap.parse_args()
    report = validate(load_jsonl(args.pack))
    report["pack"] = str(args.pack.relative_to(ROOT)) if args.pack.is_relative_to(ROOT) else str(args.pack)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
