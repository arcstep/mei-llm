#!/usr/bin/env python3
"""Validate v2 full-call packs: serializer, no Route-ID, isolation, gold schema."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import ROOT, SCRIPTS_ROOT, SFT_TRAIN, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_canonical_lib import (  # noqa: E402
    SERIALIZER_V2,
    freeze_family_splits,
    has_route_id_gold,
    leak_markers,
    load_jsonl,
    query_overlaps,
)

PACK_SMOKE = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs/mei-toolcall-v2-smoke.jsonl"
PACK_ORACLE = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-smoke.jsonl"
BANK = ROOT / "notebook/evaluation/banks/mei-toolcall-v2/eval-bank-smoke.jsonl"


def validate_rows(rows: list[dict], eval_rows: list[dict], *, require_compiler: bool) -> dict:
    errors: list[str] = []
    if not rows:
        errors.append("empty pack")
    for row in rows:
        sid = str(row.get("sample_id") or "?")
        if has_route_id_gold(row):
            errors.append(f"{sid}: route_id")
        prompt = str(row.get("prompt_text") or "")
        leaks = leak_markers(prompt)
        if leaks:
            errors.append(f"{sid}: prompt_leak:{','.join(leaks)}")
        if require_compiler:
            if row.get("serializer") != SERIALIZER_V2:
                errors.append(f"{sid}: serializer")
            answers = row.get("answers")
            if answers is None or not isinstance(answers, list):
                errors.append(f"{sid}: answers_not_list")
            elif answers:
                call = answers[0]
                if not isinstance(call, dict) or not call.get("name") or "arguments" not in call:
                    errors.append(f"{sid}: answers_schema")
                if "route_id" in call:
                    errors.append(f"{sid}: route_id_in_answers")
            retrieved = row.get("retrieved_tools") or []
            if len(retrieved) > 5:
                errors.append(f"{sid}: top5_overflow")
            if not str(row.get("query") or "").strip():
                errors.append(f"{sid}: empty_query")
    leaks = query_overlaps(rows, eval_rows)
    errors.extend(f"eval_overlap:{h}" for h in leaks)
    splits = freeze_family_splits(rows)
    if require_compiler and not splits["ok"]:
        errors.append(f"cf_cross_split:{splits['cf_cross_split']}")
    return {
        "ok": not errors,
        "n": len(rows),
        "n_errors": len(errors),
        "errors": errors[:80],
        "splits": splits,
        "require_compiler": require_compiler,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--isolation-only", action="store_true")
    ap.add_argument("--require-compiler", action="store_true")
    args = ap.parse_args()
    pack = args.pack or (PACK_ORACLE if PACK_ORACLE.is_file() and args.require_compiler else PACK_SMOKE)
    isolation = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "check_train_eval_isolation.py"), "--scope", "sft-v2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    rows = load_jsonl(pack)
    eval_rows = load_jsonl(BANK)
    require = bool(args.require_compiler or pack.name.endswith("oracle-smoke.jsonl") or "oracle" in pack.name)
    report = validate_rows(rows, eval_rows, require_compiler=require)
    report["pack"] = str(pack.relative_to(ROOT)) if pack.is_relative_to(ROOT) else str(pack)
    report["isolation_returncode"] = isolation.returncode
    if isolation.returncode != 0:
        report["ok"] = False
        report["errors"] = ["isolation_failed"] + list(report.get("errors") or [])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.isolation_only:
        return isolation.returncode
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
