#!/usr/bin/env python3
"""Validate mw-sft-v0-2k: schema-program gold, quotas, isolation. Never writes home-sft packs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from repo_paths import (
    PACK_NEEDLE_MW_SFT_2K,
    RECIPE_NEEDLE_MW_SFT,
    ROOT,
    SCHEMA_MW_GOVERNANCE,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_needle_mw_v0 import schema_errors  # noqa: E402
from needle_home_sft_lib import load_jsonl, sha256_file  # noqa: E402
from needle_mw_governance_lib import (  # noqa: E402
    V1_SHA,
    V2_SHA,
    assert_upstream_frozen,
    collect_blocked_queries,
    format_sft_user_text,
)


def validate(rows: list[dict], recipe: dict) -> dict:
    errors = schema_errors(rows, kind="sft")
    if len(rows) != int(recipe["n_total"]):
        errors.append(f"n={len(rows)} want {recipe['n_total']}")
    acts = Counter(str(r.get("act")) for r in rows)
    for act, want in recipe["act_n"].items():
        if acts.get(act, 0) != int(want):
            errors.append(f"act {act}={acts.get(act, 0)} want {want}")
    n_valid = sum(1 for r in rows if r.get("split") == "valid")
    n_train = sum(1 for r in rows if r.get("split") == "train")
    if n_train + n_valid != len(rows):
        errors.append("split not train/valid")
    # groups must not cross split
    groups: dict[str, set[str]] = {}
    for row in rows:
        g = str(row.get("cf_group") or row.get("canonical_id") or "")
        groups.setdefault(g, set()).add(str(row.get("split")))
    for g, splits in groups.items():
        if g and len(splits) > 1:
            errors.append(f"group {g} crosses splits {splits}")
    blocked = collect_blocked_queries()
    # pack itself is in blocked collector; remove self queries
    self_q = {str(r.get("query") or "") for r in rows}
    foreign = blocked - self_q
    for row in rows:
        q = str(row.get("query") or "")
        if q in foreign:
            errors.append(f"{row.get('sample_id')}: exact leak {q[:40]}")
        user = format_sft_user_text(row)
        if str(row.get("scene") or "").strip() and not user.startswith("场景："):
            errors.append(f"{row.get('sample_id')}: scene not encoded")
    tools = set()
    for row in rows:
        for c in row.get("function_calls") or []:
            tools.add(str(c.get("name")))
    return {
        "ok": not errors,
        "n": len(rows),
        "n_train": n_train,
        "n_valid": n_valid,
        "act": dict(acts),
        "tools_execute": sorted(tools),
        "n_errors": len(errors),
        "errors": errors[:80],
        "cf_n": sum(1 for r in rows if r.get("cf_group")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=PACK_NEEDLE_MW_SFT_2K)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    frozen = assert_upstream_frozen()
    recipe = json.loads(RECIPE_NEEDLE_MW_SFT.read_text(encoding="utf-8"))
    rows = load_jsonl(args.pack)
    report = validate(rows, recipe)
    report["pack"] = str(args.pack.resolve().relative_to(ROOT))
    report["sha256"] = sha256_file(args.pack) if args.pack.is_file() else None
    report["recipe_sha256"] = sha256_file(RECIPE_NEEDLE_MW_SFT)
    report["schema_sha256"] = sha256_file(SCHEMA_MW_GOVERNANCE)
    report["frozen_upstream"] = frozen
    report["v1_ok"] = frozen["v1_bank_sha256"] == V1_SHA
    report["v2_ok"] = frozen["v2_bank_sha256"] == V2_SHA
    if args.write_manifest:
        out = args.pack.with_suffix(".validate.json")
        slim = dict(report)
        out.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
