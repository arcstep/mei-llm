#!/usr/bin/env python3
"""Validate needle-vrm-mw-v0 holdout: schema, quotas, counterfactuals, isolation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_MW,
    BANK_NEEDLE_VRM_MW_LOCK,
    BANK_NEEDLE_VRM_MW_RECIPE,
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
)


def write_lock(rows: list[dict], recipe: dict, review: dict | None = None) -> dict:
    lock = {
        "bank": "needle-vrm-mw-v0",
        "holdout_version": "mw-v0",
        "not_phase1_kpi": True,
        "n_total": len(rows),
        "n_dev": sum(1 for r in rows if r.get("split") == "dev"),
        "n_eval": sum(1 for r in rows if r.get("split") == "eval"),
        "sha256": sha256_file(BANK_NEEDLE_VRM_MW),
        "recipe_sha256": sha256_file(BANK_NEEDLE_VRM_MW_RECIPE),
        "schema_sha256": sha256_file(SCHEMA_MW_GOVERNANCE),
        "toolset_sha256": sha256_file(ROOT / "eval/shared/toolsets/needle-vrm-agent-v0.json"),
        "v1_sha256": V1_SHA,
        "v2_sha256": V2_SHA,
        "distribution": {
            "act": dict(Counter(str(r.get("act")) for r in rows)),
            "cell": dict(Counter(str(r.get("cell")) for r in rows)),
            "reason_code": dict(Counter(str(r.get("reason_code")) for r in rows)),
            "split": dict(Counter(str(r.get("split")) for r in rows)),
            "cf_n": sum(1 for r in rows if r.get("cf_group")),
            "cf_frac": round(sum(1 for r in rows if r.get("cf_group")) / max(len(rows), 1), 4),
        },
        "review": review or {},
    }
    BANK_NEEDLE_VRM_MW_LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def validate(rows: list[dict], recipe: dict) -> dict:
    errors = schema_errors(rows, kind="eval")
    if len(rows) != int(recipe["n_total"]):
        errors.append(f"n={len(rows)} want {recipe['n_total']}")
    n_dev = sum(1 for r in rows if r.get("split") == "dev")
    n_eval = sum(1 for r in rows if r.get("split") == "eval")
    if n_dev != int(recipe["n_dev"]) or n_eval != int(recipe["n_eval"]):
        errors.append(f"split dev/eval={n_dev}/{n_eval} want {recipe['n_dev']}/{recipe['n_eval']}")
    acts = Counter(str(r.get("act")) for r in rows)
    for act, want in recipe["act_n"].items():
        if acts.get(act, 0) != int(want):
            errors.append(f"act {act}={acts.get(act, 0)} want {want}")
    cf_n = sum(1 for r in rows if r.get("cf_group"))
    min_cf = float(recipe.get("min_counterfactual_frac") or 0.4)
    if cf_n / max(len(rows), 1) < min_cf:
        errors.append(f"cf_frac={cf_n/len(rows):.4f} < {min_cf}")
    groups: dict[str, set[str]] = {}
    for row in rows:
        g = str(row.get("cf_group") or row.get("canonical_id") or "")
        groups.setdefault(g, set()).add(str(row.get("split")))
    for g, splits in groups.items():
        if g and len(splits) > 1:
            errors.append(f"group {g} crosses splits {splits}")
    blocked = collect_blocked_queries()
    self_q = {str(r.get("query") or "") for r in rows}
    foreign = blocked - self_q
    for row in rows:
        q = str(row.get("query") or "")
        if q in foreign:
            errors.append(f"{row.get('item_id')}: exact leak {q[:40]}")
        if not str(row.get("item_id") or "").startswith(recipe["item_id_prefix"]):
            errors.append(f"{row.get('item_id')}: prefix")
    return {
        "ok": not errors,
        "n": len(rows),
        "n_dev": n_dev,
        "n_eval": n_eval,
        "act": dict(acts),
        "cf_n": cf_n,
        "cf_frac": round(cf_n / max(len(rows), 1), 4),
        "n_errors": len(errors),
        "errors": errors[:80],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_VRM_MW)
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    frozen = assert_upstream_frozen()
    recipe = json.loads(BANK_NEEDLE_VRM_MW_RECIPE.read_text(encoding="utf-8"))
    rows = load_jsonl(args.bank)
    report = validate(rows, recipe)
    report["bank"] = str(args.bank.resolve().relative_to(ROOT))
    report["sha256"] = sha256_file(args.bank) if args.bank.is_file() else None
    report["frozen_upstream"] = frozen
    if args.write_lock and report["ok"]:
        report["lock"] = write_lock(rows, recipe)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
