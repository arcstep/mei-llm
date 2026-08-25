#!/usr/bin/env python3
"""Protocol review for MW SFT and MW holdout. Deterministic checklist, not a teacher rewrite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_MW,
    BANK_NEEDLE_VRM_MW_RECIPE,
    PACK_NEEDLE_MW_SFT_2K,
    RECIPE_NEEDLE_MW_SFT,
    ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_needle_mw_v0 import case_from_row, gold_error, gold_from_case, gold_of  # noqa: E402
from needle_home_sft_lib import dump_jsonl, load_jsonl, sha256_file  # noqa: E402
from needle_mw_governance_lib import query_banned, query_contract_ok  # noqa: E402
from validate_needle_mw_holdout_v0 import validate as validate_eval, write_lock  # noqa: E402
from validate_needle_mw_sft_v0 import validate as validate_sft  # noqa: E402


def review_row(row: dict, *, kind: str) -> dict:
    sid = row.get("item_id") or row.get("sample_id")
    case = case_from_row(row)
    errors = []
    banned = query_banned(str(row.get("query") or ""))
    if banned:
        errors.append(banned)
    ge = gold_error(case)
    if ge:
        errors.append(ge)
    else:
        if gold_from_case(case) != gold_of(row):
            errors.append("gold_mismatch")
    if not query_contract_ok(case, str(row.get("query") or "")):
        errors.append("contract")
    status = "pass" if not errors else "reject"
    return {
        "id": sid,
        "kind": kind,
        "act": row.get("act"),
        "cell": row.get("cell"),
        "reason_code": row.get("reason_code"),
        "family": row.get("family"),
        "cf_group": row.get("cf_group"),
        "cf_axis": row.get("cf_axis"),
        "split": row.get("split"),
        "high_risk": bool(row.get("high_risk")),
        "review_status": status,
        "errors": errors,
        "reviewer": "protocol-mw-v0",
        "query": row.get("query"),
        "scene": row.get("scene"),
    }


def select_sft(rows: list[dict], recipe: dict) -> list[dict]:
    chosen: list[dict] = []
    seen: set[str] = set()

    def take(row: dict) -> None:
        sid = str(row["sample_id"])
        if sid in seen:
            return
        seen.add(sid)
        chosen.append(row)

    spec = recipe.get("review") or {}
    if spec.get("include_all_escalate", True):
        for row in rows:
            if row.get("act") == "escalate":
                take(row)
    if spec.get("include_all_unknown_cell", True):
        for row in rows:
            if row.get("cell") == "Unknown":
                take(row)
    if spec.get("include_all_sequence", True):
        for row in rows:
            if row.get("family") == "sequence":
                take(row)
    if spec.get("include_all_counterfactual_canary", True):
        for row in rows:
            if row.get("cf_role") == "flip" and row.get("cf_axis") in {
                "scene_conflict",
                "illegal_pair",
                "partial_sequence",
                "unknown_slot",
                "negation",
            }:
                take(row)
    extra = int(spec.get("extra_stratified") or 80)
    by_act: dict[str, list[dict]] = {}
    for row in rows:
        by_act.setdefault(str(row.get("act")), []).append(row)
    i = 0
    while extra > 0 and by_act:
        for act, pool in by_act.items():
            if extra <= 0:
                break
            if i < len(pool):
                take(pool[i])
                extra -= 1
        i += 1
        if i > 2000:
            break
    return chosen


def select_eval(rows: list[dict], recipe: dict) -> list[dict]:
    chosen: list[dict] = []
    seen: set[str] = set()

    def take(row: dict) -> None:
        sid = str(row["item_id"])
        if sid in seen:
            return
        seen.add(sid)
        chosen.append(row)

    spec = recipe.get("review") or {}
    if spec.get("include_all_dev", True):
        for row in rows:
            if row.get("split") == "dev":
                take(row)
    if spec.get("include_all_escalate", True):
        for row in rows:
            if row.get("act") == "escalate":
                take(row)
    if spec.get("include_all_unknown_cell", True):
        for row in rows:
            if row.get("cell") == "Unknown":
                take(row)
    high = set(spec.get("high_risk_tags") or [])
    if spec.get("include_all_high_risk", True):
        for row in rows:
            tags = set(row.get("case_tags") or [])
            if row.get("high_risk") or tags & high:
                take(row)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sft", "eval", "both"], default="both")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    reports = {}
    if args.target in {"sft", "both"}:
        recipe = json.loads(RECIPE_NEEDLE_MW_SFT.read_text(encoding="utf-8"))
        rows = load_jsonl(PACK_NEEDLE_MW_SFT_2K)
        selected = select_sft(rows, recipe)
        reviews = [review_row(r, kind="sft") for r in selected]
        n_pass = sum(1 for r in reviews if r["review_status"] == "pass")
        n_rej = sum(1 for r in reviews if r["review_status"] != "pass")
        dump_jsonl(PACK_NEEDLE_MW_SFT_2K.with_name("mw-sft-v0-2k.review-sample.jsonl"), reviews)
        for row in rows:
            sid = row["sample_id"]
            hit = next((x for x in reviews if x["id"] == sid), None)
            if hit:
                row["review_status"] = hit["review_status"]
        from needle_home_sft_lib import dump_jsonl as _d

        _d(PACK_NEEDLE_MW_SFT_2K, rows)
        manifest = {
            "pack": "mw-sft-v0-2k.jsonl",
            "n": len(rows),
            "sha256": sha256_file(PACK_NEEDLE_MW_SFT_2K),
            "recipe_sha256": sha256_file(RECIPE_NEEDLE_MW_SFT),
            "review_n": len(reviews),
            "n_pass": n_pass,
            "n_rejected": n_rej,
            "pass_rate": round(n_pass / max(len(reviews), 1), 4),
            "reviewer": "protocol-mw-v0",
            "validate": validate_sft(rows, recipe),
            "not_training_entry": True,
        }
        PACK_NEEDLE_MW_SFT_2K.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports["sft"] = {"review_n": len(reviews), "n_pass": n_pass, "n_rejected": n_rej}
    if args.target in {"eval", "both"}:
        recipe = json.loads(BANK_NEEDLE_VRM_MW_RECIPE.read_text(encoding="utf-8"))
        rows = load_jsonl(BANK_NEEDLE_VRM_MW)
        selected = select_eval(rows, recipe)
        reviews = [review_row(r, kind="eval") for r in selected]
        n_pass = sum(1 for r in reviews if r["review_status"] == "pass")
        n_rej = sum(1 for r in reviews if r["review_status"] != "pass")
        dump_jsonl(BANK_NEEDLE_VRM_MW.with_name("eval-bank-v0.review.jsonl"), reviews)
        for row in rows:
            hit = next((x for x in reviews if x["id"] == row["item_id"]), None)
            row["review_status"] = hit["review_status"] if hit else row.get("review_status") or "generated"
        dump_jsonl(BANK_NEEDLE_VRM_MW, rows)
        val = validate_eval(rows, recipe)
        review_meta = {
            "n": len(reviews),
            "n_pass": n_pass,
            "n_rejected": n_rej,
            "pass_rate": round(n_pass / max(len(reviews), 1), 4),
            "reviewer": "protocol-mw-v0",
        }
        if args.freeze:
            if not val["ok"] or n_rej:
                print(json.dumps({"error": "refuse freeze", "validate": val, "review": review_meta}, ensure_ascii=False))
                return 1
            lock = write_lock(rows, recipe, review_meta)
            reports["lock"] = lock
        reports["eval"] = {"review_n": len(reviews), "n_pass": n_pass, "n_rejected": n_rej, "validate_ok": val["ok"]}
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
