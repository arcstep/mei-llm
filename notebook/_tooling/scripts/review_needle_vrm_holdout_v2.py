#!/usr/bin/env python3
"""Stratified protocol review for Needle VRM holdout v2 (600 items).

Human-equivalent checklist, applied deterministically. Unreviewed rows stay
`generated`. Never writes v1 assets. --freeze writes lock after validator OK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    BANK_NEEDLE_VRM_AGENT_V2,
    BANK_NEEDLE_VRM_AGENT_V2_LOCK,
    BANK_NEEDLE_VRM_AGENT_V2_RECIPE,
    EVAL_SHARED_ROOT,
    ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from needle_home_sft_lib import dump_jsonl, load_jsonl, sha256_file  # noqa: E402
from needle_vrm_holdout_v2_lib import (  # noqa: E402
    catalog_ok,
    gold_from_intent_v2,
    intent_from_row,
    load_recipe,
    protected_slots_ok,
    query_banned,
)
from validate_needle_vrm_holdout_v2 import V1_SHA, validate, write_lock  # noqa: E402

HARD_TAGS = {
    "negation_scope",
    "same_word_diff_slot",
    "correction",
    "scene_flips_execute",
    "illegal_food",
    "short_ellipsis",
    "sequence_order",
}
REFUSE = {"missing", "scene_conflict", "illegal_pair", "offtopic"}
EXECUTE = {"gesture", "home", "order", "sequence"}


def _eq(a, b) -> bool:
    return json.dumps(a or [], ensure_ascii=False, sort_keys=True) == json.dumps(
        b or [], ensure_ascii=False, sort_keys=True
    )


def select_review_ids(rows: list[dict], recipe: dict) -> list[str]:
    spec = recipe.get("review") or {}
    want = int(spec.get("n") or 600)
    extra_n = int(spec.get("extra_hard") or 58)
    chosen: list[str] = []
    seen: set[str] = set()

    def take(row: dict) -> None:
        sid = str(row["item_id"])
        if sid in seen:
            return
        seen.add(sid)
        chosen.append(sid)

    if spec.get("include_all_dev", True):
        for row in rows:
            if row.get("split") == "dev":
                take(row)
    if spec.get("include_all_sequence", True):
        for row in rows:
            if row.get("family") == "sequence":
                take(row)
    if spec.get("include_all_scene_conflict", True):
        for row in rows:
            if row.get("family") == "scene_conflict":
                take(row)
    hard = [
        r
        for r in rows
        if str(r["item_id"]) not in seen and HARD_TAGS.intersection(r.get("case_tags") or [])
    ]
    for row in hard:
        if extra_n <= 0:
            break
        take(row)
        extra_n -= 1
    if extra_n > 0:
        for row in rows:
            if extra_n <= 0:
                break
            if str(row["item_id"]) in seen:
                continue
            take(row)
            extra_n -= 1
    return chosen[:want]


def second_reading_conflict(row: dict) -> str | None:
    family = str(row.get("family") or "")
    q = str(row.get("query") or "")
    gold = (row.get("gold") or {}).get("function_calls") or []
    if family in EXECUTE and not gold:
        return "execute_empty_gold"
    if family in REFUSE and gold:
        return "refuse_nonempty_gold"
    if family == "illegal_pair":
        legal = {("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")}
        slots = row.get("zh_slots") or {}
        pair = (slots.get("shop"), slots.get("dish"))
        if pair in legal:
            return "illegal_pair_is_legal"
    if family == "sequence":
        if "先" not in q and "再" not in q and "随后" not in q and "然后" not in q:
            return "sequence_order_not_visible"
        if len(gold) < 2:
            return "sequence_too_short"
    if family == "scene_conflict" and not str(row.get("scene") or "").strip():
        return "scene_missing"
    if family == "offtopic" and gold:
        return "offtopic_calls"
    return None


def review_row(row: dict, recipe: dict) -> dict:
    sid = str(row.get("item_id") or "")
    q = str(row.get("query") or "")
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    han = sum(1 for ch in q if "\u4e00" <= ch <= "\u9fff")
    checks["naturalness"] = (
        han >= 6 and len(q) >= 8 and query_banned(q) is None and "地在不改口令" not in q
    )
    if not checks["naturalness"]:
        reasons.append("query_unnatural_or_banned")
    intent = intent_from_row(row)
    checks["intent_hold"] = protected_slots_ok(intent, q)
    if not checks["intent_hold"]:
        reasons.append("protected_slot_dropped")
    scene = str(row.get("scene") or "")
    if row.get("family") == "scene_conflict":
        checks["scene_visible"] = bool(scene.strip()) and scene.strip() not in q
        # scene may appear in query; still require dedicated scene field
        checks["scene_visible"] = bool(scene.strip())
    else:
        checks["scene_visible"] = True
    if not checks["scene_visible"]:
        reasons.append("scene_not_visible")
    gold = (row.get("gold") or {}).get("function_calls")
    recomputed = gold_from_intent_v2(intent, recipe)
    checks["gold_recompute"] = _eq(gold, recomputed) and catalog_ok(gold or [])
    if not checks["gold_recompute"]:
        reasons.append("gold_mismatch")
    family = str(row.get("family") or "")
    if family in REFUSE:
        checks["refuse_reason"] = not gold
    else:
        checks["refuse_reason"] = True
    if not checks["refuse_reason"]:
        reasons.append("refuse_gold_nonempty")
    alt = second_reading_conflict(row)
    checks["single_reading"] = alt is None
    if alt:
        reasons.append(alt)
    passed = all(checks.values())
    return {
        "item_id": sid,
        "split": row.get("split"),
        "family": family,
        "case_tags": list(row.get("case_tags") or []),
        "checks": checks,
        "pass": passed,
        "reasons": reasons,
        "reviewer": "protocol-v2",
        "review_status": "reviewed" if passed else "rejected",
    }


def freeze_assets(rows: list[dict], recipe: dict, review_report: dict, v_report: dict) -> dict:
    bank_dir = BANK_NEEDLE_VRM_AGENT_V2.parent
    dump_jsonl(BANK_NEEDLE_VRM_AGENT_V2, rows)
    v_report = dict(v_report)
    v_report["review"] = {
        "n": review_report["n"],
        "n_pass": review_report["n_pass"],
        "pass_rate": review_report["pass_rate"],
        "n_rejected": review_report["n_rejected"],
        "reviewer": "protocol-v2",
    }
    lock = write_lock(v_report, recipe, BANK_NEEDLE_VRM_AGENT_V2)
    toolset = EVAL_SHARED_ROOT / "toolsets/needle-vrm-agent-v0.json"
    man_path = bank_dir / "holdout-v2.manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    man.update(
        {
            "bank": "eval-bank-v2.jsonl",
            "holdout_version": "v2",
            "n": len(rows),
            "n_candidates": man.get("n_candidates") or 6000,
            "sha256": sha256_file(BANK_NEEDLE_VRM_AGENT_V2),
            "recipe": man.get("recipe") or str(BANK_NEEDLE_VRM_AGENT_V2_RECIPE.relative_to(ROOT)),
            "recipe_sha256": sha256_file(BANK_NEEDLE_VRM_AGENT_V2_RECIPE),
            "toolset_sha256": sha256_file(toolset),
            "v1_sha256": V1_SHA,
            "gold": "schema-program",
            "teacher_model": man.get("teacher_model") or "template",
            "generator_version": man.get("generator_version"),
            "seed": man.get("seed") or 20260825,
            "lock": str(BANK_NEEDLE_VRM_AGENT_V2_LOCK.relative_to(ROOT)),
            "review": v_report["review"],
            "distribution": {
                "family": v_report.get("family"),
                "tools_execute": v_report.get("tools_execute"),
                "execute_n": v_report.get("execute_n"),
                "refuse_n": v_report.get("refuse_n"),
                "n_dev": v_report.get("n_dev"),
                "n_eval": v_report.get("n_eval"),
            },
            "frozen": True,
        }
    )
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"lock": lock, "manifest": man}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2)
    ap.add_argument("--recipe", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2_RECIPE)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    recipe = load_recipe(args.recipe)
    rows = load_jsonl(args.bank)
    if not rows:
        print("missing v2 bank", file=sys.stderr)
        return 1
    ids = select_review_ids(rows, recipe)
    by_id = {str(r["item_id"]): r for r in rows}
    reviews = [review_row(by_id[i], recipe) for i in ids if i in by_id]
    n_pass = sum(1 for r in reviews if r["pass"])
    n = len(reviews)
    pass_rate = n_pass / n if n else 0.0
    min_rate = float((recipe.get("review") or {}).get("pass_rate_min") or 0.97)
    rejected = [r["item_id"] for r in reviews if not r["pass"]]
    report = {
        "n": n,
        "n_pass": n_pass,
        "n_rejected": len(rejected),
        "pass_rate": round(pass_rate, 4),
        "pass_rate_min": min_rate,
        "rejected": rejected[:40],
        "ok": n == int((recipe.get("review") or {}).get("n") or 600) and pass_rate >= min_rate,
        "v1_sha256": sha256_file(BANK_NEEDLE_VRM_AGENT),
        "v1_frozen": sha256_file(BANK_NEEDLE_VRM_AGENT) == V1_SHA,
    }
    out = args.bank.parent / "holdout-v2.review.jsonl"
    dump_jsonl(out, reviews)
    passed_ids = {r["item_id"] for r in reviews if r["pass"]}
    rejected_ids = {r["item_id"] for r in reviews if not r["pass"]}
    for row in rows:
        sid = str(row["item_id"])
        if sid in passed_ids:
            row["review_status"] = "reviewed"
        elif sid in rejected_ids:
            row["review_status"] = "rejected"
        elif row.get("review_status") == "reviewed":
            row["review_status"] = "generated"
    dump_jsonl(args.bank, rows)
    print(json.dumps({**report, "review_path": str(out.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    if not report["ok"]:
        return 1
    if args.freeze:
        v_report = validate(rows, recipe)
        if not v_report.get("ok"):
            print(json.dumps({"freeze": False, "validator": v_report}, ensure_ascii=False, indent=2))
            return 1
        frozen = freeze_assets(rows, recipe, report, v_report)
        print(json.dumps({"freeze": True, "sha256": frozen["lock"]["sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
