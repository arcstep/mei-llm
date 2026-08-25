#!/usr/bin/env python3
"""Validate Needle VRM holdout v2: schema, program gold, quotas, isolation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    BANK_NEEDLE_VRM_AGENT_V2,
    BANK_NEEDLE_VRM_AGENT_V2_LOCK,
    BANK_NEEDLE_VRM_AGENT_V2_RECIPE,
    EVAL_BANKS_ROOT,
    EVAL_SHARED_ROOT,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import format_sft_user_text, token_near_dups  # noqa: E402
from eval_needle_toolcall_v0 import schema_errors  # noqa: E402
from needle_home_sft_lib import ASK_RE, EVAL_RE, PII_RE, load_jsonl, sha256_file  # noqa: E402
from needle_vrm_holdout_v2_lib import (  # noqa: E402
    catalog_ok,
    collect_blocked_queries,
    gold_from_intent_v2,
    intent_from_row,
    load_recipe,
    query_banned,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

V1_SHA = "d12d8030a490698799712f7daea4fae057f3ff81315baa44e809fa1b6ced7ffb"
REFUSE = {"missing", "scene_conflict", "illegal_pair", "offtopic"}
EXECUTE = {"gesture", "home", "order", "sequence"}


def _eq(a, b) -> bool:
    return json.dumps(a or [], ensure_ascii=False, sort_keys=True) == json.dumps(
        b or [], ensure_ascii=False, sort_keys=True
    )


def cell_counts(rows: list[dict]) -> dict[str, int]:
    cells: Counter[str] = Counter()
    for row in rows:
        for c in (row.get("gold") or {}).get("function_calls") or []:
            name = str(c.get("name") or "")
            args = c.get("arguments") or {}
            if name == "point":
                cells[f"point_{args.get('target')}"] += 1
            elif name == "go_to":
                cells[f"go_to_{args.get('place')}"] += 1
            elif name == "set_switch":
                on = "on" if args.get("on") else "off"
                lid = str(args.get("id") or "")
                room = "kitchen" if "kitchen" in lid else "living"
                cells[f"light_{room}_{on}"] += 1
            elif name == "open_door":
                cells[f"door_{args.get('door')}_open"] += 1
            elif name == "close_door":
                cells[f"door_{args.get('door')}_close"] += 1
            elif name == "order_food":
                if args.get("shop") == "兰州拉面":
                    cells["food_lanzhou"] += 1
                if args.get("shop") == "麦当劳":
                    cells["food_mcd"] += 1
    return dict(cells)


def recipe_required_tools() -> list[str]:
    path = EVAL_SHARED_ROOT / "toolsets/needle-vrm-agent-v0.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(t["name"]) for t in data.get("tools") or [] if t.get("name")]


def validate(rows: list[dict], recipe: dict) -> dict:
    errors: list[str] = []
    errors.extend(schema_errors(rows))
    if len(rows) != int(recipe["n_total"]):
        errors.append(f"n={len(rows)} want {recipe['n_total']}")
    n_dev = sum(1 for r in rows if r.get("split") == "dev")
    n_eval = sum(1 for r in rows if r.get("split") == "eval")
    if n_dev != int(recipe["n_dev"]) or n_eval != int(recipe["n_eval"]):
        errors.append(f"split dev/eval={n_dev}/{n_eval}")
    fam = Counter(str(r.get("family")) for r in rows)
    for fam_name, want in recipe["family_n"].items():
        if fam.get(fam_name, 0) != int(want):
            errors.append(f"family {fam_name}={fam.get(fam_name, 0)} want {want}")
    tools_hit: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_qs: set[str] = set()
    tpl = Counter()
    n_exec = 0
    groups: dict[str, set[str]] = {}
    tok = ZhTokenizerV1()
    seq_len = int(recipe.get("seq_len_eval") or 256)
    for row in rows:
        sid = str(row.get("item_id") or "")
        if not sid.startswith(str(recipe["item_id_prefix"])):
            errors.append(f"{sid}: bad item_id prefix")
        if sid in seen_ids:
            errors.append(f"{sid}: duplicate id")
        seen_ids.add(sid)
        if row.get("holdout_version") != "v2":
            errors.append(f"{sid}: holdout_version")
        q = str(row.get("query") or "")
        banned = query_banned(q)
        if banned:
            errors.append(f"{sid}: query banned ({banned})")
        if ASK_RE.search(q) or EVAL_RE.search(q) or PII_RE.search(q):
            errors.append(f"{sid}: ASK/EVAL/PII in query")
        scene = str(row.get("scene") or "")
        if ASK_RE.search(scene) or EVAL_RE.search(scene) or PII_RE.search(scene):
            errors.append(f"{sid}: ASK/EVAL/PII in scene")
        sk = q.strip() + "\n" + scene.strip()
        if sk in seen_qs:
            errors.append(f"{sid}: duplicate query+scene")
        seen_qs.add(sk)
        family = str(row.get("family") or "")
        if family == "scene_conflict" and not scene.strip():
            errors.append(f"{sid}: scene_conflict missing scene")
        user = format_sft_user_text(row)
        if scene.strip() and not user.startswith("场景："):
            errors.append(f"{sid}: scene not encoded")
        if len(tok.encode(user)) > seq_len:
            errors.append(f"{sid}: token length > {seq_len}")
        intent = intent_from_row(row)
        recomputed = gold_from_intent_v2(intent, recipe)
        gold = (row.get("gold") or {}).get("function_calls")
        if not _eq(recomputed, gold):
            errors.append(f"{sid}: gold != schema-program")
        if gold and not catalog_ok(gold):
            errors.append(f"{sid}: gold catalog/enum fail")
        calls = gold or []
        if family in REFUSE and calls:
            errors.append(f"{sid}: refuse family must be []")
        if family in EXECUTE and not calls:
            errors.append(f"{sid}: execute family must have calls")
        if family == "sequence" and len(calls) < 2:
            errors.append(f"{sid}: sequence needs >=2 calls")
        for c in calls:
            if c.get("name"):
                tools_hit[str(c["name"])] += 1
        if calls:
            n_exec += 1
        tpl[str(row.get("template_id") or "")] += 1
        groups.setdefault(str(row.get("intent_id") or sid), set()).add(str(row.get("split")))
    if n_exec != int(recipe["targets"]["execute_n"]):
        errors.append(f"execute_n={n_exec} want {recipe['targets']['execute_n']}")
    refuse_n = len(rows) - n_exec
    if refuse_n != int(recipe["targets"]["refuse_n"]):
        errors.append(f"refuse_n={refuse_n} want {recipe['targets']['refuse_n']}")
    min_tool = int(recipe["min_execute_per_tool"])
    missing_tools = [t for t in recipe_required_tools() if tools_hit.get(t, 0) < min_tool]
    if missing_tools:
        errors.append(f"tools below min {min_tool}: {missing_tools}")
    cells = cell_counts(rows)
    for key, want in (recipe.get("min_cells") or {}).items():
        if cells.get(key, 0) < int(want):
            errors.append(f"cell {key}={cells.get(key, 0)} want >={want}")
    cap = float(recipe.get("max_template_share") or 0.03)
    if tpl:
        top, c = tpl.most_common(1)[0]
        if top and c / max(len(rows), 1) > cap + 1e-9:
            errors.append(f"template {top} share {c / len(rows):.3f} > {cap}")
    for key, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"intent {key} crosses split {splits}")
            break

    v1 = load_jsonl(BANK_NEEDLE_VRM_AGENT)
    v1_q = {str(r.get("query") or "").strip() for r in v1}
    exact_v1 = [str(r.get("item_id")) for r in rows if str(r.get("query") or "").strip() in v1_q]
    if exact_v1:
        errors.append(f"exact v1 query overlap n={len(exact_v1)}")
    v2_ids = [(str(r.get("item_id")), tok.encode(str(r.get("query") or ""))) for r in rows]
    v1_ids = [(str(r.get("item_id")), tok.encode(str(r.get("query") or ""))) for r in v1]
    hits_v1 = token_near_dups(v2_ids, v1_ids, threshold=0.9, limit=8)
    if hits_v1:
        errors.append(f"v2/v1 token Jaccard>=0.9 hits={len(hits_v1)}")
    dev_fmt = [
        (str(r["item_id"]), tok.encode(format_sft_user_text(r)))
        for r in rows
        if r.get("split") == "dev"
    ]
    eval_fmt = [
        (str(r["item_id"]), tok.encode(format_sft_user_text(r)))
        for r in rows
        if r.get("split") == "eval"
    ]
    cross = token_near_dups(dev_fmt, eval_fmt, threshold=0.9, limit=8)
    if cross:
        errors.append(f"dev/eval token Jaccard>=0.9 hits={len(cross)}")
    blocked = collect_blocked_queries()
    train_q = blocked - v1_q
    train_exact = [str(r.get("item_id")) for r in rows if str(r.get("query") or "").strip() in train_q]
    if train_exact:
        errors.append(f"exact train query overlap n={len(train_exact)}")
    train_ids = [(q[:24], tok.encode(q)) for q in train_q if q]
    train_hits = token_near_dups(v2_ids, train_ids, threshold=0.9, limit=8)
    if train_hits:
        errors.append(f"v2/train token Jaccard>=0.9 hits={len(train_hits)}")
    v1_lock = json.loads((EVAL_BANKS_ROOT / "needle-vrm-agent-v0/holdout-v1.lock.json").read_text(encoding="utf-8"))
    v1_ok = v1_lock.get("sha256") == V1_SHA == sha256_file(BANK_NEEDLE_VRM_AGENT)
    if not v1_ok:
        errors.append("v1 lock/bank mutated")
    return {
        "ok": not errors,
        "n": len(rows),
        "n_dev": n_dev,
        "n_eval": n_eval,
        "execute_n": n_exec,
        "refuse_n": refuse_n,
        "family": dict(fam),
        "tools_execute": dict(tools_hit),
        "cells": cells,
        "v1_sha256": v1_lock.get("sha256"),
        "v1_frozen": v1_ok,
        "errors": errors[:80],
        "n_errors": len(errors),
    }


def write_lock(report: dict, recipe: dict, bank: Path) -> dict:
    toolset = EVAL_SHARED_ROOT / "toolsets/needle-vrm-agent-v0.json"
    lock = {
        "bank": "needle-vrm-agent-v0",
        "holdout_version": "v2",
        "n_total": report["n"],
        "n_dev": report["n_dev"],
        "n_eval": report["n_eval"],
        "sha256": sha256_file(bank),
        "recipe_sha256": sha256_file(BANK_NEEDLE_VRM_AGENT_V2_RECIPE),
        "toolset_sha256": sha256_file(toolset),
        "v1_sha256": V1_SHA,
        "distribution": {
            "family": report.get("family"),
            "tools_execute": report.get("tools_execute"),
            "execute_n": report.get("execute_n"),
            "refuse_n": report.get("refuse_n"),
        },
        "review": report.get("review") or {},
    }
    BANK_NEEDLE_VRM_AGENT_V2_LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2)
    ap.add_argument("--recipe", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2_RECIPE)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if not args.bank.is_file():
        print(f"missing {args.bank}", file=sys.stderr)
        return 1
    recipe = load_recipe(args.recipe)
    rows = load_jsonl(args.bank)
    report = validate(rows, recipe)
    report["bank"] = str(args.bank.relative_to(ROOT)) if str(args.bank).startswith(str(ROOT)) else str(args.bank)
    report["sha256"] = sha256_file(args.bank)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.freeze:
        if not report["ok"]:
            print("refuse --freeze: validator not ok", file=sys.stderr)
            return 1
        lock = write_lock(report, recipe, args.bank)
        print(json.dumps({"wrote_lock": str(BANK_NEEDLE_VRM_AGENT_V2_LOCK.relative_to(ROOT)), **lock}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
