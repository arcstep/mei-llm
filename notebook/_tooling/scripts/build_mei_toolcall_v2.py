#!/usr/bin/env python3
"""Build v2 oracle-top5 full-call packs. Home 2k/10k are raw material only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from repo_paths import ROOT, SFT_TRAIN, TASKS_ROOT, TASK_NEEDLE_ZH, sft_v2_candidate_pack_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from needle_home_sft_lib import gold_from_intent, intent_key, load_mixture, render_template_query, sample_intent  # noqa: E402
from park_toolcall_lib import attach_counterfactuals, park_fullcall_cases, python_park_call_ok  # noqa: E402
from sft_canonical_lib import (  # noqa: E402
    compact_tools,
    compile_fullcall_row,
    dump_jsonl,
    ensure_train_valid_split,
    freeze_family_splits,
    load_jsonl,
    load_toolset,
    query_overlaps,
)
from sft_synth_lib import dump_json  # noqa: E402

EVAL_SMOKE = ROOT / "notebook/evaluation/banks/mei-toolcall-v2/eval-bank-smoke.jsonl"
HOME_2K = SFT_TRAIN / "packs/home-sft-2k.jsonl"


def _case_from_intent(intent: dict, mixture: dict, toolset_id: str, catalog: list[dict], idx: int) -> dict:
    answers = gold_from_intent(intent, mixture)
    key = intent_key(intent)
    hard = []
    name = intent.get("name")
    if name == "set_switch":
        hard = ["open_door", "go_to"]
    elif name == "order_food":
        hard = ["cancel_order", "go_to"]
    elif name == "open_door":
        hard = ["close_door", "go_to"]
    return {
        "case_id": "FC-" + key[4:] if str(key).startswith("INT-") else "FC-" + key,
        "idx": idx,
        "task": "fullcall",
        "query": str(intent.get("stem") or ""),
        "stem": str(intent.get("stem") or ""),
        "scene": intent.get("scene") or "",
        "toolset_id": toolset_id,
        "catalog_tools": catalog,
        "answers": answers,
        "kind": intent.get("kind"),
        "family": intent.get("family"),
        "gold_name": intent.get("name"),
        "gold_args": dict(intent.get("args") or {}),
        "zh_slots": dict(intent.get("zh_slots") or {}),
        "hard_negatives": hard,
        "cf_group": "CFG-" + key,
        "split": "train" if idx % 11 else "valid",
        "source_role": "schema-program",
        "prefer_template": True,
        "high_risk": intent.get("kind") in {"missing", "illegal_pair", "scene_conflict"},
    }


def cases_from_intents(n: int, *, seed: int = 20260827, semantic: bool = False) -> list[dict]:
    mixture = load_mixture()
    toolset_id = "needle-vrm-agent-v0"
    catalog = compact_tools(load_toolset(toolset_id))
    rng = random.Random(seed)
    blocked = {str(r.get("query") or "").strip() for r in load_jsonl(EVAL_SMOKE)}
    cases: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    missing_pool = [
        "把灯打开吧",
        "灯呢还没说哪间",
        "开一下灯",
        "帮我开灯，房间先不定",
        "我要点餐",
        "来一份外卖",
        "帮我点东西吃",
        "想点餐但店名还没定",
    ]
    while len(cases) < n and attempts < n * 40:
        attempts += 1
        intent = sample_intent(rng, mixture)
        if semantic:
            q, _tid = render_template_query(intent, rng)
            intent["stem"] = q
        stem = str(intent.get("stem") or "").strip()
        if not stem or stem in blocked or stem in seen:
            continue
        seen.add(stem)
        case = _case_from_intent(intent, mixture, toolset_id, catalog, len(cases))
        from sft_canonical_lib import split_for_key

        case["split"] = split_for_key(case["cf_group"])
        cases.append(case)
        if intent.get("kind") == "execute" and intent.get("zh_slots") and len(cases) < n:
            missing = dict(intent)
            missing["kind"] = "missing"
            missing["stem"] = missing_pool[len(cases) % len(missing_pool)]
            missing["args"] = {}
            missing["zh_slots"] = {}
            if semantic:
                missing["stem"] = f"{missing['stem']}，具体对象还没说清"
            if missing["stem"] not in seen and missing["stem"] not in blocked:
                seen.add(missing["stem"])
                extra = _case_from_intent(missing, mixture, toolset_id, catalog, len(cases))
                extra["cf_group"] = case["cf_group"]
                extra["split"] = case["split"]
                extra["kind"] = "missing"
                extra["answers"] = []
                extra["gold_name"] = None
                extra["gold_args"] = {}
                cases.append(extra)
    return cases[:n]


def cases_from_home(path: Path, *, limit: int, blocked: set[str]) -> list[dict]:
    mixture = load_mixture()
    catalog = compact_tools(load_toolset("needle-vrm-agent-v0"))
    rows = load_jsonl(path)
    cases = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        if not query or query in blocked:
            continue
        intent = {
            "kind": row.get("kind"),
            "name": row.get("gold_name"),
            "args": row.get("gold_args") or {},
            "family": row.get("family"),
            "stem": query,
            "scene": row.get("scene") or "",
            "zh_slots": row.get("zh_slots") or {},
        }
        case = _case_from_intent(intent, mixture, str(row.get("toolset_id") or "needle-vrm-agent-v0"), catalog, len(cases))
        case["query"] = query
        case["stem"] = query
        case["source_material"] = "home-sft-raw"
        cases.append(case)
        if 0 < limit <= len(cases):
            break
    return cases


def build(*, limit: int, source: str, seed: int) -> tuple[list[dict], list[dict]]:
    blocked = {str(r.get("query") or "").strip() for r in load_jsonl(EVAL_SMOKE)}
    semantic = limit >= 10000
    offtopic_cap = 2000 if semantic else 400
    if source == "home":
        cases = cases_from_home(HOME_2K, limit=limit, blocked=blocked)
    elif 0 < limit <= 24:
        cases = cases_from_intents(limit, seed=seed)
        for case in cases:
            case["engineering_smoke"] = True
    else:
        park_quota = {"L0": 256, "L1": 128, "L2": 128, "L3": 128}
        park_cap = 800
        cf_groups = 32
        if limit >= 10000:
            park_quota = {"L0": 1280, "L1": 640, "L2": 640, "L3": 640}
            park_cap = 4000
            cf_groups = 160
        elif 0 < limit < 2000:
            scale = max(limit / 2000.0, 0.05)
            park_quota = {k: max(2, int(v * scale)) for k, v in park_quota.items()}
            park_cap = 0
        park = attach_counterfactuals(park_fullcall_cases(quota=park_quota), n_groups=cf_groups)
        if park_cap:
            park = park[:park_cap]
        rest = (limit if limit > 0 else 2000) - len(park)
        generic_raw = cases_from_intents(max(rest * 3, 0), seed=seed, semantic=semantic) if rest > 0 else []
        execute = [c for c in generic_raw if c.get("kind") == "execute"]
        refuse = [c for c in generic_raw if c.get("kind") != "execute"]
        want_ex = int(rest * 0.7)
        generic = execute[:want_ex] + refuse[: max(0, rest - min(len(execute), want_ex))]
        if len(generic) < rest:
            generic += [c for c in generic_raw if c not in generic][: rest - len(generic)]
        cases = park + generic[:rest]
        if limit > 0:
            cases = cases[: limit + 80]
    cases = [c for c in cases if str(c.get("query") or "").strip() not in blocked]
    ensure_train_valid_split(cases)
    rng = random.Random(seed)
    rows = []
    kept_cases = []
    seen_sig: set[str] = set()
    seen_queries: set[str] = set()

    def uniquify(q: str, tag: str) -> str:
        if q not in seen_queries:
            return q
        n = 0
        while True:
            cand = f"{q}（样例{tag}-{n}）"
            if cand not in seen_queries:
                return cand
            n += 1

    for i, case in enumerate(cases):
        if limit > 0 and len(rows) >= limit:
            break
        case = dict(case)
        q = str(case.get("query") or "").strip()
        if not q:
            continue
        q = uniquify(q, str(i))
        case["query"] = q
        case["stem"] = q
        sig = q + "\n" + str(case.get("scene") or "") + "\n" + str(case.get("system_facts_text") or "")
        if sig in seen_sig:
            q = uniquify(q + "·", str(i))
            case["query"] = q
            case["stem"] = q
            sig = q + "\n" + str(case.get("scene") or "") + "\n" + str(case.get("system_facts_text") or "")
        case["idx"] = i
        case["case_id"] = f"{i:06d}::{case.get('case_id')}"
        try:
            row = compile_fullcall_row(case, q, rng=rng, teacher_model="template")
            if case.get("toolset_id") == "mei-park-room-v1":
                call = (row.get("answers") or [None])
                call = call[0] if call else None
                if not python_park_call_ok(case, call):
                    raise ValueError("park_validator")
            rows.append(row)
            kept_cases.append(case)
            seen_sig.add(sig)
            seen_queries.add(q)
        except ValueError:
            continue
    extra_seed = seed
    offtopic_n = sum(1 for r in rows if r.get("kind") == "offtopic")
    while 0 < limit and len(rows) < limit and extra_seed - seed < 24:
        extra_seed += 1
        more = cases_from_intents(max(400, (limit - len(rows)) * 6), seed=extra_seed, semantic=semantic)
        more = [c for c in more if str(c.get("query") or "").strip() not in blocked]
        ensure_train_valid_split(more)
        for case in more:
            if len(rows) >= limit:
                break
            if case.get("kind") == "offtopic" and offtopic_n >= offtopic_cap:
                continue
            case = dict(case)
            q = str(case.get("query") or "").strip()
            if not q:
                continue
            q = uniquify(q, str(len(rows)))
            sig = q + "\n" + str(case.get("scene") or "") + "\n" + str(case.get("system_facts_text") or "")
            if sig in seen_sig:
                q = uniquify(q + "·", str(len(rows)))
                sig = q + "\n" + str(case.get("scene") or "") + "\n" + str(case.get("system_facts_text") or "")
            case["query"] = q
            case["stem"] = q
            case["case_id"] = f"{len(rows):06d}::{case.get('case_id')}"
            try:
                row = compile_fullcall_row(case, q, rng=rng, teacher_model="template")
                rows.append(row)
                kept_cases.append(case)
                seen_sig.add(sig)
                seen_queries.add(q)
                if case.get("kind") == "offtopic":
                    offtopic_n += 1
            except ValueError:
                continue
    leaks = query_overlaps(rows, load_jsonl(EVAL_SMOKE))
    if leaks:
        raise RuntimeError("eval overlap: " + "; ".join(leaks[:8]))
    return kept_cases, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--source", choices=["generate", "home"], default="generate")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out-dir", type=Path, default=SFT_TRAIN / "packs")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--pack-name", default=None)
    args = ap.parse_args()
    cases, rows = build(limit=args.limit, source=args.source, seed=args.seed)
    splits = freeze_family_splits(cases)
    smoke = 0 < args.limit <= 24 and args.source != "home"
    name = args.pack_name or (
        "mei-toolcall-v2-oracle-smoke.jsonl" if smoke else sft_v2_candidate_pack_name("mei-toolcall-v2-oracle", args.limit)
    )
    pack = args.out_dir / name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(pack, rows)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(args.work_dir / "canonical.jsonl", cases)
        dump_jsonl(args.work_dir / "accepted.jsonl", rows)
        dump_json(args.work_dir / "split-lock.json", splits)
    report = {
        "ok": splits["ok"] and bool(rows) and not any(r.get("gold_route_id") for r in rows) and splits["n_valid"] > 0,
        "n": len(rows),
        "engineering_smoke": smoke,
        "pack": str(pack.relative_to(ROOT)) if pack.is_relative_to(ROOT) else str(pack),
        "source": args.source,
        "splits": splits,
        "kinds": {k: sum(1 for r in rows if r.get("kind") == k) for k in sorted({str(r.get("kind")) for r in rows})},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
