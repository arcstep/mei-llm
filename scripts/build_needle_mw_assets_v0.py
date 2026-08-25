#!/usr/bin/env python3
"""Build independent MW SFT 2K + MW holdout 1K. Never writes v1/v2 or home-sft packs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_MW,
    BANK_NEEDLE_VRM_MW_LOCK,
    BANK_NEEDLE_VRM_MW_RECIPE,
    EXPERIMENTS_RUNS,
    PACK_NEEDLE_MW_SFT_2K,
    RECIPE_NEEDLE_MW_SFT,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import format_sft_user_text, token_jaccard  # noqa: E402
from needle_home_sft_lib import dump_jsonl, load_jsonl, sha256_file  # noqa: E402
from needle_mw_governance_lib import (  # noqa: E402
    assert_upstream_frozen,
    assign_styles,
    collect_blocked_queries,
    enumerate_canonical_cases,
    gold_from_case,
    group_key,
    leak_against,
    load_eval_recipe,
    load_sft_recipe,
    make_eval_item,
    make_sft_row,
    query_banned,
    query_contract_ok,
    render_query,
    scene_text,
)
from tokenizer import ZhTokenizerV1  # noqa: E402


def _encode_sets(queries: set[str], tok: ZhTokenizerV1) -> list[set[int]]:
    out = []
    for q in queries:
        ids = tok.encode(q)
        if len(ids) >= 4:
            out.append(set(ids))
    return out


def generate_candidates(
    cases: list[dict],
    recipe: dict,
    rng: random.Random,
    *,
    blocked: set[str],
    blocked_sets: list[set[int]],
    tok: ZhTokenizerV1,
    seen: set[str],
    seq_len: int,
    make_row,
) -> tuple[list[dict], list[dict]]:
    overgen = int(recipe.get("candidate_overgen") or 3)
    tags = list(recipe.get("style_tags") or ["command"])
    selected_pool: list[dict] = []
    candidates: list[dict] = []
    locked_style_reasons = {"injection_rejected", "negation_cancels"}
    keep_by_act = {
        "execute": max(overgen * 10, 30),
        "expand": max(overgen * 8, 24),
        "shape": max(overgen * 10, 30),
        "escalate": max(overgen * 12, 36),
        "stop": max(overgen * 6, 18),
    }
    for i, case in enumerate(cases, start=1):
        local_kept: list[dict] = []
        keep_n = keep_by_act.get(str(case.get("act")), overgen)
        for attempt in range(keep_n + 48):
            local = dict(case)
            if local.get("reason_code") not in locked_style_reasons and "correction" not in (
                local.get("case_tags") or []
            ):
                local["style_tags"] = [tags[(i + attempt) % len(tags)]]
            salt = i * 53 + attempt * 19 + int(recipe.get("seed") or 0)
            try:
                query, tid = render_query(local, rng, salt=salt)
            except KeyError:
                continue
            reason = query_banned(query)
            if reason:
                continue
            if query in blocked:
                continue
            if not query_contract_ok(local, query):
                continue
            ids = tok.encode(query)
            if leak_against(ids, blocked_sets):
                continue
            scene = scene_text(str(local["pool"]), str(local.get("scene_kind") or ""))
            key = query.strip() + "\n" + scene
            if key in seen:
                continue
            tmp = {"query": query, "scene": scene or None, "lang": "zh"}
            if len(tok.encode(format_sft_user_text(tmp))) > seq_len:
                continue
            gold = gold_from_case(local)
            row = make_row(local, query, tid, salt)
            row["_key"] = key
            row["_group"] = group_key(local)
            row["_act"] = gold["act"]
            row["_tid"] = tid
            row["_canonical"] = local["canonical_id"]
            seen.add(key)
            local_kept.append(row)
            candidates.append(row)
            if len(local_kept) >= keep_n:
                break
        selected_pool.extend(local_kept)
    return selected_pool, candidates


def _select_by_act(
    rows: list[dict],
    want: dict[str, int],
    rng: random.Random,
    *,
    min_cf_frac: float = 0.0,
    max_template_share: float = 0.05,
) -> list[dict]:
    by_act: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_act[str(row["_act"])].append(row)
    for act in by_act:
        rng.shuffle(by_act[act])

    chosen: list[dict] = []
    chosen_keys: set[str] = set()
    taken_groups: set[str] = set()
    total_target = sum(want.values())
    cf_target = int(round(min_cf_frac * total_target)) if min_cf_frac else 0

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["_group"])].append(row)

    def take_row(row: dict) -> bool:
        if row["_key"] in chosen_keys:
            return False
        if Counter(r["_tid"] for r in chosen).get(row["_tid"], 0) + 1 > max(1, int(max_template_share * total_target)):
            return False
        chosen.append(row)
        chosen_keys.add(row["_key"])
        taken_groups.add(str(row["_group"]))
        return True

    def act_count() -> Counter:
        return Counter(r["_act"] for r in chosen)

    # Prefer complete CF groups first (star groups have cf_group).
    cf_groups = [g for g, members in groups.items() if g.startswith("CF-") and len(members) >= 2]
    rng.shuffle(cf_groups)
    for g in cf_groups:
        members = groups[g]
        preview = act_count()
        overflow = False
        for m in members:
            if preview[m["_act"]] + 1 > want.get(m["_act"], 0) + 8:
                overflow = True
                break
            preview[m["_act"]] += 1
        if overflow:
            continue
        for m in members:
            take_row(m)
        if cf_target and sum(1 for r in chosen if str(r.get("cf_group") or "").startswith("CF-")) >= cf_target:
            if all(act_count()[a] >= int(want[a] * 0.5) for a in want):
                break

    for act, n in want.items():
        pool = [r for r in by_act.get(act, []) if r["_key"] not in chosen_keys]
        rng.shuffle(pool)
        for row in pool:
            if act_count()[act] >= n:
                break
            take_row(row)

    # Trim overflow per act, never splitting remaining CF groups when possible.
    keep: list[dict] = []
    counts: Counter[str] = Counter()
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in chosen:
        by_group[str(row["_group"])].append(row)
    ordered_groups = sorted(by_group.values(), key=lambda xs: (0 if str(xs[0]["_group"]).startswith("CF-") else 1, -len(xs)))
    for members in ordered_groups:
        preview = Counter(counts)
        ok = True
        for m in members:
            preview[m["_act"]] += 1
            if preview[m["_act"]] > want.get(m["_act"], 0):
                ok = False
                break
        if not ok:
            # take a subset that fits
            for m in members:
                if counts[m["_act"]] < want.get(m["_act"], 0):
                    keep.append(m)
                    counts[m["_act"]] += 1
            continue
        keep.extend(members)
        for m in members:
            counts[m["_act"]] += 1

    # Fill leftovers
    have = {r["_key"] for r in keep}
    for act, n in want.items():
        if counts[act] >= n:
            continue
        for row in by_act.get(act, []):
            if counts[act] >= n:
                break
            if row["_key"] in have:
                continue
            keep.append(row)
            have.add(row["_key"])
            counts[act] += 1

    # Exact trim
    final: list[dict] = []
    counts = Counter()
    for row in keep:
        act = row["_act"]
        if counts[act] >= want.get(act, 0):
            continue
        final.append(row)
        counts[act] += 1
    rng.shuffle(final)
    return final


def assign_sft_split(rows: list[dict], recipe: dict, rng: random.Random) -> None:
    valid_frac = float(recipe.get("valid_frac") or 0.09)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["_group"])].append(row)
    keys = list(groups)
    rng.shuffle(keys)
    want_valid = max(1, int(round(len(rows) * valid_frac)))
    valid_ids: set[str] = set()
    n_valid = 0
    for k in keys:
        members = groups[k]
        if n_valid + len(members) > want_valid * 1.35 and n_valid >= want_valid:
            continue
        if n_valid >= want_valid:
            break
        for row in members:
            valid_ids.add(row["_key"])
        n_valid += len(members)
    for row in rows:
        row["split"] = "valid" if row["_key"] in valid_ids else "train"


def assign_eval_split(rows: list[dict], recipe: dict, rng: random.Random) -> None:
    want_n = int(recipe.get("n_dev") or 100)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["_group"])].append(row)
    items = list(groups.values())
    rng.shuffle(items)
    # Subset-sum to hit n_dev exactly without splitting groups.
    dp: dict[int, list[int]] = {0: []}
    for i, members in enumerate(items):
        sz = len(members)
        updates: dict[int, list[int]] = {}
        for total, idxs in dp.items():
            nxt = total + sz
            if nxt > want_n or nxt in dp or nxt in updates:
                continue
            updates[nxt] = idxs + [i]
        dp.update(updates)
        if want_n in dp:
            break
    if want_n in dp:
        chosen_idx = set(dp[want_n])
    else:
        best = max(dp)
        chosen_idx = set(dp[best])
        remain = want_n - best
        for i, members in enumerate(items):
            if i in chosen_idx:
                continue
            if len(members) == remain:
                chosen_idx.add(i)
                remain = 0
                break
        if remain:
            # Prefer leftover size-1 rows created by capping; otherwise leave short
            # and the caller will fail validation rather than split a group.
            for i, members in enumerate(items):
                if remain <= 0:
                    break
                if i in chosen_idx or len(members) != 1:
                    continue
                chosen_idx.add(i)
                remain -= 1
    chosen_keys = {m["_key"] for i, members in enumerate(items) if i in chosen_idx for m in members}
    for row in rows:
        row["split"] = "dev" if row["_key"] in chosen_keys else "eval"


def _strip_private(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def build_sft(recipe: dict, rng: random.Random, tok: ZhTokenizerV1, blocked: set[str], blocked_sets: list[set[int]], seen: set[str]) -> tuple[list[dict], list[dict]]:
    cases = enumerate_canonical_cases("sft")
    assign_styles(cases, recipe, rng)
    index_box = {"n": 0}

    def make_row(case, query, tid, salt):
        index_box["n"] += 1
        return make_sft_row(case, query, index=index_box["n"], template_id=tid)

    pool, cands = generate_candidates(
        cases,
        recipe,
        rng,
        blocked=blocked,
        blocked_sets=blocked_sets,
        tok=tok,
        seen=seen,
        seq_len=int(recipe.get("seq_len_train") or 256),
        make_row=make_row,
    )
    selected = _select_by_act(
        pool,
        dict(recipe["act_n"]),
        rng,
        min_cf_frac=0.25,
        max_template_share=float(recipe.get("max_template_share") or 0.04),
    )
    if len(selected) != int(recipe["n_total"]):
        # second pass from leftover candidates
        have = {r["_key"] for r in selected}
        extra = [r for r in cands if r["_key"] not in have]
        selected = _select_by_act(
            selected + extra,
            dict(recipe["act_n"]),
            rng,
            min_cf_frac=0.2,
            max_template_share=float(recipe.get("max_template_share") or 0.05),
        )
    assign_sft_split(selected, recipe, rng)
    selected.sort(key=lambda r: (0 if r["split"] == "train" else 1, r["_act"], r["_canonical"], r["query"]))
    for i, row in enumerate(selected, start=1):
        row["sample_id"] = f"TRAIN-MW-{i:06d}"
    return selected, cands


def build_eval(recipe: dict, rng: random.Random, tok: ZhTokenizerV1, blocked: set[str], blocked_sets: list[set[int]], seen: set[str]) -> tuple[list[dict], list[dict]]:
    cases = enumerate_canonical_cases("eval")
    assign_styles(cases, recipe, rng)
    index_box = {"n": 0}

    def make_row(case, query, tid, salt):
        index_box["n"] += 1
        return make_eval_item(case, query, index=index_box["n"], template_id=tid)

    pool, cands = generate_candidates(
        cases,
        recipe,
        rng,
        blocked=blocked,
        blocked_sets=blocked_sets,
        tok=tok,
        seen=seen,
        seq_len=int(recipe.get("seq_len_eval") or 256),
        make_row=make_row,
    )
    selected = _select_by_act(
        pool,
        dict(recipe["act_n"]),
        rng,
        min_cf_frac=float(recipe.get("min_counterfactual_frac") or 0.4),
        max_template_share=float(recipe.get("max_template_share") or 0.05),
    )
    if len(selected) != int(recipe["n_total"]):
        have = {r["_key"] for r in selected}
        extra = [r for r in cands if r["_key"] not in have]
        selected = _select_by_act(
            selected + extra,
            dict(recipe["act_n"]),
            rng,
            min_cf_frac=float(recipe.get("min_counterfactual_frac") or 0.4),
            max_template_share=0.08,
        )
    assign_eval_split(selected, recipe, rng)
    selected.sort(key=lambda r: (0 if r["split"] == "dev" else 1, r["_act"], r["_canonical"], r["query"]))
    for i, row in enumerate(selected, start=1):
        row["item_id"] = f"{recipe['item_id_prefix']}{i:04d}"
    return selected, cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sft", "eval", "both"], default="both")
    args = ap.parse_args()
    frozen = assert_upstream_frozen()
    tok = ZhTokenizerV1()
    blocked = collect_blocked_queries(exclude={PACK_NEEDLE_MW_SFT_2K, BANK_NEEDLE_VRM_MW} if args.only == "both" else (
        {PACK_NEEDLE_MW_SFT_2K} if args.only == "sft" else {BANK_NEEDLE_VRM_MW}
    ))
    blocked_sets = _encode_sets(blocked, tok)
    seen: set[str] = set()
    run_dir = EXPERIMENTS_RUNS / "needle-mw-v0-build"
    run_dir.mkdir(parents=True, exist_ok=True)

    sft_rows: list[dict] = []
    eval_rows: list[dict] = []
    if args.only in {"sft", "both"}:
        sft_recipe = load_sft_recipe()
        rng = random.Random(int(sft_recipe["seed"]))
        sft_rows, sft_cands = build_sft(sft_recipe, rng, tok, blocked, blocked_sets, seen)
        out = [_strip_private(r) for r in sft_rows]
        dump_jsonl(PACK_NEEDLE_MW_SFT_2K, out)
        dump_jsonl(run_dir / "mw-sft-v0.candidates.jsonl", [_strip_private(r) for r in sft_cands])
        for r in sft_rows:
            blocked.add(r["query"])
            seen.add(r["_key"])
        print(json.dumps({"sft_n": len(out), "sft_candidates": len(sft_cands), "acts": dict(Counter(r["act"] for r in out))}, ensure_ascii=False))

    if args.only in {"eval", "both"}:
        eval_recipe = load_eval_recipe()
        rng = random.Random(int(eval_recipe["seed"]))
        # refresh blocked token sets with new SFT queries
        blocked_sets = _encode_sets(blocked, tok)
        eval_rows, eval_cands = build_eval(eval_recipe, rng, tok, blocked, blocked_sets, seen)
        out = [_strip_private(r) for r in eval_rows]
        dump_jsonl(BANK_NEEDLE_VRM_MW, out)
        dump_jsonl(run_dir / "mw-eval-v0.candidates.jsonl", [_strip_private(r) for r in eval_cands])
        print(json.dumps({"eval_n": len(out), "eval_candidates": len(eval_cands), "acts": dict(Counter(r["act"] for r in out)), "splits": dict(Counter(r["split"] for r in out))}, ensure_ascii=False))

    print(json.dumps({"frozen_upstream": frozen, "ok_write": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
