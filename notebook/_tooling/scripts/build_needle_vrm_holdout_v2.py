#!/usr/bin/env python3
"""Build Needle VRM holdout v2 (2000 items). Never writes v1 bank or lock."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    BANK_NEEDLE_VRM_AGENT_V2,
    BANK_NEEDLE_VRM_AGENT_V2_RECIPE,
    EVAL_BANKS_ROOT,
    EVAL_SHARED_ROOT,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import format_sft_user_text, token_jaccard  # noqa: E402
from needle_home_sft_lib import dump_jsonl, load_jsonl, sha256_file  # noqa: E402
from needle_vrm_holdout_v2_lib import (  # noqa: E402
    GENERATOR_VERSION,
    assign_dev_split,
    catalog_ok,
    collect_blocked_queries,
    enumerate_canonical_intents,
    gold_from_intent_v2,
    intent_from_row,
    load_recipe,
    make_eval_item,
    protected_slots_ok,
    query_banned,
    render_eval_query,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

V1_SHA = "d12d8030a490698799712f7daea4fae057f3ff81315baa44e809fa1b6ced7ffb"


def assert_v1_frozen() -> dict:
    lock = EVAL_BANKS_ROOT / "needle-vrm-agent-v0" / "holdout-v1.lock.json"
    freeze = json.loads(lock.read_text(encoding="utf-8"))
    got = sha256_file(BANK_NEEDLE_VRM_AGENT)
    if freeze.get("sha256") != V1_SHA or got != V1_SHA:
        raise SystemExit(
            json.dumps(
                {"error": "v1 lock/bank hash mismatch; refuse to build v2", "lock": freeze, "bank": got},
                ensure_ascii=False,
            )
        )
    return freeze


def leak_against(ids: list[int], blocked_sets: list[set[int]]) -> bool:
    n = len(ids)
    if n < 4:
        return False
    sa = set(ids)
    for other in blocked_sets:
        if len(other) < 4:
            continue
        inter = len(sa & other)
        if inter / (n + len(other) - inter) >= 0.9:
            return True
    return False


def try_render(
    intent: dict,
    rng: random.Random,
    *,
    salt: int,
    seen: set[str],
    blocked: set[str],
    blocked_sets: list[set[int]],
    tok: ZhTokenizerV1,
    recipe: dict,
    seq_len: int,
) -> dict | None:
    query, tid = render_eval_query(intent, rng, salt=salt)
    if query_banned(query):
        return None
    if query in blocked:
        return None
    ids = tok.encode(query)
    if leak_against(ids, blocked_sets):
        return None
    if not protected_slots_ok(intent, query):
        return None
    gold = gold_from_intent_v2(intent, recipe)
    if not catalog_ok(gold):
        return None
    key = query.strip() + "\n" + str(intent.get("scene") or "").strip()
    if key in seen:
        return None
    tmp = {"query": query, "scene": intent.get("scene")}
    n_tok = len(tok.encode(format_sft_user_text(tmp)))
    if n_tok > seq_len:
        return None
    return {"query": query, "template_id": tid, "key": key}


def build(recipe: dict, *, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    tok = ZhTokenizerV1()
    blocked = collect_blocked_queries()
    blocked_sets = [set(tok.encode(q)) for q in blocked if q]
    intents = enumerate_canonical_intents(recipe)
    overgen = int(recipe.get("candidate_overgen") or 3)
    seq_len = int(recipe.get("seq_len_eval") or 256)
    selected: list[dict] = []
    candidates: list[dict] = []
    seen: set[str] = set()
    for intent_i, intent in enumerate(intents, start=1):
        local = []
        kept = None
        last_reasons: list[str] = []
        for attempt in range(overgen + 24):
            salt = intent_i * 31 + attempt * 17 + seed
            query, tid = render_eval_query(intent, rng, salt=salt)
            reason = query_banned(query)
            if reason:
                last_reasons.append(reason)
                continue
            if query in blocked:
                last_reasons.append("blocked_exact")
                continue
            ids = tok.encode(query)
            if leak_against(ids, blocked_sets):
                last_reasons.append("jaccard_train")
                continue
            if not protected_slots_ok(intent, query):
                last_reasons.append("slots")
                continue
            gold = gold_from_intent_v2(intent, recipe)
            if not catalog_ok(gold):
                last_reasons.append("catalog")
                continue
            key = query.strip() + "\n" + str(intent.get("scene") or "").strip()
            if key in seen:
                last_reasons.append("dup")
                continue
            tmp = {"query": query, "scene": intent.get("scene")}
            n_tok = len(tok.encode(format_sft_user_text(tmp)))
            if n_tok > seq_len:
                last_reasons.append("len")
                continue
            hit = {"query": query, "template_id": tid, "key": key}
            row = make_eval_item(
                intent,
                hit["query"],
                recipe=recipe,
                index=intent_i,
                template_id=hit["template_id"],
                teacher_model="template",
            )
            local.append(row)
            seen.add(hit["key"])
            if kept is None:
                kept = row
            if len(local) >= overgen:
                break
        if kept is None:
            raise RuntimeError(
                f"no unique query for family={intent['family']} stem={intent.get('stem')} reasons={last_reasons[:12]}"
            )
        selected.append(kept)
        candidates.extend(local)
    assign_dev_split(selected, recipe, random.Random(seed + 11))
    repair_cross_split_leaks(selected, recipe, rng, tok, seen, blocked, blocked_sets)
    return selected, candidates


def repair_cross_split_leaks(
    rows: list[dict],
    recipe: dict,
    rng: random.Random,
    tok: ZhTokenizerV1,
    seen: set[str],
    blocked: set[str],
    blocked_sets: list[set[int]],
) -> None:
    seq_len = int(recipe.get("seq_len_eval") or 256)
    for round_i in range(6):
        encoded = [(r, tok.encode(str(r.get("query") or ""))) for r in rows]
        pairs: list[tuple[dict, dict]] = []
        for i, (left, a) in enumerate(encoded):
            if len(a) < 4:
                continue
            for right, b in encoded[i + 1 :]:
                if left.get("split") == right.get("split"):
                    continue
                if len(b) < 4:
                    continue
                if token_jaccard(a, b) >= 0.9:
                    pairs.append((left, right))
                    if len(pairs) >= 40:
                        break
            if len(pairs) >= 40:
                break
        if not pairs:
            return
        for left, right in pairs:
            target = right if right.get("split") == "eval" else left
            intent = intent_from_row(target)
            old_key = str(target.get("query") or "").strip() + "\n" + str(target.get("scene") or "").strip()
            seen.discard(old_key)
            replaced = False
            for attempt in range(40):
                salt = 10_000 + round_i * 1000 + attempt * 13 + hash(target["item_id"]) % 997
                hit = try_render(
                    intent,
                    rng,
                    salt=salt,
                    seen=seen,
                    blocked=blocked,
                    blocked_sets=blocked_sets,
                    tok=tok,
                    recipe=recipe,
                    seq_len=seq_len,
                )
                if hit is None:
                    continue
                target["query"] = hit["query"]
                target["template_id"] = hit["template_id"]
                seen.add(hit["key"])
                replaced = True
                break
            if not replaced:
                seen.add(old_key)


def distribution(rows: list[dict]) -> dict:
    fam = Counter(str(r.get("family")) for r in rows)
    tools = Counter()
    for r in rows:
        for c in (r.get("gold") or {}).get("function_calls") or []:
            if c.get("name"):
                tools[str(c["name"])] += 1
    n_exec = sum(1 for r in rows if (r.get("gold") or {}).get("function_calls"))
    return {
        "family": dict(fam),
        "tools_execute": dict(tools),
        "n_dev": sum(1 for r in rows if r.get("split") == "dev"),
        "n_eval": sum(1 for r in rows if r.get("split") == "eval"),
        "execute_n": n_exec,
        "refuse_n": len(rows) - n_exec,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2_RECIPE)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()
    v1 = assert_v1_frozen()
    recipe = load_recipe(args.recipe)
    rows, cands = build(recipe, seed=args.seed)
    bank_dir = BANK_NEEDLE_VRM_AGENT_V2.parent
    dump_jsonl(BANK_NEEDLE_VRM_AGENT_V2, rows)
    run_dir = EXPERIMENTS_RUNS / "needle-vrm-v2-build"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(run_dir / "holdout-v2.candidates.jsonl", cands)
    dist = distribution(rows)
    toolset = EVAL_SHARED_ROOT / "toolsets" / "needle-vrm-agent-v0.json"
    man = {
        "bank": "eval-bank-v2.jsonl",
        "holdout_version": "v2",
        "n": len(rows),
        "n_candidates": len(cands),
        "sha256": sha256_file(BANK_NEEDLE_VRM_AGENT_V2),
        "recipe": str(args.recipe.relative_to(ROOT)),
        "recipe_sha256": sha256_file(args.recipe),
        "toolset_sha256": sha256_file(toolset),
        "v1_sha256": v1["sha256"],
        "gold": "schema-program",
        "teacher_model": "template",
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "distribution": dist,
        "review": {"status": "pending"},
    }
    (bank_dir / "holdout-v2.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
