#!/usr/bin/env python3
"""Build mei-tool-route-sft-v1 from leftover universe after eval freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from mei_tool_grounded_lib import build_universe, dump_jsonl, expand_train_pool
from repo_paths import EVAL_BANKS_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

LOCK = EVAL_BANKS_ROOT / "mei-tool-grounded-v1" / "holdout-grounded-v1.lock.json"
PACK_DIR = TASKS_ROOT / TASK_NEEDLE_ZH / "train" / "packs"
RECIPE_DIR = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes"


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def strip_eval_only(row: dict) -> dict:
    out = dict(row)
    out.pop("item_id", None)
    out["split"] = "train"
    if out.get("entities") and "entity_mode" not in out:
        ents = out["entities"]
        out["entity_mode"] = "all" if any(e.get("split") == "holdout" for e in ents) else "train"
    out.pop("entities", None)
    out.pop("lexicon", None)
    out.pop("param_types", None)
    out.pop("manifest", None)
    return out


def sample_with_refuse(pool: list[dict], n: int, rng: random.Random, refuse_frac: float = 0.125) -> list[dict]:
    refuse = [r for r in pool if r.get("kind") != "execute"]
    execute = [r for r in pool if r.get("kind") == "execute"]
    rng.shuffle(refuse)
    rng.shuffle(execute)
    n_ref = min(len(refuse), max(1, int(n * refuse_frac)))
    n_ex = min(len(execute), n - n_ref)
    chosen = refuse[:n_ref] + execute[:n_ex]
    rest = refuse[n_ref:] + execute[n_ex:]
    rng.shuffle(rest)
    chosen.extend(rest[: max(0, n - len(chosen))])
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["2k", "10k", "valid"], default="2k")
    args = ap.parse_args()
    if not LOCK.is_file():
        print("freeze eval lock first: build_mei_tool_grounded_eval_v1.py", flush=True)
        return 2
    lock = load_lock()
    used = {tuple(k.split("|", 1)) for k in lock.get("used_keys") or []}
    leftover = [r for r in build_universe(seed=7) if (r["query"], r["toolset_id"]) not in used]
    rng = random.Random({"2k": 11, "10k": 13, "valid": 17}[args.tier])
    target = {"2k": 2000, "10k": 10000, "valid": 200}[args.tier]
    pool = expand_train_pool(leftover, used, rng, want=max(12000, target + 400))
    rng_valid = random.Random(17)
    valid_hold = sample_with_refuse(pool, 200, rng_valid)
    valid_keys = {(r["query"], r["toolset_id"]) for r in valid_hold}
    if args.tier == "valid":
        chosen = valid_hold
        split = "valid"
        pack_path = PACK_DIR / "mei-tool-route-sft-v1-valid.jsonl"
    else:
        train_src = [r for r in pool if (r["query"], r["toolset_id"]) not in valid_keys]
        chosen = sample_with_refuse(train_src, target, rng)
        split = "train"
        pack_path = PACK_DIR / f"mei-tool-route-sft-v1-{args.tier}.jsonl"
    rows = []
    for i, row in enumerate(chosen):
        rec = strip_eval_only(row)
        rec["split"] = split
        rec["sample_id"] = f"SFT-{split}-{i:05d}"
        rows.append(rec)
    dump_jsonl(pack_path, rows)
    recipe = {
        "id": pack_path.stem,
        "n": len(rows),
        "eval_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "protocol": "mei-route-protocol-v1",
        "serializer": "mei-route-serializer-v1",
        "kind": dict(Counter(r["kind"] for r in rows)),
        "family": dict(Counter(str(r.get("family")) for r in rows)),
        "note": "Leftover universe after grounded eval freeze. 2k/10k independently sampled from cpt300m-ready pool.",
    }
    if args.tier != "valid":
        recipe_path = RECIPE_DIR / f"{pack_path.stem}.json"
        recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "n": len(rows), "pack": str(pack_path.relative_to(ROOT)), "kind": recipe["kind"], "pool": len(pool)}, indent=2))
    return 0 if len(rows) >= min(target, 180) else 1


if __name__ == "__main__":
    raise SystemExit(main())
