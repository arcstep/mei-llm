#!/usr/bin/env python3
"""Accumulate versioned colloquial synth releases. Never overwrite a parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import CORPORA_ROOT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_synth_lib import axes_of, contract_sha256, load_contract, terms_hash
from zh_pretrain_ingest import dump_json, rel


def load_cards(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    return list(obj.get("cards") or [])


def parent_release(path: Path) -> dict:
    rel_json = path / "RELEASE.json"
    if not rel_json.is_file():
        return {"id": path.name, "n_unique_train_tokens": 0, "train_shards": [], "valid_shards": []}
    return json.loads(rel_json.read_text(encoding="utf-8"))


def next_id(parents: list[str]) -> str:
    nums = []
    for name in parents:
        if name.startswith("zh-pretrain-colloquial-synth-v"):
            tail = name.rsplit("-v", 1)[-1]
            if tail.isdigit():
                nums.append(int(tail))
    n = (max(nums) + 1) if nums else 2
    return f"zh-pretrain-colloquial-synth-v{n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", action="append", default=[], help="Parent corpus dir (repeatable)")
    ap.add_argument("--cards", type=Path, default=EXPERIMENTS_RUNS / "colloquial-feedback/cards-v0.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    contract = load_contract()
    parent_dirs = []
    names = args.parent or ["zh-pretrain-colloquial-synth-v1"]
    for name in names:
        p = Path(name)
        p = p if p.is_absolute() else (name if Path(name).exists() else CORPORA_ROOT / name)
        if not p.is_absolute():
            p = ROOT / p if (ROOT / p).exists() else CORPORA_ROOT / name
        parent_dirs.append(p)
    cards_path = args.cards if args.cards.is_absolute() else ROOT / args.cards
    cards = load_cards(cards_path)
    boosted_styles = []
    boosted_scenes = []
    quota = 0
    forbidden: list[str] = []
    for card in cards:
        boosted_styles.extend(card.get("target_styles") or [])
        boosted_scenes.extend(card.get("target_scenes") or [])
        quota += int(card.get("quota_docs") or 0)
        forbidden.extend(card.get("forbidden_strings") or [])
        if "no_holdout_copy" not in (card.get("stop_conditions") or []):
            forbidden.append("MISSING_STOP_no_holdout_copy")
    axes = axes_of(contract)
    nid = next_id([p.name for p in parent_dirs])
    out = args.out_dir or (CORPORA_ROOT / nid)
    if not out.is_absolute():
        out = ROOT / out
    if any(p.resolve() == out.resolve() for p in parent_dirs):
        raise SystemExit("refuse to overwrite a parent release directory")
    out.mkdir(parents=True, exist_ok=True)
    parents_meta = []
    all_train = []
    all_valid = []
    unique = 0
    for p in parent_dirs:
        meta = parent_release(p)
        parents_meta.append(
            {
                "id": meta.get("id") or p.name,
                "path": rel(p, ROOT),
                "n_unique_train_tokens": int(meta.get("n_unique_train_tokens") or 0),
                "generator": meta.get("generator"),
            }
        )
        unique += int(meta.get("n_unique_train_tokens") or 0)
        all_train.extend(meta.get("train_shards") or [])
        all_valid.extend(meta.get("valid_shards") or [])
    recipe = {
        "id": nid,
        "parents": parents_meta,
        "accumulate": True,
        "overwrite_parent": False,
        "feedback_cards": rel(cards_path, ROOT) if cards_path.is_file() else None,
        "boosted_scenes": sorted(set(boosted_scenes)),
        "boosted_styles": sorted(set(boosted_styles)),
        "incremental_quota_docs": quota,
        "forbidden_strings": sorted(set(forbidden)),
        "axes": axes,
        "terms_hash": terms_hash(contract),
        "contract_sha256": contract_sha256(),
        "stop_conditions": ["no_holdout_copy", "no_gold_answers", "no_student_recycle"],
        "referenced_parent_shards": {"train": all_train, "valid": all_valid, "unique_tokens_so_far": unique},
    }
    dump_json(out / "recipe-lock.json", recipe)
    dump_json(
        out / "RELEASE.json",
        {
            "id": nid,
            "release_kind": "incremental_recipe",
            "parents": [m["id"] for m in parents_meta],
            "accumulate": True,
            "generated_docs": False,
            "n_unique_train_tokens": unique,
            "train_shards": all_train,
            "valid_shards": all_valid,
            "formal_cpt_eligible": False,
            "note": "Recipe for the next generator pass. Parent token bins stay in parent dirs.",
        },
    )
    (out / "README.md").write_text(
        f"# {nid}\n\nIncremental spoken-role recipe. Parent releases are referenced, never overwritten.\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "id": nid, "parents": [m["id"] for m in parents_meta], "quota_docs": quota}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
