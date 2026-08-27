#!/usr/bin/env python3
"""Build retrieval-v2 canonical cases. Gold tool is schema-program, never teacher-written."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import ROOT, SFT_TRAIN, sft_v2_candidate_pack_name

sys.path.insert(0, str(Path(__file__).resolve().parent))

from park_toolcall_lib import park_retrieval_cases  # noqa: E402
from sft_canonical_lib import (  # noqa: E402
    compile_retrieval_row,
    dump_jsonl,
    ensure_train_valid_split,
    freeze_family_splits,
    retrieval_case_bank,
    sha256_text,
    split_for_key,
)
from sft_synth_lib import dump_json  # noqa: E402

CITIES = ["南京", "上海", "北京", "杭州", "成都", "深圳", "武汉", "西安", "苏州", "青岛", "天津", "重庆"]
ROOMS = ["次卧", "主卧", "客厅", "书房", "厨房", "阳台", "玄关", "餐厅"]
TITLES = ["晨会", "周会", "评审", "站会", "月结", "年会", "复盘", "面试"]
ITEMS = ["矿泉水", "三明治", "咖啡", "面包"]
DOORS = ["前门", "后门", "侧门", "阳台门"]
PLACES = ["厨房", "客厅", "卧室", "门口", "阳台"]
MANNERS = ["麻烦", "帮我", "劳驾", "能不能", "请", "劳烦", "帮忙", ""]
POLITE = ["", "可以吗", "谢谢", "一下"]


def _park_n(limit: int) -> int:
    if limit <= 0 or limit >= 10000:
        return 2000 if limit >= 10000 else 400
    if limit >= 2000:
        return 400
    return max(8, limit * 400 // 2000)


def _rotate_negs(src: dict, i: int) -> list[str]:
    gold = src.get("gold_tool")
    names = [str(t.get("name")) for t in src.get("catalog_tools") or [] if t.get("name")]
    others = [n for n in names if n != gold]
    current = [str(x) for x in (src.get("hard_negatives") or [])]
    if len(others) < 2:
        return current
    a = others[i % len(others)]
    b = others[(i + 3) % len(others)]
    if a == b and len(others) > 1:
        b = others[(i + 1) % len(others)]
    return [a, b]


def _slot_lexicon(key: str) -> list[str]:
    if key == "city":
        return CITIES
    if key == "room":
        return ROOMS
    if key in {"title", "event"}:
        return TITLES
    if key in {"item", "sku"}:
        return ITEMS
    if key == "door":
        return DOORS
    if key in {"place", "location"}:
        return PLACES
    return []


def expand_generic(base: list[dict], n: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    def push(src: dict, query: str, slots: dict | None, i: int) -> bool:
        query = str(query or "").strip()
        if not query or query in seen:
            return False
        row = dict(src)
        row["query"] = query
        row["stem"] = query
        if slots is not None:
            row["zh_slots"] = dict(slots)
        row["hard_negatives"] = _rotate_negs(row, i)
        row["case_id"] = "RET-" + sha256_text(query + str(row.get("gold_tool")))[:16]
        row["cf_group"] = "CFG-" + sha256_text(query + "|" + str(row.get("gold_tool")))[:12]
        row["split"] = split_for_key(row["cf_group"])
        seen.add(query)
        out.append(row)
        return True

    for src in base:
        push(src, str(src.get("stem") or src.get("query") or ""), src.get("zh_slots"), 0)
        if len(out) >= n:
            return out[:n]

    i = 0
    while len(out) < n and base and i < n * 40:
        src = dict(base[i % len(base)])
        stem = str(src.get("stem") or src.get("query") or "")
        slots = dict(src.get("zh_slots") or {})
        placed = False
        for key, old in list(slots.items()):
            lex = _slot_lexicon(key)
            if not old or not lex or str(old) not in stem:
                continue
            new = lex[(i + len(key)) % len(lex)]
            if new == old:
                continue
            query = stem.replace(str(old), new, 1)
            new_slots = dict(slots)
            new_slots[key] = new
            if push(src, query, new_slots, i):
                placed = True
                break
        if not placed:
            manner = MANNERS[i % len(MANNERS)]
            polite = POLITE[(i // max(1, len(MANNERS))) % len(POLITE)]
            query = f"{manner}{stem}{polite}".strip()
            if not push(src, query, slots, i):
                query = f"{manner}{stem}可以吗{i}"
                push(src, query, slots, i)
        i += 1
    return out[:n]


def build(*, limit: int, teacher_model: str) -> tuple[list[dict], list[dict]]:
    generic = retrieval_case_bank()
    if 0 < limit <= 24:
        cases = generic[:limit]
        for case in cases:
            case["engineering_smoke"] = True
    else:
        park_n = _park_n(limit)
        park = park_retrieval_cases(park_n)
        rest = (limit if limit > 0 else 2000) - len(park)
        cases = park + expand_generic(generic, max(rest, 0))
        if limit > 0:
            cases = cases[:limit]
    ensure_train_valid_split(cases)
    rows = []
    for i, case in enumerate(cases):
        case = dict(case)
        case["idx"] = i
        rows.append(compile_retrieval_row(case, str(case["query"]), teacher_model=teacher_model))
    return cases, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=SFT_TRAIN / "packs")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--pack-name", default=None)
    args = ap.parse_args()
    cases, rows = build(limit=args.limit, teacher_model="template")
    splits = freeze_family_splits(cases)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = 0 < args.limit <= 24
    name = args.pack_name or sft_v2_candidate_pack_name("mei-retrieval-v2", args.limit)
    pack = out_dir / name
    dump_jsonl(pack, rows)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(args.work_dir / "canonical.jsonl", cases)
        dump_jsonl(args.work_dir / "accepted.jsonl", rows)
        dump_json(args.work_dir / "split-lock.json", splits)
    report = {
        "ok": splits["ok"] and len(rows) == len(cases) and bool(rows) and splits["n_valid"] > 0,
        "n": len(rows),
        "n_train": splits["n_train"],
        "n_valid": splits["n_valid"],
        "engineering_smoke": smoke,
        "pack": str(pack.relative_to(ROOT)) if pack.is_relative_to(ROOT) else str(pack),
        "splits": splits,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
