#!/usr/bin/env python3
"""Build MWDispositionHead cases. Head target is frozen reason_code only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_MW,
    PACK_NEEDLE_MW_SFT_2K,
    ROOT,
    SFT_TRAIN,
    sft_v2_candidate_pack_name,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_canonical_lib import (  # noqa: E402
    codebook_by_reason,
    compile_mw_row,
    dump_jsonl,
    ensure_train_valid_split,
    freeze_family_splits,
    load_jsonl,
    query_overlaps,
    sha256_text,
    split_for_key,
)
from sft_synth_lib import dump_json  # noqa: E402

EVAL_SMOKE = ROOT / "notebook/evaluation/banks/mei-mw-disposition-v2/eval-bank-smoke.jsonl"


def cases_from_seed(*, limit: int) -> list[dict]:
    book = codebook_by_reason()
    holdout = {str(r.get("query") or "").strip() for r in load_jsonl(BANK_NEEDLE_VRM_MW)}
    eval_q = {str(r.get("query") or "").strip() for r in load_jsonl(EVAL_SMOKE)}
    blocked = holdout | eval_q
    seen_canon: set[str] = set()
    by_reason: dict[str, list[dict]] = {k: [] for k in book}
    for row in load_jsonl(PACK_NEEDLE_MW_SFT_2K):
        q = str(row.get("query") or "").strip()
        if not q or q in blocked:
            continue
        reason = str(row.get("reason_code") or "")
        if reason not in book:
            continue
        canon = str(row.get("canonical_id") or "")
        if canon in seen_canon:
            continue
        seen_canon.add(canon)
        mapped = book[reason]
        case = {
            "case_id": canon or ("MWC-" + reason),
            "task": "mw",
            "query": q,
            "stem": q,
            "reason_code": reason,
            "act": mapped["act"],
            "cell": mapped["cell"],
            "gaps": list(row.get("gaps") or []),
            "function_calls": list(row.get("function_calls") or []),
            "zh_slots": dict(row.get("zh_slots") or {}),
            "family": row.get("family") or mapped["act"],
            "kind": row.get("kind") or mapped["act"],
            "cf_group": canon or ("MWC-" + reason),
            "split": split_for_key(canon or reason),
            "toolset_id": row.get("toolset_id") or "needle-vrm-agent-v0",
            "high_risk": bool(row.get("high_risk")),
            "source_role": "schema-program",
            "prefer_template": True,
        }
        by_reason.setdefault(reason, []).append(case)
    cases: list[dict] = []
    # Round-robin so smoke covers the codebook instead of dumping one class.
    while len(cases) < limit:
        progressed = False
        for reason, bucket in by_reason.items():
            if not bucket:
                continue
            cases.append(bucket.pop(0))
            progressed = True
            if len(cases) >= limit:
                break
        if not progressed:
            break
    return cases[:limit]


def expand_mw(cases: list[dict], limit: int) -> list[dict]:
    """MW seed uniqueness is tight after holdout; expand query-only, keep frozen reason_code."""
    if not cases:
        return []
    prefixes = ["麻烦", "帮我", "请问", "能不能", "劳驾", ""]
    suffixes = ["", "谢谢", "可以吗", "一下"]
    places = ["工位", "会议室", "走廊", "接待区", "茶水间", "门口"]
    times = ["下班前", "开会前", "刚进门时", "午休后", "有人经过时"]
    out = list(cases)
    seen = {str(c.get("query") or "").strip() for c in cases}
    i = 0
    while len(out) < limit and i < limit * 80:
        src = dict(cases[i % len(cases)])
        stem = str(src.get("query") or "").strip()
        place = places[i % len(places)]
        time = times[i % len(times)]
        frames = [
            stem,
            f"{prefixes[i % len(prefixes)]}{stem}{suffixes[(i // max(1, len(prefixes))) % len(suffixes)]}".strip(),
            f"现在在{place}，{stem}",
            f"{time}，{stem}",
            f"有同事在旁边，{stem}",
            f"别影响别人，{stem}",
            f"{time}在{place}，{stem}",
        ]
        q = frames[i % len(frames)].strip()
        i += 1
        n = 0
        while not q or q in seen:
            n += 1
            q = f"{stem}（扩{i}-{n}）"
        seen.add(q)
        src["query"] = q
        src["stem"] = q
        src["case_id"] = "MWC-" + sha256_text(q + str(src.get("reason_code")))[:16]
        src["cf_group"] = src["case_id"]
        src["split"] = split_for_key(src["cf_group"])
        out.append(src)
    return out[:limit]


def build(*, limit: int) -> tuple[list[dict], list[dict]]:
    want = limit + 16 if limit >= 2000 else limit
    pool = expand_mw(cases_from_seed(limit=max(want * 3, want)), limit=max(want * 2, want))
    cases: list[dict] = []
    rows: list[dict] = []
    seen_q: set[str] = set()
    for case in pool:
        q = str(case.get("query") or "").strip()
        if not q or q in seen_q:
            continue
        try:
            row = compile_mw_row(case, q, teacher_model="template")
        except ValueError:
            continue
        seen_q.add(q)
        cases.append(case)
        rows.append(row)
        if len(rows) >= want:
            break
    cases, rows = cases[:limit], rows[:limit]
    leaks = query_overlaps(rows, load_jsonl(BANK_NEEDLE_VRM_MW) + load_jsonl(EVAL_SMOKE))
    if leaks:
        raise RuntimeError("holdout/eval overlap: " + "; ".join(leaks[:8]))
    ensure_train_valid_split(cases)
    return cases, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=SFT_TRAIN / "packs")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--pack-name", default=None)
    args = ap.parse_args()
    cases, rows = build(limit=args.limit)
    splits = freeze_family_splits(cases)
    smoke = 0 < args.limit <= 24
    for row in rows:
        if smoke:
            row["engineering_smoke"] = True
    name = args.pack_name or (
        "mei-mw-disposition-v2-smoke.jsonl" if smoke else sft_v2_candidate_pack_name("mei-mw-disposition-v2", args.limit)
    )
    pack = args.out_dir / name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(pack, rows)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(args.work_dir / "canonical.jsonl", cases)
        dump_jsonl(args.work_dir / "accepted.jsonl", rows)
        dump_json(args.work_dir / "split-lock.json", splits)
    reasons = sorted({str(r.get("reason_code")) for r in rows})
    report = {
        "ok": bool(rows) and splits["ok"] and splits["n_valid"] > 0,
        "n": len(rows),
        "engineering_smoke": smoke,
        "pack": str(pack.relative_to(ROOT)) if pack.is_relative_to(ROOT) else str(pack),
        "n_reason_codes": len(reasons),
        "reason_codes": reasons,
        "splits": splits,
        "head_target": "reason_code",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
