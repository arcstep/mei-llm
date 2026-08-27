#!/usr/bin/env python3
"""Sample same-frame offline vs qwen pairs for non-Qwen blind review."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_synth_lib import load_contract, render_offline  # noqa: E402
from zh_pretrain_ingest import dump_json  # noqa: E402

AXES = ("scene", "styles", "relation", "mood", "n_turns")


def load_ok(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("filter") or {}).get("ok"):
                out.append(row)
    return out


def pick_stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for row in rows:
        frame = row.get("frame") or {}
        key = f"{frame.get('scene')}|{(frame.get('styles') or [''])[0]}|{frame.get('relation')}"
        by.setdefault(key, []).append(row)
    picked = []
    keys = list(by)
    rng.shuffle(keys)
    for key in keys:
        if len(picked) >= n:
            break
        bucket = by[key]
        picked.append(bucket[rng.randrange(len(bucket))])
    if len(picked) < n:
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    rows = load_ok(corpus / "raw" / "accepted.jsonl")
    sample = pick_stratified(rows, args.n, args.seed)
    pairs = []
    for row in sample:
        frame = dict(row.get("frame") or {})
        offline = render_offline(frame)
        pairs.append(
            {
                "frame_id": row.get("doc_id") or frame.get("frame_id"),
                "frame": frame,
                "qwen": row.get("text"),
                "offline": offline.get("text"),
                "axes": {k: frame.get(k) for k in AXES},
            }
        )
    out_dir = corpus / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "blind-pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    dump_json(
        out_dir / "blind-sample-meta.json",
        {
            "n": len(pairs),
            "corpus": str(corpus.relative_to(ROOT)) if corpus.is_relative_to(ROOT) else str(corpus),
            "reviewer": "gpt-5.6-sol",
            "not_qwen": True,
            "pairs_path": str(pairs_path.relative_to(ROOT)) if pairs_path.is_relative_to(ROOT) else str(pairs_path),
        },
    )
    print(json.dumps({"ok": True, "n": len(pairs), "path": str(pairs_path)}, ensure_ascii=False, indent=2))
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
