#!/usr/bin/env python3
"""Export seed chat JSONL → MLX-LM data dir (train.jsonl / valid.jsonl).

Rejects any sample_id / content that looks like EVAL-* leakage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = ROOT / "train/seed/sft-smoke-v0.jsonl"
DEFAULT_OUT = ROOT / "mlx/exports/sft-smoke-v0"
EVAL_RE = re.compile(r"\bEVAL-(?:DEV|EDGE)-\d+\b")


def load_seed(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assert_isolation(rows: list[dict]) -> None:
    bad: list[str] = []
    for r in rows:
        sid = str(r.get("sample_id") or "")
        if sid.startswith("EVAL-") or EVAL_RE.search(sid):
            bad.append(f"sample_id={sid}")
            continue
        blob = json.dumps(r, ensure_ascii=False)
        if EVAL_RE.search(blob):
            bad.append(f"{sid}: body mentions EVAL-*")
        if r.get("split") not in {"train", "valid", "test"}:
            bad.append(f"{sid}: bad split={r.get('split')}")
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            bad.append(f"{sid}: need chat messages")
    if bad:
        raise SystemExit("isolation/schema failed:\n- " + "\n- ".join(bad))


def to_mlx(row: dict) -> dict:
    return {"messages": row["messages"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--upsample-topics",
        default="",
        help="comma topics to duplicate, e.g. tp.bucket_archive,tp.bucket_ssot",
    )
    ap.add_argument("--upsample-copies", type=int, default=3, help="extra copies per matched train row")
    args = ap.parse_args()

    rows = load_seed(args.seed)
    assert_isolation(rows)

    upsample = {t.strip() for t in args.upsample_topics.split(",") if t.strip()}
    expanded: list[dict] = []
    for r in rows:
        expanded.append(r)
        if (
            upsample
            and r.get("split") == "train"
            and r.get("topic") in upsample
            and args.upsample_copies > 0
        ):
            for i in range(args.upsample_copies):
                c = json.loads(json.dumps(r))
                c["sample_id"] = f"{r['sample_id']}-dup{i+1}"
                expanded.append(c)

    by_split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for r in expanded:
        by_split[r["split"]].append(to_mlx(r))

    # mlx_lm expects valid.jsonl; if empty, duplicate a few train as valid smoke
    if not by_split["valid"] and by_split["train"]:
        by_split["valid"] = by_split["train"][: max(1, min(3, len(by_split["train"])))]

    args.out.mkdir(parents=True, exist_ok=True)
    for split, items in by_split.items():
        if split == "test" and not items:
            continue
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {len(items)} → {path}")

    meta = {
        "seed": str(args.seed.name),
        "n_train": len(by_split["train"]),
        "n_valid": len(by_split["valid"]),
        "upsample_topics": sorted(upsample),
        "upsample_copies": args.upsample_copies if upsample else 0,
        "note": "no EVAL-* ids; synthetic smoke only",
    }
    (args.out / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
