#!/usr/bin/env python3
"""Validate mei-tool-sft-v1: schema gold, whole-toolset isolation, no holdout names."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_needle_toolcall_v0 import call_schema_errors, load_toolset
from mei_tool_schema_lib import HOLDOUT_TOOLSETS, TRAIN_TOOLSETS, isolation_names, load_ts, tool_index
from repo_paths import ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from data import encode_sft_row  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    rows = load_jsonl(pack)
    errors: list[str] = []
    names = isolation_names()
    holdout_names = names["holdout_tool_names"]
    tok = ZhTokenizerV1()
    overflow = 0
    by_ts: dict[str, int] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid.startswith("EVAL-") or "EVAL-" in json.dumps(row, ensure_ascii=False):
            errors.append(f"{sid}: eval id leak")
        tid = str(row.get("toolset_id") or "")
        if tid in HOLDOUT_TOOLSETS or tid in {"mei-office-mut-v0", "mei-office-rename-v0"}:
            errors.append(f"{sid}: holdout toolset in train pack")
        if tid not in TRAIN_TOOLSETS:
            errors.append(f"{sid}: unexpected toolset {tid}")
        by_ts[tid] = by_ts.get(tid, 0) + 1
        ts = load_toolset(tid)
        answers = row.get("answers") or []
        errors.extend(call_schema_errors(sid, answers, ts))
        blob = json.dumps(row, ensure_ascii=False)
        for name in holdout_names:
            if name in blob:
                errors.append(f"{sid}: holdout tool name leak {name}")
        packed = encode_sft_row(tok, row, args.seq_len, inject_schema=True)
        if packed.get("n_unmasked", 0) <= 0:
            overflow += 1
            errors.append(f"{sid}: encode reject {packed.get('reject_reason')}")
    if len({r.get("sample_id") for r in rows}) != len(rows):
        errors.append("duplicate sample_id")
    report = {
        "ok": not errors,
        "n": len(rows),
        "by_toolset": by_ts,
        "overflow": overflow,
        "errors": errors[:40],
        "n_errors": len(errors),
        "pack": str(pack.relative_to(ROOT)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
