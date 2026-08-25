#!/usr/bin/env python3
"""Export home SFT pack to a Qwen chat JSONL for a *data teachability* bypass.

Does not prove the 24k-from-scratch student. Gold stays schema-program answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from data import format_sft_user_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="2k")
    args = ap.parse_args()
    src = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / f"home-sft-{args.tier}.jsonl"
    dst = TASKS_ROOT / TASK_NEEDLE_ZH / "train/exports" / f"home-sft-{args.tier}-qwen-chat.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as inf, dst.open("w", encoding="utf-8") as out:
        for line in inf:
            if not line.strip():
                continue
            row = json.loads(line)
            ans = json.dumps(row.get("answers") or [], ensure_ascii=False, separators=(",", ":"))
            out.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": format_sft_user_text(row)},
                            {"role": "assistant", "content": ans},
                        ],
                        "sample_id": row.get("sample_id"),
                        "bypass": "qwen-teachability-only",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(json.dumps({"src": str(src.relative_to(ROOT)), "dst": str(dst.relative_to(ROOT)), "n": n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
