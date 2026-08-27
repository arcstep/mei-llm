"""Shared colloquial fleet lanes. Unique is summed across dirs toward one 30M cap."""

from __future__ import annotations

import json
from pathlib import Path

from repo_paths import CORPORA_ROOT

POOLED_UNIQUE_TARGET = 30_000_000

FORMAL_DIR = "zh-pretrain-colloquial-synth-qwen-v1"

SIDECAR_DIRS = (
    "zh-pretrain-colloquial-synth-dsflash-v1",
    "zh-pretrain-colloquial-synth-qwen36plus-v1",
    "zh-pretrain-colloquial-synth-qwen37plus-v1",
    "zh-pretrain-colloquial-synth-glm52-v1",
    "zh-pretrain-colloquial-synth-kimi-k3-v1",
)

ALL_DIRS = (FORMAL_DIR, *SIDECAR_DIRS)


def cursor_unique(out_name: str) -> int:
    path = CORPORA_ROOT / out_name / "state" / "cursor.json"
    if not path.is_file():
        return 0
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return int(cur.get("n_unique_train_tokens") or 0)


def pooled_unique() -> dict[str, int]:
    per = {name: cursor_unique(name) for name in ALL_DIRS}
    total = sum(per.values())
    return {
        "per_dir": per,
        "total": total,
        "remain": max(0, POOLED_UNIQUE_TARGET - total),
        "target": POOLED_UNIQUE_TARGET,
    }


def remaining_for_lane(out_name: str) -> int:
    """Per-lane target = current unique + pooled remainder, so one lane can finish the cap."""
    snap = pooled_unique()
    return int(snap["per_dir"].get(out_name) or 0) + int(snap["remain"])
