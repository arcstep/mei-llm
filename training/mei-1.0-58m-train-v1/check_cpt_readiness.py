#!/usr/bin/env python3
"""CPT 1B readiness: parent hashes, lm-v2 unique coverage, schedule contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import CORPUS_LM_V2, ROOT, TOKENIZER_ZH_V1, ensure_formal_on_path

ensure_formal_on_path()
from cpt_gates import (
    CPT_INCREMENTAL_QUOTAS,
    PARENT_TOKENS_SEEN,
    refuse_cpt_parent,
    refuse_cpt_source,
)
from data import classify_schedule, file_sha256, list_source_shards, resolve_schedule_file

FROZEN_TOK_SHA = "fcd07b3d49f5174bb60e81996f4d3f2d55f458f5b8420a271aea59ac5dc58629"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def token_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        n = path.stat().st_size
        if n % 2:
            raise ValueError(f"odd shard {path}")
        total += n // 2
    return total


def report() -> dict:
    mix = _load(CORPUS_LM_V2 / "mix.json")
    schedule = _load(CORPUS_LM_V2 / "schedule-cpt-1b.json")
    hashes = _load(CORPUS_LM_V2 / "hashes.json")
    hash_ok = {
        name: (CORPUS_LM_V2 / name).is_file() and file_sha256(CORPUS_LM_V2 / name) == expected
        for name, expected in hashes.items()
    }
    struct_n = token_count(list_source_shards(CORPUS_LM_V2, "structure", "train"))
    col_n = token_count(list_source_shards(CORPUS_LM_V2, "colloquial", "train"))
    parent_err = refuse_cpt_parent()
    source_err = refuse_cpt_source(CORPUS_LM_V2, "1b")
    tok_ok = TOKENIZER_ZH_V1.is_file() and file_sha256(TOKENIZER_ZH_V1) == FROZEN_TOK_SHA
    ready = (
        parent_err is None
        and source_err is None
        and tok_ok
        and classify_schedule(schedule) == "cpt"
        and int(schedule.get("parent_tokens_seen") or 0) == PARENT_TOKENS_SEEN
        and struct_n >= CPT_INCREMENTAL_QUOTAS["structure"] + 2048
        and col_n >= CPT_INCREMENTAL_QUOTAS["colloquial"] + 2048
        and all(hash_ok.values())
        and not (CORPUS_LM_V2 / "schedule.json").exists()
        and not (CORPUS_LM_V2 / "schedule-scratch.json").exists()
        and resolve_schedule_file(CORPUS_LM_V2, "cpt") == CORPUS_LM_V2 / "schedule-cpt-1b.json"
        and mix.get("training_mode") == "cpt"
    )
    return {
        "cpt_1b_ready": ready,
        "parent_error": parent_err,
        "source_error": source_err,
        "tokenizer_ok": tok_ok,
        "hashes_ok": hash_ok,
        "structure_train_tokens": struct_n,
        "colloquial_train_tokens": col_n,
        "schedule_kind": classify_schedule(schedule),
        "current": _load(ROOT / "CURRENT.json"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-formal", action="store_true")
    args = ap.parse_args()
    payload = report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_formal:
        return 0 if payload["cpt_1b_ready"] else 2
    return 0 if payload["cpt_1b_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
