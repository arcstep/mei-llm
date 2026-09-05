#!/usr/bin/env python3
"""CPT readiness for any declared cumulative exposure rung."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import (
    CORPUS_LM_V2,
    ROOT,
    TOKENIZER_ZH_V1,
    ensure_formal_on_path,
    frozen_tokenizer_path,
)

ensure_formal_on_path()
from training.cpt.cpt_gates import (
    refuse_cpt_parent,
    refuse_cpt_source,
)
from common.data import classify_schedule, file_sha256, list_source_shards
from orchestration.lifecycle_51m import atomic_json, corpus_snapshot, rung_name, schedule_path

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


def report(corpus_dir: Path, target_exposure: int) -> dict:
    corpus_dir = corpus_dir.resolve()
    rung = rung_name(target_exposure)
    path = schedule_path(corpus_dir, target_exposure)
    mix = _load(corpus_dir / "mix.json")
    schedule = _load(path)
    hashes = _load(corpus_dir / "hashes.json")
    hash_ok = {
        name: (corpus_dir / name).is_file() and file_sha256(corpus_dir / name) == expected
        for name, expected in hashes.items()
    }
    coverage = {
        name: token_count(list_source_shards(corpus_dir, name, "train"))
        for name in (mix.get("sources") or {})
    }
    quotas = {
        name: int((row or {}).get("token_quota") or 0)
        for name, row in (schedule.get("sources") or {}).items()
    }
    parent = Path(str(schedule.get("parent_checkpoint") or ""))
    if not parent.is_absolute():
        parent = ROOT / parent
    parent_err = refuse_cpt_parent(parent, schedule)
    source_err = refuse_cpt_source(corpus_dir, rung)
    frozen = frozen_tokenizer_path()
    tok_ok = (
        (TOKENIZER_ZH_V1.is_file() and file_sha256(TOKENIZER_ZH_V1) == FROZEN_TOK_SHA)
        if frozen == TOKENIZER_ZH_V1
        else frozen.is_file()
    )
    try:
        snapshot = corpus_snapshot(corpus_dir, target_exposure)
        snapshot_error = None
    except RuntimeError as exc:
        snapshot = {}
        snapshot_error = str(exc)
    coverage_ok = all(coverage.get(name, 0) >= quota for name, quota in quotas.items())
    ready = (
        parent_err is None
        and source_err is None
        and tok_ok
        and snapshot_error is None
        and classify_schedule(schedule) == "cpt"
        and int(schedule.get("cumulative_exposure_tokens") or 0) == target_exposure
        and coverage_ok
        and all(hash_ok.values())
        and not (corpus_dir / "schedule.json").exists()
        and not (corpus_dir / "schedule-scratch.json").exists()
        and mix.get("training_mode") == "cpt"
    )
    return {
        "kind": "cpt-readiness",
        "ready": ready,
        "rung": rung,
        "target_exposure": target_exposure,
        "corpus_dir": str(corpus_dir),
        "schedule": str(path),
        "parent_error": parent_err,
        "source_error": source_err,
        "snapshot_error": snapshot_error,
        "snapshot": snapshot,
        "tokenizer_ok": tok_ok,
        "hashes_ok": hash_ok,
        "source_train_tokens": coverage,
        "source_quotas": quotas,
        "coverage_ok": coverage_ok,
        "schedule_kind": classify_schedule(schedule),
        "current": _load(ROOT / "CURRENT.json"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-formal", action="store_true")
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_LM_V2)
    ap.add_argument("--target-exposure", type=int, default=1_000_000_000)
    ap.add_argument("--receipt", type=Path, default=None)
    args = ap.parse_args()
    payload = report(args.corpus_dir, args.target_exposure)
    if args.receipt:
        atomic_json(args.receipt, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_formal:
        return 0 if payload["ready"] else 2
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
