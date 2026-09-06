#!/usr/bin/env python3
"""Curriculum resume meta checks and illegal parent rejection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.paths import CORPUS_LM_V1, ROOT, ensure_formal_on_path

ensure_formal_on_path()
from common.checkpoint import validate_expected_meta
from training.cpt.pretrain_gates import refuse_non_scratch_source


def _meta(**overrides) -> dict:
    row = {
        "tokenizer_sha256": "tok",
        "corpus_sha256": "corp",
        "seq_len": 512,
        "manifest_sha256": "man",
        "lr_horizon_tokens": 300000000,
        "batch_size": 8,
        "grad_accum": 1,
        "precision": "fp32",
        "shuffle_seed": 0,
        "schedule_sha256": "sched",
    }
    row.update(overrides)
    return row


def test_same_stage_strict_ok() -> None:
    validate_expected_meta(_meta(), _meta(), "strict")


def test_seq_change_strict_refuses() -> None:
    raised = False
    try:
        validate_expected_meta(_meta(seq_len=512), _meta(seq_len=1024, batch_size=2), "strict")
    except ValueError as exc:
        raised = True
        assert "seq_len" in str(exc)
    assert raised


def test_seq_change_curriculum_allows_layout() -> None:
    validate_expected_meta(
        _meta(seq_len=512, batch_size=8, grad_accum=1),
        _meta(seq_len=1024, batch_size=2, grad_accum=1),
        "curriculum",
    )
    validate_expected_meta(
        _meta(seq_len=1024, batch_size=2),
        _meta(seq_len=2048, batch_size=1),
        "curriculum",
    )


def test_hash_drift_refuses_even_in_curriculum() -> None:
    raised = False
    try:
        validate_expected_meta(_meta(corpus_sha256="old"), _meta(corpus_sha256="new"), "curriculum")
    except ValueError as exc:
        raised = True
        assert "corpus_sha256" in str(exc)
    assert raised


def test_missing_sampler_keys_are_trainer_contract() -> None:
    schedule = json.loads((CORPUS_LM_V1 / "schedule-scratch.json").read_text(encoding="utf-8"))
    assert schedule["sampler"] == "quota_plan"
    assert schedule.get("parent_checkpoint") in (None, "")


def test_old_cpt_paths_refused() -> None:
    dirty = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/archive/corpus/zh-pretrain-v2"
    msg = refuse_non_scratch_source(dirty, "300m")
    assert msg and "zh-pretrain-v2" in msg
    assert refuse_non_scratch_source(CORPUS_LM_V1, "300m") is None


def main() -> int:
    test_same_stage_strict_ok()
    test_seq_change_strict_refuses()
    test_seq_change_curriculum_allows_layout()
    test_hash_drift_refuses_even_in_curriculum()
    test_missing_sampler_keys_are_trainer_contract()
    test_old_cpt_paths_refused()
    print({"ok": True, "tests": 6})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
