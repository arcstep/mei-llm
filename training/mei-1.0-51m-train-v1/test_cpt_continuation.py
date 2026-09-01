#!/usr/bin/env python3
"""Quota migration, continuation LR, and hash-flex unit tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ensure_formal_on_path

ensure_formal_on_path()
from checkpoint import validate_expected_meta
from cpt_gates import (
    CPT_CUMULATIVE_QUOTAS,
    CPT_INCREMENTAL_QUOTAS,
    PARENT_SOURCE_TOKEN_CURSORS,
    PARENT_SOURCE_TOKENS_DRAWN,
    RESET_SOURCE_CURSORS,
)
from data import PackedTokenSource, QuotaPackedSources
from train_common import cosine_lr_tokens
from train_pretrain import completed_stage_tokens


def _write_bin(path: Path, n: int) -> None:
    arr = (np.arange(n, dtype=np.uint32) % 1000).astype("<u2")
    path.write_bytes(arr.tobytes())


def _sources(tmp: Path, seq: int, n_tokens: int = 50_000) -> dict[str, PackedTokenSource]:
    out: dict[str, PackedTokenSource] = {}
    for i, name in enumerate(("wiki", "hq", "structure", "colloquial")):
        path = tmp / f"{name}.bin"
        _write_bin(path, n_tokens + i * 2048)
        out[name] = PackedTokenSource([path], seq, pad_id=0)
    return out


def test_parent_migration_resets_minors() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        parent = QuotaPackedSources(
            _sources(tmp, 8),
            {"wiki": 64, "hq": 40, "structure": 16, "colloquial": 24},
            seed=0,
            seq_len=8,
            stage_id="s3",
        )
        parent.take_windows(12)
        state = parent.state_dict()
        child = QuotaPackedSources(
            _sources(tmp, 16),
            {"wiki": 48, "hq": 32, "structure": 16, "colloquial": 24},
            seed=1,
            seq_len=16,
            stage_id="cpt1",
        )
        child.migrate_from_parent(state, reset_sources=RESET_SOURCE_CURSORS)
        assert child.token_cursors["wiki"] == parent.token_cursors["wiki"]
        assert child.token_cursors["hq"] == parent.token_cursors["hq"]
        assert child.token_cursors["structure"] == 0
        assert child.token_cursors["colloquial"] == 0
        assert child.source_tokens_drawn == parent.source_tokens_drawn
        assert child.stage_tokens_drawn == {name: 0 for name in child.names}
        assert child.plan_index == 0
        win = child.take_one()
        assert win is not None
        if win["source_id"] in RESET_SOURCE_CURSORS:
            assert child.token_cursors[win["source_id"]] == 16
        else:
            assert child.token_cursors[win["source_id"]] == state["token_cursors"][win["source_id"]] + 16


def test_old_minors_not_resampled_when_quota_zero() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        packed = QuotaPackedSources(
            _sources(tmp, 8),
            {"wiki": 32, "hq": 16, "structure": 0, "colloquial": 0},
            seed=0,
            seq_len=8,
        )
        names = []
        while True:
            win = packed.take_one()
            if win is None:
                break
            names.append(win["source_id"])
        assert "structure" not in names
        assert "colloquial" not in names


def test_undeclared_hash_still_refused_in_curriculum() -> None:
    raised = False
    try:
        validate_expected_meta(
            {"tokenizer_sha256": "t", "corpus_sha256": "old", "seq_len": 2048, "manifest_sha256": "m", "lr_horizon_tokens": 1, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0},
            {"tokenizer_sha256": "t", "corpus_sha256": "new", "seq_len": 2048, "manifest_sha256": "m", "lr_horizon_tokens": 1, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0},
            "curriculum",
        )
    except ValueError as exc:
        raised = True
        assert "corpus_sha256" in str(exc)
    assert raised


def test_continuation_allows_declared_corpus_migration() -> None:
    validate_expected_meta(
        {"tokenizer_sha256": "t", "corpus_sha256": "old", "seq_len": 2048, "manifest_sha256": "m", "lr_horizon_tokens": 300000000, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0, "schedule_sha256": "s1"},
        {"tokenizer_sha256": "t", "corpus_sha256": "new", "seq_len": 2048, "manifest_sha256": "m2", "lr_horizon_tokens": 699999515, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0, "schedule_sha256": "s2"},
        "continuation",
    )


def test_continuation_allows_architecture_hash_when_identity_holds() -> None:
    validate_expected_meta(
        {
            "tokenizer_sha256": "t",
            "corpus_sha256": "old",
            "seq_len": 2048,
            "manifest_sha256": "m",
            "lr_horizon_tokens": 300000000,
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "fp32",
            "shuffle_seed": 0,
            "architecture_id": "mei-1.0-51m-arch-v1",
            "architecture_sha256": "84568ca0",
            "params": 51463797,
        },
        {
            "tokenizer_sha256": "t",
            "corpus_sha256": "new",
            "seq_len": 2048,
            "manifest_sha256": "m2",
            "lr_horizon_tokens": 699999515,
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "fp32",
            "shuffle_seed": 0,
            "architecture_id": "mei-1.0-51m-arch-v1",
            "architecture_sha256": "f02aebda",
            "params": 51463797,
        },
        "continuation",
    )


def test_continuation_refuses_architecture_hash_when_params_drift() -> None:
    raised = False
    try:
        validate_expected_meta(
            {
                "tokenizer_sha256": "t",
                "architecture_id": "mei-1.0-51m-arch-v1",
                "architecture_sha256": "old",
                "params": 58541901,
            },
            {
                "tokenizer_sha256": "t",
                "architecture_id": "mei-1.0-51m-arch-v1",
                "architecture_sha256": "new",
                "params": 51463797,
            },
            "continuation",
        )
    except ValueError as exc:
        raised = True
        assert "architecture_sha256" in str(exc) or "params" in str(exc)
    assert raised


def test_continuation_refuses_tokenizer_drift() -> None:
    raised = False
    try:
        validate_expected_meta(
            {"tokenizer_sha256": "old", "corpus_sha256": "c", "seq_len": 2048, "manifest_sha256": "m", "lr_horizon_tokens": 1, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0},
            {"tokenizer_sha256": "new", "corpus_sha256": "c", "seq_len": 2048, "manifest_sha256": "m", "lr_horizon_tokens": 1, "batch_size": 1, "grad_accum": 1, "precision": "fp32", "shuffle_seed": 0},
            "continuation",
        )
    except ValueError as exc:
        raised = True
        assert "tokenizer_sha256" in str(exc)
    assert raised


def test_segment_lr_continuous_at_parent_boundary() -> None:
    start = cosine_lr_tokens(0, 699_999_515, 3e-5, 1e-5)
    assert abs(start - 3e-5) < 1e-12
    jumped = cosine_lr_tokens(300_000_485, 1_000_000_000, 3e-4, 3e-5)
    assert jumped > 1e-4
    later = cosine_lr_tokens(699_999_515, 699_999_515, 3e-5, 1e-5)
    assert abs(later - 1e-5) < 1e-12


def test_resumed_cpt_reports_cumulative_stage_exposure() -> None:
    assert completed_stage_tokens(
        600_000_000,
        process_start_tokens=506_914_021,
        parent_tokens_seen=300_000_485,
        schedule_kind="cpt",
    ) == 299_999_515
    assert completed_stage_tokens(
        600_000_000,
        process_start_tokens=506_914_021,
        parent_tokens_seen=300_000_485,
        schedule_kind="scratch",
    ) == 93_085_979


def test_same_stage_resume_bit_for_bit_after_migration() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        parent_state = {
            "names": ["wiki", "hq", "structure", "colloquial"],
            "token_cursors": dict(PARENT_SOURCE_TOKEN_CURSORS),
            "source_tokens_drawn": dict(PARENT_SOURCE_TOKENS_DRAWN),
        }
        # Use tiny synthetic cursors so windows exist.
        parent_state["token_cursors"] = {"wiki": 16, "hq": 8, "structure": 64, "colloquial": 48}
        parent_state["source_tokens_drawn"] = {"wiki": 16, "hq": 8, "structure": 12, "colloquial": 20}
        a = QuotaPackedSources(
            _sources(tmp, 8, n_tokens=4096),
            {"wiki": 32, "hq": 16, "structure": 16, "colloquial": 16},
            seed=3,
            seq_len=8,
            stage_id="cpt1",
        )
        a.migrate_from_parent(parent_state, reset_sources=RESET_SOURCE_CURSORS)
        a.take_windows(5)
        snap = a.state_dict()
        b = QuotaPackedSources(
            _sources(tmp, 8, n_tokens=4096),
            {"wiki": 32, "hq": 16, "structure": 16, "colloquial": 16},
            seed=3,
            seq_len=8,
            stage_id="cpt1",
        )
        b.load_state_dict(snap)
        next_a = a.take_windows(6)
        next_b = b.take_windows(6)
        assert [w["source_id"] for w in next_a] == [w["source_id"] for w in next_b]
        assert [w["source_token_cursor"] for w in next_a] == [w["source_token_cursor"] for w in next_b]


def test_quota_contract_frozen() -> None:
    assert CPT_INCREMENTAL_QUOTAS["structure"] == 11_342_703
    assert CPT_INCREMENTAL_QUOTAS["colloquial"] == 70_253_893
    assert CPT_CUMULATIVE_QUOTAS["wiki"] == 555_525_480


def main() -> int:
    test_parent_migration_resets_minors()
    test_old_minors_not_resampled_when_quota_zero()
    test_undeclared_hash_still_refused_in_curriculum()
    test_continuation_allows_declared_corpus_migration()
    test_continuation_allows_architecture_hash_when_identity_holds()
    test_continuation_refuses_architecture_hash_when_params_drift()
    test_continuation_refuses_tokenizer_drift()
    test_segment_lr_continuous_at_parent_boundary()
    test_resumed_cpt_reports_cumulative_stage_exposure()
    test_same_stage_resume_bit_for_bit_after_migration()
    test_quota_contract_frozen()
    print({"ok": True, "tests": 11})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
