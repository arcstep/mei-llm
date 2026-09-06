#!/usr/bin/env python3
"""Quota sampler unit tests. No long training."""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.paths import ensure_formal_on_path

ensure_formal_on_path()
from common.data import PackedTokenSource, QuotaPackedSources, build_quota_plan, select_curriculum_stage


def _write_bin(path: Path, n: int) -> None:
    arr = (np.arange(n, dtype=np.uint32) % 1000).astype("<u2")
    path.write_bytes(arr.tobytes())


def _sources(tmp: Path, seq: int, n_tokens: int = 4096) -> dict[str, PackedTokenSource]:
    out: dict[str, PackedTokenSource] = {}
    for i, name in enumerate(("wiki", "hq", "structure", "colloquial")):
        path = tmp / f"{name}.bin"
        _write_bin(path, n_tokens + i * 16)
        out[name] = PackedTokenSource([path], seq, pad_id=0)
    return out


def test_same_seed_same_plan() -> None:
    names = ["wiki", "hq", "structure", "colloquial"]
    quotas = {"wiki": 80, "hq": 48, "structure": 16, "colloquial": 32}
    a = build_quota_plan(names, quotas, 8, 0)
    b = build_quota_plan(names, quotas, 8, 0)
    assert a == b
    counts = Counter(a)
    assert counts["wiki"] == (80 + 7) // 8
    assert counts["hq"] == (48 + 7) // 8
    assert counts["structure"] == (16 + 7) // 8
    assert counts["colloquial"] == (32 + 7) // 8


def test_different_seed_changes_interleave_not_quota() -> None:
    names = ["wiki", "hq", "structure", "colloquial"]
    quotas = {"wiki": 80, "hq": 48, "structure": 16, "colloquial": 32}
    a = build_quota_plan(names, quotas, 8, 0)
    b = build_quota_plan(names, quotas, 8, 1)
    assert a != b
    assert Counter(a) == Counter(b)


def test_take_and_resume_bit_for_bit() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        seq = 8
        sources = _sources(tmp, seq)
        quotas = {"wiki": 64, "hq": 40, "structure": 16, "colloquial": 24}
        a = QuotaPackedSources(sources, quotas, seed=0, seq_len=seq, stage_id="s1")
        first = a.take_windows(5)
        state = a.state_dict()
        b = QuotaPackedSources(_sources(tmp, seq), quotas, seed=0, seq_len=seq, stage_id="s1")
        b.load_state_dict(state)
        next_a = a.take_windows(7)
        next_b = b.take_windows(7)
        assert [w["source_id"] for w in next_a] == [w["source_id"] for w in next_b]
        assert [w["source_token_cursor"] for w in next_a] == [w["source_token_cursor"] for w in next_b]
        assert first[0]["source_id"] in quotas
        restored = QuotaPackedSources(_sources(tmp, seq), quotas, seed=0, seq_len=seq, stage_id="s1")
        restored.load_state_dict(state)
        assert restored.plan_index == a.plan_index - len(next_a)
        assert restored.token_cursors == {
            name: a.token_cursors[name] - seq * sum(1 for w in next_a if w["source_id"] == name)
            for name in quotas
        }


def test_stage_switch_cursor_continues() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        s1 = QuotaPackedSources(_sources(tmp, 8), {"wiki": 64, "hq": 32, "structure": 8, "colloquial": 16}, seed=0, seq_len=8, stage_id="s1", stage_index=0)
        s1.take_windows(6)
        state = s1.state_dict()
        s2 = QuotaPackedSources(_sources(tmp, 16), {"wiki": 48, "hq": 24, "structure": 8, "colloquial": 16}, seed=1, seq_len=16, stage_id="s2", stage_index=1)
        s2.continue_from(state)
        assert s2.token_cursors == s1.token_cursors
        assert s2.source_tokens_drawn == s1.source_tokens_drawn
        assert s2.stage_tokens_drawn == {name: 0 for name in s2.names}
        assert s2.plan_index == 0
        win = s2.take_one()
        assert win is not None
        assert s2.token_cursors[win["source_id"]] == state["token_cursors"][win["source_id"]] + 16


def test_source_exhaust_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        path = tmp / "tiny.bin"
        _write_bin(path, 20)
        src = {"wiki": PackedTokenSource([path], 8, pad_id=0)}
        packed = QuotaPackedSources(src, {"wiki": 10_000}, seed=0, seq_len=8, stage_id="s1")
        raised = False
        try:
            packed.take_windows(8)
        except RuntimeError as exc:
            raised = True
            assert "exhausted" in str(exc)
        assert raised


def test_alignment_slack_recorded() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        packed = QuotaPackedSources(
            _sources(tmp, 8, n_tokens=2048),
            {"wiki": 50, "hq": 0, "structure": 0, "colloquial": 0},
            seed=0,
            seq_len=8,
            stage_id="s1",
        )
        while True:
            win = packed.take_one()
            if win is None:
                break
        slack = packed.alignment_slack["wiki"]
        overshoot = packed.alignment_overshoot["wiki"]
        assert packed.quota_remaining["wiki"] <= 0
        assert slack == packed.stage_tokens_drawn["wiki"] - 50
        assert overshoot >= 0
        assert abs(slack) < 8 or overshoot < 8
        assert packed.stage_tokens_drawn["wiki"] >= 50


def test_select_stage() -> None:
    schedule = json_schedule()
    s2 = select_curriculum_stage(schedule, stage_id="s2")
    assert s2["id"] == "s2"
    assert s2["seq_len"] == 1024
    by_tokens = select_curriculum_stage(schedule, tokens_seen=150_000_000)
    assert by_tokens["id"] == "s2"
    by_seq = select_curriculum_stage(schedule, seq_len=2048)
    assert by_seq["id"] == "s3"


def json_schedule() -> dict:
    return {
        "curriculum": [
            {"id": "s1", "seq_len": 512, "stop_at_tokens": 150000000, "sources": {"wiki": {"token_quota": 1}}},
            {"id": "s2", "seq_len": 1024, "stop_at_tokens": 250000000, "sources": {"wiki": {"token_quota": 1}}},
            {"id": "s3", "seq_len": 2048, "stop_at_tokens": 300000000, "sources": {"wiki": {"token_quota": 1}}},
        ]
    }


def main() -> int:
    test_same_seed_same_plan()
    test_different_seed_changes_interleave_not_quota()
    test_take_and_resume_bit_for_bit()
    test_stage_switch_cursor_continues()
    test_source_exhaust_fail_closed()
    test_alignment_slack_recorded()
    test_select_stage()
    print({"ok": True, "tests": 7})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
