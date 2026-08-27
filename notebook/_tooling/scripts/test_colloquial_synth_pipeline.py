#!/usr/bin/env python3
"""Fast tests for colloquial synth frames, filters, and contract gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repo_paths import MODEL_MEI_58M, ROOT, SCRIPTS_ROOT
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(MODEL_MEI_58M))

from colloquial_synth_lib import (  # noqa: E402
    contract_sha256,
    filter_reasons,
    frame_salt,
    generate_one,
    iter_frames,
    load_contract,
    parse_turns_payload,
    render_offline,
    spend_cny,
    terms_hash,
    unique_by_first_frame,
)
from zh_pretrain_ingest import colloquial_keep, pii_or_nav  # noqa: E402
from validate_approved_colloquial import validate_spec  # noqa: E402


def test_contract() -> None:
    spec = load_contract()
    assert spec["generator"]["frozen_snapshot"] == "qwen-plus-2025-12-01"
    assert spec["formal_cpt"]["requires_unique_tokens"] == 30_000_000
    assert spec["hard_gates"]["unk_rate_max"] == 0.005
    assert spec["budget"]["production_max_spend_cny"] == 200.0
    assert spec["quality_vs_provenance"]["generator_snapshot_proves_provenance_only"] is True
    assert terms_hash(spec)
    assert contract_sha256()


def test_frame_idempotency() -> None:
    spec = load_contract()
    a = list(iter_frames(30, seed=7, axes=spec["axes"], start=0))
    b = list(iter_frames(10, seed=7, axes=spec["axes"], start=20))
    assert [x["salt"] for x in a[20:]] == [x["salt"] for x in b]
    assert [x["frame_id"] for x in a[20:]] == [x["frame_id"] for x in b]
    assert frame_salt(7, 20) == a[20]["salt"]
    ids = [x["frame_id"] for x in iter_frames(400, seed=0, start=0)]
    assert len(ids) == len(set(ids))


def test_frames_and_offline() -> None:
    spec = load_contract()
    texts = []
    for frame in iter_frames(40, seed=1, axes=spec["axes"]):
        got = generate_one(frame, generator="offline-frame-renderer", contract=spec)
        assert got["ok"]
        text = got["text"]
        reasons = filter_reasons(text, leaks=["EVAL-UNIQUE-LEAK-999"], pii_fn=pii_or_nav)
        assert not reasons, reasons
        assert colloquial_keep(text, domain="dialogue")
        texts.append(text)
        rendered = render_offline(frame)
        parsed = parse_turns_payload(json.dumps({"turns": rendered["turns"]}, ensure_ascii=False))
        assert parsed and len(parsed["turns"]) >= 2
    assert len(set(texts)) >= 36


def test_filters() -> None:
    pii = "甲：我电话是13800138000。\n乙：行。"
    assert "pii" in filter_reasons(pii, leaks=[], pii_fn=pii_or_nav)
    leak = "甲：EVAL-UNIQUE-LEAK-999 你看。\n乙：嗯。"
    assert "eval_leak" in filter_reasons(leak, leaks=["EVAL-UNIQUE-LEAK-999"], pii_fn=pii_or_nav)
    tool = '甲：{"route_id": 3}\n乙：啥？'
    assert "tool_or_eval_marker" in filter_reasons(tool, leaks=[], pii_fn=pii_or_nav)


def test_spend_uses_input_and_output() -> None:
    spec = load_contract()
    only_out = spend_cny(prompt_tokens=0, completion_tokens=1_000_000, contract=spec)
    both = spend_cny(prompt_tokens=1_000_000, completion_tokens=1_000_000, contract=spec)
    assert abs(only_out - 2.0) < 1e-9
    assert abs(both - 2.8) < 1e-9


def test_unique_by_first_frame() -> None:
    rows = [
        {"doc_id": "a", "frame": {"frame_id": "a"}, "n_tokens": 10, "split": "train"},
        {"doc_id": "a", "frame": {"frame_id": "a"}, "n_tokens": 7, "split": "train"},
        {"doc_id": "b", "frame": {"frame_id": "b"}, "n_tokens": 3, "split": "valid"},
    ]
    got = unique_by_first_frame(rows)
    assert got["unique_train_tokens"] == 10
    assert got["exposure_train_tokens"] == 17
    assert got["n_duplicate_frame_ids"] == 1


def test_formal_gate_rejects_offline_and_self_certify() -> None:
    spec = load_contract()
    report = validate_spec(
        {
            "license_cleared": True,
            "exclude_cwt2": True,
            "source_id": "zh-pretrain-colloquial-synth-v1",
            "generator": "offline-frame-renderer",
            "model_snapshot": "offline-v1",
            "terms_hash": terms_hash(spec),
            "audit_ok": True,
            "quality_ok": True,
            "isolation_ok": True,
            "blind_review_ok": True,
            "train_shards": [],
            "n_train_tokens": 5_000_000,
            "formal_cpt_eligible": True,
        },
        spec,
    )
    assert report["ok_formal_cpt"] is False
    assert report["license_cleared"] is True
    assert report["provenance_ok"] is False
    assert "generator_not_qwen_plus" in report["reasons"]

    qwen = validate_spec(
        {
            "license_cleared": True,
            "exclude_cwt2": True,
            "source_id": "zh-pretrain-colloquial-synth-qwen-v1",
            "generator": "qwen-plus",
            "model_snapshot": "qwen-plus-2025-12-01",
            "terms_hash": terms_hash(spec),
            "contract_sha256": contract_sha256(),
            "audit_ok": True,
            "quality_ok": True,
            "isolation_ok": True,
            "blind_review_ok": True,
            "train_shards": ["missing.bin"],
            "n_train_tokens": 30_000_000,
            "formal_cpt_eligible": True,
        },
        spec,
    )
    assert qwen["ok_formal_cpt"] is False
    assert "missing_train_shards" in qwen["reasons"] or qwen["n_train_shards"] == 0


def main() -> int:
    test_contract()
    test_frame_idempotency()
    test_frames_and_offline()
    test_filters()
    test_spend_uses_input_and_output()
    test_unique_by_first_frame()
    test_formal_gate_rejects_offline_and_self_certify()
    print(json.dumps({"ok": True, "tests": 7}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
