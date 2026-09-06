#!/usr/bin/env python3
"""Scratch 300M schedule isolation, hashes, and four-role contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import CORPUS_LM_V1, CORPUS_LM_V2, ROOT, ensure_formal_on_path

ensure_formal_on_path()
from training.cpt.cpt_gates import refuse_cpt_source
from common.data import classify_schedule, file_sha256, resolve_schedule_file
from training.cpt.pretrain_gates import (
    REQUIRED_ROLES,
    SCRATCH_EXPOSURE_TOKENS,
    SCRATCH_SOURCE_QUOTAS,
    refuse_non_scratch_source,
    refuse_quota_contract,
)


def test_schedules_split() -> None:
    assert not (CORPUS_LM_V1 / "schedule.json").exists()
    scratch = json.loads((CORPUS_LM_V1 / "schedule-scratch.json").read_text(encoding="utf-8"))
    assert classify_schedule(scratch) == "scratch"
    assert int(scratch["parent_tokens_seen"] or 0) == 0
    assert scratch.get("parent_checkpoint") in (None, "")
    assert scratch.get("sampler") == "quota_plan"
    assert int(scratch["exposure_tokens"]) == SCRATCH_EXPOSURE_TOKENS
    assert set(scratch["sources"]) == set(REQUIRED_ROLES)
    assert "colloquial" in scratch["sources"]
    assert all(int(cfg.get("skip_tokens") or 0) == 0 for cfg in scratch["sources"].values())
    for name, quota in SCRATCH_SOURCE_QUOTAS.items():
        assert int(scratch["sources"][name]["token_quota"]) == quota
    assert refuse_quota_contract(scratch) is None
    mix = json.loads((CORPUS_LM_V1 / "mix.json").read_text(encoding="utf-8"))
    assert mix["schedule_scratch"] == "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/cpt-delta/lm-v1/schedule-scratch.json"
    assert mix["roles_complete"] is True
    assert resolve_schedule_file(CORPUS_LM_V1, "scratch") == CORPUS_LM_V1 / "schedule-scratch.json"
    assert resolve_schedule_file(CORPUS_LM_V1, "cpt") is None
    assert resolve_schedule_file(CORPUS_LM_V1, "none") is None
    assert refuse_non_scratch_source(CORPUS_LM_V1, "300m") is None
    assert resolve_schedule_file(CORPUS_LM_V1, "cpt") is None
    cpt_err = refuse_cpt_source(CORPUS_LM_V2, "1b")
    assert cpt_err == "cpt parent is missing frozen weights or train state"
    assert resolve_schedule_file(CORPUS_LM_V2, "cpt") == CORPUS_LM_V2 / "schedule-cpt-1b.json"
    assert resolve_schedule_file(CORPUS_LM_V2, "scratch") is None


def test_resolve_cpt_schedule_unique_glob() -> None:
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as raw:
        slice_dir = _Path(raw)
        (slice_dir / "schedule-cpt-600m.json").write_text("{}", encoding="utf-8")
        assert resolve_schedule_file(slice_dir, "cpt") == slice_dir / "schedule-cpt-600m.json"
        (slice_dir / "schedule-cpt-1b.json").write_text("{}", encoding="utf-8")
        assert resolve_schedule_file(slice_dir, "cpt") is None


def test_hashes_include_scratch() -> None:
    hashes = json.loads((CORPUS_LM_V1 / "hashes.json").read_text(encoding="utf-8"))
    for name in ("mix.json", "schedule-scratch.json", "manifest.json", "RELEASE.json"):
        assert name in hashes, name
        assert file_sha256(CORPUS_LM_V1 / name) == hashes[name], name
    assert "schedule.json" not in hashes


def test_structure_ledger_not_in_published_tokens() -> None:
    rel = json.loads(
        (CORPUS_LM_V1 / "structure/zh-pretrain-v3/RELEASE.json").read_text(encoding="utf-8")
    )
    ledger = ROOT / rel["unique_ledger"]
    assert ledger.is_file()
    assert "artifacts/mei-1.0-51m/legacy/_legacy/notebook/corpus" in rel["unique_ledger"]
    assert not (CORPUS_LM_V1 / "structure/zh-pretrain-v3/unique-ledger.json").exists()


def test_contract_from_spec() -> None:
    contract = json.loads((ROOT / "model-factory/recipes/pretrain-contract.json").read_text(encoding="utf-8"))
    assert contract["config_loader"] == "NeedleZhConfig.from_spec"
    assert contract["v2_heads_in_default_pretrain"] is False
    assert contract["seq"]["train_seq_len_default"] == 512
    assert contract["seq"]["seq_curriculum"] == [512, 1024, 2048]
    assert contract["seq"]["max_positions"] == 2048
    assert contract["quantization"]["ready"] is False
    assert contract["parent_checkpoint"] is None
    spec = json.loads(
        (ROOT / "models/mei-1.0-51m/architecture/spec/model.json").read_text(encoding="utf-8")
    )
    assert spec["runtime_profile"]["ordinary_window_policy"] == "dynamic_remainder"
    assert spec["runtime_profile"]["stable_prefix_profiles"] == {
        "compact": 1024,
        "standard": 1536,
    }
    assert spec["training_aux"]["mtp"]["export"] is False
    assert contract["contracts"]["weight_geometry"].endswith("spec/model.json#architecture")


def test_current_scratch_unblocked() -> None:
    current = json.loads((ROOT / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["blocked"] == []
    assert current["plan"]["mode"] == "scratch"
    assert current["plan"]["parent_checkpoint"] is None
    assert current["plan"]["roles"] == ["wiki", "hq", "structure", "colloquial"]
    release = json.loads((CORPUS_LM_V1 / "RELEASE.json").read_text(encoding="utf-8"))
    assert release["roles_complete"] is True
    assert release["training_mode"] == "scratch"
    assert release["parent_checkpoint"] is None
    assert int(release["exposure_tokens"]) == SCRATCH_EXPOSURE_TOKENS


def main() -> int:
    test_resolve_cpt_schedule_unique_glob()
    test_schedules_split()
    test_hashes_include_scratch()
    test_structure_ledger_not_in_published_tokens()
    test_contract_from_spec()
    test_current_scratch_unblocked()
    print(json.dumps({"ok": True, "tests": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
