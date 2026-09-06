#!/usr/bin/env python3
"""CPT 1B ledger arithmetic and parent-contract unit tests. No training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import CORPUS_LM_V1, ROOT, ensure_formal_on_path

ensure_formal_on_path()
from training.cpt.cpt_gates import (
    CPT_CUMULATIVE_EXPOSURE,
    CPT_CUMULATIVE_QUOTAS,
    CPT_INCREMENTAL_EXPOSURE,
    CPT_INCREMENTAL_QUOTAS,
    PARENT_SOURCE_TOKEN_CURSORS,
    PARENT_SOURCE_TOKENS_DRAWN,
    PARENT_TOKENS_SEEN,
    assert_quota_arithmetic,
    refuse_cpt_parent,
    refuse_undeclared_hash_migration,
    refuse_weights_only_continuation,
)
from common.data import file_sha256
from training.cpt.pretrain_gates import REQUIRED_ROLES, refuse_non_scratch_source


def test_quota_sums() -> None:
    assert_quota_arithmetic()
    assert sum(CPT_CUMULATIVE_QUOTAS.values()) == 1_000_000_000
    assert sum(CPT_INCREMENTAL_QUOTAS.values()) == 699_999_515
    assert sum(PARENT_SOURCE_TOKENS_DRAWN.values()) == PARENT_TOKENS_SEEN
    for name in REQUIRED_ROLES:
        assert PARENT_SOURCE_TOKENS_DRAWN[name] + CPT_INCREMENTAL_QUOTAS[name] == CPT_CUMULATIVE_QUOTAS[name]


def test_parent_release_matches_ledger() -> None:
    release = json.loads((ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1/summary.json").read_text(encoding="utf-8"))
    assert int(summary["tokens_seen"]) == PARENT_TOKENS_SEEN
    assert release["source_tokens_drawn"] == PARENT_SOURCE_TOKENS_DRAWN
    assert summary["source_tokens_drawn"] == PARENT_SOURCE_TOKENS_DRAWN
    assert summary["source_token_cursors"] == PARENT_SOURCE_TOKEN_CURSORS
    err = refuse_cpt_parent()
    assert err is None, err


def test_scratch_line_untouched() -> None:
    assert refuse_non_scratch_source(CORPUS_LM_V1, "300m") is None
    assert not (CORPUS_LM_V1 / "schedule.json").exists()
    scratch = json.loads((CORPUS_LM_V1 / "schedule-scratch.json").read_text(encoding="utf-8"))
    assert scratch["kind"] == "scratch"
    assert int(scratch["exposure_tokens"]) == 300_000_000
    hashes = json.loads((CORPUS_LM_V1 / "hashes.json").read_text(encoding="utf-8"))
    assert file_sha256(CORPUS_LM_V1 / "schedule-scratch.json") == hashes["schedule-scratch.json"]


def test_weights_only_forbidden() -> None:
    assert refuse_weights_only_continuation("weights_only")
    assert refuse_weights_only_continuation("continuation") is None


def test_undeclared_hash_migration_refused() -> None:
    err = refuse_undeclared_hash_migration({"corpus_sha256": "a"}, {"corpus_sha256": "b"})
    assert err and "corpus_sha256" in err
    assert refuse_undeclared_hash_migration({"corpus_sha256": "a"}, {"corpus_sha256": "a"}) is None


def test_contract_matches_gates() -> None:
    contract = json.loads((ROOT / "model-factory/recipes/cpt-1b-contract.json").read_text(encoding="utf-8"))
    assert contract["parent_tokens_seen"] == PARENT_TOKENS_SEEN
    assert contract["cumulative_exposure_tokens"] == CPT_CUMULATIVE_EXPOSURE
    assert contract["incremental_exposure_tokens"] == CPT_INCREMENTAL_EXPOSURE
    assert contract["cumulative_quotas"] == CPT_CUMULATIVE_QUOTAS
    assert contract["incremental_quotas"] == CPT_INCREMENTAL_QUOTAS
    assert contract["lr"]["base"] == 3e-5
    assert contract["lr"]["final"] == 1e-5
    assert contract["allow_repeat"] is False


def test_cpt_source_gate() -> None:
    from training.cpt.cpt_gates import refuse_cpt_source
    from common.data import classify_schedule, resolve_schedule_file

    # The formal live lineage uses the immutable 600M slice.  The historical
    # lm-v2 1B schedule still names a retired parent filename and must not be
    # treated as the current source gate fixture.
    corpus = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000600m/corpus/cpt-delta/lm-v2-cpt-600m"
    err = refuse_cpt_source(corpus, "600m")
    assert err is None, err
    assert refuse_cpt_source(CORPUS_LM_V1, "1b")
    schedule = json.loads((corpus / "schedule-cpt-600m.json").read_text(encoding="utf-8"))
    assert classify_schedule(schedule) == "cpt"
    assert resolve_schedule_file(corpus, "cpt", target_tokens=600_000_000) is not None


def main() -> int:
    test_quota_sums()
    test_parent_release_matches_ledger()
    test_scratch_line_untouched()
    test_weights_only_forbidden()
    test_undeclared_hash_migration_refused()
    test_contract_matches_gates()
    test_cpt_source_gate()
    print({"ok": True, "tests": 7})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
