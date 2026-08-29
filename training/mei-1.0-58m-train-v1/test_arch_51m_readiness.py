from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from architecture import NeedleZh
from checkpoint import validate_expected_meta
from config import NeedleZhConfig
from promote_scratch_base_51m import (
    ARCHITECTURE_ID as PROMOTE_ARCHITECTURE_ID,
    EXPECTED_PARAMS as PROMOTE_EXPECTED_PARAMS,
    MODEL_ID as PROMOTE_MODEL_ID,
    RUN_NAME as PROMOTE_RUN_NAME,
    validate_identity,
)
from run_scratch_curriculum import SCRATCH_PROFILES
from train_common import train_lm_steps


def _windows(count: int = 4, length: int = 16) -> list[dict]:
    rows = []
    for row in range(count):
        ids = [2, *[((row * 17 + i) % 100) + 4 for i in range(length - 2)], 1]
        rows.append({"x": ids, "y": ids[1:] + [1], "mask": [1.0] * length})
    return rows


def test_architecture_meta_refuses_cross_version_resume() -> None:
    legacy = {
        "architecture_id": "mei-1.0-58m-arch-v1",
        "architecture_sha256": "old",
        "params": 58541901,
    }
    current = {
        "architecture_id": "mei-1.0-51m-arch-v1",
        "architecture_sha256": "new",
        "params": 51463797,
    }
    with pytest.raises(ValueError, match="architecture_id mismatch"):
        validate_expected_meta(legacy, current, "strict")


def test_legacy_arch_v2_checkpoint_is_refused() -> None:
    temporary = {
        "architecture_id": "mei-1.0-58m-arch-v2",
        "architecture_sha256": "tmp",
        "params": 51463797,
    }
    current = {
        "architecture_id": "mei-1.0-51m-arch-v1",
        "architecture_sha256": "new",
        "params": 51463797,
    }
    with pytest.raises(ValueError, match="architecture_id mismatch"):
        validate_expected_meta(temporary, current, "strict")


def test_false_mixed_precision_is_rejected() -> None:
    model = NeedleZh(NeedleZhConfig().tiny())
    with pytest.raises(NotImplementedError, match="not a real mixed-precision path"):
        train_lm_steps(model, _windows(), steps=1, lr=1e-4, precision="bf16")


def test_arch_51m_compiled_training_step_is_finite() -> None:
    mx.random.seed(29)
    model = NeedleZh(NeedleZhConfig().tiny())
    result = train_lm_steps(
        model,
        _windows(),
        steps=2,
        lr=1e-4,
        batch_size=2,
        grad_accum=1,
        compile_train=True,
        precision="fp32",
        horizon_tokens=128,
    )
    mx.eval(model.parameters())
    assert result["compile_scope"] == "full_step"
    assert result["tokens_seen"] == 64
    assert result["last_loss"] is not None
    assert float(result["last_loss"]) > 0


def test_51m_curriculum_run_id_is_explicit() -> None:
    profile = SCRATCH_PROFILES["mei-1.0-51m-arch-v1"]
    assert profile["300m_run"] == "pretrain-mei-1.0-51m-base-scratch300m-v1"
    assert profile["recipe"] == "pretrain-51m-rungs.json"
    assert profile["300m_run"] != SCRATCH_PROFILES["mei-1.0-58m-arch-v1"]["300m_run"]


def test_51m_promotion_identity_is_fail_closed() -> None:
    assert PROMOTE_ARCHITECTURE_ID == "mei-1.0-51m-arch-v1"
    assert PROMOTE_MODEL_ID == "mei-1.0-51m-base-scratch300m-v1"
    assert PROMOTE_RUN_NAME == "pretrain-mei-1.0-51m-base-scratch300m-v1"
    assert PROMOTE_EXPECTED_PARAMS == 51_463_797
    assert "58m" not in PROMOTE_ARCHITECTURE_ID
    assert "58m" not in PROMOTE_MODEL_ID
    assert "58m" not in PROMOTE_RUN_NAME

    valid = {
        "architecture_id": PROMOTE_ARCHITECTURE_ID,
        "architecture_sha256": "51m-sha",
        "params": PROMOTE_EXPECTED_PARAMS,
    }
    assert validate_identity(valid, dict(valid)) is None
    legacy = dict(valid, architecture_id="mei-1.0-58m-arch-v1")
    assert "architecture_id mismatch" in str(validate_identity(valid, legacy))


def test_legacy_cpt_remains_paused() -> None:
    root = Path(__file__).resolve().parents[2]
    stop = root / "training/runs/pretrain-1b-cpt-from-scratch300m/STOP"
    assert stop.is_file()
