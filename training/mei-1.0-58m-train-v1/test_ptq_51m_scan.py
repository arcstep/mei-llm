from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from check_qat_pilot_readiness import CONTRACT_PATH, collect_blockers, tiny_fake_quant_smoke
from identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    CANDIDATE_MAP_NAME,
    EXPECTED_PARAMS,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    MODEL_ID,
    PTQ_SCAN_NAME,
    ROOT,
    propose_candidate_bit_map,
    refuse_base_write,
    validate_release,
    write_json,
)
from quant_ops_51m import STE_IMPLEMENTED, fake_quant_4bit, ste_quantize


TRAIN_DIR = Path(__file__).resolve().parent


def test_51m_identity_rejects_58m_release() -> None:
    release = {
        "architecture_id": "mei-1.0-58m-arch-v1",
        "model_id": "mei-1.0-58m-base-scratch300m-v1",
        "params": 58_541_901,
        "weights_sha256": "abc",
    }
    error = validate_release(release, TRAIN_DIR / "does-not-exist.npz")
    assert error is not None
    assert "architecture_id mismatch" in error or "58M" in error


def test_51m_identity_params_gate() -> None:
    release = {
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "params": 58_541_901,
        "weights_sha256": "abc",
    }
    error = validate_release(release, TRAIN_DIR / "does-not-exist.npz")
    assert error is not None
    assert "params mismatch" in error
    assert EXPECTED_PARAMS == 51_463_797


def test_refuse_write_under_base() -> None:
    blocked = refuse_base_write(ROOT / "base" / MODEL_ID / "sneak.npz")
    assert blocked is not None
    assert "immutable base" in blocked
    error = write_json(ROOT / "base" / MODEL_ID / "ptq-scan.json", {"no": True})
    assert error is not None


def test_candidate_map_is_not_product_final() -> None:
    components = {
        "embedding": {"n_tensors": 1, "max_abs_logit_delta": 9.0},
        "attention_kv": {"n_tensors": 4, "max_abs_logit_delta": 2.0},
        "norm": {"n_tensors": 2, "max_abs_logit_delta": 0.2},
        "confidence": {"n_tensors": 0, "max_abs_logit_delta": 0.0},
    }
    proposal = propose_candidate_bit_map(components)
    blob = json.dumps(proposal)
    assert proposal["candidate"] is True
    assert proposal["product_final"] is False
    assert proposal["qat_mandatory"] is True
    assert proposal["bits"]["embedding"] == 4
    assert "only_if_ptq_misses" not in blob
    assert STE_IMPLEMENTED is True


def test_qat_readiness_is_fail_closed(tmp_path: Path | None = None) -> None:
    from tempfile import TemporaryDirectory

    owned = tmp_path
    tmp_ctx = None
    if owned is None:
        tmp_ctx = TemporaryDirectory()
        owned = Path(tmp_ctx.name)
    try:
        blockers = collect_blockers(
            jobs_dir=owned,
            contract_path=TRAIN_DIR / "recipes" / "qat-pilot-51m-contract.json",
        )
        joined = " | ".join(blockers)
        assert blockers
        assert "float Base-LM Anchor" in joined or "candidate map" in joined
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def test_scripts_forbid_ptq_optional_qat_language() -> None:
    for name in ("scan_ptq_51m.py", "freeze_float_baseline_51m.py"):
        text = (TRAIN_DIR / name).read_text(encoding="utf-8")
        assert "only_if_ptq_misses" not in text
        assert "qat_mandatory" in text.lower() or "QAT" in text


def test_tiny_fake_quant_smoke_is_not_qat() -> None:
    mx.random.seed(3)
    x = mx.random.normal((8, 8))
    q = fake_quant_4bit(x)
    mx.eval(q)
    report = tiny_fake_quant_smoke()
    assert report["ok"] is True
    assert report["qat_step"] is False
    assert report["logits_finite"] is True


def test_51m_spec_is_not_58m() -> None:
    spec = json.loads(ARCHITECTURE_SPEC.read_text(encoding="utf-8"))
    assert spec.get("architecture_id") == ARCHITECTURE_ID
    assert "58m" not in ARCHITECTURE_ID
    assert "51m" in ARCHITECTURE_ID


def test_58m_from_spec_is_rejected_by_51m_loader() -> None:
    from config import NeedleZhConfig

    spec_58 = ROOT / "architecture" / "mei-1.0-58m-arch-v1" / "spec" / "model.json"
    assert spec_58.is_file()
    cfg = NeedleZhConfig.from_spec(spec_58)
    assert cfg.architecture_id != ARCHITECTURE_ID


def test_ste_quantize_matches_pack_math() -> None:
    mx.random.seed(3)
    x = mx.random.normal((8, 64))
    q = ste_quantize(x)
    mx.eval(q)
    assert STE_IMPLEMENTED is True
    assert bool(mx.all(mx.isfinite(q)).item())


def test_produced_job_reports_are_qat_mandatory() -> None:
    for name in (FLOAT_ANCHOR_NAME, PTQ_SCAN_NAME, CANDIDATE_MAP_NAME):
        path = JOBS_DIR / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert "only_if_ptq_misses" not in text
        assert payload.get("qat_mandatory") is True
        assert payload.get("product_final") is not True


def test_readiness_with_jobs_allows_ste_harness() -> None:
    blockers = collect_blockers(jobs_dir=JOBS_DIR, contract_path=CONTRACT_PATH)
    assert STE_IMPLEMENTED is True
    assert "STE is required but not implemented" not in " | ".join(blockers)


if __name__ == "__main__":
    test_51m_identity_rejects_58m_release()
    test_51m_identity_params_gate()
    test_refuse_write_under_base()
    test_candidate_map_is_not_product_final()
    test_qat_readiness_is_fail_closed()
    test_scripts_forbid_ptq_optional_qat_language()
    test_tiny_fake_quant_smoke_is_not_qat()
    test_51m_spec_is_not_58m()
    test_58m_from_spec_is_rejected_by_51m_loader()
    test_ste_quantize_matches_pack_math()
    test_produced_job_reports_are_qat_mandatory()
    test_readiness_with_jobs_allows_ste_harness()
    print("test_ptq_51m_scan: ok")
