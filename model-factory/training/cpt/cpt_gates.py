#!/usr/bin/env python3
"""Fail-closed gates for every cumulative-exposure CPT hop of the 51M model."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from common._repo import (
    CORPUS_LM_V1,
    RECIPES_DIR,
    ROOT,
    TOKENIZER_ZH_V1,
    architecture_contracts,
    legacy_weight_contract_sha256,
)
from training.cpt.pretrain_gates import REQUIRED_ROLES

BASE_SCRATCH300M = ROOT / "base" / "mei-1.0-51m-base-scratch300m-v1"
CORPUS_LM_V2 = ROOT / "corpus" / "lm-v2"

CPT_CONTRACT_PATH = RECIPES_DIR / "cpt-1b-contract.json"
CPT_RUNGS_PATH = RECIPES_DIR / "cpt-1b-rungs.json"

PARENT_TOKENS_SEEN = 300_000_485
CPT_CUMULATIVE_EXPOSURE = 1_000_000_000
CPT_INCREMENTAL_EXPOSURE = 699_999_515
CPT_CUMULATIVE_QUOTAS = {
    "wiki": 555_525_480,
    "hq": 327_908_607,
    "structure": 16_203_860,
    "colloquial": 100_362_053,
}
CPT_INCREMENTAL_QUOTAS = {
    "wiki": 388_867_432,
    "hq": 229_535_487,
    "structure": 11_342_703,
    "colloquial": 70_253_893,
}
PARENT_SOURCE_TOKENS_DRAWN = {
    "wiki": 166_658_048,
    "hq": 98_373_120,
    "structure": 4_861_157,
    "colloquial": 30_108_160,
}
PARENT_SOURCE_TOKEN_CURSORS = {
    "wiki": 166_658_048,
    "hq": 98_373_120,
    "structure": 4_862_976,
    "colloquial": 30_108_160,
}
RESET_SOURCE_CURSORS = ("structure", "colloquial")
CONTINUE_SOURCE_CURSORS = ("wiki", "hq")
PARENT_HASHES = {
    "tokenizer_sha256": "fcd07b3d49f5174bb60e81996f4d3f2d55f458f5b8420a271aea59ac5dc58629",
    "weights_sha256": "97f4b537f2869ab315245fd8769eff8a61117b112eb7dce40ebfbd39e2afbd48",
    "state_sha256": "ba9eb459d57d1bad20608a4f11d1f8464394d3ad5fc8837cfa8e89ffe0c91205",
    "schedule_sha256": "e97b8c9ec1d5930369cb18204217c59ac0e3a9729ace0012439599ce9ed39628",
}
ALIGNMENT_TOLERANCE = 2048
HARD_GATES = (500_000_000, 750_000_000, 1_000_000_000)


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cpt_contract() -> dict:
    return _load(CPT_CONTRACT_PATH)


def assert_quota_arithmetic() -> None:
    if sum(CPT_CUMULATIVE_QUOTAS.values()) != CPT_CUMULATIVE_EXPOSURE:
        raise RuntimeError("cumulative 1B quotas do not sum to 1000000000")
    if sum(CPT_INCREMENTAL_QUOTAS.values()) != CPT_INCREMENTAL_EXPOSURE:
        raise RuntimeError("incremental 1B quotas do not sum to 699999515")
    if sum(PARENT_SOURCE_TOKENS_DRAWN.values()) != PARENT_TOKENS_SEEN:
        raise RuntimeError("parent drawn does not sum to 300000485")
    for name in REQUIRED_ROLES:
        got = int(PARENT_SOURCE_TOKENS_DRAWN[name]) + int(CPT_INCREMENTAL_QUOTAS[name])
        if got != int(CPT_CUMULATIVE_QUOTAS[name]):
            raise RuntimeError(f"{name} parent+incremental != cumulative")


def _resolve_cpt_schedule(corpus_dir: Path, rung: str | None = None) -> Path | None:
    if rung:
        exact = corpus_dir / f"schedule-cpt-{rung}.json"
        if exact.is_file():
            return exact
    candidates = sorted(corpus_dir.glob("schedule-cpt*.json"))
    return candidates[0] if len(candidates) == 1 else None


def refuse_cpt_parent(parent_dir: Path | None = None, schedule: dict | None = None) -> str | None:
    parent_dir = Path(parent_dir or BASE_SCRATCH300M)
    if parent_dir.is_file():
        state = parent_dir
        meta = _load(state.with_suffix(".meta.json"))
        if not state.is_file() or not meta:
            return "cpt parent state or metadata is missing"
        if str(meta.get("architecture_id") or "") != "mei-1.0-51m-arch-v1":
            return "cpt parent architecture is not the immutable 51M architecture"
        if int(meta.get("params") or 0) != 51_463_797:
            return "cpt parent parameter count is not 51463797"
        expected_weight = architecture_contracts()["weight_contract_sha256"]
        parent_weight = meta.get("weight_contract_sha256") or legacy_weight_contract_sha256(
            str(meta.get("architecture_sha256") or "")
        )
        if parent_weight != expected_weight:
            return "cpt parent weight contract is not compatible with mei-1.0-51m"
        if schedule and int(meta.get("tokens_seen") or 0) != int(schedule.get("parent_tokens_seen") or 0):
            return "cpt parent tokens_seen does not match schedule parent_tokens_seen"
        expected_hash = str((schedule or {}).get("parent_state_sha256") or "")
        if not expected_hash:
            parent_release = _load(state.parent / "RELEASE.json")
            expected_hash = str(parent_release.get("state_sha256") or "")
        if expected_hash and file_sha256(state) != expected_hash:
            return "cpt parent state hash does not match schedule"
        if not expected_hash:
            return "cpt parent state hash is not pinned by schedule or release"
        return None
    release = _load(parent_dir / "RELEASE.json")
    summary = _load(parent_dir / "summary.json")
    weights = parent_dir / str(release.get("weights") or "mei-1.0-51m-base-scratch300m-v1.npz")
    state = parent_dir / str(release.get("train_state") or "mei-1.0-51m-base-scratch300m-v1-state.npz")
    if not weights.is_file() or not state.is_file():
        return "cpt parent is missing frozen weights or train state"
    if int(summary.get("tokens_seen") or 0) != PARENT_TOKENS_SEEN:
        return f"cpt parent tokens_seen {summary.get('tokens_seen')} != {PARENT_TOKENS_SEEN}"
    if summary.get("allow_repeat"):
        return "cpt parent used repeat epochs"
    if str(summary.get("precision") or "") != "fp32":
        return "cpt parent precision is not fp32"
    expected_weight = architecture_contracts()["weight_contract_sha256"]
    parent_weight = (
        summary.get("weight_contract_sha256")
        or release.get("weight_contract_sha256")
        or legacy_weight_contract_sha256(
            str(summary.get("architecture_sha256") or release.get("architecture_sha256") or "")
        )
    )
    if parent_weight != expected_weight:
        return "cpt parent weight contract is not compatible with mei-1.0-51m"
    drawn = summary.get("source_tokens_drawn") or release.get("source_tokens_drawn") or {}
    for name, want in PARENT_SOURCE_TOKENS_DRAWN.items():
        got = int(drawn.get(name) or 0)
        if got != want:
            return f"cpt parent {name} drawn {got} != {want}"
    tok_sha = file_sha256(TOKENIZER_ZH_V1) if TOKENIZER_ZH_V1.is_file() else ""
    if tok_sha != PARENT_HASHES["tokenizer_sha256"]:
        return "cpt parent tokenizer hash drifted"
    if file_sha256(weights) != PARENT_HASHES["weights_sha256"]:
        return "cpt parent weights hash drifted"
    if file_sha256(state) != PARENT_HASHES["state_sha256"]:
        return "cpt parent state hash drifted"
    scratch_sched = CORPUS_LM_V1 / "schedule-scratch.json"
    if scratch_sched.is_file() and file_sha256(scratch_sched) != PARENT_HASHES["schedule_sha256"]:
        return "lm-v1 schedule-scratch.json drifted; 300M parent contract is broken"
    return None


def refuse_undeclared_hash_migration(ckpt_meta: dict, expected_meta: dict) -> str | None:
    """Ordinary resume (not continuation) must not silently change corpus hashes."""
    for key in ("corpus_sha256", "schedule_sha256", "manifest_sha256", "tokenizer_sha256"):
        if key not in expected_meta:
            continue
        if str(ckpt_meta.get(key) or "") != str(expected_meta.get(key) or ""):
            return f"undeclared {key} migration: ckpt={ckpt_meta.get(key)} expected={expected_meta.get(key)}"
    return None


def refuse_cpt_source(corpus_dir: Path, rung: str) -> str | None:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    if corpus_dir.resolve() == CORPUS_LM_V1.resolve():
        return "cpt refuses .local/artifacts/mei-1.0-51m/exp-000300m/corpus/cpt-delta/lm-v1; continuation consumes .local/artifacts/_legacy/corpus/planned-exp-001000m-lm-v2"
    if (corpus_dir / "schedule.json").is_file():
        return "cpt refuses archived schedule.json name; use schedule-cpt-<rung>.json"
    if (corpus_dir / "schedule-scratch.json").is_file():
        return "cpt corpus must not carry schedule-scratch.json"
    release = _load(corpus_dir / "RELEASE.json")
    schedule_path = _resolve_cpt_schedule(corpus_dir, rung)
    if schedule_path is None:
        return "cpt requires exactly one schedule-cpt*.json"
    schedule = _load(schedule_path)
    mix = _load(corpus_dir / "mix.json")
    if not release or not schedule:
        return "cpt requires RELEASE.json and a schedule-cpt*.json"
    if str(schedule.get("kind") or "") != "cpt":
        return "cpt schedule kind must be cpt"
    if schedule.get("sampler") != "quota_plan":
        return "cpt schedule must use sampler=quota_plan"
    if schedule.get("allow_repeat") is True:
        return "cpt schedule cannot allow repeats"
    parent_tokens = int(schedule.get("parent_tokens_seen") or 0)
    incremental = int(schedule.get("exposure_tokens") or 0)
    cumulative = int(schedule.get("cumulative_exposure_tokens") or 0)
    if parent_tokens <= 0 or incremental <= 0 or parent_tokens + incremental != cumulative:
        return "cpt schedule requires parent + incremental = cumulative exposure"
    parent = str(schedule.get("parent_checkpoint") or "")
    if not parent:
        return "cpt schedule must pin parent_checkpoint"
    target_by_rung = {"600m": 600_000_000, "1b": 1_000_000_000, "2b": 2_000_000_000}
    match = re.fullmatch(r"([1-9][0-9]*)(m|b)", str(rung))
    if match:
        scale = 1_000_000 if match.group(2) == "m" else 1_000_000_000
        target_by_rung[str(rung)] = int(match.group(1)) * scale
    if rung in target_by_rung and cumulative != target_by_rung[rung]:
        return f"cpt schedule cumulative exposure does not match rung {rung}"
    sources = schedule.get("sources") or {}
    if set(sources) != set(REQUIRED_ROLES):
        return "cpt schedule must contain wiki/hq/structure/colloquial"
    quota_sum = 0
    for name in REQUIRED_ROLES:
        got = int((sources.get(name) or {}).get("token_quota") or 0)
        if got <= 0:
            return f"cpt source {name} token_quota must be positive"
        quota_sum += got
        skip = int((sources.get(name) or {}).get("skip_tokens") or 0)
        if skip != 0:
            return f"cpt source {name} must keep skip_tokens=0 and continue via cursor migration"
        if float((sources.get(name) or {}).get("max_epochs") or 1.0) > 1.0:
            return f"cpt source {name} max_epochs exceeds 1"
    if quota_sum != incremental:
        return "cpt source quotas do not sum to incremental exposure"
    stages = schedule.get("curriculum") or []
    if len(stages) != 1:
        return "cpt 1B curriculum must be a single 2048 stage"
    stage = stages[0]
    if int(stage.get("seq_len") or 0) != 2048:
        return "cpt stage seq_len must be 2048"
    if int(stage.get("stop_at_tokens") or 0) != cumulative:
        return "cpt stage stop_at_tokens must equal cumulative exposure"
    if int(stage.get("stage_tokens") or 0) != incremental:
        return "cpt stage_tokens must equal incremental exposure"
    for name in REQUIRED_ROLES:
        quota = int((sources.get(name) or {}).get("token_quota") or 0)
        got = int(((stage.get("sources") or {}).get(name) or {}).get("token_quota") or 0)
        if got != quota:
            return f"cpt stage {name} token_quota mismatch"
    if mix.get("excludes") and "cwt2" not in json.dumps(mix.get("excludes")).lower():
        return "cpt mix must exclude CWT2"
    if release.get("training_mode") != "cpt":
        return "cpt release training_mode must be cpt"
    if int(release.get("cumulative_exposure_tokens") or 0) != cumulative:
        return "cpt release cumulative exposure differs from schedule"
    if "public_distribution_clearance_asserted" not in release and not (
        release.get("license") or release.get("licenses")
    ):
        return "cpt release lacks licence/clearance registration"
    if str(rung) in {"300m", "100m", "pilot-1m"}:
        return f"cpt corpus refuses scratch rung {rung}"
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = ROOT / parent_path
    parent_err = refuse_cpt_parent(parent_path, schedule)
    if parent_err:
        return parent_err
    return None


def refuse_weights_only_continuation(resume_mode: str) -> str | None:
    if resume_mode == "weights_only":
        return "cpt continuation refuses weights_only; Adam and ledger must migrate"
    return None
