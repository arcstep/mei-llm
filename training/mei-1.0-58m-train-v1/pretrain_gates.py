#!/usr/bin/env python3
"""Fail-closed gates for the scratch-pretraining line."""

from __future__ import annotations

import json
from pathlib import Path

from _repo import CORPUS_LM_V1, ROOT

DIRTY_V2 = ROOT / "notebook/archive/corpus/zh-pretrain-v2"
REQUIRED_ROLES = ("wiki", "hq", "structure", "colloquial")
SCRATCH_EXPOSURE_TOKENS = 300_000_000
SCRATCH_SOURCE_QUOTAS = {
    "wiki": 166_657_644,
    "hq": 98_372_582,
    "structure": 4_861_158,
    "colloquial": 30_108_616,
}
SCRATCH_STAGE_QUOTAS = {
    "s1": {
        "seq_len": 512,
        "stage_tokens": 150_000_000,
        "stop_at_tokens": 150_000_000,
        "sources": {
            "wiki": 83_328_822,
            "hq": 49_186_291,
            "structure": 2_430_579,
            "colloquial": 15_054_308,
        },
    },
    "s2": {
        "seq_len": 1024,
        "stage_tokens": 100_000_000,
        "stop_at_tokens": 250_000_000,
        "sources": {
            "wiki": 55_552_548,
            "hq": 32_790_861,
            "structure": 1_620_386,
            "colloquial": 10_036_205,
        },
    },
    "s3": {
        "seq_len": 2048,
        "stage_tokens": 50_000_000,
        "stop_at_tokens": 300_000_000,
        "sources": {
            "wiki": 27_776_274,
            "hq": 16_395_430,
            "structure": 810_193,
            "colloquial": 5_018_103,
        },
    },
}
MIX_UNIQUE = {
    "wiki": 649_904_474,
    "hq": 383_617_452,
    "structure": 4_861_158,
    "colloquial": 30_108_616,
}


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_non_scratch_source(corpus_dir: Path, rung: str) -> str | None:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    if corpus_dir.resolve() == DIRTY_V2.resolve() or (corpus_dir / "BLOCKED-for-1b.json").is_file():
        return "scratch pretraining refuses archived zh-pretrain-v2 and its dirty structure source"
    if corpus_dir.resolve() != CORPUS_LM_V1.resolve():
        return None
    release_path = corpus_dir / "RELEASE.json"
    schedule_path = corpus_dir / "schedule-scratch.json"
    if not release_path.is_file() or not schedule_path.is_file():
        return "scratch pretraining requires corpus/lm-v1 RELEASE.json and schedule-scratch.json"
    if (corpus_dir / "schedule.json").is_file():
        return "scratch pretraining refuses active CPT schedule.json; keep continuation plans in archive"
    release = _load(release_path)
    schedule = _load(schedule_path)
    if release.get("training_mode") != "scratch" or release.get("parent_checkpoint") is not None:
        return "active release is not a parent-free scratch release"
    if not release.get("roles_complete"):
        return "scratch release roles are incomplete"
    if schedule.get("kind") != "scratch" or int(schedule.get("parent_tokens_seen") or 0) != 0:
        return "active schedule is not parent-free scratch"
    if schedule.get("parent_checkpoint") not in (None, ""):
        return "scratch schedule cannot name a parent checkpoint"
    if schedule.get("allow_repeat") is True:
        return "scratch schedule cannot allow repeats"
    if str(schedule.get("sampler") or "") != "quota_plan":
        return "scratch schedule must use sampler=quota_plan"
    if int(schedule.get("exposure_tokens") or 0) != SCRATCH_EXPOSURE_TOKENS:
        return "scratch schedule exposure_tokens must be 300000000"
    sources = schedule.get("sources") or {}
    if set(sources) != set(REQUIRED_ROLES):
        return "scratch schedule must contain wiki/hq/structure/colloquial"
    if any(int((cfg or {}).get("skip_tokens") or 0) != 0 for cfg in sources.values()):
        return "scratch schedule cannot skip previously seen tokens"
    for name, quota in SCRATCH_SOURCE_QUOTAS.items():
        got = int((sources.get(name) or {}).get("token_quota") or 0)
        if got != quota:
            return f"scratch source {name} token_quota={got} != {quota}"
    if sum(SCRATCH_SOURCE_QUOTAS.values()) != SCRATCH_EXPOSURE_TOKENS:
        return "scratch source quotas do not sum to 300000000"
    contract_error = refuse_quota_contract(schedule)
    if contract_error:
        return contract_error
    if str(rung) == "1b" and int(schedule.get("exposure_tokens") or 0) < 1_000_000_000:
        return "1b rung exceeds the frozen 300M scratch exposure; refuse rather than repeat tokens"
    return None


def refuse_quota_contract(schedule: dict) -> str | None:
    stages = schedule.get("curriculum") or []
    if len(stages) != 3:
        return "scratch curriculum must have three stages (512, 1024, 2048)"
    seen_ids: list[str] = []
    total_by_source = {name: 0 for name in REQUIRED_ROLES}
    prev_stop = 0
    for stage, (stage_id, spec) in zip(stages, SCRATCH_STAGE_QUOTAS.items()):
        got_id = str(stage.get("id") or "")
        if got_id != stage_id:
            return f"curriculum stage id {got_id!r} != {stage_id!r}"
        if int(stage.get("seq_len") or 0) != int(spec["seq_len"]):
            return f"{stage_id} seq_len mismatch"
        if int(stage.get("stage_tokens") or 0) != int(spec["stage_tokens"]):
            return f"{stage_id} stage_tokens mismatch"
        if int(stage.get("stop_at_tokens") or 0) != int(spec["stop_at_tokens"]):
            return f"{stage_id} stop_at_tokens mismatch"
        if int(spec["stop_at_tokens"]) != prev_stop + int(spec["stage_tokens"]):
            return f"{stage_id} stop_at_tokens is not cumulative"
        src = (stage.get("sources") or {})
        stage_sum = 0
        for name in REQUIRED_ROLES:
            quota = int((src.get(name) or {}).get("token_quota") or src.get(name) or 0)
            if quota != int(spec["sources"][name]):
                return f"{stage_id} {name} token_quota mismatch"
            total_by_source[name] += quota
            stage_sum += quota
        if stage_sum != int(spec["stage_tokens"]):
            return f"{stage_id} source quotas do not sum to stage_tokens"
        seen_ids.append(got_id)
        prev_stop = int(spec["stop_at_tokens"])
    if seen_ids != ["s1", "s2", "s3"]:
        return "curriculum must be s1, s2, s3"
    for name, quota in SCRATCH_SOURCE_QUOTAS.items():
        if total_by_source[name] != quota:
            return f"curriculum {name} quotas do not sum to the 300M source quota"
    if prev_stop != SCRATCH_EXPOSURE_TOKENS:
        return "curriculum stop_at_tokens must end at 300000000"
    return None


def refuse_dirty_v2_for_1b(corpus_dir: Path, rung: str) -> str | None:
    """Compatibility alias used by archived notebook runners."""
    return refuse_non_scratch_source(corpus_dir, rung)


def write_v2_block() -> Path:
    path = DIRTY_V2 / "BLOCKED-for-1b.json"
    payload = {
        "blocked_for": ["scratch", "1b", "2b", "10b", "cpt"],
        "reason": "archived dirty zh-pretrain-v2 is not a scratch or 1B source",
        "rewrite_release": False,
        "successor": "corpus/lm-v1",
        "init_from": None,
        "init_mode": "scratch",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
