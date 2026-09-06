#!/usr/bin/env python3
"""Pretrain readiness for four-role scratch 300M."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import CORPUS_LM_V1, CORPUS_ZH_PRETRAIN, RECIPES_DIR, ROOT, TOKENIZER_ZH_V1, ensure_formal_on_path

ensure_formal_on_path()
from common.data import classify_schedule, file_sha256, list_token_shards, list_valid_set
from training.cpt.pretrain_gates import (
    MIX_UNIQUE,
    REQUIRED_ROLES,
    SCRATCH_EXPOSURE_TOKENS,
    SCRATCH_SOURCE_QUOTAS,
    refuse_non_scratch_source,
    refuse_quota_contract,
)

FROZEN_TOK_SHA = "fcd07b3d49f5174bb60e81996f4d3f2d55f458f5b8420a271aea59ac5dc58629"
LEDGER = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/corpus/lm-v1/structure/work/zh-pretrain-v3/unique-ledger.json"
EXPECTED_PARAMS = 51_463_797


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def report() -> dict:
    current = _load(ROOT / "CURRENT.json")
    release = _load(CORPUS_LM_V1 / "RELEASE.json")
    mix = _load(CORPUS_LM_V1 / "mix.json")
    hashes = _load(CORPUS_LM_V1 / "hashes.json")
    scratch = _load(CORPUS_LM_V1 / "schedule-scratch.json")
    contract = _load(RECIPES_DIR / "pretrain-contract.json")
    tok_sha = file_sha256(TOKENIZER_ZH_V1) if TOKENIZER_ZH_V1.is_file() else ""
    wiki_train = list_token_shards(CORPUS_ZH_PRETRAIN, "train")
    mix_train = list_token_shards(CORPUS_LM_V1, "train")
    mix_valid = list_token_shards(CORPUS_LM_V1, "valid")
    hash_ok = {}
    for name, expected in hashes.items():
        path = CORPUS_LM_V1 / name
        hash_ok[name] = path.is_file() and file_sha256(path) == expected
    valid_sets = {
        name: bool(list_valid_set(CORPUS_LM_V1, name))
        for name in REQUIRED_ROLES
    }
    gate = refuse_non_scratch_source(CORPUS_LM_V1, "300m")
    quota_err = refuse_quota_contract(scratch)
    cpt_present = (CORPUS_LM_V1 / "schedule.json").is_file()
    smoke_ready = bool(wiki_train) and tok_sha == FROZEN_TOK_SHA and TOKENIZER_ZH_V1.is_file()
    scratch_ready = (
        smoke_ready
        and classify_schedule(scratch) == "scratch"
        and int(scratch.get("parent_tokens_seen") or 0) == 0
        and scratch.get("parent_checkpoint") in (None, "")
        and scratch.get("sampler") == "quota_plan"
        and int(scratch.get("exposure_tokens") or 0) == SCRATCH_EXPOSURE_TOKENS
        and set((scratch.get("sources") or {})) == set(REQUIRED_ROLES)
        and all(int((cfg or {}).get("skip_tokens") or 0) == 0 for cfg in (scratch.get("sources") or {}).values())
        and all(hash_ok.values())
        and LEDGER.is_file()
        and mix.get("roles_complete") is True
        and release.get("roles_complete") is True
        and release.get("training_mode") == "scratch"
        and current.get("plan", {}).get("parent_checkpoint") is None
        and current.get("blocked") in (None, [])
        and not cpt_present
        and gate is None
        and quota_err is None
        and all(valid_sets.values())
        and bool(mix_train)
        and bool(mix_valid)
        and {name: int(mix.get(f"n_{name}_train_tokens") or 0) for name in REQUIRED_ROLES} == MIX_UNIQUE
        and {
            name: int((scratch.get("sources") or {}).get(name, {}).get("token_quota") or 0)
            for name in REQUIRED_ROLES
        }
        == SCRATCH_SOURCE_QUOTAS
    )
    return {
        "smoke_ready": smoke_ready,
        "short_rung_ready": scratch_ready,
        "formal_four_role_ready": scratch_ready,
        "scratch_300m_ready": scratch_ready,
        "tokenizer_sha256_ok": tok_sha == FROZEN_TOK_SHA,
        "expected_params": EXPECTED_PARAMS,
        "wiki_train_shards": len(wiki_train),
        "mix_train_shards": len(mix_train),
        "mix_valid_shards": len(mix_valid),
        "valid_sets_ok": valid_sets,
        "hashes_ok": hash_ok,
        "schedule_scratch_kind": classify_schedule(scratch),
        "cpt_schedule_present": cpt_present,
        "unique_ledger_ok": LEDGER.is_file(),
        "gate": gate,
        "quota_contract": quota_err,
        "current": {
            "stage": current.get("stage"),
            "blocked": current.get("blocked"),
            "base": current.get("base"),
            "runtime": current.get("runtime"),
            "plan": current.get("plan"),
        },
        "release": {
            "roles_complete": release.get("roles_complete"),
            "training_mode": release.get("training_mode"),
            "parent_checkpoint": release.get("parent_checkpoint"),
            "exposure_tokens": release.get("exposure_tokens"),
            "public_distribution_clearance_asserted": release.get("public_distribution_clearance_asserted"),
        },
        "contract": {
            "lm_pretrain": contract.get("config_loader"),
            "v2_heads_default": contract.get("v2_heads_in_default_pretrain"),
            "seq_len_default": (contract.get("seq") or {}).get("train_seq_len_default"),
            "seq_curriculum": (contract.get("seq") or {}).get("seq_curriculum"),
            "max_seq_len": (contract.get("seq") or {}).get("max_positions"),
            "quantization": (contract.get("quantization") or {}).get("product"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-smoke", action="store_true")
    ap.add_argument("--require-short-rung", action="store_true")
    ap.add_argument("--require-formal", action="store_true")
    args = ap.parse_args()
    payload = report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_formal or args.require_short_rung:
        return 0 if payload["scratch_300m_ready"] else 2
    if args.require_smoke:
        return 0 if payload["smoke_ready"] else 1
    return 0 if payload["smoke_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
