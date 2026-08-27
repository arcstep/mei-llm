#!/usr/bin/env python3
"""Admit the frozen 30M mixed-fleet colloquial pack for scratch pretraining.

This is not a claim that the pack is pure qwen-plus or cleared for public
distribution. Admission is for the internal, unlabeled CPT role and requires
the frozen mixed-fleet contract, machine quality gates, isolation, exact token
counts, and explicit product-owner approval recorded in the contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repo_paths import ROOT

CORPUS = ROOT / "corpus/lm-v1"
POOL = CORPUS / "colloquial/zh-pretrain-colloquial-synth-pooled-v1"
CONTRACT_PATH = CORPUS / "colloquial/contract-mixed-v1.json"
DRAFT = ROOT / "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1"
QUALITY_PATH = DRAFT / "reviews/quality.json"
ISOLATION_PATH = DRAFT / "reviews/isolation.json"
APPROVED_PATH = ROOT / "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4/approved-colloquial.json"
CURRENT_PATH = ROOT / "CURRENT.json"
TOKENIZER_SHA256 = "fcd07b3d49f5174bb60e81996f4d3f2d55f458f5b8420a271aea59ac5dc58629"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def token_count(paths: list[Path]) -> int:
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes % 2:
        raise ValueError("uint16 token shard has odd byte length")
    return total_bytes // 2


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def main() -> int:
    contract = load(CONTRACT_PATH)
    quality = load(QUALITY_PATH)
    isolation = load(ISOLATION_PATH)
    overall = quality.get("overall") or {}
    admission = contract.get("admission") or {}
    usage = contract.get("usage") or {}

    train_paths = sorted((POOL / "tokens").glob("colloquial-train-*.bin"))
    valid_paths = sorted((POOL / "tokens").glob("colloquial-valid-*.bin"))
    require(bool(train_paths), "missing pooled colloquial train shards")
    require(bool(valid_paths), "missing pooled colloquial valid shards")
    n_train = token_count(train_paths)
    n_valid = token_count(valid_paths)
    require(n_train == int(overall.get("unique_train_tokens_sum") or 0), "pooled train count != quality ledger")
    require(n_train >= int(admission["requires_unique_tokens"]), "pooled colloquial unique tokens below contract")
    require(bool(overall.get("quality_ok")), "pooled colloquial quality gate failed")
    require(bool(overall.get("hard_ok")), "pooled colloquial hard gate failed")
    require(bool(isolation.get("ok")), "pooled colloquial isolation gate failed")
    require(bool(usage.get("approved_for_internal_scratch_pretraining")), "product-owner approval missing")
    hard = overall.get("hard") or {}
    require(int(hard.get("pii") or 0) == 0, "PII hard gate failed")
    require(int(hard.get("eval_leak") or 0) == 0, "eval leak hard gate failed")
    require(int(hard.get("illegal_or_tool") or 0) == 0, "illegal/tool hard gate failed")
    require(int((quality.get("unique_by_first_frame") or {}).get("n_unique_frame_ids") or 0) == 178720,
            "frame ledger mismatch")

    train_rels = [rel(path) for path in train_paths]
    valid_rels = [rel(path) for path in valid_paths]
    evidence = {
        "contract": rel(CONTRACT_PATH),
        "quality": rel(QUALITY_PATH),
        "isolation": rel(ISOLATION_PATH),
        "approval_basis": usage["approval_basis"],
        "blind_review_required_for_corpus_admission": False,
        "public_distribution_clearance_asserted": False,
    }

    pool_release = load(POOL / "RELEASE.json")
    pool_release.update(
        {
            "status": "active",
            "role": "scratch_pretrain_colloquial",
            "generator": "mixed-bailian-fleet",
            "model_snapshot": "mixed-bailian-fleet",
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "n_duplicate_frame_ids": 0,
            "quality_ok": True,
            "isolation_ok": True,
            "scratch_pretrain_eligible": True,
            "admitted": True,
            "formal_cpt_eligible": False,
            "public_distribution_clearance_asserted": False,
            "train_shards": train_rels,
            "valid_shards": valid_rels,
            "evidence": evidence,
        }
    )
    dump(POOL / "RELEASE.json", pool_release)
    pool_manifest = load(POOL / "manifest.json")
    pool_manifest.update(
        {
            "status": "active",
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "train_shards": train_rels,
            "valid_shards": valid_rels,
            "scratch_pretrain_eligible": True,
        }
    )
    pool_manifest.pop("token_shards", None)
    dump(POOL / "manifest.json", pool_manifest)
    dump(
        POOL / "source-license.json",
        {
            "colloquial": {
                "license_basis": usage["license_basis"],
                "generator": "mixed-bailian-fleet",
                "snapshots": sorted((overall.get("snapshots") or {}).keys()),
                "admitted_for_internal_scratch_pretraining": True,
                "public_distribution_clearance_asserted": False,
                "cwt2": "excluded",
            }
        },
    )
    dump(
        POOL / "admission.json",
        {
            "ok": True,
            "mode": "internal_scratch_pretraining",
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "generator": "mixed-bailian-fleet",
            "quality_ok": True,
            "isolation_ok": True,
            "blind_review_required": False,
            "evidence": evidence,
        },
    )
    pool_hash_names = [
        "README.md",
        "RELEASE.json",
        "manifest.json",
        "source-license.json",
        "admission.json",
        *[path.relative_to(POOL).as_posix() for path in train_paths + valid_paths],
    ]
    dump(POOL / "hashes.json", {name: sha256(POOL / name) for name in pool_hash_names})

    approved = {
        "source_id": pool_release["id"],
        "mode": "internal_scratch_pretraining",
        "generator": "mixed-bailian-fleet",
        "model_snapshot": "mixed-bailian-fleet",
        "n_train_tokens": n_train,
        "n_valid_tokens": n_valid,
        "train_shards": train_rels,
        "valid_shards": valid_rels,
        "quality_ok": True,
        "isolation_ok": True,
        "scratch_pretrain_eligible": True,
        "formal_cpt_eligible": False,
        "public_distribution_clearance_asserted": False,
        "evidence": evidence,
    }
    dump(APPROVED_PATH, approved)

    mix = load(CORPUS / "mix.json")
    sources = mix["sources"]
    sources["colloquial"] = {
        "band": "B",
        "license_basis": usage["license_basis"],
        "excluded_cwt2": True,
        "present": True,
        "n_train_tokens": n_train,
        "n_valid_tokens": n_valid,
        "train_shards": train_rels,
        "valid_shards": valid_rels,
        "generator": "mixed-bailian-fleet",
        "model_snapshot": "mixed-bailian-fleet",
        "scratch_pretrain_eligible": True,
    }
    active_names = ["wiki", "hq", "structure", "colloquial"]
    mix["train_shards"] = [path for name in active_names for path in sources[name]["train_shards"]]
    mix["valid_shards"] = [path for name in active_names for path in sources[name]["valid_shards"]]
    mix["valid_sets"]["colloquial"] = valid_rels
    mix["n_wiki_train_tokens"] = int(sources["wiki"]["n_train_tokens"])
    mix["n_hq_train_tokens"] = int(sources["hq"]["n_train_tokens"])
    mix["n_structure_train_tokens"] = int(sources["structure"]["n_train_tokens"])
    mix["n_colloquial_train_tokens"] = n_train
    total_train = sum(int(sources[name]["n_train_tokens"]) for name in active_names)
    total_valid = sum(int(sources[name].get("n_valid_tokens") or 0) for name in active_names)
    mix["n_train_tokens"] = total_train
    mix["n_unique_train_tokens"] = total_train
    mix["n_valid_tokens"] = total_valid
    mix["roles_complete"] = True
    mix["training_mode"] = "scratch"
    mix.pop("schedule", None)
    mix["schedule_scratch"] = "corpus/lm-v1/schedule-scratch.json"
    mix["colloquial_promoted"] = True
    mix["colloquial_stored"] = {
        "id": pool_release["id"],
        "path": rel(POOL),
        "n_unique_train_tokens": n_train,
        "admitted": True,
        "status": "active",
        "reason": "mixed_fleet_contract_and_machine_gates_passed",
    }
    dump(CORPUS / "mix.json", mix)

    wiki_n = int(sources["wiki"]["n_train_tokens"])
    hq_n = int(sources["hq"]["n_train_tokens"])
    structure_n = int(sources["structure"]["n_train_tokens"])
    require(n_train == 30_108_616, "pooled colloquial unique tokens drifted from frozen 30108616")
    require(structure_n == 4_861_158, "structure unique tokens drifted from frozen 4861158")
    remain = 300_000_000 - structure_n - n_train
    wiki_q = round(remain * wiki_n / (wiki_n + hq_n))
    hq_q = remain - wiki_q
    schedule = {
        "stage_id": "scratch-four-role-300m-v1",
        "kind": "scratch",
        "formal_four_role": True,
        "parent_rung": None,
        "parent_tokens_seen": 0,
        "parent_checkpoint": None,
        "sampler": "quota_plan",
        "sampler_seed": 0,
        "skip_seen_wiki": False,
        "allow_repeat": False,
        "exposure_tokens": 300000000,
        "lr": {
            "kind": "cosine_tokens",
            "base": 0.0003,
            "final": 0.00003,
            "horizon_tokens": 300000000,
        },
        "sources": {
            "wiki": {"token_quota": wiki_q, "skip_tokens": 0, "max_epochs": round(wiki_q / wiki_n, 6)},
            "hq": {"token_quota": hq_q, "skip_tokens": 0, "max_epochs": round(hq_q / hq_n, 6)},
            "structure": {"token_quota": structure_n, "skip_tokens": 0, "max_epochs": 1.0},
            "colloquial": {"token_quota": n_train, "skip_tokens": 0, "max_epochs": 1.0},
        },
        "curriculum": [
            {
                "id": "s1",
                "seq_len": 512,
                "stage_tokens": 150000000,
                "stop_at_tokens": 150000000,
                "batch_size": 8,
                "grad_accum": 1,
                "sources": {
                    "wiki": {"token_quota": 83328822},
                    "hq": {"token_quota": 49186291},
                    "structure": {"token_quota": 2430579},
                    "colloquial": {"token_quota": 15054308},
                },
            },
            {
                "id": "s2",
                "seq_len": 1024,
                "stage_tokens": 100000000,
                "stop_at_tokens": 250000000,
                "batch_size": 2,
                "grad_accum": 1,
                "sources": {
                    "wiki": {"token_quota": 55552548},
                    "hq": {"token_quota": 32790861},
                    "structure": {"token_quota": 1620386},
                    "colloquial": {"token_quota": 10036205},
                },
            },
            {
                "id": "s3",
                "seq_len": 2048,
                "stage_tokens": 50000000,
                "stop_at_tokens": 300000000,
                "batch_size": 1,
                "grad_accum": 1,
                "sources": {
                    "wiki": {"token_quota": 27776274},
                    "hq": {"token_quota": 16395430},
                    "structure": {"token_quota": 810193},
                    "colloquial": {"token_quota": 5018103},
                },
            },
        ],
    }
    dump(CORPUS / "schedule-scratch.json", schedule)
    dump(
        CORPUS / "manifest.json",
        {
            "id": "lm-v1-scratch-four-role-v1",
            "n_train_tokens": total_train,
            "n_unique_train_tokens": total_train,
            "n_valid_tokens": total_valid,
            "roles_complete": True,
            "training_mode": "scratch",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "copies_shards": False,
            "schedule": "corpus/lm-v1/schedule-scratch.json",
        },
    )
    dump(
        CORPUS / "RELEASE.json",
        {
            "release_id": "lm-v1-scratch-four-role-v1",
            "status": "ready_for_internal_scratch_pretraining",
            "training_mode": "scratch",
            "roles_complete": True,
            "n_unique_train_tokens": total_train,
            "tokenizer": "zh-24k-v1",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "exposure_tokens": 300000000,
            "source_quotas": {
                "wiki": wiki_q,
                "hq": hq_q,
                "structure": structure_n,
                "colloquial": n_train,
            },
            "parent_checkpoint": None,
            "copies_shards": False,
            "allow_repeat": False,
            "excludes_cwt2": True,
            "excludes_v2_dirty_structure": True,
            "colloquial_release": pool_release["id"],
            "public_distribution_clearance_asserted": False,
        },
    )
    root_hash_names = ["mix.json", "schedule-scratch.json", "manifest.json", "RELEASE.json"]
    dump(CORPUS / "hashes.json", {name: sha256(CORPUS / name) for name in root_hash_names})

    current = load(CURRENT_PATH)
    current["stage"] = "scratch-pretrain-ready"
    current["blocked"] = []
    current["plan"] = {
        "mode": "scratch",
        "schedule": "corpus/lm-v1/schedule-scratch.json",
        "parent_checkpoint": None,
        "roles": ["wiki", "hq", "structure", "colloquial"],
        "exposure_tokens": 300000000,
        "seq_curriculum": [512, 1024, 2048],
        "source_quotas": {
            "wiki": wiki_q,
            "hq": hq_q,
            "structure": structure_n,
            "colloquial": n_train,
        },
    }
    dump(CURRENT_PATH, current)
    dump(DRAFT / "published.json", {"published_path": rel(POOL), "status": "active"})
    print(
        json.dumps(
            {
                "ok": True,
                "release": "lm-v1-scratch-four-role-v1",
                "n_unique_train_tokens": total_train,
                "n_colloquial_train_tokens": n_train,
                "roles_complete": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
