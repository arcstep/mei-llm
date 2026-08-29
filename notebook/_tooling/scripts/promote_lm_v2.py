#!/usr/bin/env python3
"""Publish immutable corpus/lm-v2 from fresh minors + lm-v1 wiki/HQ tails."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from repo_paths import PUBLISHED_LM_V1, PUBLISHED_LM_V2, ROOT, TOKENIZER_ZH_V1

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training/mei-1.0-58m-train-v1"))
from _repo import ensure_formal_on_path  # noqa: E402

ensure_formal_on_path()
sys.path.insert(0, str(SCRIPTS))

from cpt_gates import (  # noqa: E402
    CPT_INCREMENTAL_EXPOSURE,
    CPT_INCREMENTAL_QUOTAS,
    PARENT_SOURCE_TOKEN_CURSORS,
    PARENT_SOURCE_TOKENS_DRAWN,
    PARENT_TOKENS_SEEN,
    REQUIRED_ROLES,
    assert_quota_arithmetic,
)

CWT2_NEEDLES = ("chinesewebtext", "cwt2")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_count(paths: list[str]) -> int:
    total = 0
    for rel in paths:
        path = ROOT / rel
        n = path.stat().st_size
        if n % 2:
            raise ValueError(f"odd byte length: {rel}")
        total += n // 2
    return total


def rels_from_glob(folder: Path, pattern: str) -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in folder.glob(pattern) if p.is_file())


def refuse_cwt2(payload: dict) -> None:
    sources = payload.get("sources") or {}
    shards = list(payload.get("train_shards") or [])
    for cfg in sources.values():
        if isinstance(cfg, dict):
            shards.extend(cfg.get("train_shards") or [])
            shards.extend(cfg.get("valid_shards") or [])
    blob = " ".join(str(x) for x in shards).lower()
    for needle in CWT2_NEEDLES:
        if needle in blob:
            raise RuntimeError(f"CWT2 marker found in shards: {needle}")


def main() -> int:
    assert_quota_arithmetic()
    mix_v1 = load(PUBLISHED_LM_V1 / "mix.json")
    wiki = dict((mix_v1.get("sources") or {}).get("wiki") or {})
    hq = dict((mix_v1.get("sources") or {}).get("hq") or {})
    if not wiki.get("train_shards") or not hq.get("train_shards"):
        raise RuntimeError("lm-v1 wiki/HQ shards missing")
    struct_rel = load(PUBLISHED_LM_V2 / "structure/zh-pretrain-v4/RELEASE.json")
    col_rel = load(PUBLISHED_LM_V2 / "colloquial/zh-pretrain-colloquial-cpt-v2/RELEASE.json")
    struct_train = list(struct_rel.get("train_shards") or rels_from_glob(PUBLISHED_LM_V2 / "structure/zh-pretrain-v4/tokens", "structure-train-*.bin"))
    struct_valid = list(struct_rel.get("valid_shards") or rels_from_glob(PUBLISHED_LM_V2 / "structure/zh-pretrain-v4/tokens", "structure-valid-*.bin"))
    col_train = list(col_rel.get("train_shards") or rels_from_glob(PUBLISHED_LM_V2 / "colloquial/zh-pretrain-colloquial-cpt-v2/tokens", "colloquial-train-*.bin"))
    col_valid = list(col_rel.get("valid_shards") or rels_from_glob(PUBLISHED_LM_V2 / "colloquial/zh-pretrain-colloquial-cpt-v2/tokens", "colloquial-valid-*.bin"))
    n_struct = token_count(struct_train)
    n_col = token_count(col_train)
    n_struct_valid = token_count(struct_valid)
    n_col_valid = token_count(col_valid)
    if n_struct < CPT_INCREMENTAL_QUOTAS["structure"] + 2048:
        raise RuntimeError(f"fresh structure unique {n_struct} cannot cover quota")
    if n_col < CPT_INCREMENTAL_QUOTAS["colloquial"] + 2048:
        raise RuntimeError(f"fresh colloquial unique {n_col} cannot cover quota")
    wiki_n = int(wiki.get("n_train_tokens") or token_count(wiki["train_shards"]))
    hq_n = int(hq.get("n_train_tokens") or token_count(hq["train_shards"]))
    wiki_remain = wiki_n - int(PARENT_SOURCE_TOKEN_CURSORS["wiki"]) - 1
    hq_remain = hq_n - int(PARENT_SOURCE_TOKEN_CURSORS["hq"]) - 1
    if wiki_remain < CPT_INCREMENTAL_QUOTAS["wiki"]:
        raise RuntimeError("wiki remaining unique cannot cover 1B increment")
    if hq_remain < CPT_INCREMENTAL_QUOTAS["hq"]:
        raise RuntimeError("hq remaining unique cannot cover 1B increment")

    parent_struct_valid = list(((mix_v1.get("sources") or {}).get("structure") or {}).get("valid_shards") or [])
    parent_col_valid = list(((mix_v1.get("sources") or {}).get("colloquial") or {}).get("valid_shards") or [])
    wiki_valid = list(wiki.get("valid_shards") or [])
    hq_valid = list(hq.get("valid_shards") or [])

    mix = {
        "id": "lm-v2-cpt-1b",
        "tokenizer": "zh-24k-v1",
        "parent_release": ["lm-v1", "scratch300m"],
        "copies_shards": False,
        "excludes": ["ChineseWebText2.0", "cwt2", "zh-pretrain-v2/tokens/structure-*", "zh-pretrain-v2/tokens/colloquial-*"],
        "n_wiki_train_tokens": wiki_n,
        "n_hq_train_tokens": hq_n,
        "n_structure_train_tokens": n_struct,
        "n_colloquial_train_tokens": n_col,
        "n_train_tokens": wiki_n + hq_n + n_struct + n_col,
        "n_unique_train_tokens": wiki_n + hq_n + n_struct + n_col,
        "n_valid_tokens": token_count(wiki_valid) + token_count(hq_valid) + n_struct_valid + n_col_valid,
        "roles_complete": True,
        "training_mode": "cpt",
        "allow_repeat": False,
        "sources": {
            "wiki": {
                **{k: wiki[k] for k in ("band", "license", "train_shards", "valid_shards", "n_train_tokens", "n_valid_tokens") if k in wiki},
                "referenced_from": "corpus/lm-v1",
                "continue_cursor": PARENT_SOURCE_TOKEN_CURSORS["wiki"],
            },
            "hq": {
                **{k: hq[k] for k in ("band", "license", "train_shards", "valid_shards", "n_train_tokens", "n_valid_tokens") if k in hq},
                "referenced_from": "corpus/lm-v1",
                "continue_cursor": PARENT_SOURCE_TOKEN_CURSORS["hq"],
            },
            "structure": {
                "band": "B",
                "license": "synthetic JSON Schema / OpenAPI; internal CPT only",
                "referenced_from": "zh-pretrain-v4",
                "n_train_tokens": n_struct,
                "n_valid_tokens": n_struct_valid,
                "train_shards": struct_train,
                "valid_shards": struct_valid,
                "reset_cursor": 0,
                "forbids_v2_dirty_structure": True,
            },
            "colloquial": {
                "band": "B",
                "license_basis": "synthetic internal dialogues; no third-party web scrape; CWT2 excluded",
                "excluded_cwt2": True,
                "present": True,
                "n_train_tokens": n_col,
                "n_valid_tokens": n_col_valid,
                "train_shards": col_train,
                "valid_shards": col_valid,
                "generator": "offline-frame-renderer-v2-expanded",
                "reset_cursor": 0,
                "scratch_pretrain_eligible": False,
                "cpt_pretrain_eligible": True,
                "public_distribution_clearance_asserted": False,
            },
        },
        "valid_sets": {
            "wiki": wiki_valid,
            "hq": hq_valid,
            "structure": struct_valid,
            "colloquial": col_valid,
            "structure_parent": parent_struct_valid,
            "colloquial_parent": parent_col_valid,
        },
        "train_shards": list(wiki["train_shards"]) + list(hq["train_shards"]) + struct_train + col_train,
        "valid_shards": wiki_valid + hq_valid + struct_valid + col_valid,
        "schedule_cpt_1b": "corpus/lm-v2/schedule-cpt-1b.json",
        "published_path": "corpus/lm-v2",
        "lm_v1_immutable": True,
    }
    refuse_cwt2(mix)
    schedule = {
        "stage_id": "cpt-four-role-1b-v1",
        "kind": "cpt",
        "formal_four_role": True,
        "parent_rung": "scratch-300m",
        "parent_tokens_seen": PARENT_TOKENS_SEEN,
        "parent_checkpoint": "base/mei-1.0-58m-base-scratch300m-v1/pretrain-300m-scratch-state.npz",
        "parent_source_tokens_drawn": PARENT_SOURCE_TOKENS_DRAWN,
        "parent_source_token_cursors": PARENT_SOURCE_TOKEN_CURSORS,
        "reset_source_cursors": ["structure", "colloquial"],
        "sampler": "quota_plan",
        "sampler_seed": 1,
        "allow_repeat": False,
        "exposure_tokens": CPT_INCREMENTAL_EXPOSURE,
        "cumulative_exposure_tokens": 1_000_000_000,
        "lr": {
            "kind": "cosine_tokens_segment",
            "base": 0.00003,
            "final": 0.00001,
            "horizon_tokens": CPT_INCREMENTAL_EXPOSURE,
            "token_offset": PARENT_TOKENS_SEEN,
        },
        "sources": {
            name: {"token_quota": quota, "skip_tokens": 0, "max_epochs": 1.0}
            for name, quota in CPT_INCREMENTAL_QUOTAS.items()
        },
        "curriculum": [
            {
                "id": "cpt1",
                "seq_len": 2048,
                "stage_tokens": CPT_INCREMENTAL_EXPOSURE,
                "stop_at_tokens": 1_000_000_000,
                "batch_size": 1,
                "grad_accum": 1,
                "sources": {name: {"token_quota": quota} for name, quota in CPT_INCREMENTAL_QUOTAS.items()},
            }
        ],
    }
    refuse_cwt2(schedule)
    if set(schedule["sources"]) != set(REQUIRED_ROLES):
        raise RuntimeError("schedule missing roles")
    manifest = {
        "id": "lm-v2-cpt-1b",
        "n_train_tokens": mix["n_train_tokens"],
        "n_unique_train_tokens": mix["n_unique_train_tokens"],
        "n_valid_tokens": mix["n_valid_tokens"],
        "roles_complete": True,
        "training_mode": "cpt",
        "tokenizer_sha256": sha256(TOKENIZER_ZH_V1),
        "copies_shards": False,
        "schedule": "corpus/lm-v2/schedule-cpt-1b.json",
    }
    release = {
        "id": "lm-v2-cpt-1b",
        "training_mode": "cpt",
        "roles_complete": True,
        "parent_checkpoint": "base/mei-1.0-58m-base-scratch300m-v1",
        "parent_tokens_seen": PARENT_TOKENS_SEEN,
        "exposure_tokens": CPT_INCREMENTAL_EXPOSURE,
        "cumulative_exposure_tokens": 1_000_000_000,
        "incremental_quotas": CPT_INCREMENTAL_QUOTAS,
        "allow_repeat": False,
        "excludes": mix["excludes"],
        "public_distribution_clearance_asserted": False,
        "lm_v1_immutable": True,
        "not_a_claim": "Fresh structure/colloquial are internal unlabeled CPT shards, not a public speech corpus.",
    }
    PUBLISHED_LM_V2.mkdir(parents=True, exist_ok=True)
    dump(PUBLISHED_LM_V2 / "mix.json", mix)
    dump(PUBLISHED_LM_V2 / "schedule-cpt-1b.json", schedule)
    dump(PUBLISHED_LM_V2 / "manifest.json", manifest)
    dump(PUBLISHED_LM_V2 / "RELEASE.json", release)
    hashes = {
        name: sha256(PUBLISHED_LM_V2 / name)
        for name in ("mix.json", "schedule-cpt-1b.json", "manifest.json", "RELEASE.json")
    }
    dump(PUBLISHED_LM_V2 / "hashes.json", hashes)
    readme = """# corpus/lm-v2

Immutable CPT consumption atlas for mei-1.0-58m 300M→1B.

- Wiki/HQ shards are referenced from `corpus/lm-v1` and continue from the scratch300m cursors.
- Structure/colloquial shards are fresh (`zh-pretrain-v4`, `colloquial-cpt-v2`); parent minor shards are not resampled.
- Trainer contract: `schedule-cpt-1b.json` (`sampler=quota_plan`, `allow_repeat=false`).
- Do not restore archived `schedule.json` onto this root.
- `CURRENT.corpus` stays `corpus/lm-v1` until a later product decision; CPT training passes `--corpus-dir corpus/lm-v2`.
"""
    (PUBLISHED_LM_V2 / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "n_structure_train_tokens": n_struct,
                "n_colloquial_train_tokens": n_col,
                "wiki_remaining": wiki_remain,
                "hq_remaining": hq_remain,
                "hashes": hashes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
