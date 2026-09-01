#!/usr/bin/env python3
"""Rebuild the live lm-v1 atlas from published serving packs.

Does not copy token bins. Process ledgers go to notebook assemble work.
Writing live mix/schedule/RELEASE requires --rewrite-atlas.
Does not read archived v2 mix. Does not overwrite id=lm-v1, colloquial_stored,
or fail-closed fields unless a formal colloquial gate actually passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from repo_paths import (
    APPROVED_COLLOQUIAL,
    ASSEMBLE_V4,
    CORPUS_ZH_PRETRAIN,
    CORPUS_ZH_PRETRAIN_HQ,
    CORPUS_ZH_PRETRAIN_V3,
    CORPUS_ZH_PRETRAIN_V4,
    MODEL_MEI_51M,
    ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_MEI_51M))

from colloquial_synth_lib import load_contract  # noqa: E402
from data import file_sha256  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from validate_approved_colloquial import validate_spec  # noqa: E402
from zh_pretrain_ingest import dump_json, rel  # noqa: E402

WIKI_TRAIN = 649_904_474
WIKI_VALID = 34_610_229
WIKI_SKIP = 300_000_000
HQ_TRAIN = 383_617_452
HQ_VALID = 3_778_466


def sha_text(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def glob_bins(pack: Path, pattern: str) -> list[str]:
    return [rel(p, ROOT) for p in sorted((pack / "tokens").glob(pattern))]


def shards_or_glob(existing: list[str] | None, pack: Path, pattern: str) -> list[str]:
    kept = [s for s in (existing or []) if (ROOT / s).is_file()]
    if kept:
        return kept
    return glob_bins(pack, pattern)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colloquial-manifest", type=Path, default=None)
    ap.add_argument(
        "--rewrite-atlas",
        action="store_true",
        help="Write live corpus/lm-v1 mix/schedule/RELEASE. Default only updates assemble work.",
    )
    args = ap.parse_args()
    dest = CORPUS_ZH_PRETRAIN_V4
    dest.mkdir(parents=True, exist_ok=True)
    assemble = ASSEMBLE_V4
    assemble.mkdir(parents=True, exist_ok=True)
    tok = ZhTokenizerV1()
    existing_mix = load_json(dest / "mix.json")
    existing_release = load_json(dest / "RELEASE.json")
    existing_man = load_json(dest / "manifest.json")
    live_exists = bool(existing_mix)

    wiki_src = (existing_mix.get("sources") or {}).get("wiki") or {}
    hq_src = (existing_mix.get("sources") or {}).get("hq") or {}
    struct_src = (existing_mix.get("sources") or {}).get("structure") or {}
    wiki_train = shards_or_glob(wiki_src.get("train_shards") or existing_mix.get("train_shards"), CORPUS_ZH_PRETRAIN, "train-*.bin")
    wiki_valid = shards_or_glob(wiki_src.get("valid_shards") or existing_mix.get("valid_shards"), CORPUS_ZH_PRETRAIN, "valid-*.bin")
    hq_train = shards_or_glob(hq_src.get("train_shards"), CORPUS_ZH_PRETRAIN_HQ, "hq-train-*.bin")
    hq_valid = shards_or_glob(hq_src.get("valid_shards"), CORPUS_ZH_PRETRAIN_HQ, "hq-valid-*.bin")
    struct_train = shards_or_glob(struct_src.get("train_shards"), CORPUS_ZH_PRETRAIN_V3, "structure-train-*.bin")
    struct_valid = shards_or_glob(struct_src.get("valid_shards"), CORPUS_ZH_PRETRAIN_V3, "structure-valid-*.bin")
    v3_man = load_json(CORPUS_ZH_PRETRAIN_V3 / "manifest.json")
    n_struct_train = int(struct_src.get("n_train_tokens") or v3_man.get("n_unique_train_tokens") or 0)
    n_struct_valid = int(struct_src.get("n_valid_tokens") or v3_man.get("n_valid_tokens") or 0)

    colloquial = {
        "present": False,
        "reason": "no_license_cleared_source",
        "train_shards": [],
        "n_train_tokens": 0,
    }
    man_path = args.colloquial_manifest or APPROVED_COLLOQUIAL
    if not Path(man_path).is_absolute():
        man_path = ROOT / man_path
    if man_path.is_file():
        spec = json.loads(man_path.read_text(encoding="utf-8"))
        gate = validate_spec(spec, load_contract())
        if spec.get("source_id") in {"cwt2", "ChineseWebText2.0", "CASIA-LM/ChineseWebText2.0"}:
            colloquial = {
                "present": False,
                "reason": "cwt2_excluded_until_license_verified",
                "train_shards": [],
                "n_train_tokens": 0,
            }
        elif gate["ok_formal_cpt"]:
            shards = [s for s in list(spec.get("train_shards") or []) if (ROOT / s).is_file()]
            colloquial = {
                "present": bool(shards),
                "reason": spec.get("license") or "registered",
                "train_shards": shards,
                "valid_shards": [s for s in list(spec.get("valid_shards") or []) if (ROOT / s).is_file()],
                "n_train_tokens": int(spec.get("n_train_tokens") or 0),
                "n_valid_tokens": int(spec.get("n_valid_tokens") or 0),
                "source_id": spec.get("source_id"),
                "generator": spec.get("generator"),
                "model_snapshot": spec.get("model_snapshot"),
                "gate": gate,
            }
        else:
            colloquial = {
                "present": False,
                "reason": "approved_colloquial_not_formal:" + ",".join(gate["reasons"] or ["unknown"]),
                "engineering_candidate": bool(gate.get("ok_candidate")),
                "train_shards": [s for s in list(spec.get("train_shards") or []) if (ROOT / s).is_file()],
                "valid_shards": [s for s in list(spec.get("valid_shards") or []) if (ROOT / s).is_file()],
                "n_train_tokens": int(spec.get("n_train_tokens") or 0),
                "n_valid_tokens": int(spec.get("n_valid_tokens") or 0),
                "source_id": spec.get("source_id"),
                "generator": spec.get("generator"),
                "model_snapshot": spec.get("model_snapshot"),
                "gate": gate,
            }

    roles_complete = bool(colloquial.get("present"))
    if not roles_complete:
        roles_complete = False
        if existing_mix:
            roles_complete = bool(existing_mix.get("roles_complete")) and bool(colloquial.get("present"))
    n_col_admitted = int(colloquial.get("n_train_tokens") or 0) if roles_complete else 0
    remaining_wiki = max(0, WIKI_TRAIN - WIKI_SKIP)
    unique = remaining_wiki + (HQ_TRAIN if hq_train else 0) + n_struct_train + n_col_admitted
    weight_den = remaining_wiki + (HQ_TRAIN if hq_train else 0) + n_struct_train + n_col_admitted

    def w(n: int) -> float:
        return round(n / weight_den, 6) if weight_den else 0.0

    sources = {
        "wiki": {
            "band": "A",
            "license": "CC BY-SA 3.0/4.0 + GFDL",
            "referenced_from": "zh-pretrain-v0",
            "n_train_tokens": WIKI_TRAIN,
            "n_unique_remaining_after_300m": remaining_wiki,
            "skip_tokens_after_300m": WIKI_SKIP,
            "train_shards": wiki_train,
            "valid_shards": wiki_valid,
        },
        "hq": {
            "band": "C",
            "license": "ODC-By-1.0 + Common-Crawl-ToU",
            "referenced_from": "language/hq",
            "n_train_tokens": HQ_TRAIN if hq_train else 0,
            "n_valid_tokens": HQ_VALID if hq_valid else 0,
            "train_shards": hq_train,
            "valid_shards": hq_valid,
        },
        "structure": {
            "band": "B",
            "license": "synthetic JSON Schema / OpenAPI; registered tool descriptions",
            "referenced_from": "zh-pretrain-v3",
            "n_train_tokens": n_struct_train,
            "n_valid_tokens": n_struct_valid,
            "train_shards": struct_train,
            "valid_shards": struct_valid,
            "forbids_v2_dirty_structure": True,
        },
        "colloquial": existing_mix.get("sources", {}).get("colloquial")
        if existing_mix.get("sources") and not roles_complete
        else {
            "band": "B",
            "excluded_cwt2": True,
            "present": bool(colloquial.get("present")),
            "reason": colloquial.get("reason"),
            "n_train_tokens": int(colloquial.get("n_train_tokens") or 0) if roles_complete else 0,
            "n_valid_tokens": int(colloquial.get("n_valid_tokens") or 0) if roles_complete else 0,
            "train_shards": list(colloquial.get("train_shards") or []) if roles_complete else [],
            "valid_shards": list(colloquial.get("valid_shards") or []) if roles_complete else [],
            "generator": colloquial.get("generator"),
            "model_snapshot": colloquial.get("model_snapshot"),
            "engineering_candidate": bool(colloquial.get("engineering_candidate")),
        },
    }
    if not roles_complete and existing_mix.get("sources", {}).get("colloquial"):
        sources["colloquial"] = dict(existing_mix["sources"]["colloquial"])
        sources["colloquial"]["present"] = False
        sources["colloquial"]["n_train_tokens"] = int(sources["colloquial"].get("n_train_tokens") or 0)
    sched_sources = {
        "wiki": {"weight": w(remaining_wiki), "skip_tokens": WIKI_SKIP, "max_epochs": 1.0},
        "hq": {"weight": w(HQ_TRAIN if hq_train else 0), "skip_tokens": 0, "max_epochs": 1.0},
        "structure": {"weight": w(n_struct_train), "skip_tokens": 0, "max_epochs": 1.0},
    }
    if roles_complete:
        sched_sources["colloquial"] = {
            "weight": w(int(colloquial.get("n_train_tokens") or 0)),
            "skip_tokens": 0,
            "max_epochs": 1.0,
        }
    schedule = {
        "stage_id": "cpt-v2-independent-after-300m",
        "kind": "cpt",
        "formal_four_role": True,
        "parent_rung": "300m",
        "parent_tokens_seen": 300_000_000,
        "parent_checkpoint": "mei-1.0-51m-base-cpt300m-v1",
        "sampler_seed": 0,
        "skip_seen_wiki": True,
        "allow_repeat": False,
        "sources": sched_sources,
    }
    scratch_den = WIKI_TRAIN + (HQ_TRAIN if hq_train else 0) + n_struct_train + n_col_admitted

    def w_scratch(n: int) -> float:
        return round(n / scratch_den, 6) if scratch_den else 0.0

    scratch_sources = {
        "wiki": {"weight": w_scratch(WIKI_TRAIN), "skip_tokens": 0, "max_epochs": 1.0},
        "hq": {"weight": w_scratch(HQ_TRAIN if hq_train else 0), "skip_tokens": 0, "max_epochs": 1.0},
        "structure": {"weight": w_scratch(n_struct_train), "skip_tokens": 0, "max_epochs": 1.0},
    }
    if roles_complete:
        scratch_sources["colloquial"] = {
            "weight": w_scratch(int(colloquial.get("n_train_tokens") or 0)),
            "skip_tokens": 0,
            "max_epochs": 1.0,
        }
    scratch_schedule = {
        "stage_id": "scratch-four-role" if roles_complete else "scratch-three-role-wiki-hq-structure",
        "kind": "scratch",
        "formal_four_role": bool(roles_complete),
        "parent_rung": None,
        "parent_tokens_seen": 0,
        "parent_checkpoint": None,
        "sampler_seed": 0,
        "skip_seen_wiki": False,
        "allow_repeat": False,
        "sources": scratch_sources,
    }
    mix_id = existing_mix.get("id") or "lm-v1"
    if mix_id == "zh-pretrain-v4":
        mix_id = "lm-v1"
    mix = {
        "id": mix_id,
        "tokenizer": "zh-24k-v1",
        "parent_release": existing_mix.get("parent_release") or ["zh-pretrain-v2", "zh-pretrain-v3"],
        "copies_shards": False,
        "excludes": existing_mix.get("excludes")
        or [
            "ChineseWebText2.0",
            "cwt2",
            "zh-pretrain-v2/tokens/structure-*",
            "zh-pretrain-v2/tokens/colloquial-*",
        ],
        "n_train_tokens": unique,
        "n_unique_train_tokens": unique,
        "n_valid_tokens": WIKI_VALID,
        "n_wiki_train_tokens": WIKI_TRAIN,
        "n_hq_train_tokens": HQ_TRAIN if hq_train else 0,
        "n_colloquial_train_tokens": n_col_admitted,
        "n_structure_train_tokens": n_struct_train,
        "roles_complete": roles_complete,
        "train_shards": wiki_train + hq_train + struct_train + (list(colloquial.get("train_shards") or []) if roles_complete else []),
        "valid_shards": wiki_valid,
        "valid_sets": {
            "wiki": wiki_valid,
            "hq": hq_valid,
            "structure": struct_valid,
            "colloquial": list(colloquial.get("valid_shards") or []) if roles_complete else [],
        },
        "sources": sources,
        "schedule": rel(dest / "schedule.json", ROOT),
        "schedule_scratch": rel(dest / "schedule-scratch.json", ROOT),
        "published_path": existing_mix.get("published_path") or "corpus/lm-v1",
        "colloquial_promoted": True if roles_complete else False,
        "colloquial_stored": existing_mix.get("colloquial_stored")
        or {
            "id": "zh-pretrain-colloquial-synth-pooled-v1",
            "path": "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1",
            "admitted": bool(roles_complete),
            "status": "admitted" if roles_complete else "not_admitted",
        },
    }
    if not roles_complete and existing_mix.get("colloquial_stored"):
        mix["colloquial_stored"] = existing_mix["colloquial_stored"]
        mix["colloquial_promoted"] = bool(existing_mix.get("colloquial_promoted"))
        mix["n_colloquial_train_tokens"] = int(existing_mix.get("n_colloquial_train_tokens") or 0)
        mix["roles_complete"] = False

    source_license = {
        "wiki": {"license": "CC BY-SA 3.0/4.0 + GFDL", "role": "cpt_encyclopedia", "admitted": True},
        "hq": {"license": "ODC-By-1.0 + Common-Crawl-ToU", "role": "cpt_hq_web", "admitted": True},
        "structure": {"license": "v3 clean synthetic schema", "role": "cpt_structure_tech", "admitted": True},
        "colloquial": {
            "license": None if not roles_complete else colloquial.get("reason"),
            "role": "cpt_colloquial_synth",
            "admitted": bool(roles_complete),
            "engineering_candidate": bool(colloquial.get("engineering_candidate")),
            "cwt2": "excluded_until_license_verified",
        },
    }
    source_role = {
        "wiki": {"role": "cpt_encyclopedia", "band": "A"},
        "hq": {"role": "cpt_hq_web", "band": "C"},
        "structure": {"role": "cpt_structure_tech", "band": "B"},
        "colloquial": {"role": "cpt_colloquial_synth", "band": "B", "present": bool(roles_complete)},
    }
    unique_ledger = {
        "unique_train_tokens": unique,
        "exposure_train_tokens": unique,
        "allow_repeat": False,
        "by_source": {
            "wiki_remaining_after_300m": remaining_wiki,
            "hq": HQ_TRAIN if hq_train else 0,
            "structure_v3": n_struct_train,
            "colloquial": n_col_admitted,
        },
        "valid": {
            "wiki": WIKI_VALID,
            "hq": HQ_VALID if hq_valid else 0,
            "structure_v3": n_struct_valid,
            "colloquial": int(colloquial.get("n_valid_tokens") or 0) if roles_complete else 0,
        },
        "parent_checkpoint": "mei-1.0-51m-base-cpt300m-v1",
        "tokenizer": "zh-24k-v1",
        "note": "unique counts first-seen shards; do not multiply epochs. CWT2 excluded. v2 dirty structure excluded.",
    }
    release = {
        "release_id": existing_release.get("release_id") or "zh-pretrain-v4",
        "immutable_v2": True,
        "immutable_v3": True,
        "promote_structure": True,
        "structure_source": "zh-pretrain-v3",
        "excludes_cwt2": True,
        "excludes_v2_dirty_structure": True,
        "roles_complete": roles_complete,
        "formal_cpt": "ready" if roles_complete else "fail_closed",
        "fail_closed_reason": None if roles_complete else (
            existing_release.get("fail_closed_reason") or "missing_license_cleared_colloquial_source"
        ),
        "n_unique_train_tokens": unique,
        "n_unique_remaining_wiki": remaining_wiki,
        "tokenizer": "zh-24k-v1",
        "tokenizer_sha256": tok.model_sha256,
        "parent_checkpoint": "mei-1.0-51m-base-cpt300m-v1",
        "copies_shards": False,
        "allow_repeat": False,
    }
    if not roles_complete and existing_release:
        release["roles_complete"] = False
        release["formal_cpt"] = existing_release.get("formal_cpt") or "fail_closed"
        release["fail_closed_reason"] = existing_release.get("fail_closed_reason") or release["fail_closed_reason"]
        if existing_release.get("n_unique_train_tokens") is not None:
            release["n_unique_train_tokens"] = existing_release["n_unique_train_tokens"]

    dump_json(assemble / "source-license.json", source_license)
    dump_json(assemble / "source-role.json", source_role)
    dump_json(assemble / "unique-ledger.json", unique_ledger)
    if man_path.is_file() and man_path.resolve() != (assemble / "approved-colloquial.json").resolve():
        dump_json(assemble / "approved-colloquial.json", json.loads(man_path.read_text(encoding="utf-8")))
    dump_json(
        assemble / "receipt.json",
        {
            "kind": "assemble-pointer",
            "published_path": "corpus/lm-v1",
            "writable": False,
            "note": "Process ledgers live here. Trainer consumes corpus/lm-v1, not this directory.",
        },
    )

    wrote_live = False
    if live_exists and not args.rewrite_atlas:
        print(
            json.dumps(
                {
                    "ok": True,
                    "wrote_live": False,
                    "reason": "live_atlas_exists_pass_rewrite_atlas",
                    "roles_complete": False if not roles_complete else roles_complete,
                    "unique": unique,
                    "assemble": rel(assemble, ROOT),
                },
                indent=2,
            )
        )
        return 0

    dump_json(dest / "mix.json", mix)
    dump_json(dest / "schedule.json", schedule)
    dump_json(dest / "schedule-scratch.json", scratch_schedule)
    dump_json(
        dest / "manifest.json",
        {
            "id": existing_man.get("id") or mix_id,
            "n_train_tokens": unique,
            "n_unique_train_tokens": unique,
            "n_valid_tokens": WIKI_VALID,
            "roles_complete": roles_complete,
            "tokenizer_sha256": tok.model_sha256,
            "copies_shards": False,
        },
    )
    dump_json(dest / "RELEASE.json", release)
    meta_hashes = {
        "mix.json": file_sha256(dest / "mix.json"),
        "schedule.json": file_sha256(dest / "schedule.json"),
        "schedule-scratch.json": file_sha256(dest / "schedule-scratch.json"),
        "manifest.json": file_sha256(dest / "manifest.json"),
        "RELEASE.json": file_sha256(dest / "RELEASE.json"),
    }
    dump_json(dest / "hashes.json", meta_hashes)
    readme = """# corpus/lm-v1

Serving atlas for `mei-1.0-51m`. Trainer consumes `mix.json` + pack `tokens/*.bin`.
CPT uses `schedule.json`; from-scratch mix uses `schedule-scratch.json`.

Active sources: wiki, hq, structure. Spoken four-role stays fail-closed until a frozen qwen-plus pack is admitted.
Process ledgers live under `notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4/`.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")
    wrote_live = True
    print(
        json.dumps(
            {
                "ok": True,
                "wrote_live": wrote_live,
                "roles_complete": roles_complete,
                "unique": unique,
                "hashes": meta_hashes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
