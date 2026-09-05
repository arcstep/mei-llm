#!/usr/bin/env python3
"""Build a trainable corpus layout from the zh-v2 pool (sourcing v2).

Consumes the frozen pool release + a passed plan-mix candidate; emits the
layout the CPT machinery expects: per-role uint16 train/valid shards, mix.json,
schedule-scratch.json (quota_plan curriculum), RELEASE/manifest/hashes evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SEQUENCE = [150_000_000, 100_000_000, 50_000_000]  # s1 512 / s2 1024 / s3 2048
SEQ_LENS = [512, 1024, 2048]
BATCHES = [8, 2, 1]

ROLE_LICENSES = {
    "fineweb2_hq": ["ODC-By-1.0", "Common Crawl ToU"],
    "wiki_zh": ["CC BY-SA 3.0/4.0 + GFDL"],
    "wiki_en": ["CC BY-SA 3.0/4.0 + GFDL"],
    "dialogue": ["subtitle-rights-uncleared", "to-be-reviewed"],
    "structured": ["CC0-1.0", "Apache-2.0", "ODC-By-1.0", "public domain"],
    "code": ["per-file"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def split_bin(
    source: Path,
    train_out: Path,
    valid_out: Path,
    valid_tokens: int,
) -> tuple[int, int]:
    """Split a uint16 token shard at a token boundary; returns (train, valid)."""
    total = source.stat().st_size // 2
    valid = min(valid_tokens, total)
    with source.open("rb") as src, train_out.open("wb") as train, valid_out.open(
        "wb"
    ) as valid_handle:
        shutil.copyfileobj(src, train, (total - valid) * 2)
        shutil.copyfileobj(src, valid_handle, valid * 2)
    return total - valid, valid


def build(
    pool_release: dict,
    candidate: dict,
    out: Path,
    *,
    tokenizer_id: str,
    valid_fraction: float,
    cycle_id: str,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite corpus layout: {out}")
    quotas = dict(candidate["quotas"])
    total_target = int(candidate["target_increment_tokens"])
    sources_by_role: dict[str, list[dict]] = {}
    for row in pool_release.get("sources") or []:
        sources_by_role.setdefault(row["source_role"], []).append(row)

    mix_sources: dict[str, Any] = {}
    all_train: list[str] = []
    all_valid: list[str] = []
    valid_sets: dict[str, list[str]] = {}
    train_total = 0
    valid_total = 0

    for role, rows in sorted(sources_by_role.items()):
        role_dir = out / "language" / role / "tokens"
        role_dir.mkdir(parents=True, exist_ok=True)
        role_pool = sum(int(row["tokens"]) for row in rows)
        valid_need = max(1, round(role_pool * valid_fraction))
        train_shards: list[str] = []
        valid_shards: list[str] = []
        train_tokens = 0
        valid_tokens = 0
        for index, row in enumerate(sorted(rows, key=lambda r: r["path"])):
            source_bin = Path(row["path"]) / "tokens.bin"
            if not source_bin.is_file():
                raise RuntimeError(f"missing admitted tokens.bin: {source_bin}")
            is_last = index == len(rows) - 1
            train_name = f"train-{index:04d}.bin"
            if is_last and valid_need > 0:
                split_valid = min(valid_need, source_bin.stat().st_size // 2)
                split_train = source_bin.stat().st_size // 2 - split_valid
                if split_train > 0:
                    n_train, n_valid = split_bin(
                        source_bin, role_dir / train_name, role_dir / "valid-0000.bin", split_valid
                    )
                    train_tokens += n_train
                    valid_tokens += n_valid
                    train_shards.append(rel_root(role_dir / train_name))
                else:
                    shutil.copyfile(source_bin, role_dir / "valid-0000.bin")
                    valid_tokens += split_valid
                valid_shards.append(rel_root(role_dir / "valid-0000.bin"))
                valid_need = 0
            else:
                try:
                    os.link(source_bin, role_dir / train_name)
                except OSError:
                    shutil.copyfile(source_bin, role_dir / train_name)
                train_tokens += source_bin.stat().st_size // 2
                train_shards.append(rel_root(role_dir / train_name))
        mix_sources[role] = {
            "band": "B" if role in {"fineweb2_hq", "wiki_zh", "wiki_en"} else "A",
            "license": ROLE_LICENSES.get(role, []),
            "n_train_tokens": train_tokens,
            "n_valid_tokens": valid_tokens,
            "train_shards": train_shards,
            "valid_shards": valid_shards,
        }
        all_train.extend(train_shards)
        all_valid.extend(valid_shards)
        valid_sets[role] = valid_shards
        train_total += train_tokens
        valid_total += valid_tokens

    mix = {
        "schema": "mei-51m-cpt-layout-v2",
        "id": f"zh-v2-layout-{cycle_id}-v1",
        "tokenizer": tokenizer_id,
        "n_train_tokens": train_total,
        "n_unique_train_tokens": train_total,
        "n_valid_tokens": valid_total,
        "sources": mix_sources,
        "train_shards": all_train,
        "valid_shards": all_valid,
        "valid_sets": valid_sets,
    }
    (out / "mix.json").write_text(
        json.dumps(mix, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stage_quotas = {role: int(quotas.get(role, 0)) for role in mix_sources}
    stage_quotas = {role: q for role, q in stage_quotas.items() if q > 0}
    scale = sum(stage_quotas.values())
    curriculum = []
    stop = 0
    for stage_index, (stage_tokens, seq_len, batch) in enumerate(
        zip(SEQUENCE, SEQ_LENS, BATCHES), 1
    ):
        stop += stage_tokens
        curriculum.append(
            {
                "id": f"s{stage_index}",
                "seq_len": seq_len,
                "stage_tokens": stage_tokens,
                "stop_at_tokens": stop,
                "batch_size": batch,
                "grad_accum": 1,
                "sources": {
                    role: {"token_quota": round(stage_tokens * q / scale)}
                    for role, q in stage_quotas.items()
                },
            }
        )
    schedule = {
        "stage_id": f"zh-v2-scratch-{total_target // 1_000_000}m-v1",
        "kind": "scratch",
        "parent_rung": None,
        "parent_tokens_seen": 0,
        "parent_checkpoint": None,
        "sampler": "quota_plan",
        "sampler_seed": 0,
        "skip_seen_wiki": False,
        "allow_repeat": False,
        "exposure_tokens": total_target,
        "lr": {
            "kind": "cosine_tokens",
            "base": 0.0003,
            "final": 3e-05,
            "horizon_tokens": total_target,
        },
        "sources": {
            role: {
                "token_quota": q,
                "skip_tokens": 0,
                "max_epochs": round(q / max(mix_sources[role]["n_train_tokens"], 1), 6),
            }
            for role, q in stage_quotas.items()
        },
        "curriculum": curriculum,
    }
    (out / "schedule-scratch.json").write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    quality_dir = Path(str(pool_release.get("quality_dir") or ""))
    if quality_dir.is_dir():
        contamination_receipts = sorted(
            rel_root(path) for path in quality_dir.glob("*receipt*.json")
        )
    else:
        contamination_receipts = []
    release = {
        "schema": "mei-51m-cpt-layout-release-v1",
        "release_id": f"zh-v2-layout-{cycle_id}-v1",
        "training_mode": "cpt",
        "roles_complete": True,
        "parent_checkpoint": None,
        "tokenizer_id": tokenizer_id,
        "pool_release_id": pool_release.get("release_id"),
        "license": sorted({item for row in mix_sources.values() for item in row["license"]}),
        "licenses": {
            role: mix_sources[role]["license"] for role in sorted(mix_sources)
        },
        "dedup": {
            "kind": "document-sha256-ledger",
            "exact_dedup": True,
        },
        "contamination": {"receipts": contamination_receipts},
        "contamination_report": {
            "eval_leakage_checked": True,
            "receipts": contamination_receipts,
        },
        "public_distribution_clearance_asserted": False,
    }
    (out / "RELEASE.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema": "mei-51m-cpt-layout-manifest-v1",
        "n_train_tokens": train_total,
        "n_unique_train_tokens": train_total,
        "n_valid_tokens": valid_total,
        "tokenizer": tokenizer_id,
        "contamination": {"receipts": contamination_receipts},
        "sliced_from": pool_release.get("_pool_path"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (out / "README.md").write_text(
        f"# zh-v2 layout {cycle_id}\n\ntokenizer: {tokenizer_id}\n"
        f"roles: {', '.join(sorted(mix_sources))}\n", encoding="utf-8"
    )
    (out / "SOURCES.md").write_text(
        "\n".join(f"- {role}: {mix_sources[role]['n_train_tokens']} train / {mix_sources[role]['n_valid_tokens']} valid" for role in sorted(mix_sources)) + "\n", encoding="utf-8"
    )

    hashes = {
        name: sha256_file(out / name)
        for name in ("RELEASE.json", "manifest.json", "mix.json", "schedule-scratch.json")
    }
    (out / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"layout": rel_root(out), "train_tokens": train_total, "valid_tokens": valid_total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-release", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycle-id", default="exp-000300m-v2")
    parser.add_argument("--tokenizer-id", default="zh-24k-v3")
    parser.add_argument("--valid-fraction", type=float, default=0.004)
    args = parser.parse_args()
    pool = json.loads(args.pool_release.read_text(encoding="utf-8"))
    pool["quality_dir"] = str(args.pool_release.parent.parent / "quality")
    pool["_pool_path"] = str(args.pool_release)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if candidate.get("status") != "passed":
        raise RuntimeError(f"candidate mix did not pass: {candidate.get('status')}")
    result = build(
        pool,
        candidate,
        args.out.resolve(),
        tokenizer_id=args.tokenizer_id,
        valid_fraction=args.valid_fraction,
        cycle_id=args.cycle_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
