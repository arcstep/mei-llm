#!/usr/bin/env python3
"""Shared freeze/tokenize/leak helpers for colloquial synth releases.

formal_cpt_eligible is never set true here. Provenance can be recorded;
quality/isolation/blind review live in later gates.
"""

from __future__ import annotations

import json
from pathlib import Path

from repo_paths import BANK_NEEDLE_PRETRAIN_PROBES, ROOT, TOKENIZER_ZH_V1, all_eval_jsonl

from colloquial_synth_lib import contract_sha256, terms_hash, unique_by_first_frame
from data import leak_strings_from_rows
from tokenizer import ZhTokenizerV1
from zh_pretrain_ingest import SplitWriters, TOKENS_PER_SHARD, dump_json, file_sha256, rel


def load_leaks(*, full: bool = True) -> list[str]:
    rows: list[dict] = []
    for path in all_eval_jsonl():
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if not full and len(rows) >= 12:
                break
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        for line in BANK_NEEDLE_PRETRAIN_PROBES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [s for s in leak_strings_from_rows(rows) if s]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def tokenize_kept(out: Path, kept: list[dict], tok: ZhTokenizerV1) -> dict:
    token_dir = out / "tokens"
    if token_dir.exists():
        for p in token_dir.glob("colloquial-*"):
            p.unlink()
    token_dir.mkdir(parents=True, exist_ok=True)
    writers = SplitWriters(token_dir, "colloquial", TOKENS_PER_SHARD, ROOT)
    n_unk = 0
    seen: set[str] = set()
    for row in kept:
        fid = str(row.get("doc_id") or (row.get("frame") or {}).get("frame_id") or "")
        if fid and fid in seen:
            continue
        if fid:
            seen.add(fid)
        ids = tok.encode_document(row["text"])
        n_unk += sum(1 for t in ids if t == tok.unk_id)
        if row.get("split") == "valid":
            writers.valid.write(ids, row, tok.unk_id)
        else:
            writers.train.write(ids, row, tok.unk_id)
    writers.close()
    return {"n_unk": n_unk}


def freeze_release(
    out: Path,
    *,
    contract: dict,
    generator: str,
    kept: list[dict],
    stats: dict,
) -> dict:
    counts = unique_by_first_frame(kept)
    n_train = int(counts["unique_train_tokens"])
    n_valid = int(counts["unique_valid_tokens"])
    n_unk = int(stats.get("n_unk") or 0)
    n_tok = n_train + n_valid
    unk_rate = (n_unk / n_tok) if n_tok else 1.0
    production = generator == "qwen-plus"
    want_snap = contract["generator"]["frozen_snapshot"]
    snapshot = stats.get("model_snapshot") or (want_snap if production else "offline-v1")
    mixed = bool(stats.get("mixed_generator"))
    provenance_ok = (
        production
        and snapshot == want_snap
        and n_train >= int(contract["formal_cpt"]["requires_unique_tokens"])
        and counts["n_duplicate_frame_ids"] == 0
        and not mixed
        and n_train > 0
        and unk_rate <= float(contract["hard_gates"]["unk_rate_max"])
    )
    token_dir = out / "tokens"
    train_shards = [rel(p, ROOT) for p in sorted(token_dir.glob("colloquial-train-*.bin"))] if token_dir.is_dir() else []
    valid_shards = [rel(p, ROOT) for p in sorted(token_dir.glob("colloquial-valid-*.bin"))] if token_dir.is_dir() else []
    ledger = {
        "unique_train_tokens": n_train,
        "exposure_train_tokens": int(counts["exposure_train_tokens"]),
        "unique_valid_tokens": n_valid,
        "n_unique_frame_ids": int(counts["n_unique_frame_ids"]),
        "n_duplicate_frame_ids": int(counts["n_duplicate_frame_ids"]),
        "by_source": {"colloquial_synth": n_train},
        "valid": {"colloquial_synth": n_valid},
        "allow_repeat": False,
        "parent_checkpoint": None,
        "tokenizer": str(TOKENIZER_ZH_V1.relative_to(ROOT)),
        "seq_len": 256,
        "note": "Spoken-role unique is first frame_id only. Do not multiply epochs. Student outputs never re-enter.",
    }
    dump_json(out / "unique-ledger.json", ledger)
    reasons = []
    if not production:
        reasons.append("requires_qwen_plus_snapshot")
    if n_train < int(contract["formal_cpt"]["requires_unique_tokens"]):
        reasons.append("unique_below_30m")
    if counts["n_duplicate_frame_ids"]:
        reasons.append("duplicate_frame_id")
    if mixed:
        reasons.append("mixed_generator")
    reasons.append("awaiting_quality_isolation_blind_review")
    release = {
        "id": out.name,
        "role": "cpt_colloquial_subcorpus_only",
        "release_kind": stats.get("release_kind") or "pilot",
        "generator": generator,
        "model_snapshot": snapshot,
        "prompt_version": contract["prompt_version"],
        "terms_hash": terms_hash(contract),
        "contract_sha256": contract_sha256(),
        "excludes_cwt2": True,
        "excludes_eval_gold": True,
        "n_unique_train_tokens": n_train,
        "n_exposure_train_tokens": int(counts["exposure_train_tokens"]),
        "n_valid_tokens": n_valid,
        "n_duplicate_frame_ids": int(counts["n_duplicate_frame_ids"]),
        "unk_rate": unk_rate,
        "n_docs": int(counts["n_unique_frame_ids"]),
        "n_accepted_rows": int(counts["n_docs"]),
        "provenance_ok": provenance_ok,
        "quality_ok": False,
        "isolation_ok": False,
        "blind_review_ok": False,
        "license_cleared": False,
        "formal_cpt_eligible": False,
        "eligibility_reason": ",".join(reasons),
        "register_v4": False,
        "stats": stats,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "spend_cny": float(stats.get("spend_cny") or 0),
        "prompt_tokens": int(stats.get("prompt_tokens") or 0),
        "completion_tokens": int(stats.get("completion_tokens") or 0),
    }
    dump_json(out / "RELEASE.json", release)
    dump_json(
        out / "manifest.json",
        {
            "id": out.name,
            "n_train_tokens": n_train,
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "unk_rate": unk_rate,
            "tokenizer_sha256": file_sha256(TOKENIZER_ZH_V1) if TOKENIZER_ZH_V1.is_file() else "",
            "generator": generator,
            "model_snapshot": snapshot,
        },
    )
    dump_json(
        out / "source-license.json",
        {
            "colloquial": {
                "license": "synthetic; internal generators; no third-party web scrape",
                "generator": generator,
                "admitted_for_engineering": True,
                "admitted_for_formal_cpt": False,
                "cwt2": "excluded",
            }
        },
    )
    dump_json(out / "source-role.json", {"colloquial": {"role": "cpt_colloquial_synth", "band": "B"}})
    dump_json(
        out / "recipe-lock.json",
        {
            "prompt_version": contract["prompt_version"],
            "axes": contract["axes"],
            "sampling": contract["sampling"],
            "terms_hash": terms_hash(contract),
            "contract_sha256": contract_sha256(),
            "generator": generator,
            "model_snapshot": snapshot,
        },
    )
    meta_hashes = {
        "RELEASE.json": file_sha256(out / "RELEASE.json"),
        "unique-ledger.json": file_sha256(out / "unique-ledger.json"),
        "manifest.json": file_sha256(out / "manifest.json"),
        "recipe-lock.json": file_sha256(out / "recipe-lock.json"),
        "source-license.json": file_sha256(out / "source-license.json"),
    }
    dump_json(out / "hashes.json", meta_hashes)
    readme = """# colloquial synth release

100% synthetic spoken-role corpus for mei-1.0-58m independent CPT.

- Production generator is the frozen qwen-plus snapshot.
- Offline renderer is engineering contrast / smoke only and cannot open formal CPT.
- Generator/snapshot prove provenance only; quality, isolation, and blind review are separate gates.
- CWT2, EVAL gold, holdout, and student-model recycle are excluded.
- Token bins and raw JSONL stay local; this README does not link private docs.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return release
