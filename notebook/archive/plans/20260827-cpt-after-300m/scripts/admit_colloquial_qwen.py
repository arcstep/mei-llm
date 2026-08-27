#!/usr/bin/env python3
"""Admit a qwen colloquial release only after provenance+quality+isolation+blind.

Writes approved-colloquial.json and rebuilds zh-pretrain-v4. Never self-certifies
from formal_cpt_eligible alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    APPROVED_COLLOQUIAL,
    CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1,
    CORPUS_ZH_PRETRAIN_V4,
    ROOT,
    SCRIPTS_ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colloquial_synth_lib import contract_sha256, load_contract, terms_hash  # noqa: E402
from validate_approved_colloquial import validate_spec  # noqa: E402
from zh_pretrain_ingest import dump_json, file_sha256  # noqa: E402


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    contract = load_contract()
    release = load(corpus / "RELEASE.json")
    audit = load(corpus / "reviews" / "audit.json")
    isolation = load(corpus / "reviews" / "isolation.json")
    blind = load(corpus / "reviews" / "blind-sol.json")
    n_train = int(release.get("n_unique_train_tokens") or 0)
    provenance_ok = (
        release.get("generator") == "qwen-plus"
        and release.get("model_snapshot") == contract["generator"]["frozen_snapshot"]
        and n_train >= int(contract["formal_cpt"]["requires_unique_tokens"])
        and int(release.get("n_duplicate_frame_ids") or 0) == 0
        and float(release.get("unk_rate") or 1) <= float(contract["hard_gates"]["unk_rate_max"])
        and release.get("spend_cny", 0) <= float(contract["budget"]["production_max_spend_cny"]) + 1e-6
    )
    quality_ok = bool(audit.get("ok")) and bool(audit.get("hard_ok")) and bool(audit.get("quality_ok"))
    isolation_ok = bool(isolation.get("ok"))
    blind_ok = bool(blind.get("ok")) and float(blind.get("naturalness_pass_rate") or 0) >= 0.85
    license_cleared = provenance_ok and quality_ok and isolation_ok and blind_ok
    formal = license_cleared
    if not formal and not args.force:
        report = {
            "ok": False,
            "provenance_ok": provenance_ok,
            "quality_ok": quality_ok,
            "isolation_ok": isolation_ok,
            "blind_review_ok": blind_ok,
            "license_cleared": False,
            "formal_cpt_eligible": False,
        }
        dump_json(corpus / "reviews" / "admit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    release["provenance_ok"] = provenance_ok
    release["quality_ok"] = quality_ok
    release["isolation_ok"] = isolation_ok
    release["blind_review_ok"] = blind_ok
    release["license_cleared"] = license_cleared
    release["formal_cpt_eligible"] = formal
    release["register_v4"] = formal
    release["eligibility_reason"] = None if formal else "gates_incomplete"
    dump_json(corpus / "RELEASE.json", release)
    hashes = load(corpus / "hashes.json")
    hashes["RELEASE.json"] = file_sha256(corpus / "RELEASE.json")
    dump_json(corpus / "hashes.json", hashes)
    spec = {
        "license_cleared": license_cleared,
        "exclude_cwt2": True,
        "source_id": release.get("id") or corpus.name,
        "license": "synthetic; frozen qwen-plus-2025-12-01; no third-party web scrape",
        "generator": release.get("generator"),
        "model_snapshot": release.get("model_snapshot"),
        "prompt_version": release.get("prompt_version"),
        "terms_hash": release.get("terms_hash") or terms_hash(contract),
        "contract_sha256": release.get("contract_sha256") or contract_sha256(),
        "audit_ok": bool(audit.get("ok")),
        "quality_ok": quality_ok,
        "isolation_ok": isolation_ok,
        "blind_review_ok": blind_ok,
        "provenance_ok": provenance_ok,
        "train_shards": list(release.get("train_shards") or []),
        "valid_shards": list(release.get("valid_shards") or []),
        "n_train_tokens": n_train,
        "n_valid_tokens": int(release.get("n_valid_tokens") or 0),
        "n_duplicate_frame_ids": int(release.get("n_duplicate_frame_ids") or 0),
        "release_kind": release.get("release_kind"),
        "formal_cpt_eligible": formal,
        "spend_cny": release.get("spend_cny"),
    }
    dest = APPROVED_COLLOQUIAL
    dump_json(dest, spec)
    gate = validate_spec(spec, contract)
    dump_json(corpus / "reviews" / "admit.json", {"ok": gate["ok_formal_cpt"], "gate": gate, "spec": spec})
    if not gate["ok_formal_cpt"]:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 1
    v4 = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "build_zh_pretrain_v4.py"),
            "--colloquial-manifest",
            str(dest),
            "--rewrite-atlas",
        ],
        cwd=str(ROOT),
    )
    if v4.returncode != 0:
        return v4.returncode
    v4_rel = json.loads((CORPUS_ZH_PRETRAIN_V4 / "RELEASE.json").read_text(encoding="utf-8"))
    mix = json.loads((CORPUS_ZH_PRETRAIN_V4 / "mix.json").read_text(encoding="utf-8"))
    sched = json.loads((CORPUS_ZH_PRETRAIN_V4 / "schedule.json").read_text(encoding="utf-8"))
    shards = mix.get("sources", {}).get("colloquial", {}).get("train_shards") or []
    if not v4_rel.get("roles_complete") or "colloquial" not in (sched.get("sources") or {}):
        print("v4 roles_complete or schedule missing colloquial", file=sys.stderr)
        return 1
    if any("cwt2" in s or "colloquial-synth-v1/" in s for s in shards):
        print("v4 admitted offline or cwt2 shards", file=sys.stderr)
        return 1
    if int(mix.get("n_colloquial_train_tokens") or 0) != n_train:
        print("v4 n_colloquial_train_tokens != ledger", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "roles_complete": True,
                "n_colloquial_train_tokens": n_train,
                "shards": shards,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
