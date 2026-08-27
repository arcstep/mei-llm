#!/usr/bin/env python3
"""Validate approved-colloquial.json before zh-pretrain-v4 may flip roles_complete.

license_cleared / provenance_ok / quality_ok / isolation_ok are independent.
formal_cpt_eligible must not circularly set license_cleared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import APPROVED_COLLOQUIAL, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_synth_lib import contract_sha256, load_contract, terms_hash  # noqa: E402
from zh_pretrain_ingest import dump_json  # noqa: E402

FORBIDDEN_SOURCES = {"cwt2", "ChineseWebText2.0", "CASIA-LM/ChineseWebText2.0"}


def existing(paths: list[str]) -> list[str]:
    return [p for p in paths if (ROOT / p).is_file()]


def validate_spec(spec: dict, contract: dict) -> dict:
    reasons: list[str] = []
    source_id = str(spec.get("source_id") or "")
    if source_id in FORBIDDEN_SOURCES:
        reasons.append("forbidden_cwt2_source")
    if spec.get("exclude_cwt2") is False:
        reasons.append("exclude_cwt2_false")
    train = existing(list(spec.get("train_shards") or []))
    valid = existing(list(spec.get("valid_shards") or []))
    if not train:
        reasons.append("missing_train_shards")
    n_train = int(spec.get("n_train_tokens") or 0)
    if n_train <= 0:
        reasons.append("n_train_tokens_zero")
    snapshot = str(spec.get("model_snapshot") or "")
    generator = str(spec.get("generator") or "")
    want_snap = contract["generator"]["frozen_snapshot"]
    unique_need = int(contract["formal_cpt"]["requires_unique_tokens"])
    candidate_need = int(contract["targets"]["pilot_unique_tokens"])
    provenance_reasons = []
    if generator != "qwen-plus":
        provenance_reasons.append("generator_not_qwen_plus")
    if snapshot != want_snap:
        provenance_reasons.append("snapshot_mismatch")
    if spec.get("terms_hash") != terms_hash(contract):
        provenance_reasons.append("terms_hash_mismatch")
    if spec.get("contract_sha256") and spec.get("contract_sha256") != contract_sha256():
        provenance_reasons.append("contract_sha256_mismatch")
    if n_train < unique_need:
        provenance_reasons.append("unique_below_30m")
    if int(spec.get("n_duplicate_frame_ids") or 0) > 0:
        provenance_reasons.append("duplicate_frame_id")
    provenance_ok = not provenance_reasons and bool(train) and "forbidden_cwt2_source" not in reasons
    quality_ok = spec.get("quality_ok") is True and spec.get("audit_ok") is True and spec.get("blind_review_ok") is True
    isolation_ok = spec.get("isolation_ok") is True
    license_cleared = spec.get("license_cleared") is True
    if spec.get("audit_ok") is not True:
        reasons.append("audit_not_ok")
    if spec.get("quality_ok") is not True:
        reasons.append("quality_not_ok")
    if spec.get("blind_review_ok") is not True:
        reasons.append("blind_review_not_ok")
    if spec.get("isolation_ok") is not True:
        reasons.append("isolation_not_ok")
    if spec.get("license_cleared") is not True:
        reasons.append("license_not_cleared")
    reasons.extend(provenance_reasons)
    is_candidate = n_train >= candidate_need and "forbidden_cwt2_source" not in reasons and bool(train)
    formal = provenance_ok and quality_ok and isolation_ok and license_cleared and bool(train)
    claimed_eligible = spec.get("formal_cpt_eligible")
    if claimed_eligible and not formal:
        reasons.append("formal_cpt_eligible_self_certified")
        formal = False
    return {
        "ok_candidate": is_candidate,
        "ok_formal_cpt": formal,
        "provenance_ok": provenance_ok,
        "quality_ok": quality_ok,
        "isolation_ok": isolation_ok,
        "license_cleared": license_cleared,
        "reasons": reasons,
        "n_train_tokens": n_train,
        "n_train_shards": len(train),
        "n_valid_shards": len(valid),
        "source_id": source_id,
        "generator": generator,
        "model_snapshot": snapshot,
    }


def write_candidate_from_release(release_dir: Path, dest: Path) -> dict:
    release = json.loads((release_dir / "RELEASE.json").read_text(encoding="utf-8"))
    audit = {}
    audit_path = release_dir / "reviews" / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    isolation = {}
    iso_path = release_dir / "reviews" / "isolation.json"
    if iso_path.is_file():
        isolation = json.loads(iso_path.read_text(encoding="utf-8"))
    blind = {}
    blind_path = release_dir / "reviews" / "blind-sol.json"
    if blind_path.is_file():
        blind = json.loads(blind_path.read_text(encoding="utf-8"))
    spec = {
        "license_cleared": bool(release.get("license_cleared")),
        "exclude_cwt2": True,
        "source_id": release.get("id") or release_dir.name,
        "license": "synthetic; internal generators; no third-party web scrape",
        "generator": release.get("generator"),
        "model_snapshot": release.get("model_snapshot"),
        "prompt_version": release.get("prompt_version"),
        "terms_hash": release.get("terms_hash"),
        "contract_sha256": release.get("contract_sha256"),
        "audit_ok": bool(audit.get("ok")),
        "quality_ok": bool(audit.get("quality_ok") or audit.get("ok")) and bool(release.get("quality_ok")),
        "isolation_ok": bool(isolation.get("ok")) and bool(release.get("isolation_ok")),
        "blind_review_ok": bool(blind.get("ok")) and bool(release.get("blind_review_ok")),
        "provenance_ok": bool(release.get("provenance_ok")),
        "train_shards": list(release.get("train_shards") or []),
        "valid_shards": list(release.get("valid_shards") or []),
        "n_train_tokens": int(release.get("n_unique_train_tokens") or 0),
        "n_valid_tokens": int(release.get("n_valid_tokens") or 0),
        "n_duplicate_frame_ids": int(release.get("n_duplicate_frame_ids") or 0),
        "release_kind": release.get("release_kind"),
        "formal_cpt_eligible": bool(release.get("formal_cpt_eligible")),
    }
    dump_json(dest, spec)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--from-release", type=Path, default=None)
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--require-candidate", action="store_true")
    ap.add_argument("--require-formal", action="store_true")
    args = ap.parse_args()
    contract = load_contract()
    dest = args.manifest or APPROVED_COLLOQUIAL
    if not dest.is_absolute():
        dest = ROOT / dest
    if args.from_release:
        rel_dir = args.from_release if args.from_release.is_absolute() else ROOT / args.from_release
        write_candidate_from_release(rel_dir, dest)
    if not dest.is_file():
        print("missing approved-colloquial.json", file=sys.stderr)
        return 2
    spec = json.loads(dest.read_text(encoding="utf-8"))
    report = validate_spec(spec, contract)
    report["path"] = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "forbidden_cwt2_source" in report["reasons"]:
        return 1
    if args.require_formal:
        return 0 if report["ok_formal_cpt"] else 1
    if args.require_candidate:
        return 0 if report["ok_candidate"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
