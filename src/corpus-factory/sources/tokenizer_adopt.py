"""Adopt an audited tokenizer into a new immutable release without changing pointers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from profiling import ROOT, digest


def _write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(config: dict, out: Path) -> dict:
    candidate_dir = (ROOT / config["candidate_dir"]).resolve()
    audit_dir = (ROOT / config["runtime_audit_dir"]).resolve()
    candidate = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))
    if candidate.get("status") != "candidate_python_audited" or candidate.get("hard_errors"):
        raise ValueError("candidate Python audit did not pass")
    if audit.get("status") != "passed" or not audit.get("real_browser_passed"):
        raise ValueError("real Browser-WASM audit did not pass")
    if candidate["model_sha256"] != audit["model_sha256"]:
        raise ValueError("runtime audit was performed against a different model")
    if int(candidate["spm_vocab_size"]) != 24_000:
        raise ValueError("release must contain exactly 24000 pieces")
    out.mkdir(parents=True, exist_ok=False)
    model_name = "tokenizer.model"
    vocab_name = "tokenizer.vocab"
    shutil.copyfile(candidate_dir / model_name, out / model_name)
    shutil.copyfile(candidate_dir / vocab_name, out / vocab_name)
    if digest(out / model_name) != candidate["model_sha256"]:
        raise ValueError("copied tokenizer model hash changed")
    manifest = {
        "schema": "mei-51m-tokenizer-release-v1",
        "tokenizer_id": config["tokenizer_id"],
        "status": "frozen",
        "scope": "v1.2 training inputs; current runtime pointer unchanged",
        "vocab_size": 24_000,
        "model_file": model_name,
        "vocab_file": vocab_name,
        "model_sha256": candidate["model_sha256"],
        "vocab_sha256": candidate["vocab_sha256"],
        "special_ids": candidate["special_ids"],
        "encoding_profile_id": candidate["encoding_profile_id"],
        "normalizer": candidate["normalizer"],
        "byte_fallback": candidate["byte_fallback"],
        "split_digits": candidate["split_digits"],
        "protocol_symbols": candidate["protocol_symbols"],
        "training_sample": candidate["sample"],
        "candidate_manifest": {
            "path": config["candidate_dir"] + "/manifest.json",
            "sha256": digest(candidate_dir / "manifest.json"),
        },
        "runtime_audit": {
            "path": config["runtime_audit_dir"] + "/report.json",
            "sha256": digest(audit_dir / "report.json"),
            "python_cases": audit["python_cases"],
            "rust_passed": audit["rust_passed"],
            "real_browser_passed": audit["real_browser_passed"],
            "wasm_sha256": audit["wasm_sha256"],
        },
        "current_mutated": False,
        "a10_started": False,
    }
    _write(out / "RELEASE.json", manifest)
    manifest["release_manifest_sha256"] = digest(out / "RELEASE.json")
    return manifest
