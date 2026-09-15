"""Create a self-contained successor for a validated v1.2 CPT input release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from profiling import resolve_path, ROOT, digest
from v12_inputs_freeze import _atomic_json, validate_sampler


def _replace_paths(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _replace_paths(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_paths(child, old, new) for child in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _clone_tree(source: Path, target: Path) -> str:
    proc = subprocess.run(["cp", "-cR", str(source), str(target)], capture_output=True, text=True)
    if proc.returncode == 0:
        return "apfs-clone"
    shutil.copytree(source, target, copy_function=shutil.copy2)
    return "byte-copy"


def run(config: dict, out: Path) -> dict:
    source = (resolve_path(ROOT / config["source_release_dir"])).resolve()
    target = (resolve_path(ROOT / config["release_dir"])).resolve()
    tokenizer_manifest = (resolve_path(ROOT / config["tokenizer_manifest"])).resolve()
    out = (resolve_path(ROOT / out)).resolve() if not out.is_absolute() else out.resolve()
    if target.exists() or out.exists():
        raise FileExistsError(target if target.exists() else out)
    source_manifest = json.loads((source / "RELEASE.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "inputs_ready":
        raise ValueError("source input release is not inputs_ready")
    if tokenizer.get("status") != "frozen" or tokenizer.get("schema") != "mei-51m-tokenizer-release-v1":
        raise ValueError("tokenizer release is not frozen")
    if source_manifest.get("tokenizer", {}).get("model_sha256") != tokenizer.get("model_sha256"):
        raise ValueError("source release and tokenizer binding disagree")
    staging = target.with_name("." + target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_method = _clone_tree(source, staging)

    old_rel = str(Path(config["source_release_dir"]))
    new_rel = str(Path(config["release_dir"]))
    for name in ("schedule.json", "mix.json"):
        value = json.loads((staging / name).read_text(encoding="utf-8"))
        _atomic_json(staging / name, _replace_paths(value, old_rel, new_rel))

    model_source = tokenizer_manifest.parent / tokenizer["model_file"]
    model_target = staging / tokenizer["model_file"]
    shutil.copy2(model_source, model_target)
    if digest(model_target) != tokenizer["model_sha256"]:
        raise ValueError("bundled tokenizer model hash mismatch")
    vocab_target = None
    if tokenizer.get("vocab_file"):
        vocab_source = tokenizer_manifest.parent / tokenizer["vocab_file"]
        vocab_target = staging / tokenizer["vocab_file"]
        shutil.copy2(vocab_source, vocab_target)
        if digest(vocab_target) != tokenizer.get("vocab_sha256"):
            raise ValueError("bundled tokenizer vocab hash mismatch")
    shutil.copy2(tokenizer_manifest, staging / "TOKENIZER.json")

    manifest = json.loads((staging / "RELEASE.json").read_text(encoding="utf-8"))
    manifest.update({
        "release_id": config["release_id"],
        "status": "validating",
        "supersedes": {
            "release": old_rel,
            "release_sha256": digest(source / "RELEASE.json"),
            "reason": config.get(
                "supersede_reason", "tokenizer manifest in predecessor was not self-contained"
            ),
        },
    })
    manifest["tokenizer"].update({
        "manifest": "TOKENIZER.json",
        "manifest_sha256": digest(staging / "TOKENIZER.json"),
        "model_file": tokenizer["model_file"],
        "model_sha256": digest(model_target),
        "vocab_file": tokenizer.get("vocab_file"),
        "vocab_sha256": digest(vocab_target) if vocab_target else None,
        "runtime_audit": tokenizer.get("runtime_audit"),
    })
    manifest["schedule"] = {"path": "schedule.json", "sha256": digest(staging / "schedule.json")}
    manifest["mix"] = {"path": "mix.json", "sha256": digest(staging / "mix.json")}
    _atomic_json(staging / "RELEASE.json", manifest)
    staging.replace(target)

    sampler = validate_sampler(config, target)
    manifest = json.loads((target / "RELEASE.json").read_text(encoding="utf-8"))
    manifest["sampler_validation"] = {
        "path": "SAMPLER-VALIDATION.json",
        "sha256": digest(target / "SAMPLER-VALIDATION.json"),
        "status": sampler["status"],
    }
    manifest["status"] = "inputs_ready"
    _atomic_json(target / "RELEASE.json", manifest)

    out.mkdir(parents=True)
    receipt = {
        "schema": "mei-v12-input-rebundle-receipt-v1",
        "status": "complete",
        "release": config["release_dir"],
        "release_sha256": digest(target / "RELEASE.json"),
        "source_release": config["source_release_dir"],
        "clone_method": clone_method,
        "current_mutated": False,
        "a10_started": False,
        "cpt_started": False,
    }
    _atomic_json(out / "RECEIPT.json", receipt)
    shutil.copy2(__file__, out / "implementation.py.snapshot")
    return manifest
