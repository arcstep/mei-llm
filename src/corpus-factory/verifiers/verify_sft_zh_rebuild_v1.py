#!/usr/bin/env python3
"""Re-verify every mei-1.0-51m-exp-000600m-sft-zh-rebuild-* release (and its
eval-lock extension) against its frozen manifest: byte/hash integrity, and
CURRENT.json/base-weights non-mutation. Both v1 (140/class MW pilot) and v2
(500/class MW, after auditing the old v4-300m-v4 corpus and finding 12/20
classes had zero candidate_tool grounding) are immutable and kept -- v2
supersedes v1 for consumption, v1 is not deleted. Read-only.

Usage: .venv/bin/python corpus-factory/verifiers/verify_sft_zh_rebuild_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/corpus-factory" / "generators"))
from rebuild_zh_v1 import common as C  # noqa: E402

RELEASES = [
    ("mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1", "mei-51m-longitudinal-eval-v8-retrieval-depth"),
    ("mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2", "mei-51m-longitudinal-eval-v8-retrieval-depth-v2"),
]


def verify_manifest(base_dir: Path, manifest_path: Path) -> list[str]:
    errors = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, meta in manifest["artifacts"].items():
        path = base_dir / rel
        if not path.is_file():
            errors.append(f"missing artifact: {rel}")
            continue
        data = path.read_bytes()
        if len(data) != meta["bytes"]:
            errors.append(f"byte mismatch: {rel}")
        if C.sha256_bytes(data) != meta["sha256"]:
            errors.append(f"sha256 mismatch: {rel}")
    recomputed = C.merkle_root(manifest["artifacts"])
    if recomputed != manifest["artifact_merkle_root"]:
        errors.append(f"merkle root mismatch: recorded={manifest['artifact_merkle_root']} recomputed={recomputed}")
    return errors


def verify_current_json_untouched() -> list[str]:
    errors = []
    current = json.loads(C.CURRENT_JSON_PATH.read_text(encoding="utf-8"))
    if current.get("sft") is not None:
        errors.append(f"CURRENT.json sft is not null: {current.get('sft')!r}")
    if current.get("base") != "base/mei-1.0-51m-base-scratch300m-v1":
        errors.append(f"CURRENT.json base changed: {current.get('base')!r}")
    return errors


def verify_base_weights_untouched() -> list[str]:
    errors = []
    actual = C.sha256_file(
        ROOT / "models/mei-1.2-51m/releases/exp-000600m/base"
        / "mei-1.0-51m-base-cpt600m-clean-source-v3-v1/mei-1.0-51m-base-cpt600m-clean-source-v3-v1.npz"
    )
    if actual != C.BASE_WEIGHTS_SHA256:
        errors.append(f"600M base weights sha256 changed: expected={C.BASE_WEIGHTS_SHA256} actual={actual}")
    return errors


def main() -> int:
    all_errors: dict[str, list[str]] = {}
    for sft_id, eval_id in RELEASES:
        sft_dir = C.RELEASE_ROOT / sft_id
        eval_dir = C.EVAL_LOCK_ROOT / eval_id
        all_errors[f"{sft_id}:sft_release_manifest"] = verify_manifest(sft_dir, sft_dir / "manifests" / "artifact-manifest.json")
        all_errors[f"{sft_id}:eval_lock_manifest"] = verify_manifest(eval_dir, eval_dir / "manifests" / "artifact-manifest.json")
    all_errors["current_json"] = verify_current_json_untouched()
    all_errors["base_weights"] = verify_base_weights_untouched()

    ok = all(not v for v in all_errors.values())
    print(json.dumps({"ok": ok, "errors": all_errors}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
