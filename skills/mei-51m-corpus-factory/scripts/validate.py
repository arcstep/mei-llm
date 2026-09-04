#!/usr/bin/env python3
"""Validate one corpus release without relying on a hard-coded release list."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from _shared import find_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验指定 SFT release 的 artifact manifest、CURRENT.json 与冻结 Base。"
    )
    parser.add_argument("--release-id", required=True, help="待校验的 SFT release ID")
    parser.add_argument(
        "--eval-lock-id",
        help="可选：同时校验对应 eval-lock ID",
    )
    return parser.parse_args()


def load_verifier(root: Path) -> ModuleType:
    path = root / "corpus-factory/verifiers/verify_sft_zh_rebuild_v1.py"
    spec = importlib.util.spec_from_file_location("mei_51m_release_verifier", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 verifier：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_one(verifier: ModuleType, base_dir: Path) -> list[str]:
    manifest = base_dir / "manifests/artifact-manifest.json"
    if not manifest.is_file():
        return [f"missing manifest: {manifest}"]
    try:
        return list(verifier.verify_manifest(base_dir, manifest))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid manifest: {error}"]


def main() -> int:
    args = parse_args()
    root = find_repo_root()
    verifier = load_verifier(root)
    errors: dict[str, list[str]] = {
        f"release:{args.release_id}": verify_one(
            verifier, verifier.C.RELEASE_ROOT / args.release_id
        ),
        "current_json": list(verifier.verify_current_json_untouched()),
        "base_weights": list(verifier.verify_base_weights_untouched()),
    }
    if args.eval_lock_id:
        errors[f"eval_lock:{args.eval_lock_id}"] = verify_one(
            verifier, verifier.C.EVAL_LOCK_ROOT / args.eval_lock_id
        )

    ok = all(not values for values in errors.values())
    print(json.dumps({"ok": ok, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
