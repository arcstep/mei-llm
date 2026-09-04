#!/usr/bin/env python3
"""Build a write-once eval lock from one immutable SFT release."""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

from _shared import find_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--sft-release-id", required=True)
    parser.add_argument("--eval-lock-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="未指定时仅打印将要使用的路径",
    )
    return parser.parse_args()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("mei_51m_eval_lock_builder", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load eval lock builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"exp-\d{6}m", args.cycle_id):
        raise SystemExit("--cycle-id must look like exp-000900m")
    root = find_repo_root()
    path = root / "corpus-factory/generators/rebuild_zh_v1/build_eval_lock.py"
    builder = load_builder(path)
    corpus_root = root / ".local/artifacts/mei-1.0-51m" / args.cycle_id / "corpus"
    builder.C.CYCLE_ID = args.cycle_id
    builder.C.RELEASE_ROOT = corpus_root / "sft-suite"
    builder.C.EVAL_LOCK_ROOT = corpus_root / "eval-lock"
    builder.SFT_RELEASE_ID = args.sft_release_id
    builder.SFT_RELEASE_DIR = builder.C.RELEASE_ROOT / args.sft_release_id
    builder.EVAL_LOCK_ID = args.eval_lock_id
    builder.EVAL_LOCK_DIR = builder.C.EVAL_LOCK_ROOT / args.eval_lock_id
    print(f"sft_release={builder.SFT_RELEASE_DIR}")
    print(f"eval_lock={builder.EVAL_LOCK_DIR}")
    if not args.execute:
        return 0
    if builder.EVAL_LOCK_DIR.exists():
        raise SystemExit(f"refusing to overwrite eval lock: {builder.EVAL_LOCK_DIR}")
    return int(builder.main())


if __name__ == "__main__":
    raise SystemExit(main())
