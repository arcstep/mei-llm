#!/usr/bin/env python3
"""Create a write-once family/target overlay consumed by build.py."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _shared import find_repo_root
from build import TARGET_NAMES, load_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        action="append",
        dest="assignments",
        required=True,
        metavar="MAP.KEY=COUNT",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder_path = find_repo_root() / "corpus-factory/generators/rebuild_zh_v1/build.py"
    builder = load_builder(find_repo_root())
    targets: dict[str, dict[str, int]] = {}
    for assignment in args.assignments:
        left, separator, raw_count = assignment.partition("=")
        map_name, dot, key = left.partition(".")
        if not separator or not dot or map_name not in TARGET_NAMES:
            raise SystemExit("--set must use MAP.KEY=COUNT with a registered target map")
        current = getattr(builder, map_name)
        if key not in current:
            raise SystemExit(f"unknown target: {map_name}.{key}")
        count = int(raw_count)
        if count < 0:
            raise SystemExit("target count cannot be negative")
        targets.setdefault(map_name, {})[key] = count
    payload = {
        "schema": "mei-51m-sft-target-registration-v1",
        "generator": str(builder_path),
        "generator_sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest(),
        "targets": targets,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_bytes() != encoded:
        raise SystemExit(f"refusing to overwrite different registration: {args.out}")
    args.out.write_bytes(encoded)
    print(encoded.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
