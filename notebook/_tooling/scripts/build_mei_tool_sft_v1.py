#!/usr/bin/env python3
"""Build mei-tool-sft-v1 2k/10k packs. Does not overwrite home-sft or mw-sft."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mei_tool_schema_lib import GENERATOR_VERSION, SERIALIZER_ID, dump_jsonl, make_sft_rows
from repo_paths import ROOT, TASKS_ROOT, TASK_NEEDLE_ZH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["2k", "10k"], default="2k")
    args = ap.parse_args()
    n_train, n_valid = (1800, 200) if args.tier == "2k" else (9000, 1000)
    rows = make_sft_rows(n_train, n_valid, seed_tag=args.tier.upper())
    pack_dir = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs"
    path = pack_dir / f"mei-tool-sft-v1-{args.tier}.jsonl"
    dump_jsonl(path, rows)
    recipe = {
        "id": f"mei-tool-sft-v1-{args.tier}",
        "serializer": SERIALIZER_ID,
        "generator_version": GENERATOR_VERSION,
        "n": len(rows),
        "n_train": sum(1 for r in rows if r.get("split") == "train"),
        "n_valid": sum(1 for r in rows if r.get("split") == "valid"),
        "seq_len": 2048,
        "schema_conditioned": True,
        "holdout_toolsets_excluded": ["mei-retail-v0", "mei-office-mut-v0", "mei-office-rename-v0"],
        "does_not_overwrite": ["home-sft-2k.jsonl", "home-sft-10k.jsonl", "mw-sft-v0-2k.jsonl"],
        "pack_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pack": str(path.relative_to(ROOT)),
    }
    rec_path = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / f"mei-tool-sft-v1-{args.tier}.json"
    rec_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recipe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
