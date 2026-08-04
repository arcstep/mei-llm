#!/usr/bin/env python3
"""Summarize one family from a caller-provided corpus root."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def snapshot(corpus: Path, family_id: str) -> dict:
    manifests = corpus / "manifests"
    samples_root = corpus / "samples"
    manifest = None
    for path in manifests.glob("*.jsonl"):
        manifest = next(
            (row for row in load_jsonl(path) if row.get("family_id") == family_id),
            None,
        )
        if manifest:
            break
    samples = [
        row
        for bucket in ("build", "access", "docs")
        for row in load_jsonl(samples_root / f"{bucket}.jsonl")
        if row.get("family_id") == family_id
    ]
    return {
        "family": manifest,
        "sample_count": len(samples),
        "route_modes": Counter(r.get("route_mode", "") for r in samples).most_common(),
        "task_types": Counter(r.get("task_type", "") for r in samples).most_common(),
        "source_kinds": Counter(r.get("source_kind", "") for r in samples).most_common(),
        "truth_statuses": Counter(r.get("truth_status", "") for r in samples).most_common(),
        "sample_ids": [r.get("sample_id") for r in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--family-id", required=True)
    args = parser.parse_args()
    if not args.corpus_root.is_dir():
        parser.error(f"corpus root does not exist: {args.corpus_root}")
    print(json.dumps(snapshot(args.corpus_root.resolve(), args.family_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
