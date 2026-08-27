#!/usr/bin/env python3
"""Summarize a caller-provided MeiLang corpus root."""

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


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stats(corpus: Path) -> dict:
    samples_root = corpus / "samples"
    eval_root = corpus / "eval"
    by_bucket = {
        bucket: load_jsonl(samples_root / f"{bucket}.jsonl")
        for bucket in ("build", "access", "docs")
    }
    all_rows = [row for rows in by_bucket.values() for row in rows]
    split_counts = {}
    for split in ("train", "eval", "test"):
        data = load_split(eval_root / f"{split}.index.json")
        split_counts[split] = {
            "families": len(data.get("family_ids", [])),
            "samples": len(data.get("sample_ids", [])),
        }
    return {
        "sample_counts": {bucket: len(rows) for bucket, rows in by_bucket.items()},
        "total_samples": len(all_rows),
        "top_route_modes": Counter(r.get("route_mode", "") for r in all_rows).most_common(),
        "top_domains": Counter(r.get("domain", "") for r in all_rows).most_common(),
        "top_task_types": Counter(r.get("task_type", "") for r in all_rows).most_common(20),
        "top_families": Counter(r.get("family_id", "") for r in all_rows).most_common(20),
        "top_source_kinds": Counter(r.get("source_kind", "") for r in all_rows).most_common(),
        "splits": split_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.corpus_root.is_dir():
        parser.error(f"corpus root does not exist: {args.corpus_root}")
    print(json.dumps(stats(args.corpus_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
