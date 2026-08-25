#!/usr/bin/env python3
"""Audit structure shards for schema/OpenAPI cleanliness (v2 dirty vs v3 clean)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import CORPORA_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zh_pretrain_ingest import SCHEMA_HINT_RE, is_clean_structure_text
from data import iter_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPORA_ROOT / "zh-pretrain-v3")
    ap.add_argument("--sample", type=int, default=256)
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    reviews = corpus / "raw" / "structure.jsonl"
    idx_files = sorted((corpus / "tokens").glob("structure-*.idx.jsonl")) if (corpus / "tokens").is_dir() else []
    texts: list[str] = []
    if reviews.is_file():
        for i, row in enumerate(iter_jsonl(reviews)):
            texts.append(str(row.get("text") or ""))
            if len(texts) >= args.sample:
                break
    n = len(texts)
    n_clean = sum(1 for t in texts if is_clean_structure_text(t))
    n_hint = sum(1 for t in texts if SCHEMA_HINT_RE.search(t or ""))
    n_comment = sum(1 for t in texts if t.count("\n#") + t.count("\n//") >= 8)
    report = {
        "corpus": str(corpus.relative_to(ROOT)) if corpus.is_relative_to(ROOT) else str(corpus),
        "n_sample": n,
        "clean_rate": round(n_clean / n, 4) if n else 0.0,
        "schema_hint_rate": round(n_hint / n, 4) if n else 0.0,
        "heavy_comment_rate": round(n_comment / n, 4) if n else 0.0,
        "idx_files": len(idx_files),
        "ok": n > 0 and (n_clean / n) >= 0.9 and (n_hint / n) >= 0.9 and (n_comment / n) <= 0.05,
    }
    out = corpus / "reviews" / "structure-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
