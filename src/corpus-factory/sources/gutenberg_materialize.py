"""Fetch a pinned Gutenberg work list and preserve complete literary bodies."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil

from profiling import Budget, digest, fetch
from source_manager import load_tokenizer, tokenizer_pointer


def _write(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _body(raw: bytes, encoding: str) -> str:
    text = raw.decode(encoding, errors="strict")
    start = re.search(r"^\*\*\* START OF .*?\*\*\*\s*$", text, re.M)
    end = re.search(r"^\*\*\* END OF .*?\*\*\*\s*$", text, re.M)
    if not start or not end or start.end() >= end.start():
        raise ValueError("Gutenberg body boundaries missing")
    return text[start.end() : end.start()].strip()


def run(config: dict, out: Path, allow_network: bool) -> dict:
    if not allow_network:
        raise ValueError("Gutenberg materialization requires --allow-network")
    workers = int(config.get("download_workers", 2))
    if workers not in {1, 2}:
        raise ValueError("download concurrency must be 1 or 2")
    out.mkdir(parents=True, exist_ok=False)
    reserve = int(config.get("reserve_disk_bytes", 100 * 1024**3))
    if shutil.disk_usage(out).free < reserve:
        raise RuntimeError("disk reserve reached")
    _write(out / "config.json", config)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    catalog = Path(config["catalog_path"])
    if digest(catalog) != config["catalog_sha256"]:
        raise ValueError("Gutenberg catalog changed")
    raw_dir = out / "original-texts"
    raw_dir.mkdir()
    budget = Budget(out, int(config.get("max_network_bytes", 4 * 1024**3)), reserve)

    def acquire(work: dict) -> dict:
        raw_path = raw_dir / f"pg-{int(work['ebook_id']):07d}.txt"
        try:
            raw, headers = fetch(work["url"], budget, length=int(work.get("max_bytes", 8 << 20)))
            raw_path.write_bytes(raw)
            body = _body(raw, work.get("encoding", "utf-8-sig"))
            return {
                "work": work,
                "raw_file": raw_path.name,
                "raw_sha256": digest(raw_path),
                "raw_bytes": len(raw),
                "body": body,
                "resolved_url": headers.get("X-Mei-Resolved-URL"),
            }
        except Exception as exc:
            return {"work": work, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        acquired = list(executor.map(acquire, config["works"]))
    _write(out / "acquisition.json", [{k: v for k, v in row.items() if k != "body"} for row in acquired])

    tokenizer = load_tokenizer()
    seen: set[bytes] = set()
    counts = Counter()
    total_tokens = 0
    total_chars = 0
    by_language = Counter()
    destination = out / "literature.jsonl"
    with destination.open("x") as target:
        for row in acquired:
            if "error" in row:
                counts["acquisition_failed"] += 1
                continue
            counts["acquired"] += 1
            text = row.pop("body")
            key = hashlib.sha256(text.encode()).digest()
            if key in seen:
                counts["exact_duplicate"] += 1
                continue
            seen.add(key)
            tokens = len(tokenizer.encode_document(text))
            work = row["work"]
            record = {
                "id": f"gutenberg:{work['ebook_id']}",
                "text": text,
                "text_sha256": key.hex(),
                "source_id": "project-gutenberg",
                "tokens": tokens,
                "group_id": f"gutenberg:{work['ebook_id']}",
                "split": "candidate-unassigned",
                "origin": {
                    "catalog_path": config["catalog_path"],
                    "catalog_sha256": config["catalog_sha256"],
                    "ebook_id": work["ebook_id"],
                    "url": work["url"],
                    "resolved_url": row["resolved_url"],
                    "raw_file": row["raw_file"],
                    "raw_sha256": row["raw_sha256"],
                },
                "metadata": {
                    "title": work["title"],
                    "language": work["language"],
                    "subjects": work.get("subjects", ""),
                    "bookshelves": work.get("bookshelves", ""),
                    "content_kind": "literature-or-drama",
                    "rights_scope": "Project Gutenberg distribution; production rights review pending",
                },
                "candidate_kind": "cpt_raw",
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["written"] += 1
            total_tokens += tokens
            total_chars += len(text)
            by_language[work["language"]] += tokens

    report = {
        "schema": "mei-gutenberg-literary-candidate-v1",
        "status": "candidate_complete" if not counts["acquisition_failed"] else "candidate_partial",
        "planned_works": len(config["works"]),
        "records": counts["written"],
        "tokens": total_tokens,
        "tokens_by_catalog_language": dict(by_language),
        "characters": total_chars,
        "counts": dict(counts),
        "output": {"file": destination.name, "sha256": digest(destination), "bytes": destination.stat().st_size},
        "acquisition_sha256": digest(out / "acquisition.json"),
        "tokenizer": tokenizer_pointer(),
        "token_count_method": "encode_document; includes document boundaries",
        "network_reserved_bytes": budget.reserved,
        "release": False,
        "generated_text": False,
        "limitations": [
            "candidate preparation; production rights and semantic review remain pending",
            "catalog language is not a simplified/traditional script audit",
            "exact dedup is within this batch; cross-batch and same-work edition checks remain pending",
        ],
    }
    _write(out / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report

