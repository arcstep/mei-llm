"""Materialize pinned source archives as auditable CPT text candidates."""
from __future__ import annotations

from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from profiling import Budget, digest, fetch
from source_manager import load_tokenizer, tokenizer_pointer


def _write(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _selected(path: str, patterns: list[str], excluded: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns) and not any(
        fnmatch.fnmatch(path, pattern) for pattern in excluded
    )


def run(config: dict, out: Path, allow_network: bool) -> dict:
    if not allow_network:
        raise ValueError("archive materialization requires --allow-network")
    out.mkdir(parents=True, exist_ok=False)
    reserve = int(config.get("reserve_disk_bytes", 100 * 1024**3))
    if shutil.disk_usage(out).free < reserve:
        raise RuntimeError("disk reserve reached")
    _write(out / "config.json", config)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    raw_dir = out / "archives"
    raw_dir.mkdir()
    budget = Budget(out, int(config.get("max_network_bytes", 4 * 1024**3)), reserve)
    tokenizer = load_tokenizer()
    seen: set[bytes] = set()
    counts = Counter()
    total_tokens = 0
    total_chars = 0
    outputs = []

    for index, source in enumerate(config["sources"]):
        max_bytes = int(source["max_bytes"])
        payload, headers = fetch(source["url"], budget, length=max_bytes)
        raw_path = raw_dir / f"{index:03d}.zip"
        raw_path.write_bytes(payload)
        output_path = out / f"part-{index:03d}.jsonl"
        part = Counter()
        part_tokens = 0
        with zipfile.ZipFile(raw_path) as archive, output_path.open("x") as target:
            for member in sorted(archive.infolist(), key=lambda row: row.filename):
                if member.is_dir():
                    continue
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                    raise ValueError("unsafe or unrooted archive member")
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                if not _selected(relative, source["include"], source.get("exclude", [])):
                    continue
                part["seen"] += 1
                raw = archive.read(member)
                if len(raw) > int(source.get("max_member_bytes", 16 << 20)):
                    part["oversize"] += 1
                    continue
                try:
                    text = raw.decode(source.get("encoding", "utf-8"))
                except UnicodeDecodeError:
                    part["decode_error"] += 1
                    continue
                if not text.strip() or "\x00" in text:
                    part["empty_or_nul"] += 1
                    continue
                key = hashlib.sha256(text.encode()).digest()
                if key in seen:
                    part["exact_duplicate"] += 1
                    continue
                seen.add(key)
                tokens = len(tokenizer.encode_document(text))
                record = {
                    "id": hashlib.sha256((source["source_id"] + ":" + relative).encode()).hexdigest(),
                    "text": text,
                    "text_sha256": key.hex(),
                    "source_id": source["source_id"],
                    "tokens": tokens,
                    "group_id": source["source_id"] + ":" + str(PurePosixPath(relative).parent),
                    "split": "candidate-unassigned",
                    "origin": {
                        "repository": source["repository"],
                        "revision": source["revision"],
                        "archive_sha256": digest(raw_path),
                        "path": relative,
                    },
                    "metadata": {
                        "language": source.get("language", "structured"),
                        "content_kind": source["content_kind"],
                        "license_review": source.get("license_review", "pending"),
                    },
                    "candidate_kind": "cpt_raw",
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                part["written"] += 1
                part_tokens += tokens
                total_chars += len(text)
        counts.update(part)
        total_tokens += part_tokens
        output = {
            "source_id": source["source_id"],
            "file": output_path.name,
            "sha256": digest(output_path),
            "bytes": output_path.stat().st_size,
            "tokens": part_tokens,
            "counts": dict(part),
            "archive_file": raw_path.name,
            "archive_sha256": digest(raw_path),
            "archive_bytes": raw_path.stat().st_size,
            "resolved_url": headers.get("X-Mei-Resolved-URL"),
        }
        outputs.append(output)
        print(json.dumps(output, ensure_ascii=False), flush=True)

    report = {
        "schema": "mei-archive-cpt-candidate-v1",
        "status": "candidate_complete",
        "records": counts["written"],
        "tokens": total_tokens,
        "characters": total_chars,
        "counts": dict(counts),
        "outputs": outputs,
        "tokenizer": tokenizer_pointer(),
        "token_count_method": "encode_document; includes document boundaries",
        "network_reserved_bytes": budget.reserved,
        "release": False,
        "generated_text": False,
        "limitations": [
            "candidate preparation; semantic and license admission remain separate",
            "exact dedup is within this batch; cross-batch and near-duplicate checks remain pending",
            "whole related source files are preserved; 2048-token windowing is a later non-additive view",
        ],
    }
    _write(out / "manifest.json", report)
    return report

