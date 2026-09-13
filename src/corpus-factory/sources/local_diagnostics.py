"""Read-only source diagnostics; raw byte checks do not certify semantics."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prefix_check(full: Path, head: Path) -> dict:
    n = head.stat().st_size
    same = True
    compared = 0
    with full.open("rb") as a, head.open("rb") as b:
        while block := b.read(4 << 20):
            other = a.read(len(block))
            if other != block:
                same = False
                break
            compared += len(block)
    body_equal_bytes = 0
    with full.open("rb") as a, head.open("rb") as b:
        aa, bb = a.read(4096), b.read(4096)
        ai, bi = aa.find(b"<dblp>"), bb.find(b"<dblp>")
        if ai >= 0 and bi >= 0:
            a.seek(ai + 6); b.seek(bi + 6)
            while block := b.read(4 << 20):
                other = a.read(len(block))
                if block != other:
                    body_equal_bytes += next((i for i, (x, y) in enumerate(zip(block, other)) if x != y), min(len(block), len(other)))
                    break
                body_equal_bytes += len(block)
    def types(path):
        counts = Counter()
        pattern = re.compile(rb"<(article|inproceedings|incollection|phdthesis|mastersthesis|www|book|proceedings)\b")
        carry = b""
        with path.open("rb") as f:
            while block := f.read(4 << 20):
                data = carry + block
                cut = max(0, len(data) - 64)
                counts.update(m.group(1).decode() for m in pattern.finditer(data) if m.start() < cut)
                carry = data[cut:]
            counts.update(m.group(1).decode() for m in pattern.finditer(carry))
        return dict(counts)
    return {"full_path": str(full), "head_path": str(head), "full_bytes": full.stat().st_size,
            "head_bytes": n, "head_sha256": fingerprint(head),
            "head_is_exact_byte_prefix": same, "verified_identical_prefix_bytes": compared,
            "identical_body_prefix_bytes_after_root_open": body_equal_bytes,
            "full_record_open_tag_counts": types(full), "head_record_open_tag_counts": types(head),
            "head_byte_fraction": n / full.stat().st_size,
            "scope": "byte comparison and opening-tag counts, not XML validity or semantic coverage"}


def diagnostics(root: Path, config: dict) -> dict:
    result = {"schema": "mei-local-source-diagnostics-v1", "training_adoption_eligible": False}
    if config.get("prefix_check"):
        item = config["prefix_check"]
        result["prefix_check"] = prefix_check(root / item["full"], root / item["head"])
    if config.get("dialogue_comparison"):
        result["dialogue_comparison"] = dialogue_comparison(root, config["dialogue_comparison"])
    result["receipts"] = []
    for rel in config.get("receipts", []):
        path = root / rel
        row = json.loads(path.read_text())
        result["receipts"].append({"path": str(path), "sha256": fingerprint(path),
            "facts": {k: row[k] for k in ("status", "reviewer", "reviewed_at", "counts", "documents",
                      "tokens", "training_adoption_eligible", "ordering", "dedup", "exclusion") if k in row}})
    return result


def dialogue_comparison(root: Path, config: dict) -> dict:
    """Validate a known legacy serializer against original source lines.

    This explicitly named legacy normalization is checked against source lines
    and prepared outputs; no code from a dataset or source snapshot is executed.
    """
    snapshot = root / config["serializer_snapshot"]
    if fingerprint(snapshot) != config["serializer_sha256"]:
        raise ValueError("legacy serializer hash changed")
    def clean_turn(text):
        text = unicodedata.normalize("NFKC", text).strip()
        text = re.sub(r"(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])", "", text)
        text = re.sub(r"\s+([,.!?;:，。！？；：、])", r"\1", text)
        text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        return re.sub(r"\s+", " ", text).strip()
    samples = {}
    input_hashes = {}
    for kind in ("raw", "filtered"):
        path = root / config[f"{kind}_sample"]
        input_hashes[str(path)] = fingerprint(path)
        samples[kind] = json.loads(path.read_text())["samples"]
    wanted = {r["original"]["source_line"]: r for r in samples["filtered"]}
    matched = []
    with gzip.open(root / config["raw_jsonl_gz"], "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if number in wanted:
                original = json.loads(line)
                expected = "\n".join(clean_turn(t) for t in original)
                row = wanted[number]
                matched.append({"source_line": number, "original": original, "prepared_text": row["text"],
                                "serializer_matches": expected == row["text"], "raw_turns": len(original),
                                "derived_turns": len(row["text"].split("\n"))})
            if len(matched) == len(wanted):
                break
    valid = len(matched) == len(wanted) and all(r["serializer_matches"] and r["raw_turns"] == r["derived_turns"] for r in matched)
    def describe(rows, kind):
        turns = [r["original"] if kind == "raw" else r["text"].split("\n") for r in rows]
        lengths = sorted(sum(len(t) for t in dialog) for dialog in turns)
        return {"records": len(rows), "turn_count_histogram": dict(Counter(map(len, turns))),
                "dialogue_characters_median": lengths[len(lengths)//2] if lengths else None,
                "dialogue_characters_mean": sum(lengths)/len(lengths) if lengths else None,
                "records_with_question_mark": sum(any("?" in t or "？" in t for t in d) for d in turns)}
    return {"serializer_snapshot": str(snapshot), "serializer_sha256": fingerprint(snapshot),
            "input_sample_hashes": input_hashes, "matched_records": len(matched),
            "historical_serialization_verified_on_sample": valid,
            "raw_statistics": describe(samples["raw"], "raw"),
            "filtered_statistics": describe(samples["filtered"], "filtered") if valid else None,
            "paired_evidence": matched,
            "limitations": ["two independent record samples, not a census of filtering effects",
                            "raw text has pre-cleaning spaces; character lengths are not normalized-equivalent",
                            "question marks are lexical observations, not semantic intent labels"]}
