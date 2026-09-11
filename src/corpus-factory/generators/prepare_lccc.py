from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import sqlite3
import unicodedata
from array import array
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def clean_turn(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"\s+([,.!?;:，。！？；：、])", r"\1", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(text: str) -> bytes:
    compact = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).lower()
    return hashlib.sha256(compact.encode()).digest()[:16]


def quality_reason(turns: list[str]) -> str | None:
    if not 2 <= len(turns) <= 16:
        return "turn_count"
    if any(not 2 <= len(turn) <= 600 for turn in turns):
        return "turn_length"
    text = "\n".join(turns)
    if len(re.findall(r"[\u3400-\u9fff]", text)) / max(1, len(text)) < 0.65:
        return "low_chinese_fraction"
    if re.search(r"https?://|www\.|<\||EVAL[-_]|(?:微信|QQ|加群|代购|返利|刷单).{0,8}\d|1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text, re.I):
        return "contact_url_or_marker"
    if re.search(r"(.)\1{5,}", text):
        return "repetition"
    if len({fingerprint(turn) for turn in turns}) < len(turns):
        return "repeated_turn"
    return None


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def prepare(download_manifest: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    raw = download_manifest.parent
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    binding = {str(config_path): digest(config_path), str(download_manifest): digest(download_manifest)}
    for item in json.loads(download_manifest.read_text())["files"]:
        path = raw / item["filename"]
        actual = digest(path)
        if actual != item["sha256"] or not item["hash_pinned"]:
            raise ValueError(f"download integrity failure: {path}")
        binding[str(path)] = actual
    counts = Counter()
    excluded = set()
    for split in ("valid", "test"):
        with gzip.open(raw / f"lccc_base_{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                turns = [clean_turn(turn) for turn in json.loads(line)]
                excluded.add(fingerprint("\n".join(turns)))
                excluded.update(fingerprint(turn) for turn in turns if len(turn) >= 8)
    for value in config["baseline_raw"]:
        path = Path(value)
        binding[str(path)] = digest(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                text = json.loads(line)["text"]
                text = re.sub(r"<sup>.*?</sup>|<\|[^>]+\|>|<eo[hmtrc]>", "", text)
                if len(text) >= 8:
                    excluded.add(fingerprint(text))
    eval_grams = set()
    for folder in config["eval_directories"]:
        for path in sorted(Path(folder).rglob("*.jsonl")):
            binding[str(path)] = digest(path)
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    for text in strings(json.loads(line)):
                        compact = re.sub(r"\s+", "", text)
                        if len(compact) >= 8:
                            excluded.add(fingerprint(compact))
                        for index in range(max(0, len(compact) - 15)):
                            eval_grams.add(compact[index:index + 16])
    print(json.dumps({"excluded_fingerprints": len(excluded), "eval_16grams": len(eval_grams)}), flush=True)
    connection = sqlite3.connect(output / "selection.sqlite")
    connection.execute("CREATE TABLE docs (key BLOB PRIMARY KEY, prompt BLOB UNIQUE, source_line INTEGER, text TEXT)")
    with gzip.open(raw / "lccc_base_train.jsonl.gz", "rt", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, 1):
            counts["input_train_documents"] += 1
            turns = [clean_turn(turn) for turn in json.loads(line)]
            reason = quality_reason(turns)
            text = "\n".join(turns)
            key = fingerprint(text)
            if not reason and (key in excluded or any(fingerprint(turn) in excluded for turn in turns if len(turn) >= 8)):
                reason = "heldout_or_prior_corpus_overlap"
            compact = re.sub(r"\s+", "", text)
            if not reason and any(compact[index:index + 16] in eval_grams for index in range(max(0, len(compact) - 15))):
                reason = "eval_16gram_overlap"
            if reason:
                counts[reason] += 1
                continue
            inserted = connection.execute("INSERT OR IGNORE INTO docs VALUES (?, ?, ?, ?)",
                                          (key, fingerprint(turns[0]), source_line, text)).rowcount
            counts["accepted_before_tokenization" if inserted else "duplicate_document_or_prompt"] += 1
            if source_line % 100000 == 0:
                connection.commit()
                print(json.dumps(dict(counts)), flush=True)
    connection.commit()
    source_path = Path(__file__).resolve().parents[1] / "sources/source_manager.py"
    spec = importlib.util.spec_from_file_location("lccc_tokenizer_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.load_tokenizer()
    token_counts = Counter()
    documents = Counter()
    samples = []
    with (output / "train.bin").open("xb") as train, (output / "valid.bin").open("xb") as valid, \
            (output / "documents.jsonl").open("x", encoding="utf-8") as ledger, \
            (output / "train.jsonl").open("x", encoding="utf-8") as training_text, \
            (output / "valid.jsonl").open("x", encoding="utf-8") as validation_text:
        for key, source_line, text in connection.execute("SELECT key, source_line, text FROM docs ORDER BY key"):
            split = "valid" if hashlib.sha256(b"split-v1" + key).digest()[0] < 2 else "train"
            ids = tokenizer.encode_document(text)
            if len(ids) > 2048:
                counts["overlong_tokenized"] += 1
                continue
            encoded = array("H", ids)
            import sys
            if sys.byteorder != "little":
                encoded.byteswap()
            (train if split == "train" else valid).write(encoded.tobytes())
            record = {"text": text, "group_id": key.hex(), "source_line": source_line,
                      "split": split, "token_offset": token_counts[split], "tokens": len(ids)}
            ledger.write(json.dumps({name: value for name, value in record.items() if name != "text"}) + "\n")
            (training_text if split == "train" else validation_text).write(json.dumps(record, ensure_ascii=False) + "\n")
            token_counts[split] += len(ids)
            documents[split] += 1
            if len(samples) < 100:
                samples.append(record)
            if documents["train"] % 100000 == 0:
                print(json.dumps({"tokens": dict(token_counts), "documents": dict(documents)}), flush=True)
            if token_counts["train"] >= int(config["target_train_tokens"]):
                break
    connection.close()
    (output / "review-samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "implementation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    return {
        "schema": "mei-lccc-dialogue-preparation-v1", "status": "pending_review",
        "source_id": "lccc-base", "source_revision": "5bd582fa28cd7143f2f9c852e08e23089d677c44",
        "corpus_reuse_eligible": False, "training_adoption_eligible": False,
        "inputs": binding, "implementation_sha256": digest(Path(__file__)),
        "tokenizer_sha256": tokenizer.model_sha256, "counts": dict(counts),
        "tokens": dict(token_counts), "documents": dict(documents),
        "artifacts": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "ordering": "global normalized document hash order, independent salted hash split",
        "dedup": "full corpus normalized document and initial-prompt dedup; punctuation/spacing insensitive",
        "exclusion": "upstream valid/test and prior MOSS exact normalized turns; locked eval exact and 16-character overlap",
        "synthetic_generation_used": False, "naturalness_human_review": "pending",
        "remaining_requirements": ["named source clearance", "semantic quality sample review",
                                   "near-duplicate cross-split audit before adoption"],
        "note": "Prepared tokens are not admitted tokens. No shared seen-ledger or canonical pool was changed."
    }
