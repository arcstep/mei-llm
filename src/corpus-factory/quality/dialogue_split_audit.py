from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).lower()


def grams(text: str) -> set[bytes]:
    text = compact(text)
    return {hashlib.blake2b(text[index:index + 5].encode(), digest_size=8).digest()
            for index in range(max(1, len(text) - 4))}


def jaccard(left: set, right: set) -> float:
    return len(left & right) / max(1, len(left | right))


def audit(prepared_receipt: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    prepared = prepared_receipt.parent
    receipt = json.loads(prepared_receipt.read_text())
    for name, expected in receipt["artifacts"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"prepared artifact hash mismatch: {name}")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    inverted = defaultdict(set)
    heldout_turns = set()
    binding = {str(prepared_receipt): digest(prepared_receipt), str(config_path): digest(config_path)}

    def add_heldout(text: str) -> None:
        features = grams(text)
        index = len(documents)
        documents.append(features)
        for feature in sorted(features)[:8]:
            inverted[feature].add(index)
        for turn in text.splitlines():
            normalized = compact(turn)
            if len(normalized) >= 8:
                heldout_turns.add(hashlib.sha256(normalized.encode()).digest()[:16])

    with (prepared / "valid.jsonl").open() as stream:
        for line in stream:
            add_heldout(json.loads(line)["text"])
    for value in config["upstream_heldout"]:
        path = Path(value)
        binding[str(path)] = digest(path)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                add_heldout("\n".join(json.loads(line)))
    counts = Counter()
    train_tokens = 0
    samples = []
    with (prepared / "train.bin").open("rb") as source, \
            (prepared / "train.jsonl").open(encoding="utf-8") as rows, \
            (output / "train.bin").open("xb") as target, \
            (output / "train.jsonl").open("x", encoding="utf-8") as accepted:
        for line in rows:
            row = json.loads(line)
            counts["input_documents"] += 1
            shared = any(hashlib.sha256(compact(turn).encode()).digest()[:16] in heldout_turns
                         for turn in row["text"].splitlines() if len(compact(turn)) >= 8)
            features = grams(row["text"]) if not shared else set()
            candidates = set()
            for feature in features:
                candidates.update(inverted.get(feature, ()))
            near = any(jaccard(features, documents[index]) >= float(config["jaccard_threshold"])
                       for index in candidates)
            if shared or near:
                counts["shared_heldout_turn" if shared else "near_heldout_document"] += 1
                continue
            source.seek(row["token_offset"] * 2)
            encoded = source.read(row["tokens"] * 2)
            if len(encoded) != row["tokens"] * 2:
                raise ValueError("truncated prepared token stream")
            target.write(encoded)
            row["token_offset"] = train_tokens
            train_tokens += row["tokens"]
            counts["accepted_documents"] += 1
            accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(samples) < 100:
                samples.append(row)
            if counts["input_documents"] % 100000 == 0:
                print(json.dumps({**counts, "train_tokens": train_tokens}), flush=True)
    for name in ("valid.bin", "valid.jsonl"):
        shutil.copyfile(prepared / name, output / name)
    (output / "review-samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "implementation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    return {
        "schema": "mei-dialogue-split-audit-v1", "status": "pending_review",
        "training_adoption_eligible": False, "corpus_reuse_eligible": False,
        "inputs": binding, "implementation_sha256": digest(Path(__file__)),
        "counts": dict(counts), "train_tokens": train_tokens,
        "valid_tokens": (output / "valid.bin").stat().st_size // 2,
        "tokenizer_sha256": receipt["tokenizer_sha256"],
        "heldout_documents": len(documents),
        "near_duplicate_method": "character-5gram bottom-8 inverted candidates, exact Jaccard confirmation",
        "jaccard_threshold": config["jaccard_threshold"],
        "limitations": ["candidate search is approximate; no claim of exhaustive semantic duplicate removal",
                        "source clearance and semantic quality review remain required",
                        "historical MOSS exclusion is normalized exact-turn based"],
        "artifacts": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
    }
