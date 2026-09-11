from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def filter_candidate(receipt_path: Path, config_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    config = json.loads(config_path.read_text())
    source = receipt_path.parent
    for name, expected in receipt["artifacts"].items():
        if digest(source / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    rules = {name: re.compile(pattern, re.I) for name, pattern in config["rules"].items()}
    counts = Counter()
    tokens = Counter()
    documents = Counter()
    review = []
    for split in ("train", "valid"):
        with (source / f"{split}.bin").open("rb") as encoded_source, \
                (source / f"{split}.jsonl").open(encoding="utf-8") as rows, \
                (output / f"{split}.bin").open("xb") as encoded_target, \
                (output / f"{split}.jsonl").open("x", encoding="utf-8") as accepted:
            for line in rows:
                row = json.loads(line)
                reason = next((name for name, pattern in rules.items() if pattern.search(row["text"])), None)
                if reason:
                    counts[f"{split}:{reason}"] += 1
                    continue
                encoded_source.seek(row["token_offset"] * 2)
                encoded = encoded_source.read(row["tokens"] * 2)
                if len(encoded) != row["tokens"] * 2:
                    raise ValueError("truncated token stream")
                encoded_target.write(encoded)
                row["token_offset"] = tokens[split]
                tokens[split] += row["tokens"]
                documents[split] += 1
                accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
                if split == "train" and len(review) < 100:
                    review.append(row)
    (output / "review-samples.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "implementation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    return {
        "schema": "mei-dialogue-content-filter-v1", "status": "pending_named_review",
        "training_adoption_eligible": False, "corpus_reuse_eligible": False,
        "input_receipt": str(receipt_path), "input_receipt_sha256": digest(receipt_path),
        "config_sha256": digest(config_path), "implementation_sha256": digest(Path(__file__)),
        "counts": dict(counts), "tokens": dict(tokens), "documents": dict(documents),
        "tokenizer_sha256": receipt["tokenizer_sha256"],
        "isolation": "subset of audited v2; no split reassignment or new text",
        "artifacts": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "limitations": ["lexical filters do not prove semantic safety or factual correctness",
                        "social dialogue is intended as a CPT supplement, not task SFT or answer gold"],
    }
