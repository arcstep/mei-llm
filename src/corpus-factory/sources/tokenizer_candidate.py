"""Train and audit one immutable 24K lossless tokenizer candidate."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

from profiling import resolve_path, ROOT, digest


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def rows(path: Path):
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str):
                raise ValueError(f"missing text at {path}:{line_number}")
            text.encode("utf-8", errors="strict")
            yield value


class LosslessProcessor:
    def __init__(self, model: Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model))
        self.byte_ids = {
            byte: int(self.sp.piece_to_id(f"<0x{byte:02X}>")) for byte in range(256)
        }
        if any(value == self.sp.unk_id() for value in self.byte_ids.values()):
            raise ValueError("candidate lacks complete byte fallback")

    def encode(self, text: str) -> list[int]:
        text.encode("utf-8", errors="strict")
        result: list[int] = []
        for index, part in enumerate(text.split("▁")):
            if index:
                result.extend(self.byte_ids[byte] for byte in "▁".encode("utf-8"))
            result.extend(self.sp.encode(part, out_type=int))
        return result

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


def run(config: dict, out: Path) -> dict:
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    if int(config["vocab_size"]) != 24_000:
        raise ValueError("this production candidate route is fixed at 24K")
    train = resolve_path(ROOT / config["train_jsonl"])
    dev = resolve_path(ROOT / config["dev_jsonl"])
    sample_manifest = resolve_path(ROOT / config["sample_manifest"])
    expected = config["input_sha256"]
    actual = {"train": digest(train), "dev": digest(dev), "manifest": digest(sample_manifest)}
    if actual != expected:
        raise ValueError(f"input hashes changed: {actual}")
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError("100 GiB disk reserve reached")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "config.json", config)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    prefix = out / "tokenizer"
    started = time.time()
    options = {
        "model_prefix": str(prefix),
        "vocab_size": 24_000,
        "model_type": "unigram",
        "character_coverage": 0.9995,
        "byte_fallback": True,
        "split_digits": True,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": False,
        "add_dummy_prefix": False,
        "escape_whitespaces": True,
        "user_defined_symbols": list(config["protocol_symbols"]),
        "hard_vocab_limit": True,
        "num_threads": 2,
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "max_sentence_length": 8192,
        "seed_sentencepiece_size": 200_000,
        "max_sentencepiece_length": 16,
        "pad_id": 0,
        "eos_id": 1,
        "bos_id": 2,
        "unk_id": 3,
        "minloglevel": 1,
    }
    write_new(out / "effective-options.json", {**options, "model_prefix": "tokenizer"})
    spm.SentencePieceTrainer.train(
        sentence_iterator=(row["text"] for row in rows(train)), **options
    )
    model_path = prefix.with_suffix(".model")
    model = pb.ModelProto()
    model.ParseFromString(model_path.read_bytes())
    processor = LosslessProcessor(model_path)
    if processor.sp.vocab_size() != 24_000:
        raise ValueError(f"candidate piece size is {processor.sp.vocab_size()}, expected 24000")
    special = {
        "pad_id": processor.sp.pad_id(), "eos_id": processor.sp.eos_id(),
        "bos_id": processor.sp.bos_id(), "unk_id": processor.sp.unk_id(),
    }
    if special != {"pad_id": 0, "eos_id": 1, "bos_id": 2, "unk_id": 3}:
        raise ValueError(f"special IDs changed: {special}")
    failures = []
    domains = defaultdict(lambda: Counter(records=0, characters=0, utf8_bytes=0, tokens=0))
    source_domains = config.get("source_domains") or json.loads(
        (resolve_path(ROOT / config["source_domains_config"])).read_text(encoding="utf-8")
    )["source_domains"]
    for row in rows(dev):
        text = row["text"]
        ids = processor.encode(text)
        decoded = processor.decode(ids)
        domain = source_domains[row["source_id"]]
        cell = domains[domain]
        cell.update(records=1, characters=len(text), utf8_bytes=len(text.encode()), tokens=len(ids))
        if decoded != text:
            if len(failures) < 100:
                failures.append({
                    "source_id": row["source_id"], "file": row["file"],
                    "row_index": row["row_index"], "before": text, "after": decoded,
                })
    boundary = []
    for text in config["boundary_probes"]:
        ids = processor.encode(text)
        boundary.append({"text": text, "ids": ids, "decoded": processor.decode(ids),
                         "exact": processor.decode(ids) == text})
    hard_errors = []
    if failures:
        hard_errors.append(f"{len(failures)} sampled round-trip failures")
    if any(not item["exact"] for item in boundary):
        hard_errors.append("boundary round-trip failure")
    manifest = {
        "schema": "mei-51m-tokenizer-candidate-v1",
        "tokenizer_id": config["tokenizer_id"],
        "status": "candidate_python_audited" if not hard_errors else "candidate_failed",
        "production_ready": False,
        "encoding_profile_id": "mei-lossless-identity-v1",
        "model_type": "unigram", "spm_vocab_size": processor.sp.vocab_size(),
        "model_sha256": digest(model_path), "vocab_sha256": digest(prefix.with_suffix(".vocab")),
        "special_ids": special, "byte_fallback": True, "split_digits": True,
        "normalizer": {
            "name": model.normalizer_spec.name,
            "add_dummy_prefix": model.normalizer_spec.add_dummy_prefix,
            "remove_extra_whitespaces": model.normalizer_spec.remove_extra_whitespaces,
            "escape_whitespaces": model.normalizer_spec.escape_whitespaces,
            "literal_space_marker_policy": "UTF-8 byte fallback",
        },
        "protocol_symbols": list(config["protocol_symbols"]),
        "sample": {"train_jsonl": config["train_jsonl"], "dev_jsonl": config["dev_jsonl"],
                   "sha256": actual},
        "python_audit": {
            "dev_records": sum(value["records"] for value in domains.values()),
            "failures": failures, "boundary": boundary,
            "domains": {key: dict(value) for key, value in domains.items()},
        },
        "hard_errors": hard_errors, "elapsed_seconds": time.time() - started,
        "runtime_audit": "pending", "current_mutated": False,
    }
    write_new(out / "manifest.json", manifest)
    if hard_errors:
        raise ValueError("; ".join(hard_errors))
    return manifest
