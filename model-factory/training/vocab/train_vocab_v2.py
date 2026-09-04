#!/usr/bin/env python3
"""Train the zh-<n>k-v2 SentencePiece vocabulary from a target-distribution sample.

Flow: sample (streaming, weighted) -> train (unigram + user symbols) ->
validate (tools atomic, markers, hanzi coverage, roundtrip) -> optional freeze.
Freeze rewrites the frozen-tokenizer pointer (TOKENIZER.json) as a supersede
event and refuses to run unless validation passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_DIR = ROOT / "models/mei-1.0-51m/tokenizer"
POINTER_PATH = ROOT / "models/mei-1.0-51m/architecture/tokenizer/TOKENIZER.json"
V1_MODEL = TOKENIZER_DIR / "zh-24k-v1.model"

# Markers frozen in the v1 vocabulary; carried into v2 verbatim.
V1_MARKERS = [
    "<|im_start|>", "<|im_end|>",
    "<think>", "</think>",
    "<tools>", "</tools>",
    "<tool_call>", "</tool_call>",
    "<tool_result>", "</tool_result>",
    "<act_execute>", "<act_refuse>", "<act_expand>", "<act_shape>",
    "<act_escalate>", "<act_stop>",
]


class VocabError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_texts(source: dict[str, Any]) -> Iterator[str]:
    kind = source["kind"]
    base = Path(source["path"])
    if not base.is_absolute():
        base = ROOT / base
    if not base.exists():
        print(f"skip missing source: {base}", file=sys.stderr)
        return
    paths = sorted(base.glob(source.get("glob", "*")))
    if not paths:
        print(f"skip empty source: {base}", file=sys.stderr)
        return
    for path in paths:
        if not path.is_file():
            continue
        if kind == "jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = value.get("text")
                    if isinstance(text, str) and text.strip():
                        yield text
        elif kind == "parquet":
            import pyarrow.parquet as parquet

            table = parquet.ParquetFile(path)
            column = next(
                (name for name in ("text", "content") if name in table.schema_arrow.names),
                None,
            )
            if not column:
                continue
            for batch in table.iter_batches(batch_size=256, columns=[column]):
                for value in batch.column(0).to_pylist():
                    if isinstance(value, str) and value.strip():
                        yield value
        elif kind == "files":
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                yield text
        else:
            raise VocabError(f"unsupported source kind: {kind}")


def sample(sources: list[dict[str, Any]], out: Path, target_chars: int) -> dict[str, int]:
    """Weighted streaming sample; one document per line."""
    total_weight = sum(float(source.get("weight", 1.0)) for source in sources)
    quotas = {
        index: int(target_chars * float(source.get("weight", 1.0)) / total_weight)
        for index, source in enumerate(sources)
    }
    counters = {index: 0 for index in range(len(sources))}
    seen: set[int] = set()
    with out.open("w", encoding="utf-8") as handle:
        # Round-robin over sources so late sources are not starved.
        rounds = 0
        while True:
            rounds += 1
            progressed = False
            for index, source in enumerate(sources):
                if counters[index] >= quotas[index]:
                    continue
                iterator = source.get("_iterator")
                if iterator is None:
                    iterator = iter(_iter_texts(source))
                    source["_iterator"] = iterator
                while counters[index] < quotas[index]:
                    try:
                        text = next(iterator)
                    except StopIteration:
                        break
                    line = text.strip().replace("\n", " ")
                    if len(line) < 8:
                        continue
                    fingerprint = hash(line)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    handle.write(line + "\n")
                    counters[index] += len(line)
                    progressed = True
            if not progressed or rounds > 200:
                break
    return counters


def load_tool_symbols(tool_universes: list[Path]) -> list[str]:
    symbols: list[str] = []
    for path in tool_universes:
        value = json.loads(path.read_text(encoding="utf-8"))
        for tool in value.get("tools", []):
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                symbols.append(name.strip())
    return sorted(set(symbols))


def train(
    sample_path: Path,
    model_prefix: Path,
    vocab_size: int,
    user_symbols: list[str],
) -> None:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(sample_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        user_defined_symbols=user_symbols,
        hard_vocab_limit=False,
        byte_fallback=True,
        character_coverage=0.9995,
        split_digits=True,
        max_sentence_length=64_000_000,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
    )


def validate(
    model_path: Path,
    tool_symbols: list[str],
    marker_symbols: list[str],
    hanzi_level1: set[str],
) -> dict[str, Any]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    errors: list[str] = []
    pieces = [processor.id_to_piece(index) for index in range(processor.get_piece_size())]
    probes: dict[str, Any] = {}
    for tool in tool_symbols:
        if f"▁{tool}" not in pieces:
            errors.append(f"tool not atomic: {tool}")
    for marker in marker_symbols:
        if marker not in pieces:
            errors.append(f"marker missing: {marker}")
    covered = {piece for piece in pieces if len(piece) == 1 and piece in hanzi_level1}
    probes["hanzi_level1_covered"] = len(covered)
    if len(covered) < 3400:
        errors.append(f"hanzi level-1 coverage too low: {len(covered)}/3500")
    for text in ("北京现在天气怎么样", '{"name":"nod","arguments":{}}', "SELECT * FROM t WHERE x = 1"):
        roundtrip = processor.decode(processor.encode(text, out_type=int))
        if roundtrip != text:
            errors.append(f"roundtrip failed: {text!r} -> {roundtrip!r}")
        probes[f"tokens:{text[:20]}"] = len(processor.encode(text, out_type=int))
    probes["piece_size"] = processor.get_piece_size()
    probes["piece_size_no_byte_fallback"] = len(
        [piece for piece in pieces if not piece.startswith("<0x")]
    )
    return {"errors": errors, "probes": probes}


def load_hanzi_level1() -> set[str]:
    # Level-1 common hanzi probe set (3500 chars), reused from the v1 coverage
    # check; sampled from the v1 vocab itself when no external table is present.
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(V1_MODEL))
    return {
        processor.id_to_piece(index)
        for index in range(processor.get_piece_size())
        if len(processor.id_to_piece(index)) == 1
        and "一" <= processor.id_to_piece(index) <= "鿿"
    }


def write_manifest(
    model_path: Path,
    manifest_path: Path,
    *,
    tokenizer_id: str,
    declared_vocab_size: int,
    model_type: str,
    user_symbols_n: int,
    sample_chars: int,
    sample_sha256: str,
    validation: dict[str, Any],
    frozen_at: str,
) -> dict[str, Any]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    manifest = {
        "schema": "mei-51m-tokenizer-manifest-v1",
        "tokenizer_id": tokenizer_id,
        "spm_vocab_size": processor.get_piece_size(),
        "declared_vocab_size": declared_vocab_size,
        "model_type": model_type,
        "pad_id": 0,
        "eos_id": 1,
        "bos_id": 2,
        "unk_id": 3,
        "user_defined_symbols_n": user_symbols_n,
        "model_sha256": sha256_file(model_path),
        "sample_sha256": sample_sha256,
        "sample_chars": sample_chars,
        "validation": validation,
        "frozen_at": frozen_at,
        "supersedes": "zh-24k-v1",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def freeze_pointer(model_path: Path, manifest: dict[str, Any], pointer_path: Path) -> dict[str, Any]:
    pointer = {
        "schema": "mei-51m-tokenizer-pointer-v1",
        "current": f"tokenizer/{model_path.name}",
        "tokenizer_id": manifest["tokenizer_id"],
        "manifest": f"tokenizer-v2-manifest.json",
        "model_sha256": manifest["model_sha256"],
        "vocab_size": manifest["spm_vocab_size"],
        "status": "frozen",
        "frozen_at": manifest["frozen_at"],
        "supersedes": "zh-24k-v1",
    }
    existing: dict[str, Any] = {}
    if pointer_path.is_file():
        existing = json.loads(pointer_path.read_text(encoding="utf-8"))
        if (
            existing.get("tokenizer_id") == pointer["tokenizer_id"]
            and existing.get("model_sha256") == pointer["model_sha256"]
        ):
            return existing  # idempotent: same frozen tokenizer already pointed
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pointer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="JSON: [{path, kind, weight, glob}]")
    parser.add_argument("--sample-chars", type=int, default=400_000_000)
    parser.add_argument("--out-sample", type=Path)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--tokenizer-id", default="zh-32k-v2")
    parser.add_argument("--tool-universe", action="append", type=Path, default=[])
    parser.add_argument("--out-model-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--pointer-path", type=Path, default=POINTER_PATH)
    parser.add_argument("--freeze", action="store_true", help="rewrite TOKENIZER.json pointer (supersede event)")
    parser.add_argument("--freeze-only", action="store_true", help="skip training; freeze an already-validated model+manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze_only:
        final_model = args.out_model_dir / f"{args.tokenizer_id}.model"
        manifest_path = args.out_model_dir / "tokenizer-v2-manifest.json"
        if not final_model.is_file() or not manifest_path.is_file():
            raise VocabError("--freeze-only requires an existing model + manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pointer = freeze_pointer(final_model, manifest, args.pointer_path)
        print("pointer:", json.dumps(pointer, ensure_ascii=False, indent=2))
        return 0
    if not args.sources or not args.out_sample:
        raise VocabError("--sources and --out-sample are required unless --freeze-only")
    value = json.loads(args.sources.read_text(encoding="utf-8"))
    sources = value["sources"] if isinstance(value, dict) else value
    counters = sample(sources, args.out_sample, args.sample_chars)
    print(
        "sampled chars by source:", json.dumps(counters, ensure_ascii=False, indent=2)
    )
    tool_symbols = load_tool_symbols(args.tool_universe)
    # SentencePiece keeps user-defined symbols verbatim; tool names carry the
    # ▁ prefix so they stay atomic word pieces (v1 precedent: ▁nod).
    user_symbols = [f"▁{tool}" for tool in tool_symbols] + list(V1_MARKERS)
    with tempfile.TemporaryDirectory(prefix="vocab-v2-") as raw:
        model_prefix = Path(raw) / args.tokenizer_id
        train(args.out_sample, model_prefix, args.vocab_size, user_symbols)
        model_path = model_prefix.with_suffix(".model")
        validation = validate(model_path, tool_symbols, V1_MARKERS, load_hanzi_level1())
        print("validation:", json.dumps(validation, ensure_ascii=False, indent=2))
        if validation["errors"]:
            raise VocabError(f"vocab validation failed: {validation['errors']}")
        args.out_model_dir.mkdir(parents=True, exist_ok=True)
        final_model = args.out_model_dir / f"{args.tokenizer_id}.model"
        if final_model.exists():
            raise FileExistsError(f"refusing to overwrite tokenizer: {final_model}")
        model_path.replace(final_model)
        frozen_at = datetime.now(timezone.utc).isoformat()
        manifest = write_manifest(
            final_model,
            args.out_model_dir / "tokenizer-v2-manifest.json",
            tokenizer_id=args.tokenizer_id,
            declared_vocab_size=args.vocab_size,
            model_type="unigram",
            user_symbols_n=len(user_symbols),
            sample_chars=args.sample_chars,
            sample_sha256=sha256_file(args.out_sample),
            validation=validation,
            frozen_at=frozen_at,
        )
        print("manifest:", json.dumps(manifest, ensure_ascii=False, indent=2))
        if args.freeze:
            pointer = freeze_pointer(final_model, manifest, args.pointer_path)
            print("pointer:", json.dumps(pointer, ensure_ascii=False, indent=2))
        else:
            print("not frozen; rerun with --freeze to supersede the pointer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
