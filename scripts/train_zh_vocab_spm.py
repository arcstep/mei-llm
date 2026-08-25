#!/usr/bin/env python3
"""Train a 24k unigram SentencePiece tokenizer for needle-zh.

Default --freeze-v1 writes zh-24k-v1 with Needle-compatible control IDs:
PAD=0 EOS=1 BOS=2 UNK=3, plus protocol / VRM user-defined symbols.
The old zh-24k.model remains a retired draft and must not mix with v1 checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_VOCAB, ROOT


PROTOCOL_HEAD = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<act_execute>",
    "<act_refuse>",
    "<act_expand>",
    "<act_shape>",
    "<act_escalate>",
    "<act_stop>",
]


def load_lines(path: Path, limit: int | None = None) -> list[str]:
    if not path.is_file():
        return []
    rows = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit is not None:
        return rows[:limit]
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    try:
        import sentencepiece as spm
    except ImportError:
        print("install sentencepiece: pip install -r requirements-corpus.txt", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=24000)
    ap.add_argument("--sample", type=Path, default=CORPUS_ZH_VOCAB / "raw" / "spm-sample.txt")
    ap.add_argument("--seeds", type=Path, default=CORPUS_ZH_VOCAB / "seeds")
    ap.add_argument("--out-prefix", type=Path, default=None)
    ap.add_argument("--max-cities", type=int, default=400)
    ap.add_argument("--freeze-v1", action="store_true", help="Write zh-24k-v1 + manifest")
    args = ap.parse_args()

    if not args.sample.is_file() or args.sample.stat().st_size < 10_000:
        print(f"need spm sample at {args.sample}", file=sys.stderr)
        return 1

    if args.out_prefix:
        prefix = args.out_prefix
    elif args.freeze_v1 and args.size == 24000:
        prefix = CORPUS_ZH_VOCAB / "zh-24k-v1"
    else:
        prefix = CORPUS_ZH_VOCAB / f"zh-{args.size // 1000}k"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    reserved = load_lines(args.seeds / "reserved-tokens.txt")
    hanzi = load_lines(args.seeds / "hanzi-level1.txt")
    cities = load_lines(args.seeds / "cities-zh.txt", limit=args.max_cities)
    rooms = load_lines(args.seeds / "rooms-zh.txt")

    # Control IDs occupied by pad/eos/bos/unk. User-defined start at 4.
    uds: list[str] = []
    seen = {"<unk>", "<s>", "</s>", "<pad>", "<PAD>", "<EOS>", "<BOS>", "<UNK>"}
    for tok in PROTOCOL_HEAD + reserved + hanzi + cities + rooms:
        if tok in seen:
            continue
        if "," in tok and tok != ",":
            continue
        seen.add(tok)
        uds.append(tok)

    uds_path = prefix.parent / "raw" / "user-defined-symbols-v1.txt"
    uds_path.parent.mkdir(parents=True, exist_ok=True)
    uds_path.write_text("\n".join(uds) + "\n", encoding="utf-8")
    print(f"user_defined_symbols={len(uds)} vocab_size={args.size}")

    spm.SentencePieceTrainer.train(
        input=str(args.sample),
        model_prefix=str(prefix),
        vocab_size=args.size,
        model_type="unigram",
        character_coverage=0.9995,
        user_defined_symbols=uds,
        unk_id=3,
        bos_id=2,
        eos_id=1,
        pad_id=0,
        train_extremely_large_corpus=True,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
        minloglevel=1,
    )
    vocab_src = Path(str(prefix) + ".vocab")
    if args.freeze_v1 and args.size == 24000 and vocab_src.is_file():
        (CORPUS_ZH_VOCAB / "vocab.txt").write_text(vocab_src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied {vocab_src} → vocab.txt")
        model_path = Path(str(prefix) + ".model")
        manifest = {
            "tokenizer_id": "zh-24k-v1",
            "retired_draft": "zh-24k.model",
            "spm_vocab_size": args.size,
            "pad_id": 0,
            "eos_id": 1,
            "bos_id": 2,
            "unk_id": 3,
            "user_defined_symbols_n": len(uds),
            "model_sha256": sha256_file(model_path),
            "sample": str(args.sample.relative_to(ROOT)) if str(args.sample).startswith(str(ROOT)) else str(args.sample),
            "sample_bytes": args.sample.stat().st_size,
        }
        man_path = CORPUS_ZH_VOCAB / "tokenizer-v1-manifest.json"
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {man_path}")
    print(f"model {prefix}.model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
