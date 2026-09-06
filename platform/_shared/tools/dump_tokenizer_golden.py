#!/usr/bin/env python3
"""Freeze zh-24k-v1 SentencePiece token IDs for Rust/WASM parity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/mei-1.2-51m/tokenizer/zh-24k-v1.model"
OUTPUT = ROOT / "platform/_shared/spec/golden/tokenizer_v2.json"


CASES = [
    "",
    "你好，世界！",
    " hello  world ",
    "ＡＢＣ１２３",
    "中文 mixed ASCII 123",
    "第一行\n\n第二行\t结束",
    "{\"城市\":\"上海\",\"温度\":23.5}",
    "<|im_start|>assistant\n<tool_call>{}</tool_call><|im_end|>",
    "路径／测试：状态＝成功",
    "🙂🚀",
]


def main() -> None:
    processor = spm.SentencePieceProcessor(model_file=str(MODEL))
    rows = []
    for text in CASES:
        ids = list(processor.encode(text, out_type=int))
        rows.append(
            {
                "text": text,
                "ids": ids,
                "pieces": list(processor.encode(text, out_type=str)),
                "decoded": processor.decode(ids),
            }
        )
    payload = {
        "schema_version": "mei-tokenizer-golden-v2",
        "tokenizer_id": "zh-24k-v1",
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "sentencepiece_version": spm.__version__,
        "cases": rows,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
