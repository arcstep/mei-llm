#!/usr/bin/env python3
"""Python runtime for the zh-v2 rebuild lineage (SFT/QAT weights).

Loads the SFT float master (or the Q4 pack, dequantized on load) plus the
frozen 24K-v3 tokenizer, and exposes the text interface the SFT corpus
teaches: prompt -> generation -> parsed structured output. This is the
training-contract-accurate binding; the portable Runtime51M separate-head
package stays a later optimization step.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.checkpoint import load_params  # noqa: E402


def _frozen_tokenizer():
    from common._repo import frozen_tokenizer_path
    from training.cpt.train_pretrain import _frozen_tokenizer as _ft
    return _ft()


def load_model(weights_path: Path, *, smoke: bool = False):
    from common._repo import ARCHITECTURE_DIR
    sys.path.insert(0, str(ARCHITECTURE_DIR))
    from architecture import NeedleZh
    from config import NeedleZhConfig
    model = NeedleZh(NeedleZhConfig().tiny() if smoke else NeedleZhConfig.from_spec())
    mx.eval(model.parameters())
    report = load_params(model, weights_path, strict=False, allow_missing_prefixes=(), return_report=True)
    return model, report


def generate(model, tok, prompt: str, *, max_new: int = 128, temperature: float = 0.0) -> str:
    ids = mx.array(tok.encode(prompt), dtype=mx.int32)[None]
    generated: list[int] = []
    for _ in range(max_new):
        logits = model(ids)["logits"][0, -1, :]
        if temperature <= 0:
            next_id = int(mx.argmax(logits, axis=-1).item())
        else:
            probs = mx.softmax(logits / temperature)
            next_id = int(mx.random.categorical(probs).item())
        if next_id == tok.eos_id if hasattr(tok, "eos_id") else next_id == 0:
            break
        generated.append(next_id)
        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=-1)
    return tok.decode(generated) if hasattr(tok, "decode") else "".join(map(str, generated))


def _extract_json_objects(text: str) -> list[str]:
    """括号配平提取 JSON 对象（支持嵌套 arguments）。"""
    objects: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        depth = 0
        in_str = False
        escaped = False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : j + 1])
                    i = j + 1
                    break
        else:
            break
    return objects


def parse_tool_call(text: str) -> dict[str, Any] | None:
    for candidate in _extract_json_objects(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and ("tool" in parsed or "refuse" in parsed):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_mw(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[^{}]*\"reason_code\"[^{}]*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--smoke-row", type=Path, required=True, help="one compiled jsonl row to run through")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    tok = _frozen_tokenizer()
    model, report = load_model(args.weights, smoke=args.smoke)
    print(json.dumps({"loaded": str(args.weights), "report": str(report)[:120]}, ensure_ascii=False))

    row = json.loads(args.smoke_row.read_text(encoding="utf-8").splitlines()[0])
    from training.tool_use.train_sft_newlineage import DeployIndex, RENDERERS, render_trajectory
    deploy = DeployIndex()
    family = row["family"]
    renderer = RENDERERS[family]
    prompt, gold = renderer(row, deploy)
    print(json.dumps({"family": family, "prompt_head": prompt[:120], "gold": gold[:160]}, ensure_ascii=False, indent=1))
    output = generate(model, tok, prompt, max_new=args.max_new)
    print(json.dumps({"generated": output[:400]}, ensure_ascii=False, indent=1))
    if family in ("full_call", "agent", "trajectory"):
        print(json.dumps({"parsed_tool_call": parse_tool_call(output)}, ensure_ascii=False))
    if family in ("mw_disposition", "trajectory"):
        print(json.dumps({"parsed_mw": parse_mw(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
