#!/usr/bin/env python3
"""Quant-aware full-call SFT on the packed Q4 51M. Target is complete JSON or [].

Trains on the already-dequantized Q4 weights (deployment math). Does not
re-evaluate the float model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from common.identity_51m import JOBS_DIR, Q4_PACKAGE_DIR, fail, write_json
from tokenizer import ASSISTANT_PREFIX, TURN_END, USER_PREFIX


EXAMPLES = [
    ("把厨房灯打开", '[{"name":"light.set","arguments":{"on":true}}]'),
    ("现在几点了", "[]"),
    ("把灯关了", '[{"name":"light.set","arguments":{"on":false}}]'),
]


def encode_pair(tok, query: str, target: str) -> tuple[list[int], list[int]]:
    prompt = tok.encode(USER_PREFIX + query + ASSISTANT_PREFIX, add_bos=True)
    answer = tok.encode(target) + tok.encode(TURN_END) + [tok.eos_id]
    return prompt, answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    import sys

    sdk = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file()) / "platform/python-sdk"
    sys.path.insert(0, str(sdk))
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime, validate_call

    pkg = load_package(args.package_dir)
    runtime, _ = load_51m_runtime(pkg)
    model = runtime.model
    tok = runtime.tokenizer
    opt = optim.Adam(learning_rate=2e-4)
    pairs = [encode_pair(tok, q, a) for q, a in EXAMPLES]
    tools = [
        {
            "name": "light.set",
            "description": "开关灯",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        }
    ]

    def loss_fn(m):
        total = mx.array(0.0, dtype=mx.float32)
        n = 0
        for prompt, answer in pairs:
            ids = prompt + answer
            arr = mx.array([ids], dtype=mx.int32)
            logits = m(arr)["logits"].astype(mx.float32)
            logp = nn.log_softmax(logits, axis=-1)
            for i, tid in enumerate(answer):
                pos = len(prompt) - 1 + i
                total = total + (-logp[0, pos, int(tid)])
                n += 1
        return total / max(n, 1)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    last = None
    for _ in range(args.steps):
        loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), loss)
        last = float(loss.item())

    rows = []
    for query, gold in EXAMPLES:
        prompt, _ = encode_pair(tok, query, gold)
        decoded = runtime.greedy(prompt, tools=tools, max_new=48, decode_mode="constrained")
        text = decoded.get("text") or ""
        validated = validate_call(text, tools=tools, query=query)
        rows.append(
            {
                "query": query,
                "gold": gold,
                "generated": text,
                "validated_ok": bool(validated.get("ok")),
                "refuse": bool(validated.get("refuse")),
            }
        )
    exact = sum(1 for row in rows if row["generated"].strip() == row["gold"])
    report = {
        "stage": "P6",
        "kind": "quant-aware-full-call-sft",
        "steps": args.steps,
        "last_loss": last,
        "n_examples": len(EXAMPLES),
        "exact_overfit": exact / len(EXAMPLES),
        "turns": rows,
        "quantized_backbone": True,
        "target_is_full_call_or_empty": True,
        "qat_mandatory": True,
        "not_a_claim": "Toy overfit is not the 10k clean.v2 product SFT.",
    }
    path = args.jobs_dir / "tool-sft-51m.json"
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": True, "report": str(path), "last_loss": last, "exact": report["exact_overfit"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
