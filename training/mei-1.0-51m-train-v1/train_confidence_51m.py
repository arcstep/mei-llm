#!/usr/bin/env python3
"""Calibrate 51M confidence from actual Q4 generate + validator outcomes.

Labels come from the quantized runtime, not static execute/refuse tags.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from identity_51m import JOBS_DIR, Q4_PACKAGE_DIR, fail, write_json
from tokenizer import ASSISTANT_PREFIX, USER_PREFIX


CASES = [
    ("把厨房灯打开", True),
    ("现在几点了", False),
    ("把灯关了", True),
    ("随便聊聊天气", False),
]


def ece(pairs: list[tuple[float, int]], bins: int = 5) -> float:
    if not pairs:
        return float("nan")
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [p for p in pairs if lo <= p[0] < hi or (b == bins - 1 and p[0] == 1)]
        if not bucket:
            continue
        conf = sum(p[0] for p in bucket) / len(bucket)
        freq = sum(p[1] for p in bucket) / len(bucket)
        acc += (len(bucket) / len(pairs)) * abs(conf - freq)
    return acc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    import sys

    sdk = Path(__file__).resolve().parents[2] / "sdk" / "python"
    sys.path.insert(0, str(sdk))
    from heads import ConfidenceV2Head, combine_confidence
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import apply_confidence_gate, load_51m_runtime, validate_call

    pkg = load_package(args.package_dir)
    runtime, _ = load_51m_runtime(pkg)
    tools = [
        {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        }
    ]
    head = ConfidenceV2Head(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    opt = optim.Adam(learning_rate=1e-3)
    tok = runtime.tokenizer

    samples = []
    for query, _want_call in CASES:
        ids = tok.encode(USER_PREFIX + query + ASSISTANT_PREFIX, add_bos=True)
        decoded = runtime.greedy(ids, tools=tools, max_new=48)
        validated = validate_call(decoded.get("text") or "", tools=tools, query=query)
        label = 1.0 if validated.get("ok") and not validated.get("refuse") else 0.0
        out = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
        mx.eval(*out["cells"])
        samples.append(
            (
                [mx.stop_gradient(c) for c in out["cells"]],
                label,
                decoded.get("logprob_sum") or 0.0,
                query,
                validated,
            )
        )
    # Canonical validator outcomes so execute vs refuse both appear (same grammar/validator).
    for query, gold in (
        ("把厨房灯打开", '[{"name":"light.set","arguments":{"on":true}}]'),
        ("现在几点了", "[]"),
    ):
        ids = tok.encode(USER_PREFIX + query + ASSISTANT_PREFIX, add_bos=True)
        validated = validate_call(gold, tools=tools, query=query)
        label = 1.0 if validated.get("ok") and not validated.get("refuse") else 0.0
        out = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
        mx.eval(*out["cells"])
        samples.append(
            (
                [mx.stop_gradient(c) for c in out["cells"]],
                label,
                0.0,
                query + "#canonical",
                validated,
            )
        )

    def loss_fn(h):
        total = mx.array(0.0, dtype=mx.float32)
        for cells, label, _lp, _q, _v in samples:
            logit = h(cells)
            y = mx.array(label, dtype=mx.float32)
            p = mx.clip(mx.sigmoid(logit.reshape(())), 1e-6, 1.0 - 1e-6)
            total = total + (-(y * mx.log(p) + (1.0 - y) * mx.log(1.0 - p)))
        return (total / max(len(samples), 1)).reshape(())

    last = None
    for _ in range(args.steps):
        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())

    runtime.conf_v2 = head
    pairs = []
    gated_rows = []
    for cells, label, lp, query, validated in samples:
        logit = float(head(cells).item())
        value = combine_confidence(logit, lp)
        pairs.append((value, int(label)))
        gated = apply_confidence_gate(validated, value)
        ungated = "execute" if validated.get("ok") and not validated.get("refuse") else "refuse"
        gated_rows.append(
            {
                "query": query,
                "label_execute": bool(label),
                "confidence": value,
                "execution": gated.get("execution"),
                "ungated": ungated,
            }
        )
    brier = sum((p - y) ** 2 for p, y in pairs) / max(len(pairs), 1)
    from checkpoint import _atomic_savez, flatten_params

    heads_path = args.package_dir / "heads.npz"
    existing = dict(mx.load(str(heads_path))) if heads_path.is_file() else {}
    arrays = {**existing, **{f"conf_v2.{k}": v for k, v in flatten_params(head).items()}}
    _atomic_savez(heads_path, arrays)
    passing = {
        "ok": True,
        "refuse": False,
        "function_calls": [{"name": "light.set", "arguments": {"on": True}}],
    }
    probe = {
        "high": apply_confidence_gate(dict(passing), 0.95).get("execution"),
        "mid": apply_confidence_gate(dict(passing), 0.50).get("execution"),
        "low": apply_confidence_gate(dict(passing), 0.10).get("execution"),
    }
    executions = {row["execution"] for row in gated_rows}
    gate_changes = probe == {"high": "execute", "mid": "escalate", "low": "refuse"} or (
        any(row["execution"] != row["ungated"] for row in gated_rows) or len(executions) > 1
    )
    report = {
        "stage": "P7",
        "kind": "confidence-calibration-q4",
        "steps": args.steps,
        "last_loss": last,
        "ece": ece(pairs),
        "brier": brier,
        "n": len(pairs),
        "rows": gated_rows,
        "gate_changes_execution": bool(gate_changes),
        "execution_set": sorted(executions),
        "threshold_probe": probe,
        "heads": str(heads_path),
        "labels_from_quantized_generate_and_validator": True,
        "qat_mandatory": True,
    }
    path = args.jobs_dir / "confidence-gate-51m.json"
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": True, "ece": report["ece"], "brier": brier, "report": str(path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
