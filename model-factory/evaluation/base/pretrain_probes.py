"""Held-out pretraining probe helpers used by lifecycle training.

The probe data remains a cycle-bound artifact; this module is the maintained
evaluation implementation and therefore must not be loaded from legacy notebook
source snapshots.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_probes(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def greedy_continue(model, ids: list[int], *, max_new: int, eos_id: int) -> list[int]:
    output = list(ids)
    for _ in range(max_new):
        logits = model(mx.array([output], dtype=mx.int32))["logits"][0, -1]
        token = int(mx.argmax(logits).item())
        if token == eos_id:
            break
        output.append(token)
    return output


def probe_nll(model, tokenizer, prompt: str, target: str) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
    if not target_ids:
        return {"nll": float("nan"), "n_tokens": 0, "greedy": "", "exact": False}
    ids = prompt_ids + target_ids
    logp = nn.log_softmax(model(mx.array([ids], dtype=mx.int32))["logits"], axis=-1)
    nll = 0.0
    count = 0
    offset = len(prompt_ids) - 1
    for index, token_id in enumerate(target_ids):
        position = offset + index
        if 0 <= position < int(logp.shape[1]):
            nll += -float(logp[0, position, int(token_id)].item())
            count += 1
    generated = greedy_continue(
        model,
        prompt_ids,
        max_new=max(8, len(target_ids) + 4),
        eos_id=tokenizer.eos_id,
    )
    text = tokenizer.decode(generated[len(prompt_ids) :])
    return {
        "nll": nll / max(count, 1),
        "n_tokens": count,
        "greedy": text,
        "exact": text.strip() == target.strip(),
    }


def eval_probes(model, tokenizer, probes: list[dict]) -> dict:
    families: dict[str, list[float]] = {}
    rows = []
    for probe in probes:
        family = str(probe.get("family") or "unknown")
        prompt = str(probe.get("input") or "")
        target = str(probe.get("target") or "")
        row = probe_nll(model, tokenizer, prompt, target)
        row.update(
            {
                "probe_id": probe.get("probe_id"),
                "family": family,
                "input": prompt,
                "target": target,
            }
        )
        rows.append(row)
        families.setdefault(family, []).append(row["nll"])
    finite = [row["nll"] for row in rows if math.isfinite(row["nll"])]
    return {
        "n": len(rows),
        "mean_nll": sum(finite) / len(finite) if finite else float("nan"),
        "family_nll": {
            family: sum(values) / len(values) if values else float("nan")
            for family, values in families.items()
        },
        "exact_rate": sum(1 for row in rows if row["exact"]) / max(len(rows), 1),
        "rows": rows,
    }
