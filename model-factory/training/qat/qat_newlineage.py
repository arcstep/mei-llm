#!/usr/bin/env python3
"""QAT replay for the zh-v2 rebuild lineage (fake-quant Q4 STE).

Loads the float SFT master, replaces every ndim==2 parameter with its
4-bit fake-quantized version inside the forward (STE backward flows to the
float master), trains a short replay on the SFT corpus, then writes:
  - sft-qat-master.npz  (float master, QAT-tuned)
  - sft-qat-q4.npz      (per-tensor Q4 pack: <name>.q4 (uint8, 2 elems/byte)
                         + <name>.scale (float32))
The pack layout is simple enough for the python runtime and wasm to consume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from common.checkpoint import load_params, save_params  # noqa: E402
from training.qat.quant_ops_51m import ste_quantize  # noqa: E402

IGNORE_ID = -100
SEQ_CAP = 2048


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_quant_pairs(model: Any) -> list[str]:
    flat = dict(tree_flatten(model.parameters()))
    names = []
    for key, value in flat.items():
        if getattr(value, "ndim", 0) == 2 and key.endswith(".weight"):
            names.append(key)
    return names


def quantize_inplace(model: Any, names: list[str], bits: int = 4) -> None:
    for name in names:
        param: Any = model
        parts = name.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                param = param[int(part)]
            else:
                param = getattr(param, part)
        leaf = parts[-1]
        setattr(param, leaf, ste_quantize(getattr(param, leaf), bits=bits))


def ce_loss(model: Any, ids: mx.array, labels: mx.array) -> mx.array:
    out = model(ids)["logits"]
    logits = out[:, :-1, :]
    targets = labels[:, 1:]
    safe = mx.where(targets == IGNORE_ID, 0, targets)
    loss = nn.losses.cross_entropy(logits, safe, reduction="none")
    mask = (targets != IGNORE_ID).astype(loss.dtype)
    n = mask.sum()
    if float(n) == 0.0:
        return mx.array(0.0)
    return (loss * mask).sum() / n


def pack_q4(model: Any, names: list[str], out_path: Path) -> None:
    flat_params = dict(tree_flatten(model.parameters()))
    arrays: dict[str, Any] = {}
    for name in names:
        value = flat_params[name]
        w = mx.array(value)
        w32 = w.astype(mx.float32)
        flat_w = w32.reshape(-1)
        scale = mx.max(mx.abs(flat_w), keepdims=False) / 7.5
        scale = mx.maximum(scale, mx.array(1e-6))
        q = mx.clip(mx.round(w32 / scale), -8, 7).astype(mx.int8)
        n = q.size
        if n % 2:
            q = mx.concatenate([q, mx.array([0], dtype=mx.int8)])
        pairs = q.reshape(-1, 2)
        packed = (pairs[:, 0].astype(mx.uint8) & 0xF) | ((pairs[:, 1].astype(mx.uint8) & 0xF) << 4)
        arrays[f"{name}.q4"] = np.asarray(packed)
        arrays[f"{name}.scale"] = np.asarray(scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(out_path), **{name: mx.array(value) for name, value in arrays.items()})
    log(f"packed {len(names)} tensors -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--sft-master", type=Path, required=True, help="float SFT master npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"write-once refusal: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    import sys as _sys
    from common._repo import ARCHITECTURE_DIR  # noqa: E402
    _sys.path.insert(0, str(ARCHITECTURE_DIR))
    from architecture import NeedleZh  # noqa: E402
    from config import NeedleZhConfig  # noqa: E402

    from training.tool_use.train_sft_newlineage import (  # noqa: E402
        DeployIndex,
        RENDERERS,
        TRAINABLE_FAMILIES,
        encode_pair,
    )
    from training.cpt.train_pretrain import _frozen_tokenizer  # noqa: E402

    tok = _frozen_tokenizer()
    model = NeedleZh(NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec())
    mx.eval(model.parameters())
    log(f"loading SFT master {args.sft_master}")
    load_params(model, args.sft_master, strict=False, allow_missing_prefixes=(), return_report=True)

    deploy = DeployIndex()
    rows: list[dict[str, Any]] = []
    for family in TRAINABLE_FAMILIES:
        path = args.release_dir / "compiled" / family / "train.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, gold = RENDERERS[family](row, deploy)
            rows.append(encode_pair(tok, prompt, gold))
    if args.smoke:
        rows = rows[:64]
    log(f"replay dataset: {len(rows)} rows")

    quant_names = collect_quant_pairs(model)
    log(f"quantizing {len(quant_names)} ndim==2 weight tensors")

    lr_schedule = optim.cosine_decay(args.lr, max(args.steps, 1), args.lr * 0.1)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    def step_fn(ids, labels):
        def loss_fn(model):
            quantize_inplace(model, quant_names)
            return ce_loss(model, ids, labels)
        loss_and_grads = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grads(model)
        optimizer.update(model, grads)
        return loss

    if not args.no_compile:
        try:
            step_fn = mx.compile(step_fn)
        except Exception as exc:
            log(f"mx.compile unavailable ({exc}); eager mode")

    import random as _random
    rng = _random.Random(20260905)
    order = list(range(len(rows)))
    rng.shuffle(order)
    t0 = time.time()
    steps = min(args.steps, 30) if args.smoke else args.steps
    for step in range(steps):
        start = (step * args.batch_size) % len(rows)
        picks = order[start : start + args.batch_size]
        if len(picks) < args.batch_size:
            picks += order[: args.batch_size - len(picks)]
        encoded = [rows[i] for i in picks]
        max_len = min(max(len(ids) for ids, _ in encoded), SEQ_CAP)
        ids_b = mx.array([list(ids.tolist()[:max_len]) + [0] * (max_len - len(ids)) for ids, _ in encoded], dtype=mx.int32)
        lab_b = mx.array([list(labs.tolist()[:max_len]) + [IGNORE_ID] * (max_len - len(labs)) for _, labs in encoded], dtype=mx.int32)
        loss = step_fn(ids_b, lab_b)
        mx.eval(loss, model.parameters())
        if step % 25 == 0 or step == steps - 1:
            log(f"qat step={step}/{steps} loss={float(loss):.4f} elapsed={time.time() - t0:.0f}s")

    save_params(model, args.out_dir / "sft-qat-master.npz")
    pack_q4(model, quant_names, args.out_dir / "sft-qat-q4.npz")
    (args.out_dir / "summary.json").write_text(json.dumps({
        "steps": steps, "final_loss": float(loss), "quantized_tensors": len(quant_names),
        "pack_layout": "per-tensor int4 (2 elems/byte, low nibble first) + float32 scale",
        "sft_master": str(args.sft_master),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
