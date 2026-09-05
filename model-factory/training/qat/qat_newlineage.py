#!/usr/bin/env python3
"""QAT stage for the zh-v2 rebuild lineage (fake-quant Q4 STE).

Order (user-confirmed, matches the legacy chain): QAT BEFORE SFT. This stage
loads the float CPT base, replaces every ndim==2 parameter with its 4-bit
fake-quantized version inside the forward (STE backward flows to the float
master, which is restored before the optimizer update), and replays a short
window on the task-domain corpus so the base becomes quantization-tolerant
before SFT. Writes:
  - sft-qat-master.npz  (float master, QAT-tuned -- the SFT stage input)
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
        # 词嵌入保持 float32：检索/嵌入类操作对嵌入量化敏感
        if getattr(value, "ndim", 0) == 2 and key.endswith(".weight") and key != "embed.weight":
            names.append(key)
    return names


def _leaf_of(model: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    param: Any = model
    for part in parts[:-1]:
        if part.isdigit():
            param = param[int(part)]
        else:
            param = getattr(param, part)
    return param, parts[-1]


def quantize_inplace(model: Any, names: list[str], bits: int = 4) -> dict[str, mx.array]:
    """Replace each named weight with its fake-quantized value (STE).
    Returns the saved float masters, which the caller MUST restore after the
    backward pass so the optimizer updates the float master, not the
    quantized copy."""
    originals: dict[str, mx.array] = {}
    for name in names:
        param, leaf = _leaf_of(model, name)
        originals[name] = getattr(param, leaf)
        setattr(param, leaf, ste_quantize(originals[name], bits=bits))
    return originals


def restore_inplace(model: Any, originals: dict[str, mx.array]) -> None:
    for name, value in originals.items():
        param, leaf = _leaf_of(model, name)
        setattr(param, leaf, value)


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


def pack_model(model: Any, names: list[str], bits: int, out_path: Path) -> None:
    """按 release.quant_pack_51m 的规范格式打包（wasm 可消费）：
    量化张量按 bits 打包，其余（偏置/范数/嵌入）保持 float32。"""
    from release.quant_pack_51m import build_pack_bytes  # noqa: E402

    flat_params = dict(tree_flatten(model.parameters()))
    tensors: dict[str, np.ndarray] = {}
    bit_map: dict[str, int] = {}
    for name, value in flat_params.items():
        arr = np.asarray(mx.array(value).astype(mx.float32))
        if np.all(np.array(arr.shape) == 0):  # scalar gate 等零维张量
            continue
        tensors[name] = arr
        bit_map[name] = bits if name in names else 32
    blob, header = build_pack_bytes(tensors, bit_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    log(f"packed {len(tensors)} tensors ({len(names)}x{bits}bit, 其余 f32) -> {out_path} "
        f"({len(blob) / 1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--base-master", type=Path, required=True, help="float base/SFT master npz to quantize")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--bits", type=int, default=4, help="量化位宽：4（Q4 基准）或 2（Q2 目标）")
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
    log(f"loading SFT master {args.base_master}")
    load_params(model, args.base_master, strict=False, allow_missing_prefixes=(), return_report=True)

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
        # 每步先抓当前 float 主副本，前向用量化值（STE），反向后恢复主副本再更新
        originals: dict[str, mx.array] = {}
        for name in quant_names:
            param, leaf = _leaf_of(model, name)
            originals[name] = getattr(param, leaf)

        def loss_fn(model):
            quantize_inplace(model, quant_names, bits=args.bits)
            return ce_loss(model, ids, labels)
        loss_and_grads = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grads(model)
        restore_inplace(model, originals)
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
    pack_model(model, quant_names, args.bits, args.out_dir / f"sft-qat-q{args.bits}.pack")
    (args.out_dir / "summary.json").write_text(json.dumps({
        "steps": steps, "final_loss": float(loss), "quantized_tensors": len(quant_names),
        "pack_layout": "per-tensor int4 (2 elems/byte, low nibble first) + float32 scale",
        "sft_master": str(args.base_master),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
