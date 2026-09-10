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

IGNORE_ID = -100
SEQ_CAP = 2048


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def storage_policy(name: str) -> str:
    """旧链 cq2_policy 的逐张量存储策略：f16 / cq4 / cq2。"""
    safe = (
        name.endswith(".scale")
        or name.endswith(".bias")
        or name.endswith("attn_gate")
        or (name.startswith("engrams.") and name.endswith(".taps"))
        or name.endswith((".mlp.d1", ".mlp.d2", ".mlp.d3"))
        or name.startswith("conf_")
    )
    if safe:
        return "f16"
    if name == "embed.weight" or name.startswith("mhc_"):
        return "cq4"
    return "cq2"


def collect_quant_pairs(model: Any) -> tuple[list[str], dict[str, int]]:
    """量化张量清单 + 每张量位宽（对齐旧链 lm_storage_dtype 策略）。"""
    flat = dict(tree_flatten(model.parameters()))
    names: list[str] = []
    bits_by_name: dict[str, int] = {}
    for key, value in flat.items():
        if getattr(value, "ndim", 0) != 2 or not key.endswith(".weight"):
            continue
        policy = storage_policy(key)
        if policy == "f16":
            continue
        names.append(key)
        bits_by_name[key] = 4 if policy == "cq4" else 2
    return names, bits_by_name


def _leaf_of(model: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    param: Any = model
    for part in parts[:-1]:
        if part.isdigit():
            param = param[int(part)]
        else:
            param = getattr(param, part)
    return param, parts[-1]


def ste_cq2(weight: mx.array, name: str, bits_by_name: dict[str, int]) -> mx.array:
    """STE 量化：前向用旧链 cq2 的 g128-WHT-codebook 语义（与 runtime
    编码器一致），反向直通。group_bits 由 cq2_policy.uniform_group_bits
    按张量策略生成（每 128 一组，cq2=2 / cq4=4）。"""
    from training.qat import cq2_qat_51m as cq2  # noqa: E402
    from training.qat.cq2_policy_51m import GROUP_SIZE  # noqa: E402

    n_values = int(weight.size)
    groups = (n_values + GROUP_SIZE - 1) // GROUP_SIZE
    width = bits_by_name.get(name, 2)
    group_bits = (width,) * groups
    return cq2.fake_quant_weight(weight, group_bits, ste=True)


def quantize_inplace(model: Any, names: list[str], bits_by_name: dict[str, int]) -> dict[str, mx.array]:
    """Replace each named weight with its fake-quantized value (STE, cq2
    WHT-codebook semantics). Returns the saved float masters, which the
    caller MUST restore after the backward pass so the optimizer updates the
    float master, not the quantized copy."""
    originals: dict[str, mx.array] = {}
    for name in names:
        param, leaf = _leaf_of(model, name)
        originals[name] = getattr(param, leaf)
        setattr(param, leaf, ste_cq2(originals[name], name, bits_by_name))
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
    # compile 内不允许对 traced 标量 eval：n==0 时分子必为 0，钳到 1 语义等价
    n_safe = mx.maximum(n, 1.0)
    return (loss * mask).sum() / n_safe


def pack_model(model: Any, names: list[str], bits_by_name: dict[str, int], out_path: Path) -> None:
    """按 release.quant_pack_51m 的规范格式打包（wasm 可消费）：
    量化张量按逐张量策略位宽，其余（f16 策略张量）保持 float32。
    注意：pack 编码器仍是 block64 均匀量化，与 runtime 的 g128-WHT-codebook
    编码器不一致——仅供 python 侧加载/体积参考；wasm 导出待对齐 cq2 编码器。"""
    from release.quant_pack_51m import build_pack_bytes  # noqa: E402

    flat_params = dict(tree_flatten(model.parameters()))
    tensors: dict[str, np.ndarray] = {}
    bit_map: dict[str, int] = {}
    for name, value in flat_params.items():
        arr = np.asarray(mx.array(value).astype(mx.float32))
        if np.all(np.array(arr.shape) == 0):  # scalar gate 等零维张量
            continue
        tensors[name] = arr
        bit_map[name] = bits_by_name.get(name, 32)
    blob, header = build_pack_bytes(tensors, bit_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    log(f"packed {len(tensors)} tensors -> {out_path} ({len(blob) / 1e6:.2f} MB)")


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
    ap.add_argument("--bits", type=int, default=2, help="默认位宽（未命中策略的特殊张量）；旧链策略：f16 头/小张量、cq4 嵌入+mhc、cq2 主体")
    ap.add_argument("--family-weights", action="append", default=[],
                    help="replay 抽样权重 family=N（如 retrieval=3），未列出的族权重 1")
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"write-once refusal: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    import sys as _sys
    from common.paths import ARCHITECTURE_DIR  # noqa: E402
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
    row_families: list[str] = []
    for family in TRAINABLE_FAMILIES:
        path = args.release_dir / "compiled" / family / "train.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, gold = RENDERERS[family](row, deploy)
            rows.append(encode_pair(tok, prompt, gold))
            row_families.append(family)
    if args.smoke:
        rows = rows[:64]
        row_families = row_families[:64]
    log(f"replay dataset: {len(rows)} rows")

    quant_names, bits_by_name = collect_quant_pairs(model)
    log(f"quantizing {len(quant_names)} ndim==2 weight tensors (policy: f16/cq4/cq2 per legacy lm_storage_dtype)")

    lr_schedule = optim.cosine_decay(args.lr, max(args.steps, 1), args.lr * 0.1)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    def step_fn(ids, labels):
        # 每步先抓当前 float 主副本，前向用量化值（STE），反向后恢复主副本再更新
        originals: dict[str, mx.array] = {}
        for name in quant_names:
            param, leaf = _leaf_of(model, name)
            originals[name] = getattr(param, leaf)

        def loss_fn(model):
            quantize_inplace(model, quant_names, bits_by_name)
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
    steps = min(args.steps, 30) if args.smoke else args.steps
    family_weights: dict[str, float] = {}
    for item in args.family_weights:
        family, weight = item.split("=", 1)
        family_weights[family] = float(weight)
    if family_weights:
        order = rng.choices(
            range(len(rows)),
            weights=[family_weights.get(fam, 1.0) for fam in row_families],
            k=steps * args.batch_size,
        )
        log(f"weighted replay: {family_weights} ({len(order)} draws)")
    else:
        order = list(range(len(rows)))
        rng.shuffle(order)
    t0 = time.time()
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
    pack_model(model, quant_names, bits_by_name, args.out_dir / "sft-qat-cq.pack")
    (args.out_dir / "summary.json").write_text(json.dumps({
        "steps": steps, "final_loss": float(loss), "quantized_tensors": len(quant_names),
        "pack_layout": "per-tensor int4 (2 elems/byte, low nibble first) + float32 scale",
        "sft_master": str(args.base_master),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
