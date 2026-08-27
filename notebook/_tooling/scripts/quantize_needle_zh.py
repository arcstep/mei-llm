#!/usr/bin/env python3
"""4-bit PTQ sensitivity scan. Nibble pack is a smoke artifact, not a publish kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as xu

from architecture import NeedleZh, count_params
from checkpoint import flatten_params, load_params
from config import NeedleZhConfig
from tokenizer import ZhTokenizerV1

CKPT_300 = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-300m.npz"


def pack_int4(arr: mx.array) -> bytes:
    """Naive even-odd nibble pack for a smoke artifact (not a production kernel)."""
    x = mx.clip(mx.round(arr * 7), -8, 7).astype(mx.int8)
    flat = x.reshape(-1)
    n = int(flat.size)
    if n % 2:
        flat = mx.concatenate([flat, mx.array([0], dtype=mx.int8)])
    even = flat[0::2] & 15
    odd = flat[1::2] & 15
    packed = (even | (odd << 4)).astype(mx.uint8)
    mx.eval(packed)
    return bytes(packed.tolist()[: min(4096, int(packed.size))])


def component_of(name: str) -> str:
    if name.startswith("embed") or name.startswith("lm_head"):
        return "embedding"
    if ".attn_norm" in name or ".mlp_norm" in name or ".post_attn_norm" in name or name.startswith("final_norm"):
        return "norm"
    if name.startswith("mhc_") or "mhc" in name:
        return "mhc"
    if name.startswith("engrams") or ".engrams" in name:
        return "engram"
    if name.startswith("contrastive"):
        return "retrieval"
    if name.startswith("conf_v2") or name.startswith("conf_"):
        return "confidence"
    if ".attn." in name:
        return "attention_kv"
    return "other"


def fake_quant_4bit(arr: mx.array) -> mx.array:
    xf = arr.astype(mx.float32)
    mx_abs = mx.maximum(mx.max(mx.abs(xf)), 1e-8)
    q = mx.clip(mx.round(xf / mx_abs * 7), -8, 7)
    return (q / 7.0 * mx_abs).astype(arr.dtype)


def logits_of(model, ids: list[int]) -> mx.array:
    arr = mx.array([ids], dtype=mx.int32)
    out = model(arr)["logits"]
    mx.eval(out)
    return out


def scan(model, ids: list[int]) -> dict:
    base = logits_of(model, ids)
    params = flatten_params(model)
    grouped: dict[str, list[str]] = {}
    for name in params:
        grouped.setdefault(component_of(name), []).append(name)
    rows = {}
    for comp, names in sorted(grouped.items()):
        replaced = dict(params)
        for name in names:
            replaced[name] = fake_quant_4bit(params[name])
        model.update(xu.tree_unflatten(list(replaced.items())))
        mx.eval(model.parameters())
        q = logits_of(model, ids)
        delta = float(mx.max(mx.abs(base - q)).item())
        mse = float(mx.mean((base - q) ** 2).item())
        rows[comp] = {
            "n_tensors": len(names),
            "max_abs_logit_delta": delta,
            "mse": mse,
            "bits": 4,
        }
        model.update(xu.tree_unflatten(list(params.items())))
        mx.eval(model.parameters())
    bit_map = {k: 4 for k in rows}
    bit_map["note"] = "per-component fake PTQ; not a packed runtime kernel"
    return {"components": rows, "bit_map": bit_map}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scan", action="store_true", help="per-component 4-bit PTQ sensitivity")
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-quant-{args.bits}bit{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke and not args.scan:
        cfg = NeedleZhConfig().tiny()
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        blob = pack_int4(model.embed.weight)
        (out_dir / "weights-int4.head.bin").write_bytes(blob)
        report = {
            "bits": args.bits,
            "smoke": True,
            "params": count_params(model),
            "artifact_head_bytes": len(blob),
            "order": ["float_correctness", "4bit_ptq", "qat_or_cq2_only_if_ptq_misses"],
            "package_budget_bytes": 25 * 1024 * 1024,
            "nibble_pack_is_publish_kernel": False,
            "note": "Smoke nibble pack is not a release implementation.",
        }
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    ckpt = args.ckpt or (None if args.smoke else (CKPT_300 if CKPT_300.is_file() else None))
    if args.smoke or ckpt is None:
        cfg = NeedleZhConfig().tiny(contrastive_head_v2=True, confidence_v2=True)
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        source = "tiny_random"
    else:
        cfg = NeedleZhConfig.from_spec()
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        load_params(model, ckpt, strict=True)
        source = str(Path(ckpt).relative_to(ROOT))
    tok = ZhTokenizerV1()
    ids = tok.encode("厨房灯打开", add_bos=True, add_eos=False)[:32]
    scanned = scan(model, ids)
    report = {
        "bits": args.bits,
        "smoke": bool(args.smoke),
        "source": source,
        "params": count_params(model),
        "nibble_pack_is_publish_kernel": False,
        "first_method": "4bit_ptq",
        "qat_or_cq2": "only_if_ptq_misses_quality_gate",
        "quality_gate_registered": False,
        "new_cpt_lineage": False,
        "scan": scanned,
        "note": "PTQ scan records per-component sensitivity. Independent CPT did not produce a new float checkpoint in this handoff.",
    }
    (out_dir / "ptq-scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "scan"} | {"components": list(scanned["components"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
