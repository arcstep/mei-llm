#!/usr/bin/env python3
"""Float ↔ Q4 dequant logits / greedy / probe parity for the packed 51M package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.utils as xu
import numpy as np

from evaluation.base.freeze_float_baseline_51m import eval_probes, greedy_continue, load_probes
from common.identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    EXPECTED_PARAMS,
    JOBS_DIR,
    MODEL_ID,
    PROBE_BANK,
    Q4_PACKAGE_DIR,
    RELEASE_PATH,
    WEIGHTS_PATH,
    fail,
    load_json,
    sha256_file,
    validate_release,
    write_json,
)
from release.quant_pack_51m import load_pack_file
from tokenizer import ZhTokenizerV1

SCAN_PROMPTS = ("厨房灯打开", "[]", '{"name":')
PROBE_SUBSET_IDS = {"PROBE-COPY-001", "PROBE-JSON-001"}


def load_float_model():
    from architecture import NeedleZh, count_params
    from common.checkpoint import load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError("float param count mismatch")
    return model


def apply_numpy_params(model, arrays: dict[str, np.ndarray]) -> int:
    from common.checkpoint import flatten_params

    current = flatten_params(model)
    n = 0
    for name, val in current.items():
        if name not in arrays:
            continue
        arr = arrays[name]
        if tuple(arr.shape) != tuple(val.shape):
            raise ValueError(f"shape mismatch {name}: {arr.shape} vs {tuple(val.shape)}")
        current[name] = mx.array(arr)
        n += 1
    model.update(xu.tree_unflatten(list(current.items())))
    mx.eval(model.parameters())
    return n


def logits_of(model, ids: list[int]) -> mx.array:
    out = model(mx.array([ids], dtype=mx.int32))["logits"]
    mx.eval(out)
    return out


def greedy_tokens(model, ids: list[int], *, max_new: int, eos_id: int) -> list[int]:
    return greedy_continue(model, ids, max_new=max_new, eos_id=eos_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--max-new", type=int, default=8)
    args = parser.parse_args()
    error = validate_release(load_json(RELEASE_PATH), WEIGHTS_PATH)
    if error:
        return fail(error)
    weights = args.package_dir / "weights.q4"
    if not weights.is_file():
        return fail(f"missing Q4 package: {weights}")
    tok = ZhTokenizerV1(args.package_dir / "tokenizer.model")
    header, arrays = load_pack_file(weights)
    float_model = load_float_model()
    quant_model = load_float_model()
    n_loaded = apply_numpy_params(quant_model, arrays)

    prompt_rows = []
    for text in SCAN_PROMPTS:
        ids = tok.encode(text, add_bos=True, add_eos=False)[:32]
        base = logits_of(float_model, ids)
        quant = logits_of(quant_model, ids)
        delta = float(mx.max(mx.abs(base - quant)).item())
        mse = float(mx.mean((base - quant) ** 2).item())
        g_float = greedy_tokens(float_model, ids, max_new=args.max_new, eos_id=tok.eos_id)
        g_quant = greedy_tokens(quant_model, ids, max_new=args.max_new, eos_id=tok.eos_id)
        prompt_rows.append(
            {
                "prompt": text,
                "n_ids": len(ids),
                "max_abs_logit_delta": delta,
                "mse": mse,
                "greedy_float": tok.decode(g_float[len(ids) :]),
                "greedy_quant": tok.decode(g_quant[len(ids) :]),
                "greedy_token_match": g_float == g_quant,
            }
        )

    probes = [item for item in load_probes(PROBE_BANK) if item.get("probe_id") in PROBE_SUBSET_IDS]
    float_probes = eval_probes(float_model, tok, probes) if probes else {}
    quant_probes = eval_probes(quant_model, tok, probes) if probes else {}
    report = {
        "stage": "P0/P1",
        "kind": "q4-dequant-parity",
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "package_id": "mei-1.0-51m-base-scratch300m-q4-v1",
        "quant_math_id": header.quant_math_id,
        "n_tensors_loaded": n_loaded,
        "weights_sha256": sha256_file(weights),
        "prompts": prompt_rows,
        "probe_subset_ids": sorted(PROBE_SUBSET_IDS),
        "float_probe_mean_nll": float_probes.get("mean_nll"),
        "quant_probe_mean_nll": quant_probes.get("mean_nll"),
        "probe_delta_mean_nll": (
            None
            if float_probes.get("mean_nll") is None or quant_probes.get("mean_nll") is None
            else float(quant_probes["mean_nll"]) - float(float_probes["mean_nll"])
        ),
        "qat_mandatory": True,
        "product_final": False,
        "not_a_claim": "Logit/greedy/probe delta is a pack math baseline, not tool-calling ability.",
    }
    out = args.out_dir / "q4-dequant-parity.json"
    blocked = write_json(out, report)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": True, "report": str(out), "prompts": prompt_rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
