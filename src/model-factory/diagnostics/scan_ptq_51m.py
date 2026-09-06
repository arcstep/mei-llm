#!/usr/bin/env python3
"""Per-component 4-bit PTQ sensitivity scan for the immutable 51M base.

This is diagnostic only. It must not write into base/, start QAT, or claim a
product mixed-bit map. QAT remains mandatory regardless of PTQ deltas.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.utils as xu

from architecture import NeedleZh, count_params
from common.checkpoint import flatten_params, load_params
from config import NeedleZhConfig
from evaluation.base.freeze_float_baseline_51m import eval_probes, load_probes
from common.identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    CANDIDATE_MAP_NAME,
    EXPECTED_PARAMS,
    JOBS_DIR,
    MODEL_ID,
    PROBE_BANK,
    PTQ_SCAN_NAME,
    QAT_MANDATORY_NOTE,
    RELEASE_PATH,
    ROOT,
    WEIGHTS_PATH,
    fail,
    load_json,
    propose_candidate_bit_map,
    sha256_file,
    validate_release,
    write_json,
)
from training.qat.quant_ops_51m import component_of, fake_quant_4bit
from tokenizer import ZhTokenizerV1

SCAN_PROMPT = "厨房灯打开"
PROBE_SUBSET_IDS = {"PROBE-COPY-001", "PROBE-JSON-001"}


def logits_of(model, ids: list[int]) -> mx.array:
    arr = mx.array([ids], dtype=mx.int32)
    out = model(arr)["logits"]
    mx.eval(out)
    return out


def restore_params(model, params: dict) -> None:
    model.update(xu.tree_unflatten(list(params.items())))
    mx.eval(model.parameters())


def scan_components(model, ids: list[int]) -> dict:
    base = logits_of(model, ids)
    params = flatten_params(model)
    grouped: dict[str, list[str]] = {}
    for name in params:
        grouped.setdefault(component_of(name), []).append(name)
    rows: dict[str, dict] = {}
    for comp, names in sorted(grouped.items()):
        replaced = dict(params)
        for name in names:
            replaced[name] = fake_quant_4bit(params[name])
        restore_params(model, replaced)
        quantized = logits_of(model, ids)
        delta = float(mx.max(mx.abs(base - quantized)).item())
        mse = float(mx.mean((base - quantized) ** 2).item())
        rows[comp] = {
            "n_tensors": len(names),
            "max_abs_logit_delta": delta,
            "mse": mse,
            "bits": 4,
        }
        restore_params(model, params)
    return rows


def probe_subset_delta(model, tok, probes: list[dict], params: dict, components: dict) -> dict:
    subset = [item for item in probes if item.get("probe_id") in PROBE_SUBSET_IDS]
    if not subset:
        return {}
    base_rep = eval_probes(model, tok, subset)
    out: dict[str, dict] = {}
    grouped: dict[str, list[str]] = {}
    for name in params:
        grouped.setdefault(component_of(name), []).append(name)
    for comp, names in grouped.items():
        if int((components.get(comp) or {}).get("n_tensors") or 0) == 0:
            continue
        replaced = dict(params)
        for name in names:
            replaced[name] = fake_quant_4bit(params[name])
        restore_params(model, replaced)
        q_rep = eval_probes(model, tok, subset)
        restore_params(model, params)
        base_nll = float(base_rep.get("mean_nll") or 0.0)
        q_nll = float(q_rep.get("mean_nll") or 0.0)
        out[comp] = {
            "base_mean_nll": base_nll,
            "quant_mean_nll": q_nll,
            "delta_mean_nll": q_nll - base_nll,
        }
    return out


def load_51m_model():
    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    if cfg.architecture_id != ARCHITECTURE_ID:
        raise RuntimeError(f"from_spec loaded {cfg.architecture_id!r}, not {ARCHITECTURE_ID}")
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    params = count_params(model)
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"loaded params {params} != {EXPECTED_PARAMS}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--skip-probe-subset", action="store_true")
    args = parser.parse_args()
    release = load_json(RELEASE_PATH)
    error = validate_release(release, WEIGHTS_PATH)
    if error:
        return fail(error)

    model = load_51m_model()
    tok = ZhTokenizerV1()
    ids = tok.encode(SCAN_PROMPT, add_bos=True, add_eos=False)[:32]
    components = scan_components(model, ids)
    params = flatten_params(model)
    probe_deltas = {}
    if not args.skip_probe_subset:
        probes = load_probes(PROBE_BANK)
        probe_deltas = probe_subset_delta(model, tok, probes, params, components)
        for name, delta_row in probe_deltas.items():
            if name in components:
                components[name]["probe_subset_delta_mean_nll"] = delta_row["delta_mean_nll"]

    candidate = propose_candidate_bit_map(components)
    scan_report = {
        "stage": "M2.1",
        "kind": "ptq-scan",
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "params": EXPECTED_PARAMS,
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "bits": 4,
        "prompt": SCAN_PROMPT,
        "nibble_pack_is_publish_kernel": False,
        "product_final": False,
        "candidate": True,
        "qat_mandatory": True,
        "qat_or_cq2": "mandatory",
        "source": str(WEIGHTS_PATH.relative_to(ROOT)),
        "components": components,
        "probe_subset_ids": sorted(PROBE_SUBSET_IDS),
        "probe_subset_deltas": probe_deltas,
        "most_sensitive": candidate.get("ranking") or [],
        "note": QAT_MANDATORY_NOTE,
        "not_a_claim": "PTQ sensitivity is not QAT, packed-kernel, or tool-calling ability.",
    }
    scan_path = args.out_dir / PTQ_SCAN_NAME
    map_path = args.out_dir / CANDIDATE_MAP_NAME
    for path, payload in ((scan_path, scan_report), (map_path, candidate)):
        blocked = write_json(path, payload)
        if blocked:
            return fail(blocked)
    print(
        json.dumps(
            {
                "ok": True,
                "ptq_scan": str(scan_path),
                "candidate_map": str(map_path),
                "qat_mandatory": True,
                "product_final": False,
                "components": list(components),
                "bits": candidate.get("bits"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
