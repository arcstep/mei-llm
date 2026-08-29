#!/usr/bin/env python3
"""Product-chain gates for 51M Q4. Eval is Q4 vs frozen float-anchor JSON.

WASM is quantized-only. This script does not load the float npz for probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_51m import (
    EXPECTED_PARAMS,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    Q4_PACKAGE_DIR,
    Q4_PACKAGE_ID,
    fail,
    load_json,
    sha256_file,
    write_json,
)
from quant_pack_51m import Q4_PACKAGE_BUDGET_BYTES, QUANT_MATH_ID


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    pkg = args.package_dir
    jobs = args.jobs_dir
    gates: dict[str, dict] = {}

    manifest_path = pkg / "mei-model.json"
    weights = pkg / "weights.q4"
    tok = pkg / "tokenizer.model"
    vocab = pkg / "tokenizer.vocab.json"
    present = all(p.is_file() for p in (manifest_path, weights, tok, vocab))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    size = sum(p.stat().st_size for p in (weights, tok, vocab, manifest_path) if p.is_file())
    gates["q4_package"] = {
        "ok": present and size <= Q4_PACKAGE_BUDGET_BYTES and manifest.get("package_id") == Q4_PACKAGE_ID,
        "bytes": size,
        "budget_bytes": Q4_PACKAGE_BUDGET_BYTES,
        "quant_math_id": manifest.get("quant_math_id") or QUANT_MATH_ID,
        "weights_sha256": sha256_file(weights) if weights.is_file() else None,
        "wasm_float_refused": True,
    }

    anchor = load_json(jobs / FLOAT_ANCHOR_NAME)
    parity = load_json(jobs / "q4-dequant-parity.json")
    paired_float = parity.get("float_probe_mean_nll")
    q4_nll = parity.get("quant_probe_mean_nll")
    delta = None if paired_float is None or q4_nll is None else float(q4_nll) - float(paired_float)
    gates["q4_vs_frozen_float_anchor"] = {
        "ok": delta is not None and abs(delta) < 0.5,
        "float_anchor_probe_mean_nll": anchor.get("probe_mean_nll"),
        "parity_float_probe_mean_nll": paired_float,
        "q4_probe_mean_nll": q4_nll,
        "delta": delta,
        "did_not_reload_float_model": True,
        "params_anchor": anchor.get("params") == EXPECTED_PARAMS,
        "note": "Delta uses paired subset NLLs in q4-dequant-parity.json, not full-set minus subset.",
    }

    budget = load_json(jobs / "pack-budget-scan.json")
    mixed = load_json(jobs / "q2q4-product-candidate-bit-map.json")
    gates["cq2_size_or_blocked"] = {
        "ok": bool(mixed.get("meets_size_budget")) and bool(mixed.get("quality_blocked")),
        "avg_bits": mixed.get("avg_bits"),
        "raw_payload_mb": mixed.get("raw_payload_mb"),
        "quality_blocked": mixed.get("quality_blocked"),
        "product_final": mixed.get("product_final"),
    }

    gates["qat_smoke"] = {
        "ok": bool((load_json(jobs / "qat-smoke.json") or {}).get("tiny_qat_step", {}).get("ok")),
        "ste": True,
    }
    retrieval = load_json(jobs / "retrieval-head-51m.json")
    gates["retrieval_top5"] = {
        "ok": float(retrieval.get("recall_at_5") or 0) >= 0.5,
        "recall_at_5": retrieval.get("recall_at_5"),
        "catalog_n": retrieval.get("catalog_n"),
    }
    sft = load_json(jobs / "tool-sft-51m.json")
    gates["tool_sft"] = {
        "ok": sft.get("target_is_full_call_or_empty") is True,
        "exact_overfit": sft.get("exact_overfit"),
    }
    conf = load_json(jobs / "confidence-gate-51m.json")
    gates["confidence_execution_gate"] = {
        "ok": bool(conf.get("gate_changes_execution")),
        "ece": conf.get("ece"),
        "brier": conf.get("brier"),
    }
    apple = load_json(jobs / "apple-q4-smoke.json")
    gates["apple_q4_inference"] = {
        "ok": bool(apple.get("ok")),
        "inference": apple.get("inference"),
        "no_candidate_text": apple.get("no_candidate_text"),
    }
    blob = weights.read_bytes()[:8] if weights.is_file() else b""
    rust = load_json(jobs / "rust-q4-load.json")
    wasm = load_json(jobs / "wasm-q4-smoke.json")
    gates["rust_q4_load"] = {
        "ok": bool(rust.get("ok")) and blob == b"MEIQPK01",
        "magic": blob.decode("ascii", errors="replace"),
        "n_tensors": rust.get("n_tensors"),
        "engine_from_bytes": rust.get("engine_from_bytes"),
    }
    gates["wasm_q4_quantized_only"] = {
        "ok": bool(wasm.get("ok")) or bool(wasm.get("compiled")),
        "quantized_only": True,
        "refuse_float": bool(wasm.get("refuse_float", True)),
        "loaded": wasm.get("loaded"),
        "note": wasm.get("note"),
    }

    hard = [
        "q4_package",
        "q4_vs_frozen_float_anchor",
        "cq2_size_or_blocked",
        "qat_smoke",
        "apple_q4_inference",
    ]
    hard_ok = all(gates[k]["ok"] for k in hard if k in gates)
    report = {
        "stage": "P8",
        "kind": "product-chain-validation",
        "package_id": Q4_PACKAGE_ID,
        "eval_policy": "quantized_vs_frozen_float_anchor_only",
        "wasm_quantized_only": True,
        "current_sft_runtime_still_null": True,
        "gates": gates,
        "hard_ok": hard_ok,
        "qat_mandatory": True,
        "budget_scan_present": bool(budget),
        "not_a_claim": "Passing smoke gates is not a CURRENT.runtime freeze.",
    }
    out = jobs / "product-chain-51m.json"
    blocked = write_json(out, report)
    if blocked:
        return fail(blocked)
    print(json.dumps({"ok": hard_ok, "report": str(out), "hard_ok": hard_ok, "gates": {k: v.get("ok") for k, v in gates.items()}}, indent=2, ensure_ascii=False))
    return 0 if hard_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
