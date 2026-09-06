#!/usr/bin/env python3
"""Summarize 51M QAT product hard gates. Does not write CURRENT.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.identity_51m import JOBS_DIR, ROOT, fail, load_json, write_json

# CURRENT.json lives at repo root; identity does not export CURRENT_PATH.
CURRENT = ROOT / "CURRENT.json"


def layer(jobs: Path) -> dict:
    q4 = None
    for name in ("qat-q4-rung-0p5m.json", "qat-q4-rung-2m.json", "qat-q4-rung-5m.json"):
        row = load_json(jobs / name)
        if row.get("product_ok"):
            q4 = row
            break
        if q4 is None and row.get("quality_ok"):
            q4 = row
    if q4 is None:
        q4 = load_json(jobs / "qat-q4-rung-5m.json") or load_json(jobs / "qat-q4-rung-2m.json") or load_json(
            jobs / "qat-q4-rung-0p5m.json"
        )
    pack = load_json(jobs / "qat-q4-package-receipt.json")
    parity = load_json(jobs / "qat-q4-three-runtime-parity.json")
    mixed = load_json(jobs / "q2q4-product-bit-map.json")
    mixed_blocked = load_json(jobs / "qat-cq2-blocked.json")
    sft = load_json(jobs / "sft-ondisk-qat-51m.json")
    sft_pack = load_json(jobs / "sft-qat-q4-package-receipt.json")
    lock = load_json(jobs / "toolcall-qat-lock-v2.json")
    res = load_json(jobs / "resource-baseline-51m.json")
    q4_ok = bool(q4.get("product_ok"))
    pack_ok = bool(pack.get("ok")) and bool(pack.get("within_budget")) and bool(pack.get("within_raw_budget"))
    parity_ok = (
        bool(parity.get("ok"))
        and bool(parity.get("same_package_ok"))
        and bool(parity.get("same_prompt_tokens_ok"))
        and bool(parity.get("generated_token_ids_exact"))
        and bool(parity.get("prefill_topk_ids_exact"))
    )
    mixed_ok = bool(mixed.get("product_final")) or (
        bool(mixed_blocked.get("quality_blocked")) and not mixed.get("product_final")
    )
    mixed_status = "product" if mixed.get("product_final") else "quality_blocked_qat_q4_only"
    sft_ok = (
        bool(sft.get("fullcall_sft"))
        and bool((sft.get("isolation") or {}).get("ok"))
        and sft.get("used_51m_weights") is True
        and int(sft.get("n_loaded") or 0) == 400
        and bool(sft_pack.get("ok"))
        and bool(sft_pack.get("within_budget"))
        and bool(sft_pack.get("within_raw_budget"))
        and sft.get("weight_qat_ste") is True
        and sft.get("activation_kv_int8_ste") is True
        and sft.get("training_order")
        == ["qat_fullcall_sft", "freeze_lm", "retrieval_head", "confidence_head"]
    )
    lock_ok = bool(lock.get("hard_ok"))
    res_ok = bool(res.get("hard_ok"))
    current = load_json(CURRENT)
    gates = {
        "qat_q4_quality": {"ok": q4_ok, "source": q4.get("rung") or q4.get("kind")},
        "qat_q4_package": {"ok": pack_ok, "package_id": pack.get("package_id"), "mb": pack.get("package_files_mb")},
        "three_runtime_parity": {
            "ok": parity_ok,
            "same_package": parity.get("same_package_ok"),
            "token_ids_exact": parity.get("generated_token_ids_exact"),
            "topk_exact": parity.get("prefill_topk_ids_exact"),
        },
        "mixed_q2q4": {
            "ok": mixed_ok,
            "status": mixed_status,
            "quality_blocked": mixed_blocked.get("quality_blocked"),
        },
        "sft_ondisk": {
            "ok": sft_ok,
            "package": sft.get("sft_package"),
            "weight_qat_ste": sft.get("weight_qat_ste"),
            "activation_kv_int8_ste": sft.get("activation_kv_int8_ste"),
            "used_51m_weights": sft.get("used_51m_weights"),
            "n_loaded": sft.get("n_loaded"),
            "package_receipt_ok": sft_pack.get("ok"),
            "package_within_budget": sft_pack.get("within_budget"),
        },
        "lock_v2": {"ok": lock_ok, "hard_ok": lock.get("hard_ok")},
        "resource_baseline": {"ok": res_ok, "hard_ok": res.get("hard_ok")},
        "current_untouched": {
            "ok": current.get("sft") is None and current.get("runtime") is None,
            "sft": current.get("sft"),
            "runtime": current.get("runtime"),
        },
    }
    hard_ok = all(row["ok"] for k, row in gates.items() if k != "current_untouched") and gates["current_untouched"]["ok"]
    return {
        "kind": "qat-product-hard-gates",
        "hard_ok": hard_ok,
        "gates": gates,
        "current_sft_runtime_still_null": gates["current_untouched"]["ok"],
        "wrote_current": False,
        "note": "CURRENT.sft / CURRENT.runtime stay null until an explicit freeze order.",
        "qat_mandatory": True,
        "not_a_claim": "Passing these gates is not a CURRENT freeze.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    report = layer(args.jobs_dir)
    blocked = write_json(args.jobs_dir / "qat-product-hard-gates.json", report)
    if blocked:
        return fail(blocked)
    proposal = {
        "kind": "freeze-proposal",
        "eligible": bool(report["hard_ok"]),
        "selected_branch": report["gates"]["mixed_q2q4"]["status"],
        "requires_explicit_user_freeze_command": True,
        "writes_current": False,
        "hard_gates": "qat-product-hard-gates.json",
    }
    blocked = write_json(args.jobs_dir / "freeze-proposal.json", proposal)
    if blocked:
        return fail(blocked)
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["hard_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
