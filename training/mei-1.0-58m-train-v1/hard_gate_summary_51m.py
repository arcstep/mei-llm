#!/usr/bin/env python3
"""Summarize 51M QAT product hard gates. Does not write CURRENT.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from identity_51m import JOBS_DIR, ROOT, fail, load_json, write_json

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
    cand = load_json(jobs / "q2q4-product-candidate-bit-map.json")
    sft = load_json(jobs / "sft-ondisk-qat-51m.json")
    lock = load_json(jobs / "toolcall-qat-lock-v2.json")
    res = load_json(jobs / "resource-baseline-51m.json")
    q4_ok = bool(q4.get("quality_ok"))
    pack_ok = bool(pack.get("ok"))
    parity_ok = bool(parity.get("ok"))
    mixed_ok = bool(mixed.get("product_final")) or (
        bool(cand.get("quality_blocked")) and not mixed.get("product_final")
    )
    mixed_status = "product" if mixed.get("product_final") else "quality_blocked_qat_q4_only"
    sft_ok = bool(sft.get("fullcall_sft"))
    lock_ok = bool(lock.get("hard_ok"))
    res_ok = bool(res)
    current = load_json(CURRENT)
    gates = {
        "qat_q4_quality": {"ok": q4_ok, "source": q4.get("rung") or q4.get("kind")},
        "qat_q4_package": {"ok": pack_ok, "package_id": pack.get("package_id"), "mb": pack.get("package_files_mb")},
        "three_runtime_parity": {"ok": parity_ok},
        "mixed_q2q4": {"ok": mixed_ok, "status": mixed_status, "quality_blocked": cand.get("quality_blocked")},
        "sft_ondisk": {"ok": sft_ok, "package": sft.get("sft_package")},
        "lock_v2": {"ok": lock_ok, "hard_ok": lock.get("hard_ok")},
        "resource_baseline": {"ok": res_ok},
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
        "current_sft_runtime_still_null": True,
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
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["hard_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
