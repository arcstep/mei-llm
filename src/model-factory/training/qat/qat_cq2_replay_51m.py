#!/usr/bin/env python3
"""CQ2 mixed-bit STE replay after QAT Q4 quality gates.

If Q4 failed, keep quality_blocked and do not freeze a product bit-map.
The pack kernel is per-tensor, so the candidate's per-block Q2 counts are
coarsened to whole-tensor bits before STE and packing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.identity_51m import (
    ARCHITECTURE_SPEC,
    JOBS_DIR,
    PRODUCT_MIXED_FINAL_NAME,
    PRODUCT_MIXED_MAP_NAME,
    QAT_CQ2_PACKAGE_DIR,
    QAT_CQ2_PACKAGE_ID,
    QAT_CQ2_WEIGHTS_PATH,
    QAT_Q4_PACKAGE_DIR,
    QAT_Q4_WEIGHTS_PATH,
    fail,
    load_json,
    write_json,
)
from release.quant_pack_51m import CQ2_RAW_PAYLOAD_BUDGET, coarsen_mixed_candidate_to_tensor_bits


COARSENED_NAME = "q2q4-product-tensor-bits-coarsened.json"


def q4_passed(jobs: Path) -> bool:
    for name in ("qat-q4-rung-0p5m.json", "qat-q4-rung-2m.json", "qat-q4-rung-5m.json"):
        row = load_json(jobs / name)
        if row.get("product_ok"):
            return True
    return False


def architecture_tensor_sizes(package_dir: Path) -> dict[str, int]:
    man = load_json(package_dir / "mei-model.json")
    directory = ((man.get("weights") or {}).get("directory")) or []
    sizes = {str(row["name"]): int(row["n_params"]) for row in directory if row.get("name")}
    if sizes:
        return sizes
    from architecture import NeedleZh
    from common.checkpoint import flatten_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    return {name: int(val.size) for name, val in flatten_params(model).items()}


def write_blocked(jobs: Path, receipt: dict) -> int:
    # Do not leave a product_final map on disk if mixed is blocked.
    existing = load_json(jobs / PRODUCT_MIXED_FINAL_NAME)
    if existing.get("product_final"):
        existing["product_final"] = False
        existing["quality_blocked"] = True
        existing["kind"] = "q2q4-product-bit-map"
        write_json(jobs / PRODUCT_MIXED_FINAL_NAME, existing)
    blocked = write_json(jobs / "qat-cq2-blocked.json", receipt)
    if blocked:
        return fail(blocked)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--max-rung", default="2m", choices=["smoke", "0p5m", "2m"])
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--q4-master", type=Path, default=QAT_Q4_WEIGHTS_PATH)
    parser.add_argument("--q4-package-dir", type=Path, default=QAT_Q4_PACKAGE_DIR)
    parser.add_argument("--cq2-package-dir", type=Path, default=QAT_CQ2_PACKAGE_DIR)
    args = parser.parse_args()
    jobs = args.jobs_dir
    candidate = load_json(jobs / PRODUCT_MIXED_MAP_NAME)
    if not q4_passed(jobs):
        return write_blocked(
            jobs,
            {
                "kind": "qat-cq2-blocked",
                "quality_blocked": True,
                "reason": "QAT Q4 quality gate did not pass; mixed product map stays candidate-only.",
                "candidate_map": PRODUCT_MIXED_MAP_NAME,
                "product_final": False,
            },
        )

    sizes = architecture_tensor_sizes(args.q4_package_dir)
    coarsened = coarsen_mixed_candidate_to_tensor_bits(candidate, sizes)
    err = write_json(jobs / COARSENED_NAME, coarsened)
    if err:
        return fail(err)
    if not coarsened.get("meets_size_budget"):
        return write_blocked(
            jobs,
            {
                "kind": "qat-cq2-blocked",
                "quality_blocked": True,
                "reason": "Coarsened whole-tensor Q2 map missed the 19.3MiB raw payload budget.",
                "coarsened": COARSENED_NAME,
                "raw_payload_mb": coarsened.get("raw_payload_mb"),
                "product_final": False,
                "ship": "QAT Q4 only",
            },
        )

    from training.qat.qat_replay_51m import main as replay_main
    import sys

    bit_map_path = jobs / COARSENED_NAME
    argv = sys.argv
    sys.argv = [
        "qat_replay_51m.py",
        "--bit-map",
        str(bit_map_path),
        "--max-rung",
        args.max_rung,
        "--rung-prefix",
        "qat-cq2",
        "--init-weights",
        str(args.q4_master if args.q4_master.is_file() else jobs / "unused"),
        "--jobs-dir",
        str(jobs),
        "--save-master",
    ]
    if args.checkpoint_dir:
        sys.argv.extend(
            [
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--resume",
                "--master-out",
                str(args.checkpoint_dir / "best.npz"),
            ]
        )
    if not args.q4_master.is_file():
        sys.argv = [
            "qat_replay_51m.py",
            "--bit-map",
            str(bit_map_path),
            "--max-rung",
            args.max_rung,
            "--rung-prefix",
            "qat-cq2",
            "--jobs-dir",
            str(jobs),
        ]
    try:
        code = replay_main()
    finally:
        sys.argv = argv

    rung_2m = load_json(jobs / "qat-cq2-rung-2m.json")
    rung_0p5 = load_json(jobs / "qat-cq2-rung-0p5m.json")
    coarsened_tag = COARSENED_NAME
    if coarsened_tag in str(rung_2m.get("bit_map") or ""):
        rung = rung_2m
    elif coarsened_tag in str(rung_0p5.get("bit_map") or ""):
        rung = rung_0p5
    else:
        rung = rung_2m or rung_0p5
    if not (code == 0 and (rung.get("quality_ok") or rung.get("product_ok"))):
        return write_blocked(
            jobs,
            {
                "kind": "qat-cq2-blocked",
                "quality_blocked": True,
                "reason": "CQ2 STE replay missed the same float-anchor delta gates.",
                "eval": (rung or {}).get("eval"),
                "product_final": False,
                "ship": "QAT Q4 only",
                "coarsened": COARSENED_NAME,
            },
        )

    from release.pack_qat_51m import main as pack_main

    argv2 = sys.argv
    sys.argv = [
        "pack_qat_51m.py",
        "--master",
        str(
            (args.checkpoint_dir / "best.npz")
            if args.checkpoint_dir and (args.checkpoint_dir / "best.npz").is_file()
            else (QAT_CQ2_WEIGHTS_PATH if QAT_CQ2_WEIGHTS_PATH.is_file() else args.q4_master)
        ),
        "--out-dir",
        str(args.cq2_package_dir),
        "--package-id",
        QAT_CQ2_PACKAGE_ID,
        "--bit-map",
        str(bit_map_path),
        "--scheme",
        "cq2",
        "--budget-bytes",
        str(30 * 1024 * 1024),
        "--budget-raw-bytes",
        str(int(CQ2_RAW_PAYLOAD_BUDGET)),
        "--receipt-name",
        "qat-cq2-package-receipt.json",
    ]
    try:
        pack_code = pack_main()
    finally:
        sys.argv = argv2
    if pack_code != 0:
        return write_blocked(
            jobs,
            {
                "kind": "qat-cq2-blocked",
                "quality_blocked": True,
                "reason": "CQ2 quality passed but the packed weights missed the 19.3MiB raw budget.",
                "pack_code": pack_code,
                "product_final": False,
                "ship": "QAT Q4 only",
                "coarsened": COARSENED_NAME,
            },
        )

    product = {
        **coarsened,
        "kind": "q2q4-product-bit-map",
        "candidate": False,
        "product_final": True,
        "quality_blocked": False,
        "qat_cq2_ok": True,
        "parent": "mei-1.0-51m-base-scratch300m-qat-q4-v1",
        "source_candidate": PRODUCT_MIXED_MAP_NAME,
        "packing_granularity": "whole_tensor",
        "block_candidate_coarsened": True,
    }
    # Drop the long promoted_tensors list from the product SSOT-ish jobs file? Keep it: it's the map.
    blocked = write_json(jobs / PRODUCT_MIXED_FINAL_NAME, product)
    if blocked:
        return fail(blocked)
    print(json.dumps({"product_map": PRODUCT_MIXED_FINAL_NAME, "pack_code": pack_code, "ok": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
