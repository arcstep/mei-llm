#!/usr/bin/env python3
"""Generate the evidence-bound Chinese Needle2 mechanism alignment matrix.

Needle2 is a pinned implementation reference, not a compatibility target.  A
row is validated only from local receipts/artifacts; a declared configuration
or the mere existence of source code is never sufficient evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from common._repo import ROOT, architecture_contracts


REFERENCE = ROOT / "platform/_shared/spec/needle2-reference.json"
REQUIRED_STAGES = (
    "cq2_qat_v2",
    "retrieval_r1",
    "tool_index_final",
    "learned_top5_e2e",
    "mw_disposition_20class",
    "confidence_calibration",
    "package_v2_cq2",
    "mtp_ablation",
    "portable_runtime_gates",
    "resource_measurement",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def _receipt(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / "stages" / name / "receipt.json"
    row = load_json(path)
    if row.get("stage_id") != name or row.get("terminal_status") not in {
        "passed",
        "degraded",
    }:
        raise RuntimeError(f"alignment requires terminal receipt: {name}")
    for output, digest in (row.get("output_hashes") or {}).items():
        target = Path(output)
        if not target.is_file() or sha_file(target) != digest:
            raise RuntimeError(f"alignment receipt output drift: {name}: {output}")
    return row


def _matrix_row(
    feature: str,
    needle2: str,
    mei: str,
    *,
    implemented: bool,
    validated: bool,
    classification: str = "implemented",
    release_blocking: bool = True,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    allowed = {"implemented", "validated", "intentional_chinese_enhancement", "open"}
    if classification not in allowed:
        raise ValueError(f"bad alignment classification: {classification}")
    if validated and not implemented:
        raise ValueError(f"validated feature cannot be unimplemented: {feature}")
    return {
        "feature": feature,
        "needle2": needle2,
        "mei_51m": mei,
        "implementation_status": "implemented" if implemented else "open",
        "validation_status": "validated" if validated else "open",
        "classification": classification,
        "release_blocking_if_open": bool(release_blocking),
        "evidence": evidence or [],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_json(REFERENCE)
    if reference.get("schema") != "mei-needle2-reference-v1":
        raise RuntimeError("Needle2 reference pin is invalid")
    receipts = {name: _receipt(args.run_dir, name) for name in REQUIRED_STAGES}
    resource = load_json(args.resource_report)
    portable = load_json(args.portable_gates)

    import sys

    sys.path.insert(0, str(ROOT / "platform/python-sdk"))
    from mei_sdk.package import load_package

    package = load_package(args.package)
    manifest = package.manifest
    entries = (manifest.get("tensor_container") or {}).get("directory") or []
    tensor_names = [str(row.get("name") or "") for row in entries]
    mtp_names = [name for name in tensor_names if "mtp" in name.lower()]
    contracts = architecture_contracts()
    package_ok = (
        package.verified_hashes
        and package.tensor_identity_complete()
        and package.portable_quantization_policy_complete()
        and package.tool_index_payload_complete()
        and not package.heads.missing()
    )
    runtime_ok = bool(portable.get("all_gates_passed"))
    python_cache = (
        (portable.get("python_mlx_numeric_runtime") or {}).get("runtime_cache")
        or {}
    )
    browser_runtime = portable.get("browser_wasm_numeric_runtime") or {}
    browser_cache = browser_runtime.get("runtime_cache") or {}
    python_browser_numeric = (
        python_cache.get("kv_storage_dtype") in {"int8", "mlx.core.int8"}
        and python_cache.get("cache_growth_bounded") is True
        and browser_runtime.get("inference") is True
        and browser_cache.get("kv_storage_dtype") == "int8"
        and browser_cache.get("cache_growth_bounded") is True
    )
    kv = portable.get("bounded_int8_runtime") or {}
    bounded_kv_ok = (
        kv.get("kv_storage_dtype") == "int8"
        and kv.get("activation_quantization") == "int8-qdq"
        and kv.get("stable_prefix_tokens_max") == 1024
        and kv.get("rolling_window_tokens") == 256
        and kv.get("cache_growth_bounded") is True
    )
    resource_ok = bool(resource.get("resource_eligible"))
    confidence_metrics = receipts["confidence_calibration"].get("metrics") or {}
    confidence_class_coverage = (
        int(confidence_metrics.get("n_positive") or 0) > 0
        and int(confidence_metrics.get("n_negative") or 0) > 0
    )
    semantic = portable.get("semantic_boundaries") or {}
    mw_boundary_ok = semantic == {
        "retrieval": "independent-contrastive-head",
        "mw_disposition": "independent-20class-sidecar",
        "confidence": "independent-calibrated-binary-head",
        "mw_deviation": "deterministic-governance-gate-no-tensors",
    }

    rows = [
        _matrix_row(
            "backbone",
            "512d / 27 layers / GQA 8:4 / head 64 / RoPE 100000 / context 2048",
            "同几何；中文 24k vocabulary 使参数量为 51,463,797",
            implemented=True,
            validated=package.tensor_identity_complete(),
            evidence=["architecture contract", "portable tensor directory"],
        ),
        _matrix_row(
            "Engram",
            "sites 2,15; orders 2,3; slots 8192; sub-dim 128; 4 taps",
            "相同机制与边界 mask；中文 tokenizer 输入",
            implemented=True,
            validated=package_ok,
            evidence=["400-tensor weight contract", "Rust/Python model tests"],
        ),
        _matrix_row(
            "mHC",
            "4 lanes and 20-step Sinkhorn routing",
            "4 lanes and 20-step Sinkhorn；mHC tensors 默认 CQ4",
            implemented=True,
            validated=package_ok,
            evidence=["architecture contract", "CQ2 tensor policy"],
        ),
        _matrix_row(
            "MTP",
            "训练期下一 token 辅助路径",
            "同 parent/预算的 t+1 对照与 t+2 旁路消融；部署丢弃",
            implemented=True,
            validated=(
                receipts["mtp_ablation"]["terminal_status"] in {"passed", "degraded"}
                and not mtp_names
            ),
            evidence=[str(args.run_dir / "stages/mtp_ablation/receipt.json")],
        ),
        _matrix_row(
            "CQ2",
            "group128 WHT + Gaussian/Lloyd-Max codebook; embedding/mHC Q4, default Q2",
            "mei-cq-v2-g128-wht-codebook；逐 group bit map；完整 LM/heads 单容器",
            implemented=True,
            validated=package_ok,
            evidence=[str(args.package / "mei-model.json"), str(args.package / "tensors.bin")],
        ),
        _matrix_row(
            "KV/activation int8",
            "8-bit KV cache and activations; sliding window 256",
            "稳定前缀最多 1024 + 普通 ring 256；KV int8；activation int8 Q/DQ",
            implemented=bounded_kv_ok,
            validated=bounded_kv_ok and runtime_ok,
            classification="implemented" if bounded_kv_ok else "open",
            evidence=[str(args.portable_gates)],
        ),
        _matrix_row(
            "retrieval",
            "contrastive head, top-5, fingerprinted persistent index",
            "独立 R0/R1 contrastive head；最终 LM 后重训 R1；f16 normalized index",
            implemented=True,
            validated=package_ok and receipts["retrieval_r1"]["terminal_status"] in {"passed", "degraded"},
            evidence=[str(args.run_dir / "stages/retrieval_r1/receipt.json")],
        ),
        _matrix_row(
            "full-call SFT",
            "grammar-constrained call/refuse and multi-step continuation",
            "oracle top-5 quant-aware LM SFT + learned top-5 E2E",
            implemented=True,
            validated=receipts["learned_top5_e2e"]["terminal_status"] in {"passed", "degraded"},
            evidence=[str(args.run_dir / "stages/learned_top5_e2e/receipt.json")],
        ),
        _matrix_row(
            "MW disposition",
            "无对应 20 类 sidecar",
            "独立 20 类 reason-code sidecar；class 0 之外 fail closed",
            implemented=True,
            validated=(mw_boundary_ok and package.heads.mw_disposition.status == "ready"),
            classification="intentional_chinese_enhancement",
            evidence=[str(args.run_dir / "stages/mw_disposition_20class/receipt.json")],
        ),
        _matrix_row(
            "MW deviation",
            "无同名治理合同",
            "仅确定性 evidence-boundary 评测/治理 receipt；零 tensor、非 capability",
            implemented=True,
            validated=mw_boundary_ok and not any("deviation" in name.lower() for name in tensor_names),
            classification="intentional_chinese_enhancement",
            evidence=["frozen SFT data release governance receipt"],
        ),
        _matrix_row(
            "confidence",
            "独立 calibrated head + decode probability",
            (
                "独立 confidence head；基于真实 runtime call correctness 校准；"
                "本轮若没有正负双类 outcome，只验证执行机制，不宣称校准有效"
            ),
            implemented=True,
            validated=(
                package.heads.confidence.status == "ready"
                and confidence_class_coverage
            ),
            evidence=[str(args.run_dir / "stages/confidence_calibration/receipt.json")],
        ),
        _matrix_row(
            "runtime state machine",
            "complete/run/reset and call-result continuation",
            "complete/submit_tool_result/run/cancel/close；一次一步一次 call；最多 4/8 步",
            implemented=True,
            validated=runtime_ok,
            evidence=[str(args.portable_gates)],
        ),
        _matrix_row(
            "Python/Browser-WASM product paths",
            "Python API backed by libneedle and WASM engine",
            "Python/MLX 服务端 + Browser-WASM；Rust core 仅为 WASM 实现依赖",
            implemented=True,
            validated=runtime_ok and python_browser_numeric,
            evidence=[str(args.portable_gates)],
        ),
        _matrix_row(
            "independent Rust/Node/C SDK expansion",
            "非 Needle2 机制对齐的必要条件",
            "实现保留；独立 parity/性能/发布资格延后至更大语料 Base",
            implemented=True,
            validated=False,
            classification="open",
            release_blocking=False,
            evidence=[str(args.portable_gates)],
        ),
        _matrix_row(
            "中文 tokenizer/SFT/narration",
            "8k reference vocabulary and unconstrained reasoning text",
            "zh-24k-v1、中文 CPT/SFT、只消费 verified result 的确定性中文 narration",
            implemented=True,
            validated=runtime_ok,
            classification="intentional_chinese_enhancement",
            evidence=["tokenizer.model", "platform/_shared/runtime/narration.py"],
        ),
        _matrix_row(
            "provenance/permission/state safety",
            "evidence-only argument behavior",
            "grammar→schema→provenance/permission/state→MW disposition→confidence 确定性门序",
            implemented=True,
            validated=runtime_ok,
            classification="intentional_chinese_enhancement",
            evidence=[str(args.portable_gates)],
        ),
        _matrix_row(
            "resource targets",
            "公开配置约 14MB package / 28MB session",
            "本项目硬门 package≤18MiB、Rust≤64MiB、WASM≤96MiB",
            implemented=True,
            validated=resource_ok,
            classification="validated" if resource_ok else "open",
            evidence=[str(args.resource_report)],
        ),
        _matrix_row(
            ".cact/libneedle/Cactus ABI",
            "Needle2 原生部署格式和 ABI",
            "有意不兼容；MEI 使用 mei-model-package-v2 / mei-runtime-abi-2",
            implemented=False,
            validated=False,
            classification="open",
            release_blocking=False,
            evidence=[str(REFERENCE)],
        ),
    ]
    critical_open = [
        row["feature"]
        for row in rows
        if row["release_blocking_if_open"]
        and (row["implementation_status"] == "open" or row["validation_status"] == "open")
    ]
    report = {
        "schema": "mei-needle2-alignment-report-v1",
        "product": "mei-1.0-51m",
        "package_id": package.package_id,
        "reference": reference,
        "reference_sha256": sha_file(REFERENCE),
        "contracts": {
            key: contracts[key]
            for key in (
                "weight_contract_sha256",
                "runtime_profile_sha256",
                "training_aux_sha256",
            )
        },
        "compatibility_claimed": False,
        "confidence_calibration_class_coverage": confidence_class_coverage,
        "matrix": rows,
        "critical_open": critical_open,
        "mechanism_alignment_validated": not critical_open,
        "score_parity_claimed": False,
        "note": "机制对齐不等于 .cact/libneedle/Cactus API/ABI 兼容，也不等于分数对齐。",
    }
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# mei-1.0-51m 与 Needle2 机制对齐报告",
        "",
        f"- 候选：`{report['package_id']}`",
        f"- 机制对齐已验证：`{str(report['mechanism_alignment_validated']).lower()}`",
        "- 不声明 `.cact`、`libneedle`、Cactus API/ABI 或分数兼容。",
        "",
        "| 特性 | 实现 | 验证 | 分类 | MEI 结论 |",
        "|---|---|---|---|---|",
    ]
    for row in report["matrix"]:
        lines.append(
            "| {feature} | {implementation_status} | {validation_status} | {classification} | {mei_51m} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 剩余关键缺口",
            "",
            (
                "- 无。"
                if not report["critical_open"]
                else "\n".join(f"- {name}" for name in report["critical_open"])
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = build_report(args)
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_md.with_suffix(args.out_md.suffix + ".tmp")
    temporary.write_text(markdown(report), encoding="utf-8")
    temporary.replace(args.out_md)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--portable-gates", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
