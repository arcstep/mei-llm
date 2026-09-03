#!/usr/bin/env python3
"""Generate the recovery-chain Chinese Needle2 alignment report.

Mechanism validation and release quality are intentionally separate axes.  A
300M candidate may prove that a mechanism exists and executes portably while
remaining release-ineligible because retrieval, confidence, learned narration,
or resource quality is degraded.  This script validates the immutable upstream
adoption plus every downstream receipt; it never disguises the superseded
blocked package stage as part of the successful chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from common._repo import CURRENT_PATH, ROOT, architecture_contracts


REFERENCE = ROOT / "platform/_shared/spec/needle2-reference.json"
SOURCE_FILE = ROOT / "model-factory/orchestration/run_downstream_needle2_alignment_51m.py"
UPSTREAM_REQUIRED = (
    "float_base_lm_anchor",
    "float_task_control",
    "q4_diagnostic",
    "cq2_qat_v2",
    "retrieval_r0",
    "oracle_top5_fullcall",
    "retrieval_r1",
    "tool_index_final",
    "learned_top5_e2e",
    "mw_disposition_20class",
    "confidence_calibration",
    "narration_adapter",
)
EXPECTED_SEMANTIC_BOUNDARIES = {
    "retrieval": "independent-contrastive-head",
    "mw_disposition": "independent-20class-sidecar",
    "confidence": "independent-calibrated-binary-head",
    "mw_deviation": "deterministic-governance-gate-no-tensors",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    write_once(path, canonical_bytes(value) + b"\n")


def verify_outputs(receipt: dict[str, Any], *, label: str) -> None:
    outputs = receipt.get("output_hashes") or {}
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"{label} has no output hashes")
    for raw, digest in outputs.items():
        path = Path(raw)
        if not path.is_file() or sha_file(path) != digest:
            raise RuntimeError(f"{label} output drifted: {path}")


def verify_downstream_receipt(
    path: Path, *, stage_id: str, package_id: str | None = None
) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != "mei-productization-downstream-stage-receipt-v1"
        or receipt.get("stage_id") != stage_id
        or receipt.get("terminal_status") not in {"passed", "degraded"}
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
        or (package_id is not None and receipt.get("package_id") != package_id)
    ):
        raise RuntimeError(f"invalid downstream receipt: {stage_id}")
    verify_outputs(receipt, label=stage_id)
    return receipt


def verify_adoption(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    adoption = load_json(path)
    if (
        adoption.get("schema") != "mei-productization-upstream-adoption-receipt-v1"
        or adoption.get("status") != "passed"
        or adoption.get("adopted_through") != "narration_adapter"
        or adoption.get("current_unchanged") is not True
        or adoption.get("current_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("upstream adoption receipt is incomplete")
    source_run = Path(str(adoption.get("source_run") or "")).resolve()
    plan_path = source_run / "plan.json"
    if not plan_path.is_file() or sha_file(plan_path) != adoption.get("source_plan_sha256"):
        raise RuntimeError("adopted source plan drifted")
    plan = load_json(plan_path)
    if plan.get("run_fingerprint_sha256") != adoption.get("source_run_fingerprint_sha256"):
        raise RuntimeError("adopted source run fingerprint drifted")
    stages = {}
    for entry in adoption.get("adopted_stages") or []:
        if not isinstance(entry, dict):
            continue
        stage_id = str(entry.get("stage_id") or "")
        receipt_path = Path(str(entry.get("receipt") or ""))
        if (
            stage_id in stages
            or not receipt_path.is_file()
            or sha_file(receipt_path) != entry.get("receipt_sha256")
        ):
            raise RuntimeError(f"adopted stage receipt drifted: {stage_id}")
        receipt = load_json(receipt_path)
        if (
            receipt.get("stage_id") != stage_id
            or receipt.get("terminal_status") not in {"passed", "degraded"}
            or receipt.get("stage_fingerprint_sha256")
            != entry.get("stage_fingerprint_sha256")
        ):
            raise RuntimeError(f"adopted stage terminal state drifted: {stage_id}")
        verify_outputs(receipt, label=f"upstream:{stage_id}")
        stages[stage_id] = {"receipt": receipt, "path": receipt_path}
    if set(UPSTREAM_REQUIRED) - set(stages):
        raise RuntimeError(
            f"upstream adoption missing stages: {sorted(set(UPSTREAM_REQUIRED) - set(stages))}"
        )
    return adoption, plan, stages


def verify_package_receipt(path: Path, package_id: str | None = None) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != "mei-productization-downstream-package-receipt-v1"
        or receipt.get("stage_id") != "package_v2_cq2_downstream"
        or receipt.get("terminal_status") != "passed"
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
        or (package_id is not None and receipt.get("package_id") != package_id)
    ):
        raise RuntimeError("downstream package receipt is incomplete")
    verify_outputs(receipt, label="package_v2_cq2_downstream")
    return receipt


def require_report_output(receipt: dict[str, Any], path: Path, *, label: str) -> dict[str, Any]:
    expected = (receipt.get("output_hashes") or {}).get(str(path.resolve()))
    if expected is None:
        expected = (receipt.get("output_hashes") or {}).get(str(path))
    if expected is None or not path.is_file() or sha_file(path) != expected:
        raise RuntimeError(f"{label} report is not bound by its receipt: {path}")
    return load_json(path)


def matrix_row(
    feature: str,
    needle2: str,
    mei: str,
    *,
    implemented: bool,
    validated: bool,
    classification: str | None = None,
    mechanism_release_blocking: bool = True,
    quality_status: str = "not_applicable",
    quality_release_blocking: bool = False,
    evidence: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    classifications = {
        "implemented",
        "validated",
        "intentional_chinese_enhancement",
        "open",
    }
    if classification is None:
        classification = (
            "validated" if validated else "implemented" if implemented else "open"
        )
    quality_values = {"not_applicable", "passed", "degraded", "degenerate", "open"}
    if classification not in classifications or quality_status not in quality_values:
        raise ValueError(f"invalid alignment status for {feature}")
    if validated and not implemented:
        raise ValueError(f"validated feature cannot be unimplemented: {feature}")
    return {
        "feature": feature,
        "needle2": needle2,
        "mei_51m": mei,
        "implementation_status": "implemented" if implemented else "open",
        "validation_status": "validated" if validated else "open",
        "classification": classification,
        "mechanism_release_blocking_if_open": mechanism_release_blocking,
        "quality_status": quality_status,
        "quality_release_blocking": quality_release_blocking,
        "evidence": evidence or [],
        "note": note,
    }


def _governance_evidence(plan: dict[str, Any]) -> tuple[bool, Path | None]:
    data = (plan.get("immutable") or {}).get("data_release") or {}
    data_dir = Path(str(data.get("path") or ""))
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file() or sha_file(manifest_path) != data.get("manifest_sha256"):
        return False, None
    manifest = load_json(manifest_path)
    deviation = manifest.get("mw_deviation") or {}
    receipt_path = data_dir / str(deviation.get("receipt") or "")
    ok = (
        deviation.get("kind") == "governance_gate_not_head"
        and receipt_path.is_file()
        and sha_file(receipt_path) == deviation.get("receipt_sha256")
        and deviation.get("receipt_sha256")
        == (data.get("mw_deviation") or {}).get("receipt_sha256")
    )
    return ok, receipt_path if ok else None


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    reference = load_json(REFERENCE)
    if reference.get("schema") != "mei-needle2-reference-v1":
        raise RuntimeError("Needle2 reference pin is invalid")
    adoption, source_plan, upstream = verify_adoption(args.adoption_receipt.resolve())
    package_receipt = verify_package_receipt(args.package_receipt.resolve())
    package_id = str(package_receipt["package_id"])
    portable_receipt = verify_downstream_receipt(
        args.portable_receipt.resolve(),
        stage_id="portable_runtime_gates",
        package_id=package_id,
    )
    narration_receipt = verify_downstream_receipt(
        args.narration_receipt.resolve(),
        stage_id="narration_generation_eval",
        package_id=package_id,
    )
    mtp_receipt = verify_downstream_receipt(
        args.mtp_receipt.resolve(), stage_id="mtp_ablation", package_id=package_id
    )
    resource_receipt = verify_downstream_receipt(
        args.resource_receipt.resolve(),
        stage_id="resource_measurement",
        package_id=package_id,
    )
    base_entry_receipt = verify_downstream_receipt(
        args.base_entry_receipt.resolve(), stage_id="arbitrary_base_entry"
    )
    portable = require_report_output(
        portable_receipt,
        args.portable_report.resolve(),
        label="portable_runtime_gates",
    )
    narration = require_report_output(
        narration_receipt,
        args.narration_report.resolve(),
        label="narration_generation_eval",
    )
    resource = require_report_output(
        resource_receipt,
        args.resource_report.resolve(),
        label="resource_measurement",
    )
    base_entry = require_report_output(
        base_entry_receipt,
        args.base_entry_report.resolve(),
        label="arbitrary_base_entry",
    )

    sys.path.insert(0, str(ROOT / "platform/python-sdk"))
    from mei_sdk.package import load_package

    package = load_package(args.package.resolve())
    if package.package_id != package_id or not package.resource_measurement_verified:
        raise RuntimeError("measured package is not bound to resource evidence")
    manifest = package.manifest
    entries = (manifest.get("tensor_container") or {}).get("directory") or []
    tensor_names = [str(row.get("name") or "") for row in entries]
    lm_entries = [row for row in entries if row.get("role") == "lm"]
    mtp_names = [name for name in tensor_names if "mtp" in name.lower()]
    mw_deviation_names = [name for name in tensor_names if "deviation" in name.lower()]
    exact_lm = len(lm_entries) == 400 and sum(
        int(row.get("n_params") or 0) for row in lm_entries
    ) == 51_463_797
    package_ok = (
        package.verified_hashes
        and package.tensor_identity_complete()
        and package.portable_quantization_policy_complete()
        and package.tool_index_payload_complete()
        and not package.heads.missing()
        and exact_lm
    )
    contracts = architecture_contracts()
    expected_contracts = {
        key: contracts[key]
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    contracts_ok = manifest.get("contracts") == expected_contracts
    runtime_ok = portable.get("all_gates_passed") is True
    bounded = portable.get("bounded_int8_runtime") or {}
    bounded_ok = (
        bounded.get("kv_storage_dtype") == "int8"
        and bounded.get("activation_quantization") == "int8-qdq"
        and bounded.get("stable_prefix_tokens_max") == 1024
        and bounded.get("rolling_window_tokens") == 256
        and bounded.get("cache_growth_bounded") is True
    )
    semantic_ok = portable.get("semantic_boundaries") == EXPECTED_SEMANTIC_BOUNDARIES
    governance_ok, governance_path = _governance_evidence(source_plan)
    mw_boundary_ok = semantic_ok and governance_ok and not mw_deviation_names
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
    learned_metrics = upstream["learned_top5_e2e"]["receipt"].get("metrics") or {}
    learned_quality = "passed" if learned_metrics.get("quality_ok") is True else "degraded"
    confidence_metrics = upstream["confidence_calibration"]["receipt"].get("metrics") or {}
    confidence_n = int(confidence_metrics.get("n") or 0)
    confidence_non_degenerate = (
        confidence_n > 0
        and int(confidence_metrics.get("n_positive") or 0) > 0
        and int(confidence_metrics.get("n_negative") or 0) > 0
    )
    mw_metrics = upstream["mw_disposition_20class"]["receipt"].get("metrics") or {}
    mw_classes_complete = set((mw_metrics.get("class_counts") or {}).keys()) == {
        str(index) for index in range(20)
    }
    mw_loss = float(mw_metrics.get("last_loss") or math.inf)
    mw_quality = "degraded" if not math.isfinite(mw_loss) or mw_loss > math.log(20) else "passed"
    narration_metrics = narration.get("metrics") or {}
    narration_mechanism_ok = (
        narration.get("terminal_status") == "passed"
        and narration.get("packed_cq2_evaluated") is True
        and int(narration_metrics.get("rows") or 0) == 600
        and narration_metrics.get("mechanism_status") == "passed"
        and float(narration_metrics.get("deterministic_fallback_exact_rate") or 0.0)
        == 1.0
    )
    narration_quality = (
        "passed"
        if narration_metrics.get("learned_adapter_quality") == "validated"
        else "degraded"
    )
    base_entry_ok = (
        base_entry.get("terminal_status") == "passed"
        and (base_entry.get("claims") or {}).get("productizer_is_frozen_base_agnostic")
        is True
        and (base_entry.get("claims") or {}).get(
            "600m_or_900m_artifact_availability_claimed"
        )
        is False
    )
    agent = ((source_plan.get("immutable") or {}).get("data_release") or {}).get(
        "agent_continuation"
    ) or {}
    agent_data_ok = (
        int(agent.get("nonempty_tool_result_rows") or 0) > 0
        and set(agent.get("target_kinds") or []) == {"call", "respond"}
        and agent.get("group_aware") is True
        and agent.get("family_aware") is True
    )
    resource_ok = resource.get("resource_eligible") is True

    evidence = {
        "adoption": str(args.adoption_receipt.resolve()),
        "package": str(args.package_receipt.resolve()),
        "portable": str(args.portable_receipt.resolve()),
        "narration": str(args.narration_receipt.resolve()),
        "mtp": str(args.mtp_receipt.resolve()),
        "resource": str(args.resource_receipt.resolve()),
        "base_entry": str(args.base_entry_receipt.resolve()),
    }
    rows = [
        matrix_row(
            "backbone",
            "512d / 27 layers / GQA 8:4 / head 64 / RoPE 100000 / context 2048",
            "同几何；zh-24k-v1；部署 LM 精确 51,463,797 参数",
            implemented=True,
            validated=package_ok and contracts_ok,
            evidence=[evidence["package"]],
        ),
        matrix_row(
            "Engram",
            "sites 2,15；orders 2,3；slots 8192；sub-dim 128；4 taps",
            "相同机制与边界 mask，使用中文 tokenizer 输入",
            implemented=True,
            validated=package_ok and runtime_ok,
            evidence=[evidence["package"], evidence["portable"]],
        ),
        matrix_row(
            "mHC",
            "4 lanes 与 20-step Sinkhorn routing",
            "4 lanes 与 20-step Sinkhorn；mHC tensor 默认 CQ4",
            implemented=True,
            validated=package_ok,
            evidence=[evidence["package"]],
        ),
        matrix_row(
            "MTP",
            "训练期辅助预测路径",
            "固定预算 t+1 control / t+2 MTP 单变量消融；部署零残留",
            implemented=True,
            validated=(
                mtp_receipt.get("training_only") is True
                and mtp_receipt.get("temporary_heads_exported") is False
                and mtp_receipt.get("deployment_mtp_tensor_count") == 0
                and not mtp_names
            ),
            evidence=[evidence["mtp"], evidence["package"]],
        ),
        matrix_row(
            "CQ2",
            "group128 WHT + Gaussian/Lloyd-Max；embedding/mHC Q4、默认 Q2",
            "真正 mei-cq-v2-g128-wht-codebook；逐 group bit map；LM+heads 单容器",
            implemented=True,
            validated=(
                package_ok
                and package_receipt.get("quant_math_id")
                == "mei-cq-v2-g128-wht-codebook"
            ),
            evidence=[evidence["package"]],
        ),
        matrix_row(
            "KV/activation int8",
            "8-bit KV/activation；sliding window 256",
            "稳定前缀≤1024 + ring 256；KV int8；activation int8 Q/DQ",
            implemented=bounded_ok,
            validated=bounded_ok and runtime_ok and python_browser_numeric,
            evidence=[evidence["portable"], evidence["resource"]],
        ),
        matrix_row(
            "retrieval",
            "独立 contrastive head、top-5、持久化 fingerprint index",
            "R0→full-call LM→R1；最终 LM 后重训并重建归一化 f16 index",
            implemented=package.heads.contrastive.status == "ready",
            validated=(package.heads.contrastive.status == "ready" and runtime_ok),
            quality_status=learned_quality,
            quality_release_blocking=True,
            evidence=[str(upstream["retrieval_r1"]["path"]), str(upstream["learned_top5_e2e"]["path"])],
            note=(
                f"learned recall@5={learned_metrics.get('recall_at_5_learned')}; "
                f"lexical={learned_metrics.get('recall_at_5_lexical')}"
            ),
        ),
        matrix_row(
            "full-call / multi-step Agent SFT",
            "constrained call/refuse 与 call-result continuation",
            "oracle top-5 quant-aware SFT + 非空 ToolResult 多步 call→respond",
            implemented=agent_data_ok and manifest.get("capabilities", {}).get("full_call") is True,
            validated=agent_data_ok and runtime_ok,
            quality_status=learned_quality,
            quality_release_blocking=True,
            evidence=[str(upstream["oracle_top5_fullcall"]["path"]), str(upstream["learned_top5_e2e"]["path"])],
        ),
        matrix_row(
            "MW disposition",
            "无对应 20 类原因 sidecar",
            "独立 20 类 reason-code sidecar；仅 class 0 可继续且不能覆盖确定性门禁",
            implemented=package.heads.mw_disposition.status == "ready",
            validated=mw_classes_complete and semantic_ok and runtime_ok,
            classification="intentional_chinese_enhancement",
            quality_status=mw_quality,
            quality_release_blocking=True,
            evidence=[str(upstream["mw_disposition_20class"]["path"])],
            note=f"训练末 loss={mw_loss}; 机制验证与分类质量分开记录",
        ),
        matrix_row(
            "MW deviation",
            "无同名治理合同",
            "确定性 evidence-boundary 治理 receipt；零 tensor、非 head、非 capability",
            implemented=governance_ok,
            validated=mw_boundary_ok,
            classification="intentional_chinese_enhancement",
            evidence=[str(governance_path) if governance_path else evidence["adoption"]],
        ),
        matrix_row(
            "confidence head",
            "独立 calibrated head + decode probability",
            "独立 binary sidecar；真实 runtime correctness 作为校准标签",
            implemented=package.heads.confidence.status == "ready",
            validated=package.heads.confidence.status == "ready" and runtime_ok,
            quality_status="passed" if confidence_non_degenerate else "degenerate",
            quality_release_blocking=True,
            evidence=[str(upstream["confidence_calibration"]["path"]), evidence["portable"]],
            note=(
                f"n={confidence_n}, positive={confidence_metrics.get('n_positive')}, "
                f"negative={confidence_metrics.get('n_negative')}; 当前校准集退化不能证明概率质量"
            ),
        ),
        matrix_row(
            "runtime state machine",
            "stepwise complete/run/reset 与 call-result continuation",
            "complete/submit_tool_result/run/cancel/close；call→result→call|respond|refuse|error",
            implemented=True,
            validated=runtime_ok and agent_data_ok,
            evidence=[evidence["portable"]],
        ),
        matrix_row(
            "Python/Browser-WASM product paths",
            "Python/libneedle 与 WASM engine",
            "Python/MLX 服务端 + Browser-WASM；Rust core 仅为 WASM 实现依赖",
            implemented=True,
            validated=runtime_ok and python_browser_numeric,
            evidence=[evidence["portable"]],
        ),
        matrix_row(
            "independent Rust/Node/C SDK expansion",
            "非 Needle2 机制对齐的必要条件",
            "实现保留；独立 parity/性能/发布资格延后至更大语料 Base",
            implemented=True,
            validated=False,
            classification="open",
            mechanism_release_blocking=False,
            evidence=[evidence["portable"]],
        ),
        matrix_row(
            "确定性中文 narration fallback",
            "无对应中文终态模板层",
            "仅消费 verified ToolResult；终态一次生成；失败/取消也可解释",
            implemented=True,
            validated=narration_mechanism_ok,
            classification="intentional_chinese_enhancement",
            evidence=[evidence["narration"]],
        ),
        matrix_row(
            "rank-16 narration generation sidecar",
            "无对应独立中文 sidecar",
            "冻结 51M 主干的 392,192 参数词表 residual；CQ2 包内约 102KiB",
            implemented=package.heads.narration_adapter.status == "ready",
            validated=narration_mechanism_ok and runtime_ok,
            classification="intentional_chinese_enhancement",
            quality_status=narration_quality,
            # The deterministic fallback proves the mechanism remains safe,
            # but the learned sidecar is still a declared product capability.
            # Keep its quality axis aligned with the final audit: a degraded
            # adapter does not block process completion, but it does make the
            # candidate release-ineligible.
            quality_release_blocking=True,
            evidence=[str(upstream["narration_adapter"]["path"]), evidence["narration"]],
            note=(
                f"adapter exact={narration_metrics.get('adapter_exact_rate')}; "
                f"fallback={narration_metrics.get('adapter_fallback_rate')}"
            ),
        ),
        matrix_row(
            "provenance/permission/state safety",
            "evidence-only arguments 与 constrained schema",
            "grammar→schema→provenance/permission/state→MW disposition→confidence",
            implemented=True,
            validated=runtime_ok and mw_boundary_ok,
            classification="intentional_chinese_enhancement",
            evidence=[evidence["portable"], str(governance_path) if governance_path else evidence["adoption"]],
        ),
        matrix_row(
            "arbitrary frozen base entry",
            "同架构 checkpoint 可复用产品化机制",
            "以 weight/tokenizer contract 选择 base；300M 实跑，600M/900M 控制面 fixture",
            implemented=base_entry_ok,
            validated=base_entry_ok,
            classification="intentional_chinese_enhancement",
            evidence=[evidence["base_entry"]],
            note="不声称尚不存在的 600M/900M artifact 或质量已验证",
        ),
        matrix_row(
            "resource targets",
            "参考实现报告约 14MB package / 28MB session",
            "硬门：package≤18MiB、Rust≤64MiB、WASM≤96MiB",
            implemented=True,
            validated=resource_ok,
            mechanism_release_blocking=False,
            quality_status="passed" if resource_ok else "open",
            quality_release_blocking=True,
            evidence=[evidence["resource"]],
        ),
        matrix_row(
            ".cact/libneedle/Cactus ABI",
            "Needle2 原生格式与 ABI",
            "有意不兼容；MEI 使用 mei-model-package-v2 / mei-runtime-abi-2",
            implemented=False,
            validated=False,
            classification="open",
            mechanism_release_blocking=False,
            evidence=[str(REFERENCE)],
        ),
    ]
    critical_mechanism_open = [
        row["feature"]
        for row in rows
        if row["mechanism_release_blocking_if_open"]
        and (
            row["implementation_status"] == "open"
            or row["validation_status"] == "open"
        )
    ]
    quality_open = [
        row["feature"]
        for row in rows
        if row["quality_release_blocking"]
        and row["quality_status"] in {"degraded", "degenerate", "open"}
    ]
    report = {
        "schema": "mei-needle2-alignment-report-v2",
        "product": "mei-1.0-51m",
        "package_id": package_id,
        "base_id": source_plan["immutable"]["base"]["base_id"],
        "base_exposure_tokens": source_plan["immutable"]["base"]["tokens_seen_exposure"],
        "reference": reference,
        "reference_sha256": sha_file(REFERENCE),
        "contracts": expected_contracts,
        "upstream_adoption_receipt_sha256": sha_file(args.adoption_receipt.resolve()),
        "evidence_chain": evidence,
        "matrix": rows,
        "critical_mechanism_open": critical_mechanism_open,
        "quality_or_resource_open": quality_open,
        "mechanism_alignment_validated": not critical_mechanism_open,
        "release_quality_validated": not quality_open,
        "process_evidence_complete": True,
        "compatibility_claimed": False,
        "score_parity_claimed": False,
        "note": (
            "机制执行/跨运行时验证与质量分数严格分轴；不声明 .cact、libneedle、"
            "Cactus API/ABI 或分数兼容。"
        ),
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    input_hashes = {
        str(path.resolve()): sha_file(path.resolve())
        for path in (
            args.adoption_receipt,
            args.package_receipt,
            args.portable_receipt,
            args.portable_report,
            args.narration_receipt,
            args.narration_report,
            args.mtp_receipt,
            args.resource_receipt,
            args.resource_report,
            args.base_entry_receipt,
            args.base_entry_report,
            REFERENCE,
            SOURCE_FILE,
        )
    }
    return report, input_hashes


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# mei-1.0-51m 中文增强 Needle2 机制对齐报告",
        "",
        f"- 候选：`{report['package_id']}`",
        f"- 验证 base：`{report['base_id']}`（累计 exposure `{report['base_exposure_tokens']}`）",
        f"- 关键机制验证：`{str(report['mechanism_alignment_validated']).lower()}`",
        f"- 发布质量验证：`{str(report['release_quality_validated']).lower()}`",
        "- 不声明 `.cact`、`libneedle`、Cactus API/ABI 或分数兼容。",
        "",
        "| 特性 | 实现 | 机制验证 | 质量 | 分类 | MEI 结论 |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["matrix"]:
        lines.append(
            "| {feature} | {implementation_status} | {validation_status} | "
            "{quality_status} | {classification} | {mei_51m} |".format(**row)
        )
    lines.extend(["", "## 关键机制未闭环", ""])
    lines.extend(
        [f"- {name}" for name in report["critical_mechanism_open"]]
        or ["- 无。"]
    )
    lines.extend(["", "## 质量或资源发布缺口", ""])
    lines.extend(
        [f"- {name}" for name in report["quality_or_resource_open"]]
        or ["- 无。"]
    )
    lines.extend(["", "## 结论边界", "", f"{report['note']}", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report, input_hashes = build_report(args)
    fingerprint = hashlib.sha256(canonical_bytes(input_hashes)).hexdigest()
    report["stage_fingerprint_sha256"] = fingerprint
    out_dir = args.out_dir.resolve()
    json_path = out_dir / "needle2-alignment.json"
    markdown_path = out_dir / "needle2-alignment.zh-CN.md"
    receipt_path = out_dir / "receipt.json"
    write_json_once(json_path, report)
    write_once(markdown_path, render_markdown(report).encode("utf-8"))
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "needle2_alignment_report",
        "terminal_status": "passed",
        "process_complete": True,
        "product": "mei-1.0-51m",
        "package_id": report["package_id"],
        "stage_fingerprint_sha256": fingerprint,
        "mechanism_alignment_validated": report["mechanism_alignment_validated"],
        "release_quality_validated": report["release_quality_validated"],
        "critical_mechanism_open": report["critical_mechanism_open"],
        "quality_or_resource_open": report["quality_or_resource_open"],
        "input_hashes": input_hashes,
        "output_hashes": {
            str(json_path): sha_file(json_path),
            str(markdown_path): sha_file(markdown_path),
        },
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_json_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adoption-receipt", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--portable-receipt", type=Path, required=True)
    parser.add_argument("--portable-report", type=Path, required=True)
    parser.add_argument("--narration-receipt", type=Path, required=True)
    parser.add_argument("--narration-report", type=Path, required=True)
    parser.add_argument("--mtp-receipt", type=Path, required=True)
    parser.add_argument("--resource-receipt", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--base-entry-receipt", type=Path, required=True)
    parser.add_argument("--base-entry-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
