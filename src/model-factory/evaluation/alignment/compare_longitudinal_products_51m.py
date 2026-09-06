#!/usr/bin/env python3
"""Hash-bound paired comparison for mei-1.0-51m productization runs.

The report deliberately has no aggregate score.  It verifies that the two
products used the same frozen task/evaluation contracts, then reports each
Base, QAT, task, sidecar, package, and resource metric independently.  A
completed pair may still be ineligible for causal exposure attribution when
lineage, corpus diversity, or source provenance is confounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file())
EXPECTED_PARAMS = 51_463_797
EXPECTED_PLAN_SCHEMA = "mei-51m-sft-v4-productization-plan-v1"

# These files change numerical/training/evaluation semantics.  Control-plane
# receipt and eligibility fixes are reported separately and do not silently
# count as semantic-source equality.
SEMANTIC_SOURCE_SUFFIXES = {
    "src/architecture/mei-1.2-51m/heads.py",
    "platform/_shared/runtime/kv_manager.py",
    "platform/_shared/runtime/tool_index.py",
    "platform/python-sdk/mei_sdk/runtime_51m.py",
    ".internal/src/mei_llm/training/pipelines/cq2_qat_51m.py",
    ".internal/src/mei_llm/training/pipelines/freeze_longitudinal_eval_v7_51m.py",
    ".internal/src/mei_llm/training/pipelines/longitudinal_eval_metrics_51m.py",
    ".internal/src/mei_llm/training/pipelines/pack_cq2_v2_51m.py",
    ".internal/src/mei_llm/training/pipelines/sft_v3_eval_51m.py",
    ".internal/src/mei_llm/training/pipelines/sft_v3_training_51m.py",
    ".internal/src/mei_llm/training/pipelines/sft_v4_contract_51m.py",
    "src/model-factory/training/qat/cq2_qat_51m.py",
    "src/model-factory/evaluation/tool_use/freeze_longitudinal_eval_v7_51m.py",
    "src/model-factory/evaluation/tool_use/longitudinal_eval_metrics_51m.py",
    "src/model-factory/release/pack_cq2_v2_51m.py",
    "src/model-factory/evaluation/tool_use/sft_v3_eval_51m.py",
    "src/model-factory/training/tool_use/sft_v3_training_51m.py",
    "src/model-factory/contracts/sft_v4_contract_51m.py",
}

# The QAT launcher validates and records the selected Base, recipe, corpus,
# group policy, and worker receipt.  Changes confined to this launcher are
# therefore control/provenance drift when all of those frozen inputs and the
# numerical worker sources remain exact.  Every other QAT source file is part
# of the numerical/training contract and remains fail-closed for a scale curve.
QAT_CONTROL_SOURCE_SUFFIXES = {
    ".internal/src/mei_llm/training/pipelines/qat_cq2_v2_51m.py",
    "src/model-factory/training/qat/qat_cq2_v2_51m.py",
}


METRICS: tuple[tuple[str, str, str], ...] = (
    ("base_anchor", "validation_nll", "lower"),
    ("base_anchor", "domain_nll_hq", "lower"),
    ("base_anchor", "domain_nll_colloquial", "lower"),
    ("base_anchor", "domain_nll_structure", "lower"),
    ("base_anchor", "probe_mean_nll", "lower"),
    ("base_anchor", "task_control_slice.balanced_accuracy", "higher"),
    ("base_anchor", "task_control_slice.execute_tool_name_exact", "higher"),
    ("base_anchor", "task_control_slice.execute_arguments_exact", "higher"),
    ("base_anchor", "task_control_slice.refusal_accuracy", "higher"),
    ("float_task_control", "last_loss", "lower"),
    ("float_task_control", "task_control_slice.balanced_accuracy", "higher"),
    ("float_task_control", "task_control_slice.execute_tool_name_exact", "higher"),
    ("float_task_control", "task_control_slice.execute_arguments_exact", "higher"),
    ("float_task_control", "task_control_slice.refusal_accuracy", "higher"),
    ("float_task_control", "task_control_slice.false_execute_rate", "lower"),
    ("float_task_control", "task_control_slice.error_rate", "lower"),
    ("qat", "metrics.last_train_loss", "lower"),
    ("qat", "metrics.valid_loss", "lower"),
    ("scorecard", "metrics.retrieval.recall_at_1", "higher"),
    ("scorecard", "metrics.retrieval.recall_at_5", "higher"),
    ("scorecard", "metrics.retrieval.mrr", "higher"),
    ("scorecard", "metrics.retrieval.ndcg_at_5", "higher"),
    ("scorecard", "metrics.fullcall.balanced_accuracy", "higher"),
    ("scorecard", "metrics.fullcall.execute_tool_name_exact", "higher"),
    ("scorecard", "metrics.fullcall.execute_arguments_exact", "higher"),
    ("scorecard", "metrics.fullcall.refusal_accuracy", "higher"),
    ("scorecard", "metrics.fullcall.false_execute_rate", "lower"),
    ("scorecard", "metrics.fullcall.error_rate", "lower"),
    ("scorecard", "metrics.multi_step.trajectory_success", "higher"),
    ("scorecard", "metrics.multi_step.step_accuracy", "higher"),
    ("scorecard", "metrics.multi_step.terminal_response_accuracy", "higher"),
    ("scorecard", "metrics.multi_step.call_id_integrity", "higher"),
    ("scorecard", "metrics.multi_step.by_trajectory_length.2", "higher"),
    ("scorecard", "metrics.multi_step.by_trajectory_length.3", "higher"),
    ("scorecard", "metrics.multi_step.by_trajectory_length.4", "higher"),
    ("scorecard", "metrics.mw_disposition.accuracy", "higher"),
    ("scorecard", "metrics.mw_disposition.macro_f1", "higher"),
    ("scorecard", "metrics.mw_disposition.class_0_false_continue_rate", "lower"),
    ("scorecard", "metrics.mw_disposition.prediction_error_rate", "lower"),
    ("scorecard", "metrics.confidence.auroc", "higher"),
    ("scorecard", "metrics.confidence.auprc", "higher"),
    ("scorecard", "metrics.confidence.ece_10", "lower"),
    ("scorecard", "metrics.confidence.brier", "lower"),
    ("scorecard", "metrics.narration.required_fact_recall", "higher"),
    ("scorecard", "metrics.narration.number_preservation", "higher"),
    ("scorecard", "metrics.narration.polarity_accuracy", "higher"),
    ("scorecard", "metrics.narration.fallback_exact", "higher"),
    ("scorecard", "metrics.narration.delivered_answer_correctness", "higher"),
    ("scorecard", "metrics.natural_cross_generator.retrieval.recall_at_1", "higher"),
    ("scorecard", "metrics.natural_cross_generator.retrieval.recall_at_5", "higher"),
    ("scorecard", "metrics.natural_cross_generator.retrieval.mrr", "higher"),
    (
        "scorecard",
        "metrics.natural_cross_generator.fullcall.balanced_accuracy",
        "higher",
    ),
    (
        "scorecard",
        "metrics.natural_cross_generator.fullcall.execute_tool_name_exact",
        "higher",
    ),
    (
        "scorecard",
        "metrics.natural_cross_generator.fullcall.execute_arguments_exact",
        "higher",
    ),
    (
        "scorecard",
        "metrics.natural_cross_generator.fullcall.refusal_accuracy",
        "higher",
    ),
    ("scorecard", "metrics.whole_schema_holdout.retrieval.recall_at_1", "higher"),
    ("scorecard", "metrics.whole_schema_holdout.retrieval.recall_at_5", "higher"),
    ("scorecard", "metrics.whole_schema_holdout.retrieval.mrr", "higher"),
    ("scorecard", "metrics.whole_schema_holdout.fullcall.balanced_accuracy", "higher"),
    (
        "scorecard",
        "metrics.whole_schema_holdout.fullcall.execute_tool_name_exact",
        "higher",
    ),
    (
        "scorecard",
        "metrics.whole_schema_holdout.fullcall.execute_arguments_exact",
        "higher",
    ),
    ("scorecard", "metrics.whole_schema_holdout.fullcall.refusal_accuracy", "higher"),
    ("scorecard", "metrics.whole_schema_holdout.confidence.auroc", "higher"),
    ("scorecard", "metrics.whole_schema_holdout.confidence.ece_10", "lower"),
    ("scorecard", "metrics.whole_schema_holdout.confidence.brier", "lower"),
    ("package", "package_bytes", "lower"),
    ("package", "tensor_count", "identity"),
    ("package", "lm_tensor_count", "identity"),
    ("package", "mtp_tensor_count", "identity"),
    ("resources", "measurements.package_bytes", "lower"),
    ("resources", "measurements.wasm_heap_peak_bytes", "lower"),
    ("resources", "wasm_raw.decode_tok_s", "higher"),
    ("resources", "measurements.rust_session_peak_bytes", "lower"),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def resolve_path(raw: str | Path, run_dir: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [ROOT / path]
    if run_dir is not None:
        candidates.append(run_dir / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if not expected_sha256 or actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected_sha256} actual={actual} path={path}"
        )


def verify_output_hashes(
    receipt: dict[str, Any], run_dir: Path, label: str
) -> dict[str, str]:
    outputs = receipt.get("output_hashes") or {}
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"{label} receipt has no output hashes")
    verified: dict[str, str] = {}
    for raw, digest in outputs.items():
        if not isinstance(raw, str) or not isinstance(digest, str):
            raise RuntimeError(f"{label} receipt contains an invalid output binding")
        path = resolve_path(raw, run_dir)
        verify_file(path, digest, f"{label} output")
        verified[str(path)] = digest
    return verified


def verify_frozen_eval(eval_dir: Path, expected_lock_sha256: str) -> dict[str, Any]:
    lock_path = eval_dir / "lock.json"
    verify_file(lock_path, expected_lock_sha256, "eval lock")
    lock = load_json(lock_path)
    if lock.get("schema") != "mei-51m-longitudinal-eval-lock-v4":
        raise RuntimeError("paired comparison requires longitudinal eval-v7")
    for name, spec in (lock.get("artifacts") or {}).items():
        if not isinstance(spec, dict) or not spec.get("sha256"):
            raise RuntimeError(f"eval lock artifact lacks SHA: {name}")
        verify_file(eval_dir / name, str(spec["sha256"]), f"eval artifact {name}")
    isolation = load_json(eval_dir / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("eval isolation receipt is not passed")
    return lock


def _path_sha_pairs(value: Any) -> Iterable[tuple[Path, str]]:
    if isinstance(value, dict):
        raw_path = value.get("path")
        raw_sha = value.get("sha256")
        if isinstance(raw_path, str) and isinstance(raw_sha, str):
            yield Path(raw_path), raw_sha
        for key, child in value.items():
            if key in {"output_hashes", "predecessor_output_hashes"} and isinstance(
                child, dict
            ):
                for raw, digest in child.items():
                    if isinstance(raw, str) and isinstance(digest, str):
                        yield Path(raw), digest
            yield from _path_sha_pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_sha_pairs(child)


def find_bound_artifact(
    plan: dict[str, Any], run_dir: Path, basename: str
) -> tuple[Path, str]:
    declared: list[tuple[Path, str]] = []
    for raw, digest in _path_sha_pairs(plan):
        if raw.name == basename:
            path = resolve_path(raw, run_dir)
            declared.append((path, digest))
    unique = {(str(path), digest) for path, digest in declared}
    if len(unique) == 1:
        path_raw, digest = next(iter(unique))
        path = Path(path_raw)
        verify_file(path, digest, basename)
        return path, digest
    if len(unique) > 1:
        raise RuntimeError(f"multiple declared {basename} artifacts: {sorted(unique)}")
    matches = sorted(run_dir.glob(f"stages/**/{basename}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {basename} below {run_dir}, found {len(matches)}"
        )
    path = matches[0]
    receipt_path = path.parent / "receipt.json"
    receipt = load_json(receipt_path)
    status = receipt.get("terminal_status") or receipt.get("status")
    if status not in {"passed", "degraded"}:
        raise RuntimeError(f"{basename} stage is not terminal: {status}")
    expected = (receipt.get("output_hashes") or {}).get(str(path.resolve()))
    if expected is None:
        expected = (receipt.get("output_hashes") or {}).get(str(path))
    if not expected:
        raise RuntimeError(f"{basename} is not hash-bound by its stage receipt")
    verify_file(path, str(expected), basename)
    return path.resolve(), str(expected)


def verify_manifest(spec: dict[str, Any], label: str, run_dir: Path) -> Path:
    root = resolve_path(str(spec.get("path") or ""), run_dir)
    path = root if root.name == "manifest.json" else root / "manifest.json"
    verify_file(path, str(spec.get("manifest_sha256") or ""), f"{label} manifest")
    return path


def nested(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _semantic_source(source_manifest: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in source_manifest.items()
        if any(key.endswith(suffix) for suffix in SEMANTIC_SOURCE_SUFFIXES)
    }


def _qat_numerical_source(source_files: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in source_files.items()
        if not any(key.endswith(suffix) for suffix in QAT_CONTROL_SOURCE_SUFFIXES)
    }


def _base_confounds(release: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    assurance = str(release.get("lineage_assurance") or "")
    if assurance in {"hybrid_recovery", "reconstructed"}:
        out.append({"code": "lineage_" + assurance, "severity": "material"})
    quality = release.get("corpus_quality") or {}
    if quality.get("corpus_diversity_degraded") is True:
        out.append({"code": "corpus_diversity_degraded", "severity": "material"})
    if release.get("pure_single_source_exposure_comparison") is False:
        out.append({"code": "not_pure_single_source_lineage", "severity": "material"})
    numeric = release.get("numeric_integrity") or {}
    if release.get("kind") == "base-cpt-candidate" and (
        numeric.get("schema") != "mei-cpt-numeric-integrity-v1"
        or numeric.get("status") != "passed"
        or numeric.get("all_numeric_tensors_finite") is not True
        or numeric.get("parameter_names_and_order_exact") is not True
        or numeric.get("parameter_shapes_exact") is not True
        or int(numeric.get("parameter_tensor_count") or 0) != 400
        or int(numeric.get("parameter_count") or 0) != EXPECTED_PARAMS
        or int(numeric.get("train_state_tokens_seen") or 0)
        != int(release.get("tokens_seen_exposure") or 0)
    ):
        out.append(
            {"code": "base_numeric_integrity_unverified", "severity": "material"}
        )
    for blocker in release.get("release_blockers") or []:
        code = str(blocker)
        if not any(item["code"] == code for item in out):
            out.append({"code": code, "severity": "material"})
    return out


def load_product(
    run_dir: Path, downstream_run_dir: Path | None = None
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    downstream_run_dir = (
        downstream_run_dir.resolve() if downstream_run_dir is not None else run_dir
    )
    plan_path = run_dir / "plan.json"
    plan = load_json(plan_path)
    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise RuntimeError(f"unsupported productization plan: {plan.get('schema')}")
    immutable = plan.get("immutable") or {}
    base = immutable.get("base") or {}
    release_path = resolve_path(str(base.get("release") or ""), run_dir)
    weights_path = resolve_path(str(base.get("weights") or ""), run_dir)
    verify_file(release_path, str(base.get("release_sha256") or ""), "Base RELEASE")
    verify_file(weights_path, str(base.get("weights_sha256") or ""), "Base weights")
    release = load_json(release_path)
    if release.get("model_id") != base.get("base_id"):
        raise RuntimeError("Base model ID differs between plan and RELEASE")
    if int(release.get("params") or 0) != EXPECTED_PARAMS:
        raise RuntimeError("Base deployed parameter identity is not 51,463,797")
    if int(release.get("tokens_seen_exposure") or 0) != int(
        base.get("tokens_seen_exposure") or 0
    ):
        raise RuntimeError("Base exposure differs between plan and RELEASE")
    if release.get("weights_sha256") != base.get("weights_sha256"):
        raise RuntimeError("Base weights identity differs between plan and RELEASE")
    if release.get("tokenizer_sha256") != base.get("tokenizer_sha256"):
        raise RuntimeError("Base tokenizer identity differs between plan and RELEASE")
    if release.get("kind") == "base-cpt-candidate" and base.get(
        "numeric_integrity"
    ) != release.get("numeric_integrity"):
        raise RuntimeError(
            "Base numeric-integrity evidence differs between plan and RELEASE"
        )
    verify_manifest(immutable.get("data_release") or {}, "SFT data", run_dir)
    verify_manifest(
        immutable.get("linguistic_augmentation") or {}, "linguistic", run_dir
    )
    verify_manifest(immutable.get("narration_release") or {}, "narration", run_dir)
    eval_spec = immutable.get("eval_lock") or {}
    eval_dir = resolve_path(str(eval_spec.get("path") or ""), run_dir)
    eval_lock = verify_frozen_eval(eval_dir, str(eval_spec.get("lock_sha256") or ""))
    if eval_lock.get("evaluation_fingerprint") != eval_spec.get(
        "evaluation_fingerprint"
    ):
        raise RuntimeError("eval fingerprint differs between plan and lock")

    anchor_path, anchor_sha = find_bound_artifact(
        plan, run_dir, "float-base-lm-anchor.json"
    )
    control_path, control_sha = find_bound_artifact(
        plan, run_dir, "float-task-control.json"
    )
    anchor = load_json(anchor_path)
    control = load_json(control_path)
    if anchor.get("weights_sha256") != base.get("weights_sha256"):
        raise RuntimeError("Float Base-LM Anchor belongs to another Base")

    qat_spec = immutable.get("qat_import") or {}
    qat_candidate_path = resolve_path(
        str(qat_spec.get("candidate_receipt") or ""), run_dir
    )
    verify_file(
        qat_candidate_path,
        str(qat_spec.get("candidate_receipt_sha256") or ""),
        "QAT import candidate",
    )
    qat_candidate = load_json(qat_candidate_path)
    candidate_base = qat_candidate.get("base") or {}
    candidate_master = qat_candidate.get("master") or {}
    candidate_worker = qat_candidate.get("worker_receipt") or {}
    candidate_contracts = qat_candidate.get("contracts")
    if (
        qat_candidate.get("schema") != "mei-cq2-qat-import-candidate-receipt-v1"
        or qat_candidate.get("status") != "passed"
        or candidate_base.get("model_id") != base.get("base_id")
        or candidate_base.get("weights_sha256") != base.get("weights_sha256")
        or candidate_master.get("sha256") != qat_spec.get("master_sha256")
        or candidate_worker.get("sha256") != qat_spec.get("worker_receipt_sha256")
        or qat_candidate.get("quant_math_id") != qat_spec.get("quant_math_id")
        or (
            candidate_contracts is not None
            and candidate_contracts != immutable.get("contracts")
        )
    ):
        raise RuntimeError("QAT import candidate is not bound to this product/Base")
    qat_receipt_path = resolve_path(str(qat_spec.get("worker_receipt") or ""), run_dir)
    verify_file(
        qat_receipt_path,
        str(qat_spec.get("worker_receipt_sha256") or ""),
        "QAT receipt",
    )
    qat = load_json(qat_receipt_path)
    if qat.get("terminal_status") != "passed":
        raise RuntimeError("CQ2 QAT worker did not pass")
    if qat.get("parent_id") != base.get("base_id"):
        raise RuntimeError("CQ2 QAT worker belongs to another Base")
    if qat.get("contracts") != immutable.get("contracts"):
        raise RuntimeError("CQ2 QAT architecture contracts differ from product plan")
    if qat.get("quant_math_id") != qat_spec.get("quant_math_id"):
        raise RuntimeError("CQ2 QAT math differs from product plan")
    master_path = resolve_path(str(qat_spec.get("master") or ""), run_dir)
    verify_file(master_path, str(qat_spec.get("master_sha256") or ""), "QAT master")
    qat_plan_path = qat_receipt_path.with_name("plan.json")
    qat_plan = load_json(qat_plan_path)
    if qat_plan.get("stage_fingerprint_sha256") != qat.get("stage_fingerprint_sha256"):
        raise RuntimeError("QAT plan/receipt fingerprint mismatch")

    locked_dir = run_dir / "stages/locked_test_eval_v4"
    locked_path = locked_dir / "locked-test-eval-v4.json"
    scorecard_path = locked_dir / "longitudinal-scorecard.test.json"
    locked_receipt_path = locked_dir / "receipt.json"
    locked_receipt = load_json(locked_receipt_path)
    locked_status = locked_receipt.get("terminal_status") or locked_receipt.get(
        "status"
    )
    if locked_status not in {"passed", "degraded"}:
        raise RuntimeError("locked test evaluation is not terminal")
    locked_outputs = verify_output_hashes(locked_receipt, run_dir, "locked test")
    for path in (locked_path, scorecard_path):
        expected = locked_outputs.get(str(path.resolve()))
        if not expected:
            raise RuntimeError(f"locked test artifact is not hash-bound: {path}")
    locked = load_json(locked_path)
    scorecard = load_json(scorecard_path)
    for artifact in (locked, scorecard):
        if artifact.get("evaluation_fingerprint") != eval_spec.get(
            "evaluation_fingerprint"
        ):
            raise RuntimeError("locked result used another evaluation fingerprint")

    package_dir = run_dir / "stages/package_v2_cq2_v4"
    package_path = package_dir / "package-v2-cq2-v4.json"
    package_receipt_path = package_dir / "receipt.json"
    package_receipt = load_json(package_receipt_path)
    package_outputs = verify_output_hashes(package_receipt, run_dir, "package")
    package_expected = package_outputs.get(str(package_path.resolve()))
    if not package_expected:
        raise RuntimeError("package report is not hash-bound")
    package = load_json(package_path)
    if package.get("lm_tensor_count") != 400 or package.get("mtp_tensor_count") != 0:
        raise RuntimeError("package changed the deployed 51M LM/MTP tensor identity")

    package_verification_path = (
        downstream_run_dir / "downstream/package-verification.json"
    )
    package_verification = load_json(package_verification_path)
    source_binding = package_verification.get("source_binding") or {}
    if (
        package_verification.get("schema")
        != "mei-productization-downstream-package-receipt-v1"
        or package_verification.get("terminal_status") != "passed"
        or package_verification.get("package_id") != package.get("package_id")
        or package_verification.get("source_run_fingerprint_sha256")
        != plan.get("run_fingerprint_sha256")
    ):
        raise RuntimeError("downstream package verification belongs to another product")
    verify_output_hashes(
        package_verification, downstream_run_dir, "downstream package verification"
    )
    if downstream_run_dir != run_dir and (
        package_verification.get("current_source_reevaluation") is not True
        or package_verification.get("release_claim_from_reevaluation") is not False
        or source_binding.get("current_source_reevaluation") is not True
    ):
        raise RuntimeError(
            "separate downstream run is not an explicit current-source reevaluation"
        )

    resource_dir = downstream_run_dir / "downstream/resources"
    resource_path = resource_dir / "resource-measurement.json"
    resource_receipt_path = resource_dir / "receipt.json"
    resource_receipt = load_json(resource_receipt_path)
    resource_outputs = verify_output_hashes(resource_receipt, run_dir, "resource")
    resource_expected = resource_outputs.get(str(resource_path.resolve()))
    if not resource_expected:
        raise RuntimeError("resource report is not hash-bound")
    resources = load_json(resource_path)

    current_path = ROOT / "CURRENT.json"
    current_sha256 = sha256_file(current_path)
    expected_current = str(immutable.get("current_baseline_sha256") or "")
    if (
        not expected_current
        or current_sha256 != expected_current
        or resource_receipt.get("current_unchanged") is not True
        or resource_receipt.get("current_sha256") != expected_current
    ):
        raise RuntimeError("CURRENT.json changed across productization")
    verify_file(
        package_verification_path,
        str(resource_receipt.get("package_receipt_sha256") or ""),
        "downstream package verification receipt",
    )
    portable_receipt_path = downstream_run_dir / "downstream/portable-gates/receipt.json"
    verify_file(
        portable_receipt_path,
        str(resource_receipt.get("portable_receipt_sha256") or ""),
        "portable gate receipt",
    )
    portable_receipt = load_json(portable_receipt_path)
    if portable_receipt.get("terminal_status") != "passed":
        raise RuntimeError("portable runtime gates are not passed")

    source_manifest = immutable.get("source_manifest") or {}
    runtime_source_manifest = package_verification.get("source_hashes") or {}
    if not runtime_source_manifest:
        raise RuntimeError("current runtime source manifest is missing")
    process_complete = (
        resource_receipt.get("process_complete") is True
        and resource_receipt.get("terminal_status") == "passed"
        and package_receipt.get("terminal_status") == "passed"
        and locked_status in {"passed", "degraded"}
    )
    return {
        "run_dir": str(run_dir),
        "downstream_run_dir": str(downstream_run_dir),
        "plan_sha256": sha256_file(plan_path),
        "run_fingerprint_sha256": plan.get("run_fingerprint_sha256"),
        "base": {
            "id": base.get("base_id"),
            "exposure_tokens": int(base.get("tokens_seen_exposure") or 0),
            "release_sha256": base.get("release_sha256"),
            "weights_sha256": base.get("weights_sha256"),
            "lineage_assurance": release.get("lineage_assurance")
            or "legacy_frozen_base",
            "corpus_quality": release.get("corpus_quality"),
            "release_eligible": release.get("release_eligible"),
            "release_blockers": release.get("release_blockers") or [],
            "numeric_integrity": release.get("numeric_integrity"),
        },
        "contracts": immutable.get("contracts") or {},
        "tokenizer_sha256": immutable.get("tokenizer_sha256")
        or base.get("tokenizer_sha256"),
        "data_release": immutable.get("data_release") or {},
        "linguistic_augmentation": immutable.get("linguistic_augmentation") or {},
        "narration_release": immutable.get("narration_release") or {},
        "eval_lock": eval_spec,
        "recipe": immutable.get("recipe") or {},
        "validation_scope": immutable.get("validation_scope") or {},
        "source_manifest": source_manifest,
        "semantic_source_manifest": _semantic_source(source_manifest),
        "runtime_source_manifest": runtime_source_manifest,
        "runtime_source_reevaluation": package_verification.get(
            "current_source_reevaluation"
        )
        is True,
        "historical_source_drift": source_binding.get("source_drift") or [],
        "qat_recipe": qat_plan.get("recipe") or {},
        "qat_corpus_files": qat_plan.get("corpus_files") or {},
        "qat_group_policy": qat_plan.get("group_policy") or {},
        "qat_source_files": qat_plan.get("source_files") or {},
        "qat_numerical_source_files": _qat_numerical_source(
            qat_plan.get("source_files") or {}
        ),
        "quant_math_id": qat.get("quant_math_id"),
        "base_anchor": anchor,
        "float_task_control": control,
        "qat": qat,
        "locked_eval": locked,
        "scorecard": scorecard,
        "package": package,
        "resources": resources,
        "gates": locked.get("gates") or {},
        "process_complete": process_complete,
        "release_eligible": resource_receipt.get("release_eligible") is True,
        "artifact_hashes": {
            "float_anchor": anchor_sha,
            "float_task_control": control_sha,
            "qat_import_candidate": qat_spec.get("candidate_receipt_sha256"),
            "qat_receipt": qat_spec.get("worker_receipt_sha256"),
            "qat_master": qat_spec.get("master_sha256"),
            "locked_eval": sha256_file(locked_path),
            "scorecard": sha256_file(scorecard_path),
            "package": str(package_expected),
            "resources": str(resource_expected),
            "current": current_sha256,
            "portable_receipt": resource_receipt.get("portable_receipt_sha256"),
            "downstream_package_receipt": sha256_file(package_verification_path),
        },
        "confounds": _base_confounds(release)
        + (
            [
                {
                    "code": "historical_training_source_drift_recorded_for_reevaluation",
                    "severity": "provenance",
                }
            ]
            if source_binding.get("source_drift")
            else []
        ),
    }


def _frozen_contract(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "contracts": record["contracts"],
        "tokenizer_sha256": record["tokenizer_sha256"],
        "data_manifest_sha256": record["data_release"].get("manifest_sha256"),
        "data_release_fingerprint": record["data_release"].get("release_fingerprint"),
        "linguistic_manifest_sha256": record["linguistic_augmentation"].get(
            "manifest_sha256"
        ),
        "linguistic_release_fingerprint": record["linguistic_augmentation"].get(
            "release_fingerprint"
        ),
        "narration_manifest_sha256": record["narration_release"].get("manifest_sha256"),
        "eval_fingerprint": record["eval_lock"].get("evaluation_fingerprint"),
        "eval_lock_sha256": record["eval_lock"].get("lock_sha256"),
        "recipe": record["recipe"],
        "validation_scope": record["validation_scope"],
        "quant_math_id": record["quant_math_id"],
        "qat_recipe": record["qat_recipe"],
        "qat_corpus_files": record["qat_corpus_files"],
        "qat_group_policy": record["qat_group_policy"],
    }


def metric_rows(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact, path, direction in METRICS:
        left = nested(baseline.get(artifact), path)
        right = nested(candidate.get(artifact), path)
        if not isinstance(left, (int, float)) or isinstance(left, bool):
            continue
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            continue
        left_float = float(left)
        right_float = float(right)
        if not (math.isfinite(left_float) and math.isfinite(right_float)):
            continue
        delta = right_float - left_float
        relative = None if left_float == 0 else delta / abs(left_float)
        if abs(delta) <= 1e-12:
            outcome = "equal"
        elif direction == "higher":
            outcome = "improved" if delta > 0 else "regressed"
        elif direction == "lower":
            outcome = "improved" if delta < 0 else "regressed"
        else:
            outcome = "changed"
        rows.append(
            {
                "metric": f"{artifact}.{path}",
                "direction": direction,
                "baseline": left,
                "candidate": right,
                "candidate_minus_baseline": delta,
                "relative_delta": relative,
                "outcome": outcome,
            }
        )
    return rows


def _gate_rows(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    names = sorted(set(baseline.get("gates") or {}) | set(candidate.get("gates") or {}))
    return [
        {
            "gate": name,
            "baseline_passed": nested(baseline, f"gates.{name}.passed"),
            "candidate_passed": nested(candidate, f"gates.{name}.passed"),
        }
        for name in names
    ]


def build_comparison(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    left_contract = _frozen_contract(baseline)
    right_contract = _frozen_contract(candidate)
    comparisons = {
        key: {
            "baseline": left_contract[key],
            "candidate": right_contract[key],
            "exact": left_contract[key] == right_contract[key],
        }
        for key in left_contract
    }
    frozen_exact = all(item["exact"] for item in comparisons.values())
    source_exact = baseline.get("source_manifest") == candidate.get("source_manifest")
    semantic_source_exact = bool(
        baseline.get("semantic_source_manifest")
    ) and baseline.get("semantic_source_manifest") == candidate.get(
        "semantic_source_manifest"
    )
    qat_source_exact = baseline.get("qat_source_files") == candidate.get(
        "qat_source_files"
    )
    qat_numerical_source_exact = bool(
        baseline.get("qat_numerical_source_files")
    ) and baseline.get("qat_numerical_source_files") == candidate.get(
        "qat_numerical_source_files"
    )
    runtime_source_exact = bool(
        baseline.get("runtime_source_manifest")
    ) and baseline.get("runtime_source_manifest") == candidate.get(
        "runtime_source_manifest"
    )
    distinct_weights = (
        baseline["base"]["weights_sha256"] != candidate["base"]["weights_sha256"]
    )
    ordered_exposure = (
        baseline["base"]["exposure_tokens"] < candidate["base"]["exposure_tokens"]
    )
    confounds = [
        {"side": "baseline", **item} for item in baseline.get("confounds") or []
    ] + [{"side": "candidate", **item} for item in candidate.get("confounds") or []]
    if not source_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "source_manifest_not_exact",
                "severity": "provenance",
            }
        )
    if not semantic_source_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "semantic_source_not_exact",
                "severity": "material",
            }
        )
    if not qat_source_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "qat_source_files_not_exact",
                "severity": "provenance",
            }
        )
    if not qat_numerical_source_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "qat_numerical_source_not_exact",
                "severity": "material",
            }
        )
    if not runtime_source_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "runtime_evaluation_source_not_exact",
                "severity": "material",
            }
        )
    if not frozen_exact:
        confounds.append(
            {
                "side": "pair",
                "code": "frozen_productization_contract_drift",
                "severity": "material",
            }
        )
    if not distinct_weights or not ordered_exposure:
        confounds.append(
            {
                "side": "pair",
                "code": "base_pair_identity_or_order_invalid",
                "severity": "material",
            }
        )
    pair_complete = (
        baseline.get("process_complete") is True
        and candidate.get("process_complete") is True
    )
    has_material = any(item.get("severity") == "material" for item in confounds)
    main_curve = (
        pair_complete
        and frozen_exact
        and semantic_source_exact
        and qat_numerical_source_exact
        and runtime_source_exact
        and distinct_weights
        and ordered_exposure
        and not has_material
    )
    causal = main_curve and source_exact and qat_source_exact
    return {
        "schema": "mei-51m-longitudinal-paired-comparison-v1",
        "product": "mei-1.0-51m",
        "comparison_policy": {
            "aggregate_score": False,
            "metric_directions_preregistered": True,
            "quality_resource_and_lineage_axes_separate": True,
        },
        "paired_experiment_complete": pair_complete,
        "main_scale_curve_eligible": main_curve,
        "causal_exposure_attribution_eligible": causal,
        "comparability": {
            "frozen_productization_contract_exact": frozen_exact,
            "source_manifest_exact": source_exact,
            "semantic_source_manifest_exact": semantic_source_exact,
            "qat_source_files_exact": qat_source_exact,
            "qat_numerical_source_files_exact": qat_numerical_source_exact,
            "runtime_evaluation_source_exact": runtime_source_exact,
            "distinct_base_weights": distinct_weights,
            "candidate_exposure_greater": ordered_exposure,
            "fields": comparisons,
        },
        "baseline": baseline,
        "candidate": candidate,
        "confounds": confounds,
        "metric_deltas": metric_rows(baseline, candidate),
        "gate_transitions": _gate_rows(baseline, candidate),
    }


def write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"refusing to overwrite different comparison report: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-downstream-run", type=Path)
    parser.add_argument("--candidate-downstream-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_product(args.baseline_run, args.baseline_downstream_run)
    candidate = load_product(args.candidate_run, args.candidate_downstream_run)
    report = build_comparison(baseline, candidate)
    write_once(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
