#!/usr/bin/env python3
"""Receipt-driven SFT-v4 productization runner, defaulting to the 300M Base.

The runner refuses to claim Metal while a CPT worker is live.  It imports the
explicitly selected Base's verified CQ2-QAT master, then executes the frozen
v4 task chain.  It never mutates Base releases, CURRENT.json, or historical
runs.  Exposure labels are metadata, never stage-graph selectors.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import longitudinal_eval_metrics_51m as longitudinal_metrics
import productize_51m as lifecycle
import sft_v4_contract_51m as contract
import sft_v3_eval_51m as evaluation
import sft_v3_training_51m as training
from _repo import CURRENT_PATH, ROOT, TOKENIZER_ZH_V1, architecture_contracts


RUNNER_ID = "mei-51m-sft-v4-productizer-v1-quality-schema"
DEFAULT_BASE_DIR = ROOT / "base/mei-1.0-51m-base-scratch300m-v1"
DEFAULT_BASE_RELEASE = DEFAULT_BASE_DIR / "RELEASE.json"
DEFAULT_BASE_WEIGHTS = DEFAULT_BASE_DIR / "mei-1.0-51m-base-scratch300m-v1.npz"
DEFAULT_DATA_RELEASE = (
    contract.DEFAULT_RELEASE_ROOT / contract.RELEASE_ID
)
DEFAULT_LINGUISTIC_AUGMENTATION = (
    contract.DEFAULT_RELEASE_ROOT / contract.LINGUISTIC_AUGMENTATION_ID
)
DEFAULT_EVAL_LOCK = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID
DEFAULT_QAT_IMPORT_RECEIPT = (
    DEFAULT_DATA_RELEASE / "qat-import-candidate-receipt.json"
)
DEFAULT_NARRATION_RELEASE = contract.NARRATION_RELEASE_DIR
DEFAULT_RUN_DIR = (
    ROOT
    / "training/runs/mei-1.0-51m/productize-scratch300m-sft-v4-quality-schema-cq2-v2"
)
DEFAULT_PACKAGE_ID = "mei-1.0-51m-scratch300m-tool-sft-cq2-v2-sftv4-quality-schema"

STAGES = (
    "float_base_lm_anchor",
    "float_task_control_v4",
    "cq2_qat_import",
    "retrieval_r0_v4",
    "oracle_top5_fullcall_v4",
    "oracle_top5_agent_v4",
    "retrieval_r1_v4",
    "tool_index_v4",
    "single_step_eval_v4",
    "multistep_eval_v4",
    "crossgen_schema_eval_v4",
    "mw_disposition_v4",
    "confidence_harvest_v4",
    "confidence_head_v4",
    "narration_adapter_v4",
    "sidecar_eval_v4",
    "locked_test_eval_v4",
    "package_v2_cq2_v4",
)
TRAINING_PREFIX_STAGES = STAGES[:8]
TRAINING_PREFIX_ADOPTION_STAGE = "adopt_training_prefix_v4"
DOWNSTREAM_STAGES = STAGES[8:]
PRODUCTIZATION_PREFIX_ADOPTION_STAGE = "adopt_productization_prefix_v4"
PRODUCTIZATION_PREFIX_STAGES = (
    TRAINING_PREFIX_ADOPTION_STAGE,
    *STAGES[8:15],
)
FINALIZATION_STAGES = STAGES[15:]


def _verify_training_prefix_run(
    prefix_run: Path,
    expected_immutable: dict[str, Any],
) -> dict[str, Any]:
    """Verify and describe a completed immutable LM/R1/index prefix.

    The parent run remains untouched and its sources are not relabelled as the
    current sources.  The returned evidence becomes an immutable input to a new
    downstream run, which is the only safe way to continue after evaluator
    source changes without replaying already-completed model training.
    """

    prefix_run = prefix_run.resolve()
    plan_path = prefix_run / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError("training-prefix adoption lacks a parent plan")
    parent = contract.load_json(plan_path)
    parent_immutable = parent.get("immutable") or {}
    if (
        parent.get("schema") != "mei-51m-sft-v4-productization-plan-v1"
        or parent.get("run_fingerprint_sha256")
        != contract.sha_bytes(contract.canonical_bytes(parent_immutable))
    ):
        raise RuntimeError("training-prefix parent plan identity is invalid")

    shared_fields = (
        "base",
        "data_release",
        "linguistic_augmentation",
        "eval_lock",
        "narration_release",
        "qat_import",
        "contracts",
        "tokenizer_sha256",
        "current_baseline_sha256",
        "validation_scope",
    )
    for field in shared_fields:
        if parent_immutable.get(field) != expected_immutable.get(field):
            raise RuntimeError(f"training-prefix immutable input differs: {field}")
    prefix_recipe_fields = (
        "float_control_steps",
        "float_control_sampler",
        "retrieval_r0_steps",
        "retrieval_sampler",
        "fullcall_steps",
        "fullcall_sampler",
        "linguistic_source_balanced",
        "whole_schema_holdout",
        "agent_steps",
        "retrieval_r1_steps",
        "seed",
    )
    parent_recipe = parent_immutable.get("recipe") or {}
    expected_recipe = expected_immutable.get("recipe") or {}
    for field in prefix_recipe_fields:
        if parent_recipe.get(field) != expected_recipe.get(field):
            raise RuntimeError(f"training-prefix recipe differs: {field}")

    parent_stages = parent.get("stages") or []
    if [row.get("stage_id") for row in parent_stages[:8]] != list(
        TRAINING_PREFIX_STAGES
    ):
        raise RuntimeError("training-prefix parent stage order is invalid")
    preflight = parent.get("execution_preflight") or {}
    preflight_path = Path(str(preflight.get("path") or ""))
    if (
        preflight.get("status") != "passed"
        or not preflight_path.is_file()
        or contract.sha_file(preflight_path) != preflight.get("sha256")
    ):
        raise RuntimeError("training-prefix parent preflight is unavailable or drifted")
    preflight_receipt = contract.load_json(preflight_path)
    if (
        preflight_receipt.get("status") != "passed"
        or preflight_receipt.get("run_fingerprint_sha256")
        != parent.get("run_fingerprint_sha256")
    ):
        raise RuntimeError("training-prefix preflight is not bound to its parent run")

    stages: dict[str, Any] = {}
    output_union: dict[str, str] = {}
    stage_rows = lifecycle.stage_map(parent)
    for stage_id in TRAINING_PREFIX_STAGES:
        receipt_path = prefix_run / "stages" / stage_id / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"training-prefix stage receipt missing: {stage_id}")
        receipt = contract.load_json(receipt_path)
        expected_stage = stage_rows[stage_id]
        input_fingerprint, _input_evidence = lifecycle._stage_input_fingerprint(
            prefix_run, parent, stage_id
        )
        outputs = receipt.get("output_hashes") or {}
        if (
            receipt.get("terminal_status") not in {"passed", "degraded"}
            or receipt.get("run_fingerprint_sha256")
            != parent.get("run_fingerprint_sha256")
            or receipt.get("stage_fingerprint_sha256")
            != expected_stage.get("stage_fingerprint_sha256")
            or receipt.get("stage_input_fingerprint_sha256") != input_fingerprint
            or not outputs
            or any(
                not Path(path).is_file()
                or contract.sha_file(Path(path)) != digest
                for path, digest in outputs.items()
            )
        ):
            raise RuntimeError(f"training-prefix stage is not reusable: {stage_id}")
        stages[stage_id] = {
            "receipt": str(receipt_path),
            "receipt_sha256": contract.sha_file(receipt_path),
            "terminal_status": receipt["terminal_status"],
            "stage_fingerprint_sha256": receipt["stage_fingerprint_sha256"],
            "stage_input_fingerprint_sha256": input_fingerprint,
            "output_hashes": outputs,
        }
        output_union.update({str(Path(path).resolve()): digest for path, digest in outputs.items()})

    parent_paths = _stage_paths(prefix_run)
    artifacts = {
        "final_lm_master": parent_paths["agent"],
        "retrieval_r1_head": parent_paths["r1"],
        "tool_index": parent_paths["index"],
    }
    artifact_evidence: dict[str, Any] = {}
    for name, path in artifacts.items():
        resolved = str(path.resolve())
        if (
            not path.is_file()
            or resolved not in output_union
            or contract.sha_file(path) != output_union[resolved]
        ):
            raise RuntimeError(f"training-prefix required artifact is unbound: {name}")
        artifact_evidence[name] = {
            "path": resolved,
            "sha256": output_union[resolved],
        }

    body = {
        "schema": "mei-sft-v4-training-prefix-adoption-input-v1",
        "status": "verified",
        "parent_run_dir": str(prefix_run),
        "parent_plan": str(plan_path),
        "parent_plan_sha256": contract.sha_file(plan_path),
        "parent_run_fingerprint_sha256": parent["run_fingerprint_sha256"],
        "parent_source_manifest": parent_immutable.get("source_manifest") or {},
        "parent_preflight": {
            "path": str(preflight_path),
            "sha256": preflight["sha256"],
            "preflight_fingerprint_sha256": preflight.get(
                "preflight_fingerprint_sha256"
            ),
        },
        "stage_ids": list(TRAINING_PREFIX_STAGES),
        "stages": stages,
        "artifacts": artifact_evidence,
        "source_transition": "parent-training-sources-to-current-downstream-sources",
    }
    return {
        **body,
        "adoption_input_fingerprint_sha256": contract.sha_bytes(
            contract.canonical_bytes(body)
        ),
    }


def _verify_productization_prefix_run(
    prefix_run: Path,
    expected_immutable: dict[str, Any],
) -> dict[str, Any]:
    """Verify a run through narration without relabelling its old sources.

    This is deliberately stricter than copying checkpoints.  It validates the
    parent plan, preflight, recursive training-prefix adoption, every stage
    input fingerprint and every output hash through ``narration_adapter_v4``.
    A new run may then execute only sidecar eval, locked test and packaging.
    """

    prefix_run = prefix_run.resolve()
    plan_path = prefix_run / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError("productization-prefix adoption lacks a parent plan")
    parent = contract.load_json(plan_path)
    parent_immutable = parent.get("immutable") or {}
    if (
        parent.get("schema") != "mei-51m-sft-v4-productization-plan-v1"
        or parent.get("run_fingerprint_sha256")
        != contract.sha_bytes(contract.canonical_bytes(parent_immutable))
    ):
        raise RuntimeError("productization-prefix parent plan identity is invalid")

    shared_fields = (
        "base",
        "data_release",
        "linguistic_augmentation",
        "eval_lock",
        "narration_release",
        "qat_import",
        "contracts",
        "tokenizer_sha256",
        "current_baseline_sha256",
        "validation_scope",
        "recipe",
    )
    for field in shared_fields:
        if parent_immutable.get(field) != expected_immutable.get(field):
            raise RuntimeError(f"productization-prefix immutable input differs: {field}")

    parent_stage_ids = [
        str(row.get("stage_id") or "") for row in parent.get("stages") or []
    ]
    full_prefix = tuple(STAGES[:15])
    nested_training: dict[str, Any] | None = None
    if tuple(parent_stage_ids[: len(full_prefix)]) == full_prefix:
        active_prefix = full_prefix
    elif tuple(parent_stage_ids[: len(PRODUCTIZATION_PREFIX_STAGES)]) == (
        PRODUCTIZATION_PREFIX_STAGES
    ):
        active_prefix = PRODUCTIZATION_PREFIX_STAGES
        nested_training = parent_immutable.get("training_prefix_adoption")
        if not isinstance(nested_training, dict):
            raise RuntimeError(
                "productization-prefix parent lacks its training-prefix evidence"
            )
        observed_training = _verify_training_prefix_run(
            Path(str(nested_training.get("parent_run_dir") or "")),
            expected_immutable,
        )
        if contract.canonical_bytes(observed_training) != contract.canonical_bytes(
            nested_training
        ):
            raise RuntimeError("nested training-prefix adoption evidence drifted")
    else:
        raise RuntimeError("productization-prefix parent stage order is invalid")

    preflight = parent.get("execution_preflight") or {}
    preflight_path = Path(str(preflight.get("path") or ""))
    if (
        preflight.get("status") != "passed"
        or not preflight_path.is_file()
        or contract.sha_file(preflight_path) != preflight.get("sha256")
    ):
        raise RuntimeError(
            "productization-prefix parent preflight is unavailable or drifted"
        )
    preflight_receipt = contract.load_json(preflight_path)
    if (
        preflight_receipt.get("status") != "passed"
        or preflight_receipt.get("run_fingerprint_sha256")
        != parent.get("run_fingerprint_sha256")
    ):
        raise RuntimeError(
            "productization-prefix preflight is not bound to its parent run"
        )

    stages: dict[str, Any] = {}
    component_stages: dict[str, Any] = dict(
        (nested_training or {}).get("stages") or {}
    )
    output_union: dict[str, str] = {}
    stage_rows = lifecycle.stage_map(parent)
    for stage_id in active_prefix:
        receipt_path = prefix_run / "stages" / stage_id / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(
                f"productization-prefix stage receipt missing: {stage_id}"
            )
        receipt = contract.load_json(receipt_path)
        expected_stage = stage_rows[stage_id]
        input_fingerprint, _input_evidence = lifecycle._stage_input_fingerprint(
            prefix_run, parent, stage_id
        )
        outputs = receipt.get("output_hashes") or {}
        if (
            receipt.get("terminal_status") not in {"passed", "degraded"}
            or receipt.get("run_fingerprint_sha256")
            != parent.get("run_fingerprint_sha256")
            or receipt.get("stage_fingerprint_sha256")
            != expected_stage.get("stage_fingerprint_sha256")
            or receipt.get("stage_input_fingerprint_sha256") != input_fingerprint
            or not outputs
            or any(
                not Path(path).is_file()
                or contract.sha_file(Path(path)) != digest
                for path, digest in outputs.items()
            )
        ):
            raise RuntimeError(
                f"productization-prefix stage is not reusable: {stage_id}"
            )
        evidence = {
            "receipt": str(receipt_path),
            "receipt_sha256": contract.sha_file(receipt_path),
            "terminal_status": receipt["terminal_status"],
            "stage_fingerprint_sha256": receipt["stage_fingerprint_sha256"],
            "stage_input_fingerprint_sha256": input_fingerprint,
            "output_hashes": outputs,
        }
        stages[stage_id] = evidence
        if stage_id != TRAINING_PREFIX_ADOPTION_STAGE:
            component_stages[stage_id] = evidence
        output_union.update(
            {str(Path(path).resolve()): digest for path, digest in outputs.items()}
        )

    training_prefix_run = (
        Path(str(nested_training["parent_run_dir"]))
        if nested_training is not None
        else None
    )
    parent_paths = _stage_paths(prefix_run, training_prefix_run)
    required_paths = {
        "agent": parent_paths["agent"],
        "r1": parent_paths["r1"],
        "index": parent_paths["index"],
        "mw": parent_paths["mw"],
        "confidence": parent_paths["confidence"],
        "calibration": parent_paths["calibration"],
        "narration": parent_paths["narration"],
        "confidence_train_outcomes": prefix_run
        / "stages/confidence_harvest_v4/train-outcomes.jsonl",
        "confidence_valid_outcomes": prefix_run
        / "stages/confidence_harvest_v4/valid-outcomes.jsonl",
        "confidence_dev_outcomes": prefix_run
        / "stages/confidence_harvest_v4/dev-outcomes.jsonl",
    }
    artifact_evidence: dict[str, Any] = {}
    for name, path in required_paths.items():
        resolved = str(path.resolve())
        if (
            not path.is_file()
            or resolved not in output_union
            or contract.sha_file(path) != output_union[resolved]
        ):
            raise RuntimeError(
                f"productization-prefix required artifact is unbound: {name}"
            )
        artifact_evidence[name] = {
            "path": resolved,
            "sha256": output_union[resolved],
        }

    body = {
        "schema": "mei-sft-v4-productization-prefix-adoption-input-v1",
        "status": "verified",
        "parent_run_dir": str(prefix_run),
        "parent_plan": str(plan_path),
        "parent_plan_sha256": contract.sha_file(plan_path),
        "parent_run_fingerprint_sha256": parent["run_fingerprint_sha256"],
        "parent_source_manifest": parent_immutable.get("source_manifest") or {},
        "parent_preflight": {
            "path": str(preflight_path),
            "sha256": preflight["sha256"],
            "preflight_fingerprint_sha256": preflight.get(
                "preflight_fingerprint_sha256"
            ),
        },
        "stage_ids": list(active_prefix),
        "stages": stages,
        "component_stages": component_stages,
        "artifacts": artifact_evidence,
        "source_transition": (
            "parent-productization-sources-to-current-finalization-sources"
        ),
    }
    return {
        **body,
        "adoption_input_fingerprint_sha256": contract.sha_bytes(
            contract.canonical_bytes(body)
        ),
    }


def _source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("sft_v4_contract_51m.py"),
        Path(__file__).with_name("sft_v3_training_51m.py"),
        Path(__file__).with_name("sft_v3_eval_51m.py"),
        Path(__file__).with_name("freeze_sft_linguistic_augmentation_v2_51m.py"),
        Path(__file__).with_name("freeze_sft_v4_release_51m.py"),
        Path(__file__).with_name("freeze_longitudinal_eval_v7_51m.py"),
        Path(__file__).with_name("longitudinal_eval_metrics_51m.py"),
        Path(__file__).with_name("preflight_sft_v3_51m.py"),
        Path(__file__).with_name("productize_51m.py"),
        Path(__file__).with_name("cq2_qat_51m.py"),
        Path(__file__).with_name("pack_cq2_v2_51m.py"),
        ROOT / "architecture/mei-1.0-51m-arch-v1/heads.py",
        ROOT / "sdk/python/mei_sdk/runtime_51m.py",
        ROOT / "runtime/_shared/kv_manager.py",
        ROOT / "runtime/_shared/tool_index.py",
    ]
    return {
        str(path.relative_to(ROOT)): contract.sha_file(path)
        for path in paths
    }


def _validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    base = contract.load_json(args.base_release)
    base_identity = lifecycle.validate_base(args.base_release, args.base_weights)
    if int(base_identity.get("tokens_seen_exposure") or 0) <= 0:
        raise RuntimeError("selected Base lacks a positive cumulative exposure")
    data = training.verify_release_contract(args.data_release)
    lock = longitudinal_metrics.verify_lock(args.eval_lock)
    if (
        data.get("schema") != "mei-sft-data-release-v4"
        or data.get("release_id") != contract.RELEASE_ID
        or lock.get("schema") != "mei-51m-longitudinal-eval-lock-v4"
        or lock.get("id") != contract.EVAL_ID
    ):
        raise RuntimeError("productizer requires SFT-v4-v3 and eval-v7")
    if (data.get("evaluation") or {}).get("evaluation_fingerprint") != lock.get(
        "evaluation_fingerprint"
    ):
        raise RuntimeError("SFT release and eval-v7 fingerprint differ")
    linguistic = contract.load_json(args.linguistic_augmentation / "manifest.json")
    if (
        linguistic.get("schema") != "mei-sft-linguistic-augmentation-v2"
        or linguistic.get("status") != "frozen"
        or linguistic.get("release_id") != contract.LINGUISTIC_AUGMENTATION_ID
        or (linguistic.get("evaluation") or {}).get("id") != lock.get("id")
        or ((data.get("sources") or {}).get("linguistic_release_manifest") or {}).get(
            "sha256"
        )
        != contract.sha_file(args.linguistic_augmentation / "manifest.json")
    ):
        raise RuntimeError("linguistic augmentation identity or parent drifted")
    for name, spec in (linguistic.get("artifacts") or {}).items():
        path = args.linguistic_augmentation / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"linguistic augmentation artifact drifted: {path}")
        if name.endswith(".jsonl") and len(contract.load_jsonl(path)) != int(
            spec.get("rows") or -1
        ):
            raise RuntimeError(f"linguistic augmentation row count drifted: {path}")
    linguistic_quality = contract.load_json(
        args.linguistic_augmentation / "quality-audit-receipt.json"
    )
    linguistic_budget = contract.load_json(
        args.linguistic_augmentation / "token-budget-receipt.json"
    )
    if (
        linguistic_quality.get("status") != "passed"
        or linguistic_budget.get("status") != "passed"
        or linguistic_quality.get("train_valid_query_overlap") != 0
        or linguistic_quality.get("eval_query_overlap") != 0
    ):
        raise RuntimeError("linguistic augmentation quality/isolation did not pass")
    qat = contract.load_json(args.qat_import_receipt)
    master = ROOT / str((qat.get("master") or {}).get("path") or "")
    worker_receipt = ROOT / str((qat.get("worker_receipt") or {}).get("path") or "")
    qat_base = qat.get("base") or {}
    if (
        qat.get("schema") != "mei-cq2-qat-import-candidate-receipt-v1"
        or qat.get("status") != "passed"
        or qat_base.get("model_id") != base_identity["base_id"]
        or int(qat_base.get("tokens_seen_exposure") or 0)
        != base_identity["tokens_seen_exposure"]
        or qat_base.get("weights_sha256") != base_identity["weights_sha256"]
        or not master.is_file()
        or contract.sha_file(master) != (qat.get("master") or {}).get("sha256")
        or not worker_receipt.is_file()
        or contract.sha_file(worker_receipt)
        != (qat.get("worker_receipt") or {}).get("sha256")
    ):
        raise RuntimeError("CQ2-QAT import candidate drifted or belongs to another Base")
    worker = contract.load_json(worker_receipt)
    contracts = architecture_contracts()
    expected_contracts = {
        name: str(contracts[name])
        for name in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    if (
        worker.get("terminal_status") != "passed"
        or worker.get("parent_id") != base_identity["base_id"]
        or worker.get("quant_math_id") != qat.get("quant_math_id")
        or worker.get("contracts") != expected_contracts
        or (worker.get("outputs") or {}).get("master_sha256")
        != (qat.get("master") or {}).get("sha256")
    ):
        raise RuntimeError("CQ2-QAT worker receipt is not bound to the selected Base")
    narration = contract.load_json(args.narration_release / "manifest.json")
    if (
        narration.get("schema") != "mei-narration-sft-data-release-v2"
        or narration.get("release_id")
        != (data.get("narration_source") or {}).get("release_id")
    ):
        raise RuntimeError("narration release identity drifted")
    for name, spec in (narration.get("files") or {}).items():
        path = args.narration_release / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"narration release artifact drifted: {path}")
    narration_isolation = contract.load_json(
        args.narration_release / "isolation-receipt.json"
    )
    if narration_isolation.get("status") != "passed":
        raise RuntimeError("narration isolation gate did not pass")
    return {
        "base": base,
        "base_identity": base_identity,
        "data": data,
        "linguistic": linguistic,
        "linguistic_quality": linguistic_quality,
        "linguistic_budget": linguistic_budget,
        "eval": lock,
        "qat": qat,
        "qat_master": master,
        "qat_worker_receipt": worker_receipt,
        "qat_worker": worker,
        "narration": narration,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _validate_inputs(args)
    live = lifecycle.live_cpt_workers()
    contracts_raw = architecture_contracts()
    base_identity = inputs["base_identity"]
    immutable = {
        "runner_id": RUNNER_ID,
        "product": contract.PRODUCT_ID,
        "package_id": args.package_id,
        "base": {
            "base_id": base_identity["base_id"],
            "weights": base_identity["weights"],
            "weights_sha256": base_identity["weights_sha256"],
            "release": base_identity["release"],
            "release_sha256": base_identity["release_sha256"],
            "tokens_seen_exposure": base_identity["tokens_seen_exposure"],
            "tokenizer_sha256": base_identity["tokenizer_sha256"],
        },
        "data_release": {
            "path": str(args.data_release.resolve()),
            "release_id": inputs["data"]["release_id"],
            "release_fingerprint": inputs["data"]["release_fingerprint"],
            "manifest_sha256": contract.sha_file(args.data_release / "manifest.json"),
        },
        "linguistic_augmentation": {
            "path": str(args.linguistic_augmentation.resolve()),
            "release_id": inputs["linguistic"]["release_id"],
            "release_fingerprint": inputs["linguistic"]["release_fingerprint"],
            "manifest_sha256": contract.sha_file(
                args.linguistic_augmentation / "manifest.json"
            ),
            "quality_receipt_sha256": contract.sha_file(
                args.linguistic_augmentation / "quality-audit-receipt.json"
            ),
            "token_budget_receipt_sha256": contract.sha_file(
                args.linguistic_augmentation / "token-budget-receipt.json"
            ),
            "training_files": sorted(
                name
                for name in (inputs["linguistic"].get("artifacts") or {})
                if name.startswith("linguistic-") and name.endswith(".train.jsonl")
            ),
            "natural_teacher_scope": "historical-provenance-bound-subset",
            "offline_program_scope": "deterministic-linguistic-coverage-not-natural-user",
        },
        "eval_lock": {
            "path": str(args.eval_lock.resolve()),
            "id": inputs["eval"]["id"],
            "evaluation_fingerprint": inputs["eval"]["evaluation_fingerprint"],
            "lock_sha256": contract.sha_file(args.eval_lock / "lock.json"),
        },
        "narration_release": {
            "path": str(args.narration_release.resolve()),
            "release_id": inputs["narration"]["release_id"],
            "manifest_sha256": contract.sha_file(
                args.narration_release / "manifest.json"
            ),
            "isolation_receipt_sha256": contract.sha_file(
                args.narration_release / "isolation-receipt.json"
            ),
        },
        "qat_import": {
            "candidate_receipt": str(args.qat_import_receipt.resolve()),
            "candidate_receipt_sha256": contract.sha_file(args.qat_import_receipt),
            "master": str(inputs["qat_master"].resolve()),
            "master_sha256": inputs["qat"]["master"]["sha256"],
            "worker_receipt": str(inputs["qat_worker_receipt"].resolve()),
            "worker_receipt_sha256": inputs["qat"]["worker_receipt"]["sha256"],
            "worker_stage_fingerprint_sha256": inputs["qat_worker"].get(
                "stage_fingerprint_sha256"
            ),
            "base_id": base_identity["base_id"],
            "base_exposure_tokens": base_identity["tokens_seen_exposure"],
            "base_weights_sha256": base_identity["weights_sha256"],
            "quant_math_id": inputs["qat"]["quant_math_id"],
        },
        "contracts": {
            name: str(contracts_raw[name])
            for name in (
                "weight_contract_sha256",
                "runtime_profile_sha256",
                "training_aux_sha256",
            )
        },
        "tokenizer_sha256": contract.sha_file(TOKENIZER_ZH_V1),
        "current_baseline_sha256": contract.sha_file(CURRENT_PATH),
        "source_manifest": _source_manifest(),
        "recipe": {
            "float_control_steps": args.float_control_steps,
            "float_control_sampler": training.BANKED_FULLCALL_SAMPLER_ID,
            "retrieval_r0_steps": args.retrieval_r0_steps,
            "retrieval_sampler": training.RETRIEVAL_SAMPLER_ID,
            "fullcall_steps": args.fullcall_steps,
            "fullcall_sampler": training.BANKED_FULLCALL_SAMPLER_ID,
            "linguistic_source_balanced": True,
            "whole_schema_holdout": True,
            "agent_steps": args.agent_steps,
            "retrieval_r1_steps": args.retrieval_r1_steps,
            "mw_steps": args.mw_steps,
            "confidence_train_limit": args.confidence_train_limit,
            "confidence_valid_limit": args.confidence_valid_limit,
            "confidence_steps": args.confidence_steps,
            "narration_steps": args.narration_steps,
            "narration_eval_limit": args.narration_eval_limit,
            "eval_limit": args.eval_limit,
            "seed": 51,
        },
        "validation_scope": {
            "profile": "python-browser-wasm",
            "required_runtime_bindings": ["python", "browser-wasm"],
            "deferred_bindings": ["c", "node", "rust-public-sdk"],
        },
    }
    if (
        args.adopt_training_prefix_run is not None
        and args.adopt_productization_prefix_run is not None
    ):
        raise RuntimeError(
            "training-prefix and productization-prefix adoption are mutually exclusive"
        )
    active_stages: tuple[str, ...] = STAGES
    if args.adopt_productization_prefix_run is not None:
        immutable["productization_prefix_adoption"] = (
            _verify_productization_prefix_run(
                args.adopt_productization_prefix_run,
                immutable,
            )
        )
        immutable["stage_graph_mode"] = (
            "verified-productization-prefix-continuation-v1"
        )
        active_stages = (
            PRODUCTIZATION_PREFIX_ADOPTION_STAGE,
            *FINALIZATION_STAGES,
        )
    elif args.adopt_training_prefix_run is not None:
        immutable["training_prefix_adoption"] = _verify_training_prefix_run(
            args.adopt_training_prefix_run,
            immutable,
        )
        immutable["stage_graph_mode"] = "verified-training-prefix-continuation-v1"
        active_stages = (
            TRAINING_PREFIX_ADOPTION_STAGE,
            *DOWNSTREAM_STAGES,
        )
    else:
        immutable["stage_graph_mode"] = "full-productization-v1"
    run_fingerprint = contract.sha_bytes(contract.canonical_bytes(immutable))
    rows: list[dict[str, Any]] = []
    predecessor = base_identity["weights_sha256"]
    for index, stage in enumerate(active_stages, 1):
        fingerprint = contract.sha_bytes(
            contract.canonical_bytes(
                {
                    "run_fingerprint": run_fingerprint,
                    "stage": stage,
                    "index": index,
                    "predecessor": predecessor,
                }
            )
        )
        rows.append(
            {
                "index": index,
                "stage_id": stage,
                "stage_fingerprint_sha256": fingerprint,
                "predecessor_fingerprint_sha256": predecessor,
            }
        )
        predecessor = fingerprint
    return {
        "schema": "mei-51m-sft-v4-productization-plan-v1",
        "run_fingerprint_sha256": run_fingerprint,
        "immutable": immutable,
        "stages": rows,
        "live_cpt_workers": live,
        "heavy_execution_deferred": bool(live),
        "process_complete": False,
        "release_eligible": False,
    }


def _validated_preflight(
    args: argparse.Namespace, plan: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    root = ROOT / "notebook/evaluation/jobs/mei-1.0-51m"
    candidates = (
        [args.preflight_receipt]
        if args.preflight_receipt is not None
        else sorted(root.glob("sft-v4-*-preflight-*.json"), reverse=True)
    )
    expected_inputs = {
        "base_release_sha256": contract.sha_file(args.base_release),
        "base_weights_sha256": contract.sha_file(args.base_weights),
        "qat_import_receipt_sha256": contract.sha_file(args.qat_import_receipt),
        "data_manifest_sha256": contract.sha_file(args.data_release / "manifest.json"),
        "linguistic_augmentation_manifest_sha256": contract.sha_file(
            args.linguistic_augmentation / "manifest.json"
        ),
        "eval_lock_sha256": contract.sha_file(args.eval_lock / "lock.json"),
        "current_sha256": contract.sha_file(CURRENT_PATH),
    }
    for path in candidates:
        if path is None or not path.is_file():
            continue
        receipt = contract.load_json(path)
        inputs = receipt.get("inputs") or {}
        invariants = receipt.get("invariants") or {}
        exposure = receipt.get("sampler_exposure") or {}
        if (
            receipt.get("schema")
            != "mei-51m-sft-v4-exhaustive-preflight-receipt-v1"
            or receipt.get("status") != "passed"
            or receipt.get("run_fingerprint_sha256")
            != plan["run_fingerprint_sha256"]
            or any(inputs.get(name) != value for name, value in expected_inputs.items())
            or invariants.get("mlx_imported") is not False
            or invariants.get("metal_claimed") is not False
            or invariants.get("current_mutated") is not False
            or invariants.get("frozen_release_mutated") is not False
            or invariants.get("all_hashes_verified") is not True
            or invariants.get("all_lm_rows_encoded") is not True
            or invariants.get("all_fixed_step_lm_rows_exposed") is not True
            or set(exposure) != {"fullcall", "retrieval_r0", "retrieval_r1", "agent"}
            or not all(
                bool((exposure.get(name) or {}).get("all_rows_exposed"))
                for name in exposure
            )
        ):
            continue
        return path, receipt
    raise RuntimeError(
        "no current passed SFT-v4 preflight receipt matches this run fingerprint"
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    lifecycle.write_json(path, value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    contract.write_once(path, contract.jsonl_bytes(rows))


def _release_runtime(runtime: Any) -> None:
    try:
        import mlx.core as mx

        del runtime
        gc.collect()
        mx.clear_cache()
    except ImportError:
        pass


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return contract.load_jsonl(path)


def _with_training_bank(
    rows: list[dict[str, Any]], bank: str | None = None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        tagged = dict(row)
        selected = bank
        if selected is None:
            selected = (
                "linguistic_natural"
                if row.get("linguistic_layer") == "historical_admitted_natural"
                else "linguistic_offline"
            )
        tagged["_training_bank"] = selected
        output.append(tagged)
    return output


def _catalog(eval_lock: Path) -> list[dict[str, Any]]:
    return evaluation.catalog_from_lock(eval_lock)


def _stage_paths(
    run_dir: Path, training_prefix_run: Path | None = None
) -> dict[str, Path]:
    paths = {
        "float_control": run_dir / "stages/float_task_control_v4/float-control.npz",
        "r0": run_dir / "stages/retrieval_r0_v4/retrieval-r0.npz",
        "fullcall": run_dir / "stages/oracle_top5_fullcall_v4/fullcall-master.npz",
        "agent": run_dir / "stages/oracle_top5_agent_v4/agent-master.npz",
        "r1": run_dir / "stages/retrieval_r1_v4/retrieval-r1.npz",
        "index": run_dir / "stages/tool_index_v4/tool-index.json",
        "mw": run_dir / "stages/mw_disposition_v4/mw-disposition.npz",
        "confidence": run_dir / "stages/confidence_head_v4/confidence.npz",
        "calibration": run_dir / "stages/confidence_head_v4/confidence-calibration.json",
        "narration": run_dir / "stages/narration_adapter_v4/narration-adapter.npz",
    }
    if training_prefix_run is not None:
        parent = _stage_paths(training_prefix_run)
        for name in ("float_control", "r0", "fullcall", "agent", "r1", "index"):
            paths[name] = parent[name]
    return paths


def _save(value: Any, path: Path) -> None:
    lifecycle._save_model_or_head(value, path)


def _load_runtime(master: Path, **kwargs: Any) -> Any:
    runtime = lifecycle._load_runtime(master, **kwargs)
    if kwargs.get("retrieval_head") is not None:
        training.configure_retrieval_encoding_v3(runtime)
    return runtime


def _metric_floors(eval_lock: Path) -> dict[str, Any]:
    return contract.load_json(eval_lock / "metric-contract.json")


def execute(args: argparse.Namespace, plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if plan["live_cpt_workers"]:
        raise RuntimeError("live CPT owns Metal; SFT-v4 execution is deferred")
    if _source_manifest() != plan["immutable"]["source_manifest"]:
        raise RuntimeError(
            "productization source drifted after planning; use a new run fingerprint"
        )
    live_workers = lifecycle.live_cpt_workers()
    if live_workers:
        raise RuntimeError(f"live CPT reclaimed Metal before execution: {live_workers}")
    import mlx.core as mx
    from checkpoint import save_params
    from cq2_qat_51m import explicit_group_map
    from train_sft_ondisk_51m import save_product_heads, train_narration_adapter

    data_dir = args.data_release
    linguistic_dir = args.linguistic_augmentation
    catalog = _catalog(args.eval_lock)
    deploy_by_name = {str(tool["name"]): tool for tool in catalog}
    training_universe = contract.load_json(data_dir / "training-tool-universe.json")
    training_catalog = [
        contract.compact_tool(tool) for tool in training_universe.get("tools") or []
    ]
    training_tools_by_name = {
        str(tool["name"]): tool for tool in training_catalog
    }
    schema_holdout_document = contract.load_json(
        args.eval_lock / "schema-holdout-tool-universe.json"
    )
    schema_holdout_catalog = [
        contract.compact_tool(tool)
        for tool in schema_holdout_document.get("tools") or []
    ]
    schema_eval_catalog = [*catalog, *schema_holdout_catalog]
    schema_eval_tools_by_name = {
        str(tool["name"]): tool for tool in schema_eval_catalog
    }
    if (
        len(catalog) != 147
        or len(training_catalog) != 211
        or len(schema_holdout_catalog) != 32
        or len(schema_eval_tools_by_name) != 179
    ):
        raise RuntimeError("SFT-v4 requires 147 deploy tools and 211 training tools")
    structural_fullcall_train = _with_training_bank(
        _load_rows(data_dir / "full-call.train.jsonl"), "structural"
    )
    schema_fullcall_train = _with_training_bank(
        _load_rows(data_dir / "schema-full-call.train.jsonl"), "schema"
    )
    linguistic_fullcall_train = _with_training_bank(
        _load_rows(linguistic_dir / "linguistic-full-call.train.jsonl")
    )
    fullcall_train = [
        *structural_fullcall_train,
        *schema_fullcall_train,
        *linguistic_fullcall_train,
    ]
    fullcall_valid = [
        *_load_rows(data_dir / "full-call.valid.jsonl"),
        *_load_rows(data_dir / "schema-full-call.valid.jsonl"),
        *_load_rows(linguistic_dir / "linguistic-full-call.valid.jsonl"),
    ]
    agent_train = _load_rows(data_dir / "agent-continuation.train.jsonl")
    structural_retrieval_train = _with_training_bank(
        _load_rows(data_dir / "retrieval.train.jsonl"), "structural"
    )
    schema_retrieval_train = _with_training_bank(
        _load_rows(data_dir / "schema-retrieval.train.jsonl"), "schema"
    )
    linguistic_retrieval_train = _with_training_bank(
        _load_rows(linguistic_dir / "linguistic-retrieval.train.jsonl")
    )
    retrieval_train = [
        *structural_retrieval_train,
        *schema_retrieval_train,
        *linguistic_retrieval_train,
    ]
    mw_train = _load_rows(data_dir / "mw-disposition.train.jsonl")
    confidence_train = _load_rows(data_dir / "confidence-harvest.train.jsonl")
    confidence_valid = _load_rows(data_dir / "confidence-harvest.valid.jsonl")
    confidence_dev = [
        *_load_rows(args.eval_lock / "confidence.dev.jsonl"),
        *_load_rows(args.eval_lock / "schema-confidence.dev.jsonl"),
    ]
    fullcall_dev = _load_rows(args.eval_lock / "fullcall.dev.jsonl")
    control_fullcall_dev = [
        *evaluation.balanced_fullcall_sample(fullcall_dev),
        *_load_rows(args.eval_lock / "natural-fullcall.dev.jsonl"),
        *_load_rows(args.eval_lock / "schema-fullcall.dev.jsonl"),
    ]
    if len(fullcall_train) > args.fullcall_steps:
        raise RuntimeError(
            "fixed full-call budget no longer exposes every v4 full-call row"
        )
    adoption = plan["immutable"].get("training_prefix_adoption")
    productization_adoption = plan["immutable"].get(
        "productization_prefix_adoption"
    )
    training_prefix_run = (
        Path(str(adoption["parent_run_dir"])) if adoption is not None else None
    )
    paths = _stage_paths(run_dir, training_prefix_run)
    if productization_adoption is not None:
        adopted_artifacts = productization_adoption.get("artifacts") or {}
        for name in (
            "agent",
            "r1",
            "index",
            "mw",
            "confidence",
            "calibration",
            "narration",
        ):
            paths[name] = Path(str(adopted_artifacts[name]["path"]))
    receipts: dict[str, dict[str, Any]] = {}

    def run(name: str, action: Any) -> dict[str, Any]:
        if _source_manifest() != plan["immutable"]["source_manifest"]:
            raise RuntimeError(
                "productization source drifted after planning; use a new run fingerprint"
            )
        live_workers = lifecycle.live_cpt_workers()
        if live_workers:
            raise RuntimeError(
                f"live CPT reclaimed Metal before stage {name}: {live_workers}"
            )
        receipt = lifecycle.run_stage(
            run_dir=run_dir,
            plan=plan,
            name=name,
            action=action,
            resume=args.resume,
        )
        receipts[name] = receipt
        return receipt

    if productization_adoption is not None:

        def productization_adoption_action(directory: Path):
            observed = _verify_productization_prefix_run(
                Path(str(productization_adoption["parent_run_dir"])),
                plan["immutable"],
            )
            if contract.canonical_bytes(observed) != contract.canonical_bytes(
                productization_adoption
            ):
                raise RuntimeError(
                    "productization-prefix adoption evidence drifted after planning"
                )
            report = {
                "status": "passed",
                "schema": (
                    "mei-sft-v4-productization-prefix-adoption-receipt-v1"
                ),
                "adoption_input_fingerprint_sha256": productization_adoption[
                    "adoption_input_fingerprint_sha256"
                ],
                "parent_run_fingerprint_sha256": productization_adoption[
                    "parent_run_fingerprint_sha256"
                ],
                "adopted_stage_ids": productization_adoption["stage_ids"],
                "component_stages": productization_adoption["component_stages"],
                "artifacts": productization_adoption["artifacts"],
                "source_transition": productization_adoption["source_transition"],
                "parent_run_mutated": False,
            }
            report_path = directory / "productization-prefix-adoption.json"
            _write_json(report_path, report)
            artifact_paths = [
                Path(str(spec["path"]))
                for spec in (
                    productization_adoption.get("artifacts") or {}
                ).values()
            ]
            return report, [report_path, *artifact_paths]

        run(
            PRODUCTIZATION_PREFIX_ADOPTION_STAGE,
            productization_adoption_action,
        )

    elif adoption is not None:

        def adoption_action(directory: Path):
            observed = _verify_training_prefix_run(
                Path(str(adoption["parent_run_dir"])), plan["immutable"]
            )
            if contract.canonical_bytes(observed) != contract.canonical_bytes(adoption):
                raise RuntimeError("training-prefix adoption evidence drifted after planning")
            report = {
                "status": "passed",
                "schema": "mei-sft-v4-training-prefix-adoption-receipt-v1",
                "adoption_input_fingerprint_sha256": adoption[
                    "adoption_input_fingerprint_sha256"
                ],
                "parent_run_fingerprint_sha256": adoption[
                    "parent_run_fingerprint_sha256"
                ],
                "adopted_stage_ids": adoption["stage_ids"],
                "artifacts": adoption["artifacts"],
                "source_transition": adoption["source_transition"],
                "parent_run_mutated": False,
            }
            report_path = directory / "training-prefix-adoption.json"
            _write_json(report_path, report)
            artifact_paths = [
                Path(str(spec["path"]))
                for spec in (adoption.get("artifacts") or {}).values()
            ]
            return report, [report_path, *artifact_paths]

        run(TRAINING_PREFIX_ADOPTION_STAGE, adoption_action)

    def anchor_action(directory: Path):
        base = contract.load_json(args.base_release)
        runtime = _load_runtime(args.base_weights, quantized=False)
        control_report, control_predictions = evaluation.evaluate_fullcall(
            runtime,
            control_fullcall_dev,
            schema_eval_catalog,
            retrieval_mode="oracle_top5",
        )
        report = {
            "status": "passed",
            "base_id": base["model_id"],
            "weights_sha256": base["weights_sha256"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
            "validation_nll": base.get("valid_loss"),
            "domain_nll_hq": base.get("valid_loss_hq"),
            "domain_nll_colloquial": base.get("valid_loss_colloquial"),
            "domain_nll_structure": base.get("valid_loss_structure"),
            "probe_mean_nll": base.get("final_probe_mean_nll"),
            "immutable_base": True,
            "task_control_slice": control_report,
        }
        path = directory / "float-base-lm-anchor.json"
        _write_json(path, report)
        predictions_path = directory / "float-base-control.predictions.jsonl"
        _write_jsonl(predictions_path, control_predictions)
        _release_runtime(runtime)
        return report, [path, predictions_path]

    if adoption is None and productization_adoption is None:
        run("float_base_lm_anchor", anchor_action)

    def float_control_action(directory: Path):
        runtime = _load_runtime(args.base_weights, quantized=False)
        report = training.train_lm_sft_v3(
            runtime,
            fullcall_train,
            training_tools_by_name,
            steps=args.float_control_steps,
            lr=2e-4,
            group_map=None,
            activation_ste=False,
            sampler=training.BANKED_FULLCALL_SAMPLER_ID,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="float_task_control_v4",
        )
        save_params(runtime.model, paths["float_control"])
        control_report, control_predictions = evaluation.evaluate_fullcall(
            runtime,
            control_fullcall_dev,
            schema_eval_catalog,
            retrieval_mode="oracle_top5",
        )
        report["task_control_slice"] = control_report
        report_path = directory / "float-task-control.json"
        _write_json(report_path, report)
        predictions_path = directory / "float-task-control.predictions.jsonl"
        _write_jsonl(predictions_path, control_predictions)
        _release_runtime(runtime)
        return report, [paths["float_control"], report_path, predictions_path]

    if adoption is None and productization_adoption is None:
        run("float_task_control_v4", float_control_action)

    qat_master = Path(plan["immutable"]["qat_import"]["master"])
    qat_receipt = Path(plan["immutable"]["qat_import"]["worker_receipt"])

    def qat_import_action(directory: Path):
        report = {
            "status": "passed",
            "mode": "strict_existing_artifact_import",
            "master_sha256": contract.sha_file(qat_master),
            "worker_receipt_sha256": contract.sha_file(qat_receipt),
            "quant_math_id": plan["immutable"]["qat_import"]["quant_math_id"],
            "not_retrained": True,
        }
        path = directory / "cq2-qat-import.json"
        _write_json(path, report)
        return report, [qat_master, qat_receipt, path]

    if adoption is None and productization_adoption is None:
        run("cq2_qat_import", qat_import_action)

    def r0_action(directory: Path):
        runtime = _load_runtime(qat_master, quantized=True)
        report = training.train_retrieval_v3(
            runtime,
            retrieval_train,
            training_tools_by_name,
            steps=args.retrieval_r0_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="r0-v4",
        )
        _save(runtime.contrastive, paths["r0"])
        report_path = directory / "retrieval-r0.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["r0"], report_path]

    if adoption is None and productization_adoption is None:
        run("retrieval_r0_v4", r0_action)

    def fullcall_action(directory: Path):
        runtime = _load_runtime(qat_master, quantized=False)
        group_map = explicit_group_map(runtime.model.parameters())
        report = training.train_lm_sft_v3(
            runtime,
            fullcall_train,
            training_tools_by_name,
            steps=args.fullcall_steps,
            lr=2e-4,
            group_map=group_map,
            activation_ste=True,
            sampler=training.BANKED_FULLCALL_SAMPLER_ID,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="oracle_top5_fullcall_v4",
        )
        save_params(runtime.model, paths["fullcall"])
        report_path = directory / "oracle-top5-fullcall.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["fullcall"], report_path]

    if adoption is None and productization_adoption is None:
        run("oracle_top5_fullcall_v4", fullcall_action)

    def agent_action(directory: Path):
        runtime = _load_runtime(paths["fullcall"], quantized=False)
        group_map = explicit_group_map(runtime.model.parameters())
        report = training.train_lm_sft_v3(
            runtime,
            agent_train,
            training_tools_by_name,
            steps=args.agent_steps,
            lr=1e-4,
            group_map=group_map,
            activation_ste=True,
            sampler=training.AGENT_SAMPLER_ID,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="oracle_top5_agent_v4",
        )
        save_params(runtime.model, paths["agent"])
        report_path = directory / "oracle-top5-agent.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["agent"], report_path]

    if adoption is None and productization_adoption is None:
        run("oracle_top5_agent_v4", agent_action)

    def r1_action(directory: Path):
        runtime = _load_runtime(paths["agent"], quantized=True)
        report = training.train_retrieval_v3(
            runtime,
            retrieval_train,
            training_tools_by_name,
            steps=args.retrieval_r1_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="r1-v4",
        )
        _save(runtime.contrastive, paths["r1"])
        report_path = directory / "retrieval-r1.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["r1"], report_path]

    if adoption is None and productization_adoption is None:
        run("retrieval_r1_v4", r1_action)

    def index_action(directory: Path):
        runtime = _load_runtime(paths["agent"], quantized=True, retrieval_head=paths["r1"])
        report = training.finalize_tool_index_v3(
            runtime,
            catalog,
            paths["index"],
            model_sha256=contract.sha_file(paths["agent"]),
            head_sha256=contract.sha_file(paths["r1"]),
            tokenizer_sha256=contract.sha_file(TOKENIZER_ZH_V1),
        )
        report_path = directory / "tool-index-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["index"], report_path]

    if adoption is None and productization_adoption is None:
        run("tool_index_v4", index_action)

    def single_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        retrieval_rows = _load_rows(args.eval_lock / "retrieval.dev.jsonl")
        fullcall_rows = _load_rows(args.eval_lock / "fullcall.dev.jsonl")
        retrieval_report, retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, retrieval_rows, catalog
        )
        oracle_report, oracle_predictions = evaluation.evaluate_fullcall(
            runtime,
            fullcall_rows,
            catalog,
            retrieval_mode="oracle_top5",
            limit=args.eval_limit,
        )
        learned_report, learned_predictions = evaluation.evaluate_fullcall(
            runtime,
            fullcall_rows,
            catalog,
            retrieval_mode="learned_top5",
            limit=args.eval_limit,
        )
        floors = _metric_floors(args.eval_lock)
        gates = {
            "retrieval": evaluation.threshold_status(
                retrieval_report, floors["retrieval"]["release_floor"]
            ),
            "fullcall_learned": evaluation.threshold_status(
                learned_report, floors["fullcall"]["release_floor"]
            ),
        }
        report = {
            "retrieval": retrieval_report,
            "fullcall_oracle": oracle_report,
            "fullcall_learned": learned_report,
            "gates": gates,
            "quality_ok": all(value["passed"] for value in gates.values()),
        }
        report["degraded"] = not report["quality_ok"]
        report_path = directory / "single-step-eval-v4.json"
        _write_json(report_path, report)
        prediction_paths = []
        for name, rows in (
            ("retrieval.dev.predictions.jsonl", retrieval_predictions),
            ("fullcall.oracle.dev.predictions.jsonl", oracle_predictions),
            ("fullcall.learned.dev.predictions.jsonl", learned_predictions),
        ):
            path = directory / name
            _write_jsonl(path, rows)
            prediction_paths.append(path)
        _release_runtime(runtime)
        return report, [report_path, *prediction_paths]

    if productization_adoption is None:
        run("single_step_eval_v4", single_eval_action)

    def multistep_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        rows = _load_rows(args.eval_lock / "multistep.dev.jsonl")
        learned_report, learned_predictions = evaluation.evaluate_multistep(
            runtime,
            rows,
            catalog,
            retrieval_mode="learned_top5",
        )
        floors = _metric_floors(args.eval_lock)["multi_step"]["release_floor"]
        gate = evaluation.threshold_status(learned_report, floors)
        report = {
            "learned_top5": learned_report,
            "gate": gate,
            "quality_ok": gate["passed"],
            "degraded": not gate["passed"],
        }
        report_path = directory / "multistep-eval-v4.json"
        predictions_path = directory / "multistep.learned.dev.predictions.jsonl"
        _write_json(report_path, report)
        _write_jsonl(predictions_path, learned_predictions)
        _release_runtime(runtime)
        return report, [report_path, predictions_path]

    if productization_adoption is None:
        run("multistep_eval_v4", multistep_eval_action)

    def crossgen_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        natural_retrieval_rows = _load_rows(
            args.eval_lock / "natural-retrieval.dev.jsonl"
        )
        natural_fullcall_rows = _load_rows(
            args.eval_lock / "natural-fullcall.dev.jsonl"
        )
        schema_retrieval_rows = _load_rows(
            args.eval_lock / "schema-retrieval.dev.jsonl"
        )
        schema_fullcall_rows = _load_rows(
            args.eval_lock / "schema-fullcall.dev.jsonl"
        )
        natural_retrieval, natural_retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, natural_retrieval_rows, catalog
        )
        natural_oracle, natural_oracle_predictions = evaluation.evaluate_fullcall(
            runtime,
            natural_fullcall_rows,
            catalog,
            retrieval_mode="oracle_top5",
        )
        natural_learned, natural_learned_predictions = evaluation.evaluate_fullcall(
            runtime,
            natural_fullcall_rows,
            catalog,
            retrieval_mode="learned_top5",
        )
        schema_retrieval, schema_retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, schema_retrieval_rows, schema_eval_catalog
        )
        schema_oracle, schema_oracle_predictions = evaluation.evaluate_fullcall(
            runtime,
            schema_fullcall_rows,
            schema_eval_catalog,
            retrieval_mode="oracle_top5",
        )
        schema_learned, schema_learned_predictions = evaluation.evaluate_fullcall(
            runtime,
            schema_fullcall_rows,
            schema_eval_catalog,
            retrieval_mode="learned_top5",
        )
        floors = _metric_floors(args.eval_lock)
        gates = {
            "natural_retrieval": evaluation.threshold_status(
                natural_retrieval, floors["retrieval"]["release_floor"]
            ),
            "natural_fullcall_learned": evaluation.threshold_status(
                natural_learned, floors["fullcall"]["release_floor"]
            ),
            "schema_retrieval": evaluation.threshold_status(
                schema_retrieval, floors["retrieval"]["release_floor"]
            ),
            "schema_fullcall_learned": evaluation.threshold_status(
                schema_learned, floors["fullcall"]["release_floor"]
            ),
        }
        report = {
            "linguistic_augmentation_release_fingerprint": plan["immutable"][
                "linguistic_augmentation"
            ]["release_fingerprint"],
            "split": "dev",
            "natural": {
                "retrieval": natural_retrieval,
                "fullcall_oracle": natural_oracle,
                "fullcall_learned": natural_learned,
            },
            "whole_schema_holdout": {
                "holdout_tool_count": len(schema_holdout_catalog),
                "retrieval": schema_retrieval,
                "fullcall_oracle": schema_oracle,
                "fullcall_learned": schema_learned,
            },
            "gates": gates,
            "quality_ok": all(value["passed"] for value in gates.values()),
        }
        report["degraded"] = not report["quality_ok"]
        report_path = directory / "crossgen-schema-eval-v4.json"
        _write_json(report_path, report)
        prediction_paths: list[Path] = []
        for name, rows in (
            ("natural-retrieval.dev.predictions.jsonl", natural_retrieval_predictions),
            ("natural-fullcall.oracle.dev.predictions.jsonl", natural_oracle_predictions),
            ("natural-fullcall.learned.dev.predictions.jsonl", natural_learned_predictions),
            ("schema-retrieval.dev.predictions.jsonl", schema_retrieval_predictions),
            ("schema-fullcall.oracle.dev.predictions.jsonl", schema_oracle_predictions),
            ("schema-fullcall.learned.dev.predictions.jsonl", schema_learned_predictions),
        ):
            path = directory / name
            _write_jsonl(path, rows)
            prediction_paths.append(path)
        _release_runtime(runtime)
        return report, [report_path, *prediction_paths]

    if productization_adoption is None:
        run("crossgen_schema_eval_v4", crossgen_eval_action)

    def mw_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        train_report = training.train_mw_disposition_v3(
            runtime,
            mw_train,
            deploy_by_name,
            catalog,
            steps=args.mw_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        _save(runtime.mw_disposition_head, paths["mw"])
        eval_rows = _load_rows(args.eval_lock / "mw.dev.jsonl")
        oracle_report, oracle_predictions = evaluation.evaluate_mw_disposition(
            runtime, eval_rows, catalog, retrieval_mode="oracle_top5"
        )
        learned_report, learned_predictions = evaluation.evaluate_mw_disposition(
            runtime, eval_rows, catalog, retrieval_mode="learned_top5"
        )
        floors = _metric_floors(args.eval_lock)["mw_disposition"]["release_floor"]
        gates = {
            "oracle": evaluation.threshold_status(oracle_report, floors),
            "learned": evaluation.threshold_status(learned_report, floors),
        }
        report = {
            "train": train_report,
            "eval_oracle": oracle_report,
            "eval_learned": learned_report,
            "gates": gates,
            "degraded": not all(value["passed"] for value in gates.values()),
            "mw_deviation_tensor_count": 0,
        }
        report_path = directory / "mw-disposition-v4.json"
        _write_json(report_path, report)
        oracle_path = directory / "mw.oracle.dev.predictions.jsonl"
        learned_path = directory / "mw.learned.dev.predictions.jsonl"
        _write_jsonl(oracle_path, oracle_predictions)
        _write_jsonl(learned_path, learned_predictions)
        _release_runtime(runtime)
        return report, [paths["mw"], report_path, oracle_path, learned_path]

    if productization_adoption is None:
        run("mw_disposition_v4", mw_action)

    confidence_train_path = run_dir / "stages/confidence_harvest_v4/train-outcomes.jsonl"
    confidence_valid_path = run_dir / "stages/confidence_harvest_v4/valid-outcomes.jsonl"
    confidence_dev_path = run_dir / "stages/confidence_harvest_v4/dev-outcomes.jsonl"
    if productization_adoption is not None:
        adopted_artifacts = productization_adoption["artifacts"]
        confidence_train_path = Path(
            str(adopted_artifacts["confidence_train_outcomes"]["path"])
        )
        confidence_valid_path = Path(
            str(adopted_artifacts["confidence_valid_outcomes"]["path"])
        )
        confidence_dev_path = Path(
            str(adopted_artifacts["confidence_dev_outcomes"]["path"])
        )

    def confidence_harvest_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        train_outcomes, train_receipt = training.harvest_confidence_outcomes_v3(
            runtime,
            confidence_train,
            training_tools_by_name,
            training_catalog,
            limit=args.confidence_train_limit,
        )
        valid_outcomes, valid_receipt = training.harvest_confidence_outcomes_v3(
            runtime,
            confidence_valid,
            training_tools_by_name,
            training_catalog,
            limit=args.confidence_valid_limit,
        )
        dev_outcomes, dev_receipt = training.harvest_confidence_outcomes_v3(
            runtime,
            confidence_dev,
            schema_eval_tools_by_name,
            schema_eval_catalog,
            limit=len(confidence_dev),
        )
        _write_jsonl(confidence_train_path, train_outcomes)
        _write_jsonl(confidence_valid_path, valid_outcomes)
        _write_jsonl(confidence_dev_path, dev_outcomes)
        report = {
            "train": train_receipt,
            "valid": valid_receipt,
            "fixed_dev": dev_receipt,
            "degraded": False,
        }
        report_path = directory / "confidence-harvest-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [
            confidence_train_path,
            confidence_valid_path,
            confidence_dev_path,
            report_path,
        ]

    if productization_adoption is None:
        run("confidence_harvest_v4", confidence_harvest_action)

    def confidence_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
        )
        train_outcomes = _load_rows(confidence_train_path)
        valid_outcomes = _load_rows(confidence_valid_path)
        report, calibration = training.train_confidence_v3(
            runtime,
            train_outcomes,
            valid_outcomes,
            steps=args.confidence_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        _save(runtime.conf_v2, paths["confidence"])
        _write_json(paths["calibration"], {"score_contract": training.CONFIDENCE_SCORE_ID, **calibration})
        floors = _metric_floors(args.eval_lock)["confidence"]["release_floor"]
        gate = evaluation.threshold_status(report["valid_metrics"], floors)
        report["gate"] = gate
        report["degraded"] = not gate["passed"]
        report_path = directory / "confidence-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["confidence"], paths["calibration"], report_path]

    if productization_adoption is None:
        run("confidence_head_v4", confidence_action)

    def narration_action(directory: Path):
        runtime = _load_runtime(paths["agent"], quantized=True)
        report = train_narration_adapter(
            runtime,
            _load_rows(args.narration_release / "narration.train.jsonl"),
            _load_rows(args.narration_release / "narration.valid.jsonl"),
            steps=args.narration_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        _save(runtime.narration_adapter, paths["narration"])
        report_path = directory / "narration-adapter-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["narration"], report_path]

    if productization_adoption is None:
        run("narration_adapter_v4", narration_action)

    def sidecar_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
            mw_head=paths["mw"],
            confidence_head=paths["confidence"],
            narration_head=paths["narration"],
        )
        calibration_document = contract.load_json(paths["calibration"])
        calibration = {
            "scale": float(calibration_document["scale"]),
            "bias": float(calibration_document["bias"]),
        }
        confidence_outcomes = _load_rows(confidence_dev_path)
        confidence_report, confidence_predictions = evaluation.evaluate_confidence(
            runtime,
            confidence_outcomes,
            calibration,
            candidates=confidence_dev,
        )
        narration_rows = _load_rows(args.eval_lock / "narration.dev.jsonl")
        narration_report, narration_predictions = evaluation.evaluate_narration(
            runtime,
            narration_rows,
            limit=args.narration_eval_limit,
        )
        floors = _metric_floors(args.eval_lock)
        gates = {
            "confidence": evaluation.threshold_status(
                confidence_report, floors["confidence"]["release_floor"]
            ),
            "narration": evaluation.threshold_status(
                narration_report, floors["narration"]["release_floor"]
            ),
        }
        report = {
            "confidence": confidence_report,
            "narration": narration_report,
            "gates": gates,
            "quality_ok": all(value["passed"] for value in gates.values()),
        }
        report["degraded"] = not report["quality_ok"]
        report_path = directory / "sidecar-eval-v4.json"
        confidence_path = directory / "confidence.dev.predictions.jsonl"
        narration_path = directory / "narration.dev.predictions.jsonl"
        _write_json(report_path, report)
        _write_jsonl(confidence_path, confidence_predictions)
        _write_jsonl(narration_path, narration_predictions)
        _release_runtime(runtime)
        return report, [report_path, confidence_path, narration_path]

    run("sidecar_eval_v4", sidecar_eval_action)

    def locked_test_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r1"],
            tool_index=paths["index"],
            mw_head=paths["mw"],
            confidence_head=paths["confidence"],
            narration_head=paths["narration"],
        )
        prediction_dir = directory / "predictions" / "test"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_paths: list[Path] = []

        def save_prediction(name: str, rows: list[dict[str, Any]]) -> None:
            path = prediction_dir / name
            _write_jsonl(path, rows)
            prediction_paths.append(path)

        structural_retrieval_rows = _load_rows(
            args.eval_lock / "retrieval.test.jsonl"
        )
        structural_fullcall_rows = _load_rows(
            args.eval_lock / "fullcall.test.jsonl"
        )
        retrieval_report, retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, structural_retrieval_rows, catalog
        )
        fullcall_report, fullcall_predictions = evaluation.evaluate_fullcall(
            runtime,
            structural_fullcall_rows,
            catalog,
            retrieval_mode="learned_top5",
        )
        save_prediction("retrieval.test.jsonl", retrieval_predictions)
        save_prediction("fullcall.test.jsonl", fullcall_predictions)

        multistep_rows = _load_rows(args.eval_lock / "multistep.test.jsonl")
        multistep_report, multistep_predictions = evaluation.evaluate_multistep(
            runtime,
            multistep_rows,
            catalog,
            retrieval_mode="learned_top5",
        )
        save_prediction("multistep.test.jsonl", multistep_predictions)

        natural_retrieval_rows = _load_rows(
            args.eval_lock / "natural-retrieval.test.jsonl"
        )
        natural_fullcall_rows = _load_rows(
            args.eval_lock / "natural-fullcall.test.jsonl"
        )
        natural_retrieval, natural_retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, natural_retrieval_rows, catalog
        )
        natural_fullcall, natural_fullcall_predictions = evaluation.evaluate_fullcall(
            runtime,
            natural_fullcall_rows,
            catalog,
            retrieval_mode="learned_top5",
        )
        save_prediction("natural-retrieval.test.jsonl", natural_retrieval_predictions)
        save_prediction("natural-fullcall.test.jsonl", natural_fullcall_predictions)

        schema_retrieval_rows = _load_rows(
            args.eval_lock / "schema-retrieval.test.jsonl"
        )
        schema_fullcall_rows = _load_rows(
            args.eval_lock / "schema-fullcall.test.jsonl"
        )
        schema_retrieval, schema_retrieval_predictions = evaluation.evaluate_retrieval(
            runtime, schema_retrieval_rows, schema_eval_catalog
        )
        schema_fullcall, schema_fullcall_predictions = evaluation.evaluate_fullcall(
            runtime,
            schema_fullcall_rows,
            schema_eval_catalog,
            retrieval_mode="learned_top5",
        )
        save_prediction("schema-retrieval.test.jsonl", schema_retrieval_predictions)
        save_prediction("schema-fullcall.test.jsonl", schema_fullcall_predictions)

        mw_rows = _load_rows(args.eval_lock / "mw.test.jsonl")
        mw_report, mw_predictions = evaluation.evaluate_mw_disposition(
            runtime, mw_rows, catalog, retrieval_mode="learned_top5"
        )
        save_prediction("mw.test.jsonl", mw_predictions)

        calibration_document = contract.load_json(paths["calibration"])
        calibration = {
            "scale": float(calibration_document["scale"]),
            "bias": float(calibration_document["bias"]),
        }
        core_confidence_rows = _load_rows(args.eval_lock / "confidence.test.jsonl")
        core_outcomes, core_harvest = training.harvest_confidence_outcomes_v3(
            runtime,
            core_confidence_rows,
            deploy_by_name,
            catalog,
            limit=len(core_confidence_rows),
            minimum_class_rows=100,
        )
        core_confidence, core_confidence_predictions = evaluation.evaluate_confidence(
            runtime,
            core_outcomes,
            calibration,
            candidates=core_confidence_rows,
            minimum_class_rows=100,
        )
        save_prediction("confidence.test.jsonl", core_confidence_predictions)

        schema_confidence_rows = _load_rows(
            args.eval_lock / "schema-confidence.test.jsonl"
        )
        schema_outcomes, schema_harvest = training.harvest_confidence_outcomes_v3(
            runtime,
            schema_confidence_rows,
            schema_eval_tools_by_name,
            schema_eval_catalog,
            limit=len(schema_confidence_rows),
            minimum_class_rows=1,
        )
        schema_confidence, schema_confidence_predictions = evaluation.evaluate_confidence(
            runtime,
            schema_outcomes,
            calibration,
            candidates=schema_confidence_rows,
            minimum_class_rows=1,
        )
        save_prediction(
            "schema-confidence.test.jsonl", schema_confidence_predictions
        )

        narration_rows = _load_rows(args.eval_lock / "narration.test.jsonl")
        narration_report, narration_predictions = evaluation.evaluate_narration(
            runtime, narration_rows
        )
        save_prediction("narration.test.jsonl", narration_predictions)

        scorecard = longitudinal_metrics.score_split(
            args.eval_lock, prediction_dir, "test"
        )
        scorecard_path = directory / "longitudinal-scorecard.test.json"
        _write_json(scorecard_path, scorecard)
        floors = _metric_floors(args.eval_lock)
        gates = {
            "structural_retrieval": evaluation.threshold_status(
                retrieval_report, floors["retrieval"]["release_floor"]
            ),
            "structural_fullcall": evaluation.threshold_status(
                fullcall_report, floors["fullcall"]["release_floor"]
            ),
            "natural_retrieval": evaluation.threshold_status(
                natural_retrieval, floors["retrieval"]["release_floor"]
            ),
            "natural_fullcall": evaluation.threshold_status(
                natural_fullcall, floors["fullcall"]["release_floor"]
            ),
            "schema_retrieval": evaluation.threshold_status(
                schema_retrieval, floors["retrieval"]["release_floor"]
            ),
            "schema_fullcall": evaluation.threshold_status(
                schema_fullcall, floors["fullcall"]["release_floor"]
            ),
            "multi_step": evaluation.threshold_status(
                multistep_report, floors["multi_step"]["release_floor"]
            ),
            "mw_disposition": evaluation.threshold_status(
                mw_report, floors["mw_disposition"]["release_floor"]
            ),
            "confidence": evaluation.threshold_status(
                core_confidence, floors["confidence"]["release_floor"]
            ),
            "schema_confidence_diagnostic": evaluation.threshold_status(
                schema_confidence, floors["confidence"]["release_floor"]
            ),
            "narration": evaluation.threshold_status(
                narration_report, floors["narration"]["release_floor"]
            ),
        }
        report = {
            "status": "passed",
            "split": "test",
            "evaluation_fingerprint": plan["immutable"]["eval_lock"][
                "evaluation_fingerprint"
            ],
            "metrics": {
                "retrieval": retrieval_report,
                "fullcall": fullcall_report,
                "multi_step": multistep_report,
                "natural_retrieval": natural_retrieval,
                "natural_fullcall": natural_fullcall,
                "schema_retrieval": schema_retrieval,
                "schema_fullcall": schema_fullcall,
                "mw_disposition": mw_report,
                "confidence": core_confidence,
                "schema_confidence": schema_confidence,
                "narration": narration_report,
            },
            "confidence_harvest": {
                "structural": core_harvest,
                "whole_schema_holdout": schema_harvest,
            },
            "gates": gates,
            "quality_ok": all(value["passed"] for value in gates.values()),
        }
        report["degraded"] = not report["quality_ok"]
        report_path = directory / "locked-test-eval-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [
            report_path,
            scorecard_path,
            *prediction_paths,
        ]

    run("locked_test_eval_v4", locked_test_eval_action)

    package_dir = run_dir / "candidate" / args.package_id

    def component_stage_evidence(stage: str) -> dict[str, Any]:
        if stage in receipts:
            receipt = receipts[stage]
            return {
                "stage_id": stage,
                "stage_fingerprint_sha256": receipt[
                    "stage_fingerprint_sha256"
                ],
                "status": "passed",
                "terminal_status": receipt["terminal_status"],
                "source_run_fingerprint_sha256": plan["run_fingerprint_sha256"],
                "adopted": False,
            }
        adoption_source = productization_adoption or adoption or {}
        parent_stage = (
            adoption_source.get("component_stages")
            or adoption_source.get("stages")
            or {}
        ).get(stage)
        if parent_stage is None:
            raise RuntimeError(f"component stage evidence is unavailable: {stage}")
        return {
            "stage_id": stage,
            "stage_fingerprint_sha256": parent_stage[
                "stage_fingerprint_sha256"
            ],
            "status": "passed",
            "terminal_status": parent_stage["terminal_status"],
            "source_run_fingerprint_sha256": adoption_source[
                "parent_run_fingerprint_sha256"
            ],
            "source_receipt_sha256": parent_stage["receipt_sha256"],
            "adopted": True,
            "adopted_via_stage": (
                PRODUCTIZATION_PREFIX_ADOPTION_STAGE
                if productization_adoption is not None
                else TRAINING_PREFIX_ADOPTION_STAGE
            ),
        }

    def package_action(directory: Path):
        from pack_cq2_v2_51m import export as export_cq2

        runtime = _load_runtime(
            paths["agent"],
            quantized=False,
            retrieval_head=paths["r1"],
            mw_head=paths["mw"],
            confidence_head=paths["confidence"],
            narration_head=paths["narration"],
        )
        heads_path = directory / "product-heads.npz"
        heads_report = save_product_heads(runtime, heads_path)
        evidence = {
            component: component_stage_evidence(stage)
            for component, stage in {
                "lm": "oracle_top5_agent_v4",
                "contrastive": "retrieval_r1_v4",
                "mw_disposition": "mw_disposition_v4",
                "confidence": "confidence_head_v4",
                "narration_adapter": "narration_adapter_v4",
            }.items()
        }
        evidence_path = directory / "stage-evidence.json"
        _write_json(evidence_path, evidence)
        report = export_cq2(
            argparse.Namespace(
                master=paths["agent"],
                heads=heads_path,
                tool_index=paths["index"],
                stage_evidence=evidence_path,
                out_dir=package_dir,
                package_id=args.package_id,
                parent_package_id=plan["immutable"]["base"]["base_id"],
            )
        )
        report["heads"] = heads_report
        report["confidence_calibration_source"] = {
            "path": str(paths["calibration"]),
            "sha256": contract.sha_file(paths["calibration"]),
            "packaging_pending_runtime_auxiliary_contract": True,
        }
        report["component_stage_terminal_status"] = {
            component: component_stage_evidence(stage)["terminal_status"]
            for component, stage in {
                "lm": "oracle_top5_agent_v4",
                "contrastive": "retrieval_r1_v4",
                "mw_disposition": "mw_disposition_v4",
                "confidence": "confidence_head_v4",
                "narration_adapter": "narration_adapter_v4",
            }.items()
        }
        crossgen_evidence = component_stage_evidence("crossgen_schema_eval_v4")
        report["crossgen_schema_eval"] = {
            "stage_fingerprint_sha256": crossgen_evidence[
                "stage_fingerprint_sha256"
            ],
            "terminal_status": crossgen_evidence["terminal_status"],
            "adopted": crossgen_evidence["adopted"],
            "linguistic_augmentation_release_fingerprint": plan["immutable"][
                "linguistic_augmentation"
            ]["release_fingerprint"],
        }
        report["locked_test_eval"] = {
            "stage_fingerprint_sha256": receipts["locked_test_eval_v4"][
                "stage_fingerprint_sha256"
            ],
            "terminal_status": receipts["locked_test_eval_v4"][
                "terminal_status"
            ],
            "evaluation_fingerprint": plan["immutable"]["eval_lock"][
                "evaluation_fingerprint"
            ],
        }
        report["degraded"] = bool(
            int(report.get("package_bytes") or 0) > 18 * 1024 * 1024
        )
        report_path = directory / "package-v2-cq2-v4.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        outputs = [path for path in package_dir.iterdir() if path.is_file()]
        return report, [heads_path, evidence_path, report_path, *outputs]

    run("package_v2_cq2_v4", package_action)
    return {
        "schema": "mei-51m-sft-v4-productization-progress-v1",
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "run_dir": str(run_dir),
        "stages": {name: receipt["terminal_status"] for name, receipt in receipts.items()},
        "candidate": str(package_dir),
        "process_complete": False,
        "release_eligible": False,
        "remaining": ["python_runtime_gate", "browser_wasm_gate", "longitudinal_history", "final_audit"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        "--qat-import-receipt",
        type=Path,
        default=DEFAULT_QAT_IMPORT_RECEIPT,
    )
    parser.add_argument("--data-release", type=Path, default=DEFAULT_DATA_RELEASE)
    parser.add_argument(
        "--linguistic-augmentation",
        "--natural-augmentation",
        dest="linguistic_augmentation",
        type=Path,
        default=DEFAULT_LINGUISTIC_AUGMENTATION,
    )
    parser.add_argument("--eval-lock", type=Path, default=DEFAULT_EVAL_LOCK)
    parser.add_argument("--narration-release", type=Path, default=DEFAULT_NARRATION_RELEASE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--package-id", default=DEFAULT_PACKAGE_ID)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument(
        "--adopt-training-prefix-run",
        type=Path,
        help=(
            "verify and adopt passed Float/QAT/R0/Full-call/Agent/R1/index "
            "stages from an immutable parent run, then execute downstream stages"
        ),
    )
    parser.add_argument(
        "--adopt-productization-prefix-run",
        type=Path,
        help=(
            "verify and adopt immutable stages through narration_adapter_v4, "
            "then execute only sidecar eval, locked test and package"
        ),
    )
    parser.add_argument("--float-control-steps", type=int, default=7_200)
    parser.add_argument("--retrieval-r0-steps", type=int, default=1_600)
    parser.add_argument("--fullcall-steps", type=int, default=7_200)
    parser.add_argument("--agent-steps", type=int, default=contract.AGENT_TRAIN_STEPS)
    parser.add_argument("--retrieval-r1-steps", type=int, default=1_600)
    parser.add_argument("--mw-steps", type=int, default=2_000)
    parser.add_argument("--confidence-train-limit", type=int, default=5_255)
    parser.add_argument("--confidence-valid-limit", type=int, default=2_020)
    parser.add_argument("--confidence-steps", type=int, default=800)
    parser.add_argument("--narration-steps", type=int, default=1_200)
    parser.add_argument("--narration-eval-limit", type=int, default=600)
    parser.add_argument("--eval-limit", type=int, default=1_176)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.endswith("steps") or name.endswith("limit"):
            if int(value) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    result: dict[str, Any] = {
        "plan": plan,
        "requested_run_dir": str(args.run_dir),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if plan["heavy_execution_deferred"]:
        result["status"] = "deferred_live_cpt"
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 75
    preflight_path, preflight = _validated_preflight(args, plan)
    plan["execution_preflight"] = {
        "path": str(preflight_path.resolve()),
        "sha256": contract.sha_file(preflight_path),
        "preflight_fingerprint_sha256": preflight[
            "preflight_fingerprint_sha256"
        ],
        "status": preflight["status"],
    }
    run_dir = lifecycle.choose_run_dir(args.run_dir, plan, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.write_json(run_dir / "plan.json", plan)
    progress = execute(args, plan, run_dir)
    lifecycle.write_json(run_dir / "progress.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
