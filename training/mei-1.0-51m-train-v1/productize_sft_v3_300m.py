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
    "package_v2_cq2_v4",
)


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
        Path(__file__).with_name("productize_51m.py"),
        Path(__file__).with_name("cq2_qat_51m.py"),
        Path(__file__).with_name("pack_cq2_v2_51m.py"),
        ROOT / "architecture/mei-1.0-51m-arch-v1/heads.py",
        ROOT / "sdk/python/mei_sdk/runtime_51m.py",
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
    run_fingerprint = contract.sha_bytes(contract.canonical_bytes(immutable))
    rows: list[dict[str, Any]] = []
    predecessor = base_identity["weights_sha256"]
    for index, stage in enumerate(STAGES, 1):
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


def _stage_paths(run_dir: Path) -> dict[str, Path]:
    return {
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
    paths = _stage_paths(run_dir)
    receipts: dict[str, dict[str, Any]] = {}

    def run(name: str, action: Any) -> dict[str, Any]:
        receipt = lifecycle.run_stage(
            run_dir=run_dir,
            plan=plan,
            name=name,
            action=action,
            resume=args.resume,
        )
        receipts[name] = receipt
        return receipt

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

    run("mw_disposition_v4", mw_action)

    confidence_train_path = run_dir / "stages/confidence_harvest_v4/train-outcomes.jsonl"
    confidence_valid_path = run_dir / "stages/confidence_harvest_v4/valid-outcomes.jsonl"
    confidence_dev_path = run_dir / "stages/confidence_harvest_v4/dev-outcomes.jsonl"

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

    package_dir = run_dir / "candidate" / args.package_id

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
            component: {
                "stage_id": stage,
                "stage_fingerprint_sha256": receipts[stage]["stage_fingerprint_sha256"],
                "status": "passed",
            }
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
            component: receipts[stage]["terminal_status"]
            for component, stage in {
                "lm": "oracle_top5_agent_v4",
                "contrastive": "retrieval_r1_v4",
                "mw_disposition": "mw_disposition_v4",
                "confidence": "confidence_head_v4",
                "narration_adapter": "narration_adapter_v4",
            }.items()
        }
        report["crossgen_schema_eval"] = {
            "stage_fingerprint_sha256": receipts["crossgen_schema_eval_v4"][
                "stage_fingerprint_sha256"
            ],
            "terminal_status": receipts["crossgen_schema_eval_v4"]["terminal_status"],
            "linguistic_augmentation_release_fingerprint": plan["immutable"][
                "linguistic_augmentation"
            ]["release_fingerprint"],
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
    run_dir = lifecycle.choose_run_dir(args.run_dir, plan, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.write_json(run_dir / "plan.json", plan)
    progress = execute(args, plan, run_dir)
    lifecycle.write_json(run_dir / "progress.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
