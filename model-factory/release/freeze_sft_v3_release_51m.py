#!/usr/bin/env python3
"""Freeze the balanced, exposure-portable SFT-v3 release for mei-1.0-51m.

The first consumer is the immutable 300M Base.  The corpus itself is bound to
the 51M weight contract and zh-24k-v1 tokenizer rather than to a token exposure
allowlist, so the exact same release and budget can form the main longitudinal
comparison for later 600M..2.1B Bases.

Confidence files in this release are *harvest candidates*, not labels.  Labels
must be generated from actual final-runtime outcomes after R1 and full-call
training.  This prevents the historical single-class calibration failure from
being baked into the data release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import contracts.sft_v3_contract_51m as contract


BASE_RELEASE_PATH = (
    contract.ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json"
)
QAT_STAGE_DIR = (
    contract.ROOT
    / "artifacts/mei-1.0-51m/legacy/exp-000300m/runs/productize-scratch300m-agent-cq2-v2-ff204182428e"
    / "stages/cq2_qat_v2"
)
QAT_MASTER_PATH = QAT_STAGE_DIR / "worker/stages/cq2_qat_v2/final-master.npz"
QAT_WORKER_RECEIPT_PATH = QAT_STAGE_DIR / "worker/stages/cq2_qat_v2/receipt.json"
QAT_MASTER_SHA256 = "b76ed28923fb315e84a9716cbd2a767f5b09ead9eae4b4c8227219d9055f5268"


def _artifact_spec(payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": contract.sha_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _source_spec(path: Path) -> dict[str, Any]:
    try:
        display_path = str(path.relative_to(contract.ROOT))
    except ValueError:
        display_path = "external-fixture/" + path.name
    return {
        "path": display_path,
        "sha256": contract.sha_file(path),
        "bytes": path.stat().st_size,
    }


def _all_eval_text_hashes(eval_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted(eval_dir.glob("*.jsonl")):
        result |= contract.query_hashes(contract.load_jsonl(path))
    return result


def _copy_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    rows = contract.load_jsonl(path)
    return rows, path.read_bytes()


def _load_tokenizer() -> Any:
    architecture_dir = contract.ROOT / "models/mei-1.0-51m/architecture"
    text = str(architecture_dir)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from tokenizer import ZhTokenizerV1

        return ZhTokenizerV1()
    finally:
        if added:
            sys.path.remove(text)


def _normalize_agent_rows(
    source_rows: list[dict[str, Any]],
    split: str,
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in source_rows:
        row = dict(raw)
        row["source_sample_id"] = raw.get("sample_id")
        row["source_generator_version"] = raw.get("generator_version")
        row["permissions"] = {}
        row["slot_provenance"] = []
        row["split"] = split
        row["prompt_framing"] = contract.PROMPT_FRAMING_ID
        row["generator_version"] = "mei-agent-continuation-portable-input-v2"
        answers = row.get("answers") or []
        row["target_text"] = contract.serialize_tool_target(answers)
        output.append(row)
    audit = contract.audit_agent_rows(
        output,
        tools,
        expected_split=split,
        minimum_unique_call_tools=len(tools),
        minimum_terminal_tools=len(tools),
    )
    return output, audit


MW_GENERATED_CLASS_IDS = {
    "ready_to_execute": 0,
    "missing_slot": 1,
    "ambiguous_scope": 4,
    "mixed_intent": 5,
    "unknown_slot_value": 16,
    "offtopic": 17,
    "negation_cancels": 18,
}


def _normalize_mw_rows(
    rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        output.append(
            {
                "sample_id": raw.get("sample_id"),
                "source_sample_id": raw.get("sample_id"),
                "case_id": raw.get("case_id") or raw.get("sample_id"),
                "cf_group": raw.get("cf_group") or raw.get("case_id") or raw.get("sample_id"),
                "task": "mw_disposition",
                "split": split,
                "family": raw.get("family") or "unknown",
                "kind": raw.get("kind") or "unknown",
                "query": raw.get("query") or "",
                "system_facts": raw.get("system_facts") or raw.get("scene") or "",
                "reason_code": raw.get("reason_code"),
                "reason_class_id": int(raw["reason_class_id"]),
                "retrieved_tools": list(raw.get("retrieved_tools") or []),
                "generator_version": "mei-mw-disposition-clean-input-v3",
                "source_generator_version": raw.get("generator_version"),
                "status": "accepted",
            }
        )
    return output


def _generated_mw_rows(
    fullcall_rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in fullcall_rows:
        reason = str(raw.get("reason_code") or "")
        if reason not in MW_GENERATED_CLASS_IDS:
            raise RuntimeError(f"unmapped generated MW reason: {reason}")
        sample_id = contract.stable_id("MW3", split, raw["sample_id"])
        output.append(
            {
                "sample_id": sample_id,
                "source_sample_id": raw["sample_id"],
                "case_id": sample_id,
                "cf_group": raw.get("cf_group") or sample_id,
                "task": "mw_disposition",
                "split": split,
                "family": raw.get("family") or "unknown",
                "kind": raw.get("kind"),
                "query": raw.get("query"),
                "system_facts": "",
                "candidate_tool": raw.get("candidate_tool"),
                "reason_code": reason,
                "reason_class_id": MW_GENERATED_CLASS_IDS[reason],
                "retrieved_tools": list(raw.get("retrieved_tools") or []),
                "generator_version": "mei-sft-v3-fullcall-to-mw-disposition-v1",
                "status": "accepted",
            }
        )
    return output


def _confidence_candidates(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        expected_call = None
        answers = row.get("answers") or []
        if row.get("kind") == "execute" and answers:
            expected_call = answers[0]
        output.append(
            {
                "sample_id": contract.stable_id("CONF3", split, row["sample_id"]),
                "source_sample_id": row["sample_id"],
                "split": split,
                "task": "confidence_outcome_harvest",
                "query": row["query"],
                "oracle_top5": row["oracle_top5"],
                "retrieved_tools": row["retrieved_tools"],
                "permissions": {},
                "slot_provenance": [],
                "family": row.get("family") or "unknown",
                "kind": row.get("kind"),
                "expected_kind": "call" if expected_call else "refuse",
                "expected_call": expected_call,
                "candidate_tool": row.get("candidate_tool"),
                "reason_code": row.get("reason_code"),
                "label": None,
                "label_state": "pending_actual_frozen_runtime_outcome",
                "label_contract": "exact_call_or_correct_refusal_after_r1",
                "sampling_contract": "kind_then_family_then_tool_stratified_v1",
                "score_contract": "mei-confidence-combined-score-v2",
                "generator_version": contract.GENERATOR_ID,
            }
        )
    return output


def _training_schedule() -> dict[str, Any]:
    return {
        "schema": "mei-sft-v3-training-schedule-v3",
        "id": "mei-51m-productization-v3-fixed-budget-v3",
        "sampler": contract.SAMPLER_ID,
        "ordering": [
            {
                "stage": "float_task_control",
                "input": "full-call.valid.jsonl",
                "steps": 512,
                "quant_aware": False,
                "purpose": "trained float control from the same immutable Base",
            },
            {
                "stage": "cq2_qat",
                "input": "immutable Base LM corpus stream",
                "token_budget": 5_000_000,
                "purpose": "quantization adaptation before task SFT",
                "reuse_policy": "reuse only when Base, corpus, quant math, contracts and output SHA match",
            },
            {
                "stage": "retrieval_r0",
                "input": "retrieval.train.jsonl",
                "steps": 1200,
                "batch_size": 8,
                "sampler": "tool_uniform_all_four_hard_negatives_plus_inbatch_positives",
                "objective": "InfoNCE",
                "temperature": 0.07,
                "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
                "lm_frozen": True,
            },
            {
                "stage": "oracle_top5_fullcall_single_step",
                "input": "full-call.train.jsonl",
                "steps": 4000,
                "sampler": "kind_then_tool_uniform_without_replacement_per_epoch",
                "quant_aware": True,
            },
            {
                "stage": "oracle_top5_agent_continuation",
                "input": "agent-continuation.train.jsonl",
                "steps": contract.AGENT_TRAIN_STEPS,
                "sampler": "trajectory_then_step_uniform_without_replacement",
                "quant_aware": True,
                "terminal_success_tool_coverage": 147,
                "cross_tool_chain_source": "reviewed_historical_bank_only",
            },
            {
                "stage": "retrieval_r1",
                "input": "retrieval.train.jsonl",
                "steps": 1200,
                "batch_size": 8,
                "sampler": "tool_uniform_all_four_hard_negatives_plus_inbatch_positives",
                "objective": "InfoNCE",
                "temperature": 0.07,
                "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
                "lm_frozen": True,
                "after": "all LM weight changes",
            },
            {
                "stage": "learned_top5_end_to_end",
                "input": contract.EVAL_ID + " dev",
                "mode": "evaluation_only",
            },
            {
                "stage": "mw_disposition",
                "input": "mw-disposition.train.jsonl",
                "steps": 2000,
                "sampler": "class_uniform",
                "lm_frozen": True,
                "semantic_boundary": "20_class_sidecar_not_mw_deviation_gate",
                "evaluation_modes": ["oracle_top5_pure_head", "learned_top5_combined"],
            },
            {
                "stage": "confidence_outcome_harvest",
                "input": "confidence-harvest.train.jsonl",
                "mode": "actual_final_runtime",
                "minimum_positive": 100,
                "minimum_negative": 100,
                "single_class": "hard_fail",
                "sampler": "kind_then_family_then_tool_stratified_v1",
                "score_contract": "mei-confidence-combined-score-v2",
            },
            {
                "stage": "confidence_head",
                "input": "runtime-harvested labels only",
                "steps": 800,
                "sampler": "label_uniform",
                "lm_frozen": True,
            },
            {
                "stage": "narration_adapter",
                "input": "mei-1.0-51m-narration-sft-agent300m-v3",
                "steps": 1200,
                "rank": 16,
                "lm_frozen": True,
                "must_retrain_after_final_lm_change": True,
            },
            {
                "stage": "final_package_and_longitudinal_test",
                "input": "all final LM/head tensors and frozen test lock",
                "mode": "terminal_receipts",
            },
        ],
        "longitudinal_policy": {
            "exposures": [
                300_000_000,
                600_000_000,
                900_000_000,
                1_200_000_000,
                1_500_000_000,
                2_100_000_000,
            ],
            "main_curve": "reuse this exact release and schedule",
            "arbitrary_exposure_supported": True,
            "adaptive_budget_experiments": "separate branch and separate scorecard only",
        },
    }


def _data_contract() -> dict[str, Any]:
    return {
        "schema": "mei-sft-data-contract-v3",
        "id": contract.CONTRACT_ID,
        "model_visible": {
            "single_step": [
                "query",
                "portable selected tool schemas",
                "real context/evidence/history/tool_results/permissions/state",
            ],
            "forbidden": [
                "gold_name",
                "gold_args",
                "answers",
                "target_text",
                "reason_code",
                "reason_class_id",
                "gold-derived slot_provenance",
            ],
            "permissions_default": {},
        },
        "gold": {
            "storage": "offline labels only",
            "execute_binding": "every argument must be literal in query or a verified prior ToolResultV2",
            "target_serializer": contract.SERIALIZER_ID,
            "refusal_target": "[]",
        },
        "retrieval": {
            "encoding_id": contract.RETRIEVAL_ENCODING_ID,
            "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
            "tool_uniform": True,
            "hard_negatives_per_row": 4,
            "distinct_negatives_per_tool_train": 27,
            "objective": "InfoNCE temperature=0.07 with all row negatives and in-batch positives",
        },
        "fullcall": {
            "execute_refuse_balanced_per_tool": True,
            "optional_argument_variants": ["required_only", "one_optional_present"],
            "negative_reason_codes": sorted(
                {
                    "negation_cancels",
                    "missing_slot",
                    "ambiguous_scope",
                    "unknown_slot_value",
                    "mixed_intent",
                    "offtopic",
                }
            ),
            "prompt_framing_id": contract.PROMPT_FRAMING_ID,
            "stable_prefix_tokens_max": contract.STABLE_PREFIX_TOKENS_MAX,
            "rolling_window_tokens": contract.ROLLING_WINDOW_TOKENS,
        },
        "agent_continuation": {
            "reviewed_cross_tool_chains_retained": True,
            "terminal_success_tool_coverage": 147,
            "generated_cross_tool_chains": False,
            "trusted_result_required": True,
            "trajectory_terminal_required": True,
        },
        "capability_separation": {
            "retrieval": "independent contrastive R0/R1 head",
            "tool_call_and_agent": "shared autoregressive LM",
            "mw_disposition": "independent 20-class sidecar",
            "confidence": "independent binary sidecar on actual final-runtime outcomes",
            "narration": "independent rank-16 terminal generation adapter",
            "mw_deviation": "deterministic governance gate with zero learned tensors",
        },
        "isolation": {
            "group_aware": True,
            "family_aware": True,
            "train_valid_eval_query_overlap": 0,
            "forbidden_eval_markers": True,
        },
        "longitudinal": {
            "main_curve": "same release, sampler and fixed budget for every immutable Base exposure",
            "supported_exposures": [
                300_000_000,
                600_000_000,
                900_000_000,
                1_200_000_000,
                1_500_000_000,
                2_100_000_000,
            ],
            "arbitrary_future_exposure_supported": True,
            "adaptive_budget_branch": "separate release and scorecard only",
        },
    }


def _verify_eval_lock(eval_dir: Path) -> dict[str, Any]:
    lock = contract.load_json(eval_dir / "lock.json")
    if (
        lock.get("schema") != "mei-51m-longitudinal-eval-lock-v3"
        or lock.get("id") != contract.EVAL_ID
        or lock.get("status") != "frozen"
    ):
        raise RuntimeError("SFT-v3 requires the frozen longitudinal eval lock")
    for name, spec in (lock.get("artifacts") or {}).items():
        path = eval_dir / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"longitudinal eval artifact drift: {path}")
    isolation = contract.load_json(eval_dir / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("longitudinal eval isolation failed")
    return lock


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    eval_lock = _verify_eval_lock(args.eval_dir)
    projected_universe, projection_receipt = contract.portable_universe_document(
        args.universe
    )
    frozen_universe = contract.load_json(args.eval_dir / "tool-universe.json")
    if contract.canonical_bytes(projected_universe) != contract.canonical_bytes(
        frozen_universe
    ):
        raise RuntimeError("raw tool universe projection differs from eval-v2 lock")
    tools = list(frozen_universe.get("tools") or [])
    tools_by_name = {str(tool["name"]): tool for tool in tools}
    eval_hashes = _all_eval_text_hashes(args.eval_dir)
    train = contract.build_single_step_rows(
        tools,
        split="train",
        retrieval_per_tool=contract.TRAIN_RETRIEVAL_PER_TOOL,
        execute_per_tool=contract.TRAIN_EXECUTE_PER_TOOL,
        refuse_per_tool=contract.TRAIN_REFUSE_PER_TOOL,
    )
    train_audit = contract.audit_single_step_rows(
        train,
        tools,
        expected_split="train",
        min_retrieval_per_tool=contract.TRAIN_RETRIEVAL_PER_TOOL,
        min_execute_per_tool=contract.TRAIN_EXECUTE_PER_TOOL,
        min_refuse_per_tool=contract.TRAIN_REFUSE_PER_TOOL,
        forbidden_query_hashes=eval_hashes,
    )
    if train_audit["status"] != "passed":
        raise RuntimeError("SFT-v3 train coverage/isolation failed: " + train_audit["errors"][0])
    valid = contract.build_single_step_rows(
        tools,
        split="valid",
        retrieval_per_tool=contract.VALID_RETRIEVAL_PER_TOOL,
        execute_per_tool=contract.VALID_EXECUTE_PER_TOOL,
        refuse_per_tool=contract.VALID_REFUSE_PER_TOOL,
    )
    valid_audit = contract.audit_single_step_rows(
        valid,
        tools,
        expected_split="valid",
        min_retrieval_per_tool=contract.VALID_RETRIEVAL_PER_TOOL,
        min_execute_per_tool=contract.VALID_EXECUTE_PER_TOOL,
        min_refuse_per_tool=contract.VALID_REFUSE_PER_TOOL,
        forbidden_query_hashes=eval_hashes | set(train_audit["query_hashes"]),
    )
    if valid_audit["status"] != "passed":
        raise RuntimeError("SFT-v3 valid coverage/isolation failed: " + valid_audit["errors"][0])
    token_budget = contract.token_budget_audit(
        {
            "retrieval": train["retrieval"] + valid["retrieval"],
            "fullcall": train["fullcall"] + valid["fullcall"],
        },
        tools,
        _load_tokenizer(),
    )
    if token_budget["status"] != "passed":
        raise RuntimeError("SFT-v3 token budget failed: " + token_budget["errors"][0])

    parent = args.parent_release
    copied_paths = {
        "agent-continuation.train.jsonl": parent / "agent-continuation.train.jsonl",
        "agent-continuation.valid.jsonl": parent / "agent-continuation.valid.jsonl",
        "mw-disposition.train.jsonl": parent / "mw-disposition.train.jsonl",
        "mw-disposition.valid.jsonl": parent / "mw-disposition.valid.jsonl",
        "host-simulator-v1.json": parent / "host-simulator-v1.json",
        "agent-isolation-receipt.json": parent / "agent-isolation-receipt.json",
        "mw-deviation-audit-receipt.json": parent / "mw-deviation-audit-receipt.json",
        "mw-reason-definitions-v2-20class.json": parent / "mw-reason-definitions-v2-20class.json",
        "mw-disposition-codebook-v1.json": (
            contract.ROOT
            / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/recipes/mw-disposition-codebook-v1.json"
        ),
    }
    payloads: dict[str, bytes] = {
        "tool-universe.json": contract.canonical_bytes(frozen_universe) + b"\n",
        "tool-universe-projection-receipt.json": (
            contract.canonical_bytes(projection_receipt) + b"\n"
        ),
        "token-budget-receipt.json": contract.canonical_bytes(token_budget) + b"\n",
        "retrieval.train.jsonl": contract.jsonl_bytes(train["retrieval"]),
        "retrieval.valid.jsonl": contract.jsonl_bytes(valid["retrieval"]),
        "full-call.train.jsonl": contract.jsonl_bytes(train["fullcall"]),
        "full-call.valid.jsonl": contract.jsonl_bytes(valid["fullcall"]),
        "confidence-harvest.train.jsonl": contract.jsonl_bytes(
            _confidence_candidates(train["fullcall"], "train")
        ),
        "confidence-harvest.valid.jsonl": contract.jsonl_bytes(
            _confidence_candidates(valid["fullcall"], "valid")
        ),
        "data-contract.json": contract.canonical_bytes(_data_contract()) + b"\n",
        "training-schedule.json": contract.canonical_bytes(_training_schedule()) + b"\n",
    }
    copied_rows: dict[str, list[dict[str, Any]]] = {}
    agent_audits: dict[str, dict[str, Any]] = {}
    for name, path in copied_paths.items():
        if name.startswith("agent-continuation."):
            split = name.split(".")[1]
            source = contract.load_jsonl(path)
            variants_per_tool = (
                contract.TRAIN_AGENT_TERMINAL_PER_TOOL
                if split == "train"
                else contract.VALID_AGENT_TERMINAL_PER_TOOL
            )
            source.extend(
                contract.build_agent_terminal_rows(
                    tools,
                    split=split,
                    variants_per_tool=variants_per_tool,
                )
            )
            rows, audit = _normalize_agent_rows(source, split, tools)
            if audit["status"] != "passed":
                raise RuntimeError("Agent input audit failed: " + audit["errors"][0])
            agent_audits[split] = audit
            payload = contract.jsonl_bytes(rows)
            copied_rows[name] = rows
            payloads[name] = payload
        elif name.startswith("mw-disposition."):
            split = name.split(".")[1]
            rows = _normalize_mw_rows(contract.load_jsonl(path), split)
            supplement = train["fullcall"] if split == "train" else valid["fullcall"]
            rows.extend(_generated_mw_rows(supplement, split))
            payload = contract.jsonl_bytes(rows)
            copied_rows[name] = rows
            payloads[name] = payload
        elif path.suffix == ".jsonl":
            rows, payload = _copy_rows(path)
            copied_rows[name] = rows
            payloads[name] = payload
        else:
            payloads[name] = path.read_bytes()

    simulator = contract.load_json(parent / "host-simulator-v1.json")
    simulator["coverage_extension"] = {
        "id": "mei-agent-terminal-success-coverage-v1",
        "tool_count": len(tools),
        "train_trajectories_per_tool": contract.TRAIN_AGENT_TERMINAL_PER_TOOL,
        "valid_trajectories_per_tool": contract.VALID_AGENT_TERMINAL_PER_TOOL,
        "semantics": "one exact call then one verified successful result then terminal respond",
        "cross_tool_chains": "reviewed_historical_bank_only",
    }
    payloads["host-simulator-v1.json"] = contract.canonical_bytes(simulator) + b"\n"

    agent_input_audit = {
        "schema": "mei-agent-continuation-input-audit-receipt-v3",
        "status": "passed",
        "splits": agent_audits,
        "trusted_tool_results_only": True,
        "gold_slot_provenance_model_visible": False,
        "permissions_default": {},
    }
    payloads["agent-input-audit-receipt.json"] = (
        contract.canonical_bytes(agent_input_audit) + b"\n"
    )

    mw_train = copied_rows["mw-disposition.train.jsonl"]
    mw_valid = copied_rows["mw-disposition.valid.jsonl"]
    mw_train_counts = Counter(int(row["reason_class_id"]) for row in mw_train)
    mw_valid_counts = Counter(int(row["reason_class_id"]) for row in mw_valid)
    if set(mw_train_counts) != set(range(20)) or set(mw_valid_counts) != set(range(20)):
        raise RuntimeError("MW train/valid must cover all 20 disposition classes")
    mw_generated_train_tools = {
        str(row.get("candidate_tool"))
        for row in mw_train
        if row.get("generator_version") == "mei-sft-v3-fullcall-to-mw-disposition-v1"
    }
    mw_generated_valid_tools = {
        str(row.get("candidate_tool"))
        for row in mw_valid
        if row.get("generator_version") == "mei-sft-v3-fullcall-to-mw-disposition-v1"
    }
    if mw_generated_train_tools != set(tools_by_name) or mw_generated_valid_tools != set(
        tools_by_name
    ):
        raise RuntimeError("MW generated supplement must cover all 147 candidate tools")
    agent_train = copied_rows["agent-continuation.train.jsonl"]
    agent_valid = copied_rows["agent-continuation.valid.jsonl"]
    auxiliary_overlap = {
        "mw_train_eval": len(contract.query_hashes(mw_train) & eval_hashes),
        "agent_train_eval": len(contract.query_hashes(agent_train) & eval_hashes),
        "mw_train_valid": len(
            contract.query_hashes(mw_train) & contract.query_hashes(mw_valid)
        ),
        "agent_train_valid": len(
            contract.query_hashes(agent_train) & contract.query_hashes(agent_valid)
        ),
    }
    if any(auxiliary_overlap.values()):
        raise RuntimeError(f"SFT-v3 auxiliary isolation failed: {auxiliary_overlap}")

    legacy_retrieval = contract.load_jsonl(parent / "retrieval.train.jsonl")
    legacy_fullcall = contract.load_jsonl(parent / "full-call.train.jsonl")
    legacy_retrieval_tools = {
        str(row.get("gold_tool")) for row in legacy_retrieval if row.get("gold_tool")
    }
    legacy_fullcall_tools = {
        str(row.get("gold_name")) for row in legacy_fullcall if row.get("gold_name")
    }
    coverage = {
        "schema": "mei-sft-v3-coverage-receipt-v2",
        "status": "passed",
        "historical_v2": {
            "retrieval_rows": len(legacy_retrieval),
            "retrieval_unique_gold_tools": len(legacy_retrieval_tools),
            "fullcall_rows": len(legacy_fullcall),
            "fullcall_unique_gold_tools": len(legacy_fullcall_tools),
            "tool_universe_size": len(tools),
        },
        "v3": {
            "train": {
                key: value for key, value in train_audit.items() if key != "query_hashes"
            },
            "valid": {
                key: value for key, value in valid_audit.items() if key != "query_hashes"
            },
            "mw_train_class_counts": {
                str(key): value for key, value in sorted(mw_train_counts.items())
            },
            "mw_valid_class_counts": {
                str(key): value for key, value in sorted(mw_valid_counts.items())
            },
            "mw_train_rows": len(mw_train),
            "mw_valid_rows": len(mw_valid),
            "mw_generated_train_unique_candidate_tools": len(mw_generated_train_tools),
            "mw_generated_valid_unique_candidate_tools": len(mw_generated_valid_tools),
            "agent_train_rows": len(agent_train),
            "agent_train_trajectories": agent_audits["train"]["trajectories"],
            "agent_train_unique_call_tools": agent_audits["train"][
                "unique_call_tools"
            ],
            "agent_train_unique_terminal_tools": agent_audits["train"][
                "unique_terminal_tools"
            ],
            "agent_noninitial_result_rows": sum(
                1 for row in agent_train if row.get("tool_results")
            ),
            "agent_input_audits": agent_audits,
            "token_budget": token_budget,
            "portable_projection": projection_receipt["output"],
            "confidence_harvest": {
                "train_candidates": len(train["fullcall"]),
                "valid_candidates": len(valid["fullcall"]),
                "labels_embedded": 0,
                "sampling_contract": "kind_then_family_then_tool_stratified_v1",
            },
        },
        "fixed_defects": [
            "retrieval gold coverage 25/147 plus no-match -> 147/147",
            "full-call gold coverage 31/147 -> 147/147",
            "single-step execute/refuse balanced per tool",
            "each retrieval row retains the nearest available negative and rotates catalog distractors",
            "16 retrieval variants cover 27 distinct hard negatives per tool",
            "retrieval schema encoding expanded from truncating 96 to audited 384 tokens",
            "gold-derived slot provenance removed from every model-visible row",
            "empty permission arrays replaced by unrestricted permission objects",
            "portable schema projection replaces non-portable regex escapes",
            "multi-step continuation is a dedicated training stage instead of an appended tail",
            "Agent arguments require query or verified ToolResultV2 grounding",
            "MW inputs stripped of teacher metadata and augmented across all 147 tools",
            "confidence labels deferred to actual final-runtime outcomes",
            "MW disposition remains an independent 20-class sidecar",
            "configurable tools no longer receive executable empty-argument no-op labels",
            "zero-argument tools no longer receive inapplicable missing-slot or unknown-slot refusals",
            "mechanical double-imperative Chinese templates are rejected",
            "each tool/variant execute-refuse pair shares one counterfactual isolation group",
            "Agent call-to-result-to-terminal coverage expanded from 11 to 147 tools without inventing cross-tool semantics",
        ],
    }
    isolation = {
        "schema": "mei-sft-v3-isolation-receipt-v2",
        "status": "passed",
        "eval_lock_id": eval_lock["id"],
        "evaluation_fingerprint": eval_lock["evaluation_fingerprint"],
        "train_valid_eval_query_overlap": 0,
        "auxiliary_overlap": auxiliary_overlap,
        "group_aware": True,
        "family_aware": True,
        "marker_contamination": 0,
        "gold_slot_provenance_model_visible_rows": 0,
        "portable_tool_universe_fingerprint": frozen_universe["fingerprint"],
    }
    payloads["coverage-receipt.json"] = contract.canonical_bytes(coverage) + b"\n"
    payloads["isolation-receipt.json"] = contract.canonical_bytes(isolation) + b"\n"

    base = contract.load_json(args.base_release)
    if (
        base.get("params") != 51_463_797
        or base.get("tokens_seen_exposure") != 300_000_485
        or not base.get("weights_sha256")
    ):
        raise RuntimeError("primary 300M Base identity drifted")
    qat_worker = contract.load_json(QAT_WORKER_RECEIPT_PATH)
    actual_qat_sha = contract.sha_file(QAT_MASTER_PATH)
    qat_import_ok = (
        qat_worker.get("terminal_status") == "passed"
        and qat_worker.get("parent_id") == base.get("model_id")
        and qat_worker.get("quant_math_id") == "mei-cq-v2-g128-wht-codebook"
        and (qat_worker.get("outputs") or {}).get("master_sha256") == QAT_MASTER_SHA256
        and actual_qat_sha == QAT_MASTER_SHA256
        and int((qat_worker.get("metrics") or {}).get("tokens_seen_qat") or 0)
        >= 5_000_000
    )
    if not qat_import_ok:
        raise RuntimeError("300M CQ2-QAT reuse candidate failed strict identity checks")
    qat_import = {
        "schema": "mei-cq2-qat-import-candidate-receipt-v1",
        "status": "passed",
        "base": {
            "model_id": base["model_id"],
            "weights_sha256": base["weights_sha256"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
        },
        "quant_math_id": qat_worker["quant_math_id"],
        "master": {
            "path": str(QAT_MASTER_PATH.relative_to(contract.ROOT)),
            "sha256": actual_qat_sha,
            "bytes": QAT_MASTER_PATH.stat().st_size,
        },
        "worker_receipt": {
            "path": str(QAT_WORKER_RECEIPT_PATH.relative_to(contract.ROOT)),
            "sha256": contract.sha_file(QAT_WORKER_RECEIPT_PATH),
        },
        "tokens_seen_qat": (qat_worker.get("metrics") or {}).get("tokens_seen_qat"),
        "valid_loss": (qat_worker.get("metrics") or {}).get("valid_loss"),
        "reuse_policy": (
            "candidate only; productizer must revalidate Base, corpus, quant math, "
            "contracts, stage fingerprint and output SHA before import"
        ),
        "later_base_policy": "600M and later immutable Bases require their own CQ2-QAT stage",
    }
    payloads["qat-import-candidate-receipt.json"] = (
        contract.canonical_bytes(qat_import) + b"\n"
    )

    outputs: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        rows = None
        if name.endswith(".jsonl"):
            rows = sum(1 for line in payload.splitlines() if line)
        outputs[name] = _artifact_spec(payload, rows=rows)

    source_specs = {
        "primary_base_release": _source_spec(args.base_release),
        "parent_sft_manifest": _source_spec(parent / "manifest.json"),
        "longitudinal_eval_lock": _source_spec(args.eval_dir / "lock.json"),
        "tool_universe": _source_spec(args.universe),
        "narration_release_manifest": _source_spec(
            args.narration_release / "manifest.json"
        ),
        "qat_worker_receipt": _source_spec(QAT_WORKER_RECEIPT_PATH),
        "generator_source": _source_spec(
            contract.ROOT / "model-factory/contracts/sft_v3_contract_51m.py"
        ),
        "freezer_source": _source_spec(Path(__file__)),
    }
    release_fingerprint = contract.sha_bytes(
        contract.canonical_bytes(
            {
                "contract": contract.CONTRACT_ID,
                "generator": contract.GENERATOR_ID,
                "sampler": contract.SAMPLER_ID,
                "sources": source_specs,
                "outputs": outputs,
            }
        )
    )
    manifest = {
        "schema": "mei-sft-data-release-v3",
        "release_id": args.release_id,
        "status": "frozen",
        "product": contract.PRODUCT_ID,
        "contract_id": contract.CONTRACT_ID,
        "release_fingerprint": release_fingerprint,
        "primary_base": {
            "model_id": base["model_id"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
            "weights_sha256": base["weights_sha256"],
            "release_sha256": source_specs["primary_base_release"]["sha256"],
        },
        "base_compatibility": {
            "weight_contract_id": contract.WEIGHT_CONTRACT_ID,
            "deployed_parameter_count": 51_463_797,
            "tokenizer_id": contract.TOKENIZER_ID,
            "tokenizer_sha256": base["tokenizer_sha256"],
            "exposure_allowlist": None,
            "arbitrary_cumulative_exposure_supported": True,
            "require_immutable_base_and_matching_contract": True,
        },
        "serializer": contract.SERIALIZER_ID,
        "prompt_framing": contract.PROMPT_FRAMING_ID,
        "retrieval_encoding": {
            "id": contract.RETRIEVAL_ENCODING_ID,
            "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        },
        "runtime_context": {
            "stable_prefix_tokens_max": contract.STABLE_PREFIX_TOKENS_MAX,
            "rolling_window_tokens": contract.ROLLING_WINDOW_TOKENS,
        },
        "wire": contract.WIRE_ID,
        "schema_subset": contract.SCHEMA_SUBSET_ID,
        "grammar": contract.GRAMMAR_ID,
        "generator": contract.GENERATOR_ID,
        "sampler": contract.SAMPLER_ID,
        "sources": source_specs,
        "outputs": outputs,
        "evaluation": {
            "lock_id": eval_lock["id"],
            "evaluation_fingerprint": eval_lock["evaluation_fingerprint"],
            "lock_sha256": source_specs["longitudinal_eval_lock"]["sha256"],
        },
        "capability_boundaries": {
            "retrieval": "independent contrastive head R0/R1",
            "fullcall": "LM constrained tool-call/refusal generation",
            "agent_continuation": "same LM, dedicated quant-aware continuation stage",
            "mw_disposition": "independent 20-class sidecar",
            "mw_deviation": "deterministic governance gate; never a learned head",
            "confidence": "independent binary sidecar trained only on actual runtime outcomes",
            "narration": "independent rank-16 terminal adapter; no executor access",
        },
        "agent_continuation": {
            "schema": "mei-agent-continuation-sft-v2",
            "simulator_id": "mei-agent-host-simulator-v1",
            "train_file": "agent-continuation.train.jsonl",
            "valid_file": "agent-continuation.valid.jsonl",
            "eval_bank": contract.EVAL_ID + "/multistep.test.jsonl",
            "target_kinds": ["call", "respond"],
            "unique_call_tools": agent_audits["train"]["unique_call_tools"],
            "unique_terminal_tools": agent_audits["train"]["unique_terminal_tools"],
            "terminal_success_coverage": "147_of_147_tools",
            "generated_cross_tool_chains": False,
            "nonempty_tool_result_rows": sum(
                1 for row in agent_train if row.get("tool_results")
            ),
            "group_aware": True,
            "family_aware": True,
            "simulator_sha256": outputs["host-simulator-v1.json"]["sha256"],
            "isolation_receipt_sha256": outputs["agent-isolation-receipt.json"]["sha256"],
            "input_audit_receipt_sha256": outputs["agent-input-audit-receipt.json"]["sha256"],
        },
        "mw_disposition": {
            "n_classes": 20,
            "class_ids": list(range(20)),
            "codebook_file": "mw-disposition-codebook-v1.json",
            "codebook_sha256": outputs["mw-disposition-codebook-v1.json"]["sha256"],
            "definitions_file": "mw-reason-definitions-v2-20class.json",
            "definitions_sha256": outputs["mw-reason-definitions-v2-20class.json"]["sha256"],
            "trainable_sidecar": True,
        },
        "mw_deviation": {
            "kind": "governance_gate_not_head",
            "receipt": "mw-deviation-audit-receipt.json",
            "receipt_sha256": outputs["mw-deviation-audit-receipt.json"]["sha256"],
            "trainable": False,
            "tensor_count": 0,
        },
        "narration_source": {
            "release_id": contract.load_json(args.narration_release / "manifest.json").get("release_id"),
            "manifest_sha256": source_specs["narration_release_manifest"]["sha256"],
            "retrain_on_final_lm": True,
        },
        "confidence_label_contract": {
            "source": "actual final runtime after fullcall and R1",
            "train_candidates": "confidence-harvest.train.jsonl",
            "valid_candidates": "confidence-harvest.valid.jsonl",
            "minimum_positive": 100,
            "minimum_negative": 100,
            "single_class": "hard_fail",
            "sampling": "kind_then_family_then_tool_stratified_v1",
            "score_contract": "mei-confidence-combined-score-v2",
        },
        "portable_tool_universe": {
            "file": "tool-universe.json",
            "sha256": outputs["tool-universe.json"]["sha256"],
            "fingerprint": frozen_universe["fingerprint"],
            "projection_receipt": "tool-universe-projection-receipt.json",
            "projection_receipt_sha256": outputs[
                "tool-universe-projection-receipt.json"
            ]["sha256"],
        },
        "data_contract": {
            "file": "data-contract.json",
            "sha256": outputs["data-contract.json"]["sha256"],
        },
        "token_budget_receipt": {
            "file": "token-budget-receipt.json",
            "sha256": outputs["token-budget-receipt.json"]["sha256"],
        },
        "primary_base_qat_import_candidate": {
            "file": "qat-import-candidate-receipt.json",
            "sha256": outputs["qat-import-candidate-receipt.json"]["sha256"],
            "master_sha256": QAT_MASTER_SHA256,
            "automatic_reuse": False,
        },
        "coverage_receipt": {
            "file": "coverage-receipt.json",
            "sha256": outputs["coverage-receipt.json"]["sha256"],
        },
        "isolation_receipt": {
            "file": "isolation-receipt.json",
            "sha256": outputs["isolation-receipt.json"]["sha256"],
        },
        "training_schedule": {
            "file": "training-schedule.json",
            "sha256": outputs["training-schedule.json"]["sha256"],
        },
    }
    payloads["manifest.json"] = contract.canonical_bytes(manifest) + b"\n"
    return payloads, manifest


def _verify_existing(target: Path, payloads: dict[str, bytes]) -> None:
    actual = {path.name for path in target.iterdir() if path.is_file()}
    if actual != set(payloads):
        raise RuntimeError(f"existing SFT-v3 release file set differs: {target}")
    for name, payload in payloads.items():
        if (target / name).read_bytes() != payload:
            raise RuntimeError(f"existing SFT-v3 artifact differs: {target / name}")


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    payloads, manifest = build_payloads(args)
    args.release_root.mkdir(parents=True, exist_ok=True)
    target = args.release_root / args.release_id
    if target.exists():
        _verify_existing(target, payloads)
        return {
            "ok": True,
            "reused": True,
            "path": str(target),
            "release_fingerprint": manifest["release_fingerprint"],
        }
    with tempfile.TemporaryDirectory(prefix=f".{args.release_id}-", dir=args.release_root) as name:
        temporary = Path(name) / args.release_id
        temporary.mkdir()
        for filename, payload in payloads.items():
            (temporary / filename).write_bytes(payload)
        temporary.replace(target)
    return {
        "ok": True,
        "reused": False,
        "path": str(target),
        "release_fingerprint": manifest["release_fingerprint"],
        "artifacts": len(payloads),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=contract.DEFAULT_RELEASE_ROOT)
    parser.add_argument("--release-id", default=contract.RELEASE_ID)
    parser.add_argument("--parent-release", type=Path, default=contract.PARENT_RELEASE_DIR)
    parser.add_argument("--eval-dir", type=Path, default=contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID)
    parser.add_argument("--universe", type=Path, default=contract.TOOL_UNIVERSE_PATH)
    parser.add_argument("--narration-release", type=Path, default=contract.NARRATION_RELEASE_DIR)
    parser.add_argument("--base-release", type=Path, default=BASE_RELEASE_PATH)
    return parser.parse_args()


def main() -> int:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
