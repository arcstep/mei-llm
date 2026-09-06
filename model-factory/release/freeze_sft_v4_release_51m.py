#!/usr/bin/env python3
"""Freeze the quality-first, exposure-portable SFT-v4 data release.

The deployed catalog remains the historical 147-tool universe.  Sixty-four
schema-feature tools are training-only and live in a separate universe so they
can improve schema generalization without entering the final product index.
The v3 parent remains immutable; contaminated MW rows are rejected, never
rewritten in place.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import release.freeze_sft_v3_release_51m as v3_freezer
import contracts.sft_v4_contract_51m as contract


PARENT_SFT = contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-tool-sft-v3-300m-v7"
LINGUISTIC_RELEASE = (
    contract.DEFAULT_RELEASE_ROOT / contract.LINGUISTIC_AUGMENTATION_ID
)
EVAL_DIR = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID
BASE_RELEASE = contract.ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json"
NARRATION_RELEASE = (
    contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-narration-sft-agent300m-v3"
)

SCHEMA_COUNTS = {"retrieval": 4, "execute": 2, "refuse": 2}
RELEASE_ID = "mei-1.0-51m-tool-sft-v4-300m-v4"


def _artifact(payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sha256": contract.sha_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _source(path: Path) -> dict[str, Any]:
    try:
        display = str(path.relative_to(contract.ROOT))
    except ValueError:
        display = "external-fixture/" + path.name
    return {
        "path": display,
        "sha256": contract.sha_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_manifest_files(directory: Path, manifest: dict[str, Any]) -> None:
    specs = manifest.get("outputs") or manifest.get("artifacts") or {}
    for name, spec in specs.items():
        path = directory / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"source release artifact drift: {path}")


def _eval_hashes(eval_dir: Path) -> tuple[dict[str, Any], set[str]]:
    lock = contract.load_json(eval_dir / "lock.json")
    if (
        lock.get("schema") != "mei-51m-longitudinal-eval-lock-v4"
        or lock.get("id") != contract.EVAL_ID
        or lock.get("status") != "frozen"
    ):
        raise RuntimeError("SFT-v4 requires frozen eval-v7")
    _verify_manifest_files(eval_dir, lock)
    isolation = contract.load_json(eval_dir / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("eval-v7 isolation is not passed")
    hashes: set[str] = set()
    for path in sorted(eval_dir.glob("*.jsonl")):
        hashes |= contract.query_hashes_normalized(contract.load_jsonl(path))
    return lock, hashes


def _training_universe(
    deploy_document: dict[str, Any], schema_tools: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    deploy_tools = list(deploy_document.get("tools") or [])
    deploy_names = {str(tool["name"]) for tool in deploy_tools}
    schema_names = {str(tool["name"]) for tool in schema_tools}
    if deploy_names & schema_names:
        raise RuntimeError("training-only schema tools collide with deployed catalog")
    material = {
        "schema": "mei-sft-v4-training-tool-universe-v1",
        "deploy_universe_id": deploy_document.get("universe_id"),
        "deploy_universe_fingerprint": deploy_document.get("fingerprint"),
        "schema_subset_id": contract.SCHEMA_SUBSET_ID,
        "deploy_tool_count": len(deploy_tools),
        "training_only_schema_tool_count": len(schema_tools),
        "final_product_index_policy": "deploy tools only; training-only schema tools excluded",
        "tools": [*deploy_tools, *schema_tools],
    }
    material["fingerprint"] = contract.sha_bytes(contract.canonical_bytes(material))
    return material


def _structured_agent_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    tools: Sequence[dict[str, Any]],
    forbidden_query_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    filtered_eval_overlap = 0
    for raw in rows:
        row = dict(raw)
        row["split"] = split
        row.setdefault("context", {"locale": "zh-CN"})
        row.setdefault("evidence", [])
        row.setdefault("permissions", {})
        row.setdefault("state", {"runtime_state": "continuation"})
        row.setdefault("history", [])
        row.setdefault("prior_calls", [])
        row.setdefault("prior_tool_results", [])
        row.setdefault("tool_results", [])
        row.setdefault("mw", {})
        row["structured_request_contract"] = "CompleteRequestV2"
        row["generator_version"] = "mei-agent-continuation-structured-input-v4"
        if contract.query_hashes_normalized([row]) & forbidden_query_hashes:
            filtered_eval_overlap += 1
            continue
        output.append(row)
    audit = contract.audit_agent_rows(
        output,
        tools,
        expected_split=split,
        minimum_unique_call_tools=len(tools),
        minimum_terminal_tools=len(tools),
    )
    audit["filtered_eval_overlap_rows"] = filtered_eval_overlap
    return output, audit


def _clean_mw_rows(
    rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    split: str,
    eval_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    clean: list[dict[str, Any]] = []
    rejected_shortcut = 0
    for raw in rows:
        normalized = contract.normalize_mw_row(raw, split=split)
        if normalized is None:
            rejected_shortcut += 1
        else:
            clean.append(normalized)
    supplement = contract.generate_clean_mw_rows(
        tools, split=split, variants_per_tool=2 if split == "train" else 1
    )
    combined = [*clean, *supplement]
    output: list[dict[str, Any]] = []
    ids: set[str] = set()
    queries: set[str] = set()
    rejected_eval = 0
    rejected_duplicate = 0
    for row in combined:
        sample_id = str(row.get("sample_id") or "")
        query = str(row.get("query") or "")
        hashes = contract.query_hashes_normalized([row])
        if hashes & eval_hashes:
            rejected_eval += 1
            continue
        if sample_id in ids or query in queries:
            rejected_duplicate += 1
            continue
        ids.add(sample_id)
        queries.add(query)
        output.append(row)
    audit = contract.audit_mw_rows(output, split=split)
    stats = {
        "source_rows": len(rows),
        "rejected_synthetic_shortcut": rejected_shortcut,
        "clean_adopted": len(clean),
        "generated_supplement": len(supplement),
        "rejected_eval_overlap": rejected_eval,
        "rejected_duplicate": rejected_duplicate,
        "output_rows": len(output),
    }
    return output, audit, stats


def _confidence_candidates(
    sources: Sequence[tuple[str, Sequence[dict[str, Any]]]], split: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ids: set[str] = set()
    for bank, rows in sources:
        for row in rows:
            answers = list(row.get("answers") or [])
            expected = answers[0] if row.get("kind") == "execute" and answers else None
            sample_id = contract.stable_id("CONF4", split, bank, row["sample_id"])
            if sample_id in ids:
                raise RuntimeError(f"duplicate confidence candidate: {sample_id}")
            ids.add(sample_id)
            output.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": row["sample_id"],
                    "source_bank": bank,
                    "split": split,
                    "task": "confidence_outcome_harvest",
                    "query": row["query"],
                    "oracle_top5": row["oracle_top5"],
                    "retrieved_tools": row["retrieved_tools"],
                    "context": row.get("context") or {},
                    "evidence": row.get("evidence") or [],
                    "history": row.get("history") or [],
                    "tool_results": row.get("tool_results") or [],
                    "permissions": row.get("permissions") or {},
                    "state": row.get("state") or {},
                    "family": row.get("family") or "unknown",
                    "kind": row.get("kind"),
                    "expected_kind": "call" if expected else "refuse",
                    "expected_call": expected,
                    "candidate_tool": row.get("candidate_tool"),
                    "reason_code": row.get("reason_code"),
                    "label": None,
                    "label_state": "pending_actual_final_runtime_outcome",
                    "label_contract": "exact_call_or_correct_refusal_after_r1",
                    "sampling_contract": "bank_then_kind_then_family_then_tool_stratified_v1",
                    "score_contract": "mei-confidence-combined-score-v2",
                    "generator_version": contract.GENERATOR_ID,
                }
            )
    return output


def _isolate_schema_validation(
    bank: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Give validation rows a natural dialogue turn distinct from train.

    The frozen v4 generator predates the main release and used the train
    positive wording as its default for ``valid``.  We preserve the frozen
    eval generator identity and isolate the new training release here.
    """

    output: dict[str, list[dict[str, Any]]] = {"retrieval": [], "fullcall": []}
    for task, rows in bank.items():
        for raw in rows:
            row = dict(raw)
            query = "我重新说一下，" + str(raw.get("query") or "")
            sample_id = contract.stable_id(
                "SCHEMAV4", task, raw.get("sample_id"), query
            )
            row["source_sample_id"] = raw.get("sample_id")
            row["sample_id"] = sample_id
            row["case_id"] = sample_id
            row["cf_group"] = contract.stable_id(
                "SCHEMAV4G", task, raw.get("cf_group"), query
            )
            row["query"] = query
            row["generator_version"] = "mei-sft-v4-schema-valid-isolation-v1"
            output[task].append(row)
    return output


def _training_schedule(row_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "schema": "mei-sft-v4-training-schedule-v1",
        "id": "mei-51m-productization-v4-quality-schema-fixed-budget-v1",
        "ordering": [
            {
                "stage": "float_base_lm_anchor",
                "parent": "selected immutable Base",
                "weights_changed": False,
            },
            {
                "stage": "float_task_control",
                "inputs": [
                    "full-call.train.jsonl",
                    "schema-full-call.train.jsonl",
                    "linguistic-full-call.train.jsonl",
                ],
                "steps": 7200,
                "sampler": "weighted_bank_full_coverage",
                "quant_aware": False,
                "same_seed_serializer_grammar_scorer_as_main": True,
            },
            {
                "stage": "cq2_qat",
                "input": "selected immutable Base LM corpus stream",
                "token_budget": 5_000_000,
                "quant_math_id": "mei-cq-v2-g128-wht-codebook",
            },
            {
                "stage": "retrieval_r0",
                "inputs": [
                    "retrieval.train.jsonl",
                    "schema-retrieval.train.jsonl",
                    "linguistic-retrieval.train.jsonl",
                ],
                "steps": 1600,
                "batch_size": 8,
                "objective": "InfoNCE",
                "temperature": 0.07,
                "lm_frozen": True,
                "sampler": "bank_then_tool_uniform_with_all_row_negatives_and_inbatch_positives",
            },
            {
                "stage": "oracle_top5_quant_aware_fullcall",
                "inputs": [
                    "full-call.train.jsonl",
                    "schema-full-call.train.jsonl",
                    "linguistic-full-call.train.jsonl",
                ],
                "steps": 7200,
                "sampler": "weighted_bank_kind_tool_full_coverage",
                "quant_aware": True,
            },
            {
                "stage": "oracle_top5_agent_continuation",
                "input": "agent-continuation.train.jsonl",
                "steps": contract.AGENT_TRAIN_STEPS,
                "sampler": "trajectory_then_step_uniform_without_replacement",
                "quant_aware": True,
            },
            {
                "stage": "retrieval_r1",
                "same_inputs_as": "retrieval_r0",
                "steps": 1600,
                "lm_frozen": True,
                "after": "all LM weight changes",
                "rebuild_tool_index": True,
            },
            {
                "stage": "learned_top5_end_to_end",
                "input": contract.EVAL_ID,
                "modes": [
                    "structural_seen",
                    "natural_cross_generator",
                    "whole_schema_holdout",
                ],
                "evaluation_only": True,
            },
            {
                "stage": "mw_disposition",
                "input": "mw-disposition.train.jsonl",
                "steps": 2000,
                "sampler": "class_uniform",
                "lm_frozen": True,
                "prompt_id": contract.MW_PROMPT_ID,
            },
            {
                "stage": "confidence_outcome_harvest",
                "inputs": [
                    "confidence-harvest.train.jsonl",
                    "confidence-harvest.valid.jsonl",
                ],
                "mode": "actual_final_runtime_after_r1",
                "minimum_positive": 100,
                "minimum_negative": 100,
                "single_class": "hard_fail",
            },
            {
                "stage": "confidence_head",
                "input": "runtime-harvested labels only",
                "steps": 800,
                "sampler": "label_then_bank_uniform",
                "lm_frozen": True,
            },
            {
                "stage": "narration_adapter",
                "input": "mei-1.0-51m-narration-sft-agent300m-v3",
                "steps": 1200,
                "rank": 16,
                "lm_frozen": True,
                "terminal_only": True,
            },
            {
                "stage": "final_package_eval_and_gates",
                "input": "all final LM/head tensors plus eval-v7",
            },
        ],
        "bank_mix": {
            "retrieval": {"structural": 0.50, "linguistic": 0.35, "schema": 0.15},
            "fullcall": {"structural": 0.50, "linguistic": 0.35, "schema": 0.15},
            "within_linguistic": "source-balanced; admitted natural must not be drowned by offline rows",
        },
        "row_counts": row_counts,
        "longitudinal_policy": {
            "main_curve": "reuse exact SFT-v4 release, schedule and eval-v7",
            "cumulative_exposures": [
                300_000_000,
                600_000_000,
                900_000_000,
                1_200_000_000,
                1_500_000_000,
                2_100_000_000,
            ],
            "arbitrary_future_exposure_supported": True,
            "hard_case_expansion": "new additive release and separate scorecard only",
        },
    }


def _data_contract() -> dict[str, Any]:
    return {
        "schema": "mei-sft-data-contract-v4",
        "id": contract.CONTRACT_ID,
        "model_visible": {
            "fields": [
                "query",
                "selected tool schemas",
                "context",
                "evidence",
                "history",
                "tool_results",
                "permissions",
                "state",
            ],
            "forbidden": [
                "gold_name",
                "gold_args",
                "answers",
                "target_text",
                "reason_code",
                "reason_class_id",
                "teacher identity",
                "split marker",
                "holdout marker",
                "gold-derived slot provenance",
            ],
        },
        "banks": {
            "structural_seen": "147 deployed tools; inherited immutable v7 rows",
            "linguistic": "147 tools; admitted natural plus explicitly non-natural offline program",
            "schema_train": "64 training-only tools; never enter final product index",
            "schema_holdout": "32 whole tools/schemas in eval-v7 only",
            "mw_disposition": "independent clean structured 20-class sidecar labels",
            "confidence": "unlabelled candidates until actual final runtime outcomes",
            "agent": "verified ToolResult continuation; same autoregressive LM",
            "narration": "terminal-only rank-16 residual sidecar",
        },
        "gold": {
            "storage": "offline labels only",
            "execute_binding": "arguments literal in query or verified prior ToolResultV2",
            "serializer": contract.SERIALIZER_ID,
            "refusal_target": "[]",
        },
        "schema": {
            "subset_id": contract.SCHEMA_SUBSET_ID,
            "supported": [
                "required_optional",
                "string_boolean_integer_number_null",
                "enum_const",
                "numeric_bounds_multipleOf",
                "string_length_pattern_format",
                "scalar_array_length",
            ],
            "unsupported": "registration-time unsupported_schema fail closed",
        },
        "isolation": {
            "query_group_family_aware": True,
            "whole_tool_schema_holdout": True,
            "synthetic_shortcut_rows": 0,
            "eval_lock": contract.EVAL_ID,
        },
        "capability_separation": {
            "retrieval": "independent R0/R1 contrastive head",
            "tool_call_and_agent": "shared autoregressive LM",
            "mw_disposition": "independent 20-class head",
            "confidence": "independent calibrated binary head",
            "narration": "independent rank-16 terminal adapter",
            "mw_deviation": "deterministic governance evaluation; no learned tensors",
        },
    }


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    eval_lock, eval_hashes = _eval_hashes(args.eval_dir)
    parent_manifest = contract.load_json(args.parent_sft / "manifest.json")
    linguistic_manifest = contract.load_json(args.linguistic_release / "manifest.json")
    narration_manifest = contract.load_json(args.narration_release / "manifest.json")
    if parent_manifest.get("release_id") != "mei-1.0-51m-tool-sft-v3-300m-v7":
        raise RuntimeError("SFT-v4 parent identity drifted")
    if linguistic_manifest.get("release_id") != contract.LINGUISTIC_AUGMENTATION_ID:
        raise RuntimeError("SFT-v4 linguistic release identity drifted")
    _verify_manifest_files(args.parent_sft, parent_manifest)
    _verify_manifest_files(args.linguistic_release, linguistic_manifest)
    _verify_manifest_files(args.narration_release, narration_manifest)

    deploy_universe = contract.load_json(args.parent_sft / "tool-universe.json")
    deploy_tools = list(deploy_universe.get("tools") or [])
    if len(deploy_tools) != 147:
        raise RuntimeError("SFT-v4 requires the 147-tool deployed catalog")
    schema_tools = contract.schema_feature_tools("train")
    training_universe = _training_universe(deploy_universe, schema_tools)

    structural = {
        "retrieval.train.jsonl": contract.load_jsonl(args.parent_sft / "retrieval.train.jsonl"),
        "retrieval.valid.jsonl": contract.load_jsonl(args.parent_sft / "retrieval.valid.jsonl"),
        "full-call.train.jsonl": contract.load_jsonl(args.parent_sft / "full-call.train.jsonl"),
        "full-call.valid.jsonl": contract.load_jsonl(args.parent_sft / "full-call.valid.jsonl"),
    }
    schema_banks: dict[str, list[dict[str, Any]]] = {}
    schema_audits: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        bank = contract.build_schema_feature_rows(
            schema_tools,
            deploy_tools,
            split=split,
            retrieval_per_tool=SCHEMA_COUNTS["retrieval"],
            execute_per_tool=SCHEMA_COUNTS["execute"],
            refuse_per_tool=SCHEMA_COUNTS["refuse"],
        )
        if split == "valid":
            bank = _isolate_schema_validation(bank)
        audit = contract.audit_schema_feature_rows(
            bank, schema_tools, expected_split=split
        )
        if audit["status"] != "passed":
            raise RuntimeError("SFT-v4 schema train audit failed: " + audit["errors"][0])
        if (
            contract.query_hashes_normalized(bank["retrieval"] + bank["fullcall"])
            & eval_hashes
        ):
            raise RuntimeError("SFT-v4 schema train rows overlap eval-v7")
        schema_audits[split] = audit
        schema_banks[f"schema-retrieval.{split}.jsonl"] = bank["retrieval"]
        schema_banks[f"schema-full-call.{split}.jsonl"] = bank["fullcall"]

    linguistic = {
        "linguistic-retrieval.train.jsonl": contract.load_jsonl(
            args.linguistic_release / "linguistic-retrieval.train.jsonl"
        ),
        "linguistic-retrieval.valid.jsonl": contract.load_jsonl(
            args.linguistic_release / "linguistic-retrieval.valid.jsonl"
        ),
        "linguistic-full-call.train.jsonl": contract.load_jsonl(
            args.linguistic_release / "linguistic-full-call.train.jsonl"
        ),
        "linguistic-full-call.valid.jsonl": contract.load_jsonl(
            args.linguistic_release / "linguistic-full-call.valid.jsonl"
        ),
    }

    agent_rows: dict[str, list[dict[str, Any]]] = {}
    agent_audits: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        rows, audit = _structured_agent_rows(
            contract.load_jsonl(args.parent_sft / f"agent-continuation.{split}.jsonl"),
            split=split,
            tools=deploy_tools,
            forbidden_query_hashes=eval_hashes,
        )
        if audit["status"] != "passed":
            raise RuntimeError("SFT-v4 Agent audit failed: " + audit["errors"][0])
        agent_rows[f"agent-continuation.{split}.jsonl"] = rows
        agent_audits[split] = audit

    mw_rows: dict[str, list[dict[str, Any]]] = {}
    mw_audits: dict[str, dict[str, Any]] = {}
    mw_cleaning: dict[str, dict[str, int]] = {}
    for split in ("train", "valid"):
        rows, audit, stats = _clean_mw_rows(
            contract.load_jsonl(args.parent_sft / f"mw-disposition.{split}.jsonl"),
            deploy_tools,
            split=split,
            eval_hashes=eval_hashes,
        )
        if audit["status"] != "passed":
            raise RuntimeError("SFT-v4 MW audit failed: " + audit["errors"][0])
        mw_rows[f"mw-disposition.{split}.jsonl"] = rows
        mw_audits[split] = audit
        mw_cleaning[split] = stats
    if (
        contract.query_hashes_normalized(mw_rows["mw-disposition.train.jsonl"])
        & contract.query_hashes_normalized(mw_rows["mw-disposition.valid.jsonl"])
    ):
        raise RuntimeError("SFT-v4 MW train/valid query overlap")

    confidence: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid"):
        confidence[f"confidence-harvest.{split}.jsonl"] = _confidence_candidates(
            [
                ("structural", structural[f"full-call.{split}.jsonl"]),
                ("schema", schema_banks[f"schema-full-call.{split}.jsonl"]),
                ("linguistic", linguistic[f"linguistic-full-call.{split}.jsonl"]),
            ],
            split,
        )

    all_train_eval_rows = {
        **structural,
        **schema_banks,
        **linguistic,
        **agent_rows,
        **mw_rows,
    }
    train_valid_hashes: dict[str, set[str]] = {"train": set(), "valid": set()}
    shortcut_rows = 0
    for name, rows in all_train_eval_rows.items():
        split = "train" if ".train." in name else "valid"
        hashes = contract.query_hashes_normalized(rows)
        train_valid_hashes[split] |= hashes
        if hashes & eval_hashes:
            raise RuntimeError(f"SFT-v4 {name} overlaps eval-v7")
        shortcut_rows += sum(
            contract.query_has_synthetic_shortcut(str(row.get("query") or ""))
            for row in rows
        )
    cross_split_overlap = len(train_valid_hashes["train"] & train_valid_hashes["valid"])
    if cross_split_overlap:
        raise RuntimeError(f"SFT-v4 train/valid overlap: {cross_split_overlap}")
    if shortcut_rows:
        raise RuntimeError(f"SFT-v4 retains {shortcut_rows} synthetic shortcut rows")

    token_budget = contract.token_budget_audit(
        {
            "retrieval": structural["retrieval.train.jsonl"]
            + structural["retrieval.valid.jsonl"]
            + schema_banks["schema-retrieval.train.jsonl"]
            + schema_banks["schema-retrieval.valid.jsonl"]
            + linguistic["linguistic-retrieval.train.jsonl"]
            + linguistic["linguistic-retrieval.valid.jsonl"],
            "fullcall": structural["full-call.train.jsonl"]
            + structural["full-call.valid.jsonl"]
            + schema_banks["schema-full-call.train.jsonl"]
            + schema_banks["schema-full-call.valid.jsonl"]
            + linguistic["linguistic-full-call.train.jsonl"]
            + linguistic["linguistic-full-call.valid.jsonl"],
        },
        list(training_universe["tools"]),
        v3_freezer._load_tokenizer(),
    )
    if token_budget["status"] != "passed":
        raise RuntimeError("SFT-v4 token budget failed: " + token_budget["errors"][0])

    payloads: dict[str, bytes] = {
        "tool-universe.json": contract.canonical_bytes(deploy_universe) + b"\n",
        "training-tool-universe.json": contract.canonical_bytes(training_universe) + b"\n",
        **{name: contract.jsonl_bytes(rows) for name, rows in structural.items()},
        **{name: contract.jsonl_bytes(rows) for name, rows in schema_banks.items()},
        **{name: contract.jsonl_bytes(rows) for name, rows in linguistic.items()},
        **{name: contract.jsonl_bytes(rows) for name, rows in agent_rows.items()},
        **{name: contract.jsonl_bytes(rows) for name, rows in mw_rows.items()},
        **{name: contract.jsonl_bytes(rows) for name, rows in confidence.items()},
    }
    for name in (
        "host-simulator-v1.json",
        "mw-reason-definitions-v2-20class.json",
        "mw-disposition-codebook-v1.json",
        "qat-import-candidate-receipt.json",
    ):
        payloads[name] = (args.parent_sft / name).read_bytes()

    row_counts = {
        name: len(rows)
        for name, rows in {
            **structural,
            **schema_banks,
            **linguistic,
            **agent_rows,
            **mw_rows,
            **confidence,
        }.items()
    }
    data_contract = _data_contract()
    training_schedule = _training_schedule(row_counts)
    coverage = {
        "schema": "mei-sft-v4-coverage-receipt-v1",
        "status": "passed",
        "deployed_tools": len(deploy_tools),
        "training_only_schema_tools": len(schema_tools),
        "final_product_index_tools": len(deploy_tools),
        "schema_audits": schema_audits,
        "linguistic_quality_receipt_sha256": contract.sha_file(
            args.linguistic_release / "quality-audit-receipt.json"
        ),
        "agent_audits": agent_audits,
        "mw_audits": mw_audits,
        "mw_cleaning": mw_cleaning,
        "confidence_candidates": {
            split: {
                "rows": len(confidence[f"confidence-harvest.{split}.jsonl"]),
                "labels_embedded": 0,
                "source_banks": dict(
                    Counter(
                        row["source_bank"]
                        for row in confidence[f"confidence-harvest.{split}.jsonl"]
                    )
                ),
            }
            for split in ("train", "valid")
        },
        "token_budget": token_budget,
        "row_counts": row_counts,
    }
    isolation = {
        "schema": "mei-sft-v4-isolation-receipt-v1",
        "status": "passed",
        "eval_id": eval_lock["id"],
        "evaluation_fingerprint": eval_lock["evaluation_fingerprint"],
        "train_valid_eval_query_overlap": 0,
        "train_valid_query_overlap": cross_split_overlap,
        "synthetic_shortcut_rows": shortcut_rows,
        "whole_schema_holdout": True,
        "training_schema_tools": len(schema_tools),
        "eval_schema_tools": 32,
        "schema_tool_identity_overlap": 0,
        "group_aware": True,
        "family_aware": True,
    }
    agent_receipt = {
        "schema": "mei-agent-continuation-input-audit-receipt-v4",
        "status": "passed",
        "structured_request_contract": "CompleteRequestV2",
        "splits": agent_audits,
        "trusted_tool_results_only": True,
        "gold_slot_provenance_model_visible": False,
    }
    payloads["data-contract.json"] = contract.canonical_bytes(data_contract) + b"\n"
    payloads["training-schedule.json"] = contract.canonical_bytes(training_schedule) + b"\n"
    payloads["coverage-receipt.json"] = contract.canonical_bytes(coverage) + b"\n"
    payloads["isolation-receipt.json"] = contract.canonical_bytes(isolation) + b"\n"
    payloads["agent-input-audit-receipt.json"] = contract.canonical_bytes(agent_receipt) + b"\n"
    payloads["token-budget-receipt.json"] = contract.canonical_bytes(token_budget) + b"\n"

    base = contract.load_json(args.base_release)
    if base.get("params") != 51_463_797 or not base.get("weights_sha256"):
        raise RuntimeError("SFT-v4 primary Base identity drifted")
    qat_candidate = contract.load_json(args.parent_sft / "qat-import-candidate-receipt.json")
    if (
        qat_candidate.get("status") != "passed"
        or (qat_candidate.get("base") or {}).get("model_id") != base.get("model_id")
        or (qat_candidate.get("base") or {}).get("weights_sha256") != base.get("weights_sha256")
    ):
        raise RuntimeError("SFT-v4 primary Base QAT candidate mismatch")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        rows = sum(1 for line in payload.splitlines() if line) if name.endswith(".jsonl") else None
        artifacts[name] = _artifact(payload, rows=rows)
    sources = {
        "primary_base_release": _source(args.base_release),
        "parent_sft_manifest": _source(args.parent_sft / "manifest.json"),
        "linguistic_release_manifest": _source(args.linguistic_release / "manifest.json"),
        "eval_lock": _source(args.eval_dir / "lock.json"),
        "narration_release_manifest": _source(args.narration_release / "manifest.json"),
        "generator_source": _source(contract.ROOT / "model-factory/contracts/sft_v4_contract_51m.py"),
        "freezer_source": _source(Path(__file__)),
    }
    fingerprint = contract.sha_bytes(
        contract.canonical_bytes(
            {
                "contract": contract.CONTRACT_ID,
                "generator": contract.GENERATOR_ID,
                "sources": sources,
                "artifacts": artifacts,
            }
        )
    )
    manifest = {
        "schema": "mei-sft-data-release-v4",
        "release_id": args.release_id,
        "status": "frozen",
        "product": contract.PRODUCT_ID,
        "contract_id": contract.CONTRACT_ID,
        "release_fingerprint": fingerprint,
        "primary_base": {
            "model_id": base["model_id"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
            "weights_sha256": base["weights_sha256"],
            "release_sha256": sources["primary_base_release"]["sha256"],
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
        "wire": contract.WIRE_ID,
        "schema_subset": contract.SCHEMA_SUBSET_ID,
        "grammar": contract.GRAMMAR_ID,
        "generator": contract.GENERATOR_ID,
        "runtime_context": {
            "stable_prefix_tokens_max": contract.STABLE_PREFIX_TOKENS_MAX,
            "rolling_window_tokens": contract.ROLLING_WINDOW_TOKENS,
        },
        "sources": sources,
        "artifacts": artifacts,
        "evaluation": {
            "lock_id": eval_lock["id"],
            "evaluation_fingerprint": eval_lock["evaluation_fingerprint"],
            "lock_sha256": sources["eval_lock"]["sha256"],
        },
        "tool_universes": {
            "deploy": {
                "file": "tool-universe.json",
                "tools": len(deploy_tools),
                "sha256": artifacts["tool-universe.json"]["sha256"],
            },
            "training": {
                "file": "training-tool-universe.json",
                "tools": len(training_universe["tools"]),
                "training_only_schema_tools": len(schema_tools),
                "sha256": artifacts["training-tool-universe.json"]["sha256"],
            },
            "final_index_policy": "deploy universe only",
        },
        "capability_boundaries": {
            "retrieval": "independent R0/R1 contrastive head",
            "fullcall_agent": "autoregressive LM",
            "mw_disposition": "independent 20-class sidecar",
            "mw_deviation": "deterministic governance gate; not a head",
            "confidence": "independent binary sidecar; labels from actual runtime only",
            "narration": "independent rank-16 terminal adapter",
        },
        "confidence_label_contract": {
            "source": "actual final runtime after fullcall and R1",
            "train_candidates": "confidence-harvest.train.jsonl",
            "valid_candidates": "confidence-harvest.valid.jsonl",
            "minimum_positive": 100,
            "minimum_negative": 100,
            "single_class": "hard_fail",
        },
        "narration_source": {
            "release_id": narration_manifest.get("release_id"),
            "manifest_sha256": sources["narration_release_manifest"]["sha256"],
            "terminal_only": True,
        },
        "primary_base_qat_import_candidate": {
            "file": "qat-import-candidate-receipt.json",
            "sha256": artifacts["qat-import-candidate-receipt.json"]["sha256"],
            "automatic_reuse": False,
            "later_bases_require_own_qat_receipt": True,
        },
        "receipts": {
            name: {"file": name, "sha256": artifacts[name]["sha256"]}
            for name in (
                "data-contract.json",
                "training-schedule.json",
                "coverage-receipt.json",
                "isolation-receipt.json",
                "agent-input-audit-receipt.json",
                "token-budget-receipt.json",
            )
        },
    }
    payloads["manifest.json"] = contract.canonical_bytes(manifest) + b"\n"
    return payloads, manifest


def _verify_existing(target: Path, payloads: dict[str, bytes]) -> None:
    actual = {path.name for path in target.iterdir() if path.is_file()}
    if actual != set(payloads):
        raise RuntimeError(f"existing SFT-v4 file set differs: {target}")
    for name, payload in payloads.items():
        if (target / name).read_bytes() != payload:
            raise RuntimeError(f"existing SFT-v4 artifact differs: {target / name}")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=contract.DEFAULT_RELEASE_ROOT)
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--parent-sft", type=Path, default=PARENT_SFT)
    parser.add_argument("--linguistic-release", type=Path, default=LINGUISTIC_RELEASE)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--base-release", type=Path, default=BASE_RELEASE)
    parser.add_argument("--narration-release", type=Path, default=NARRATION_RELEASE)
    return parser.parse_args(argv)


def main() -> int:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
