#!/usr/bin/env python3
"""CPU-only exhaustive readiness check for the frozen mei-51m SFT-v4 graph."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import freeze_sft_v3_release_51m as freeze_sft
import longitudinal_eval_metrics_51m as longitudinal
import productize_sft_v3_300m as productizer
import sft_v4_contract_51m as contract
import sft_v3_training_51m as training
from _repo import CURRENT_PATH, ROOT


PREFLIGHT_ID = "mei-51m-sft-v4-exhaustive-preflight-v1-quality-schema"
CURRENT_BASELINE_SHA256 = (
    "5b0b68eeb8322bb9cdbef112777b1b346b69a91f3bce7234b1c6370389a42607"
)
DEFAULT_OUT_ROOT = ROOT / "notebook/evaluation/jobs/mei-1.0-51m"


def _quantiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return contract.load_jsonl(path)


def _encode_fullcall_bank(
    rows: list[dict[str, Any]], tokenizer: Any, tools_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    stable_lengths: list[int] = []
    ordinary_lengths: list[int] = []
    answer_lengths: list[int] = []
    for row in rows:
        prompt, answer, stats = training.encode_fullcall_row(
            tokenizer, row, tools_by_name
        )
        if len(prompt) != stats["prompt_tokens"] or len(answer) != stats["answer_tokens"]:
            raise RuntimeError(f"encoded length receipt mismatch: {row.get('sample_id')}")
        prompt_lengths.append(len(prompt))
        stable_lengths.append(int(stats["stable_prefix_tokens"]))
        ordinary_lengths.append(int(stats["ordinary_tokens_retained"]))
        answer_lengths.append(len(answer))
    return {
        "rows": len(rows),
        "prompt_tokens": _quantiles(prompt_lengths),
        "stable_prefix_tokens": _quantiles(stable_lengths),
        "ordinary_tokens_retained": _quantiles(ordinary_lengths),
        "answer_tokens": _quantiles(answer_lengths),
        "stable_prefix_over_budget": sum(
            value > contract.STABLE_PREFIX_TOKENS_MAX for value in stable_lengths
        ),
        "rolling_window_over_budget": sum(
            value > contract.ROLLING_WINDOW_TOKENS for value in ordinary_lengths
        ),
        "answer_over_budget": sum(value > 128 for value in answer_lengths),
    }


def _mw_bank(
    rows: list[dict[str, Any]], tokenizer: Any, tools_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    stable_lengths: list[int] = []
    ordinary_lengths: list[int] = []
    classes: set[int] = set()
    for row in rows:
        if row.get("retrieved_tools"):
            selected, label, _reason = training.mw_training_view(row, tools_by_name)
        else:
            names = [
                str(item.get("name") if isinstance(item, dict) else item)
                for item in row.get("oracle_top5") or []
            ]
            selected = [tools_by_name[name] for name in names if name in tools_by_name]
            label = int(row["reason_class_id"])
            if len(selected) != 5 or len({str(tool["name"]) for tool in selected}) != 5:
                raise RuntimeError(f"MW eval row lacks five oracle schemas: {row.get('sample_id')}")
        if not 0 <= label < 20:
            raise RuntimeError(f"MW label outside 20-class codebook: {row.get('sample_id')}")
        classes.add(label)
        rendered = training.render_mw_prompt_parts(row, selected)
        _ids, stats = training.encode_stable_ring_prompt(tokenizer, rendered)
        stable_lengths.append(int(stats["stable_prefix_tokens"]))
        ordinary_lengths.append(int(stats["ordinary_tokens_retained"]))
    return {
        "rows": len(rows),
        "classes": sorted(classes),
        "stable_prefix_tokens": _quantiles(stable_lengths),
        "ordinary_tokens_retained": _quantiles(ordinary_lengths),
        "stable_prefix_over_budget": sum(
            value > contract.STABLE_PREFIX_TOKENS_MAX for value in stable_lengths
        ),
        "rolling_window_over_budget": sum(
            value > contract.ROLLING_WINDOW_TOKENS for value in ordinary_lengths
        ),
    }


def _retrieval_bank(
    rows: list[dict[str, Any]], tokenizer: Any, tools_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids: set[str] = set()
    query_lengths: list[int] = []
    tools: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in ids:
            raise RuntimeError(f"retrieval bank has duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        gold = str(row.get("gold_tool") or "")
        catalog = [str(item.get("name") or "") for item in row.get("catalog_tools") or []]
        if gold not in tools_by_name or len(catalog) != 5 or len(set(catalog)) != 5 or gold not in catalog:
            raise RuntimeError(f"retrieval candidate contract failed: {sample_id}")
        tools.add(gold)
        query_lengths.append(
            len(tokenizer.encode(str(row.get("query") or ""), add_bos=True))
        )
    return {
        "rows": len(rows),
        "unique_gold_tools": len(tools),
        "query_tokens": _quantiles(query_lengths),
        "query_over_retrieval_budget": sum(
            value > contract.RETRIEVAL_MAX_TOKENS for value in query_lengths
        ),
    }


def _confidence_bank(
    rows: list[dict[str, Any]], tools_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids: set[str] = set()
    kinds: dict[str, int] = {"call": 0, "refuse": 0}
    tools: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in ids:
            raise RuntimeError(f"confidence bank has duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        if row.get("label") is not None:
            raise RuntimeError(f"confidence candidate embeds a synthetic label: {sample_id}")
        kind = str(row.get("expected_kind") or "")
        if kind not in kinds:
            raise RuntimeError(f"confidence candidate has invalid expected kind: {sample_id}")
        kinds[kind] += 1
        tool_name = str(row.get("candidate_tool") or "")
        if tool_name not in tools_by_name:
            raise RuntimeError(f"confidence candidate references unknown tool: {sample_id}")
        tools.add(tool_name)
        expected = row.get("expected_call")
        if kind == "call":
            if not isinstance(expected, dict) or expected.get("name") != tool_name:
                raise RuntimeError(f"confidence call target is malformed: {sample_id}")
            if not contract.arguments_match_schema(
                expected.get("arguments") or {}, tools_by_name[tool_name]["parameters"]
            ):
                raise RuntimeError(f"confidence call target violates schema: {sample_id}")
        elif expected is not None:
            raise RuntimeError(f"confidence refusal embeds a call target: {sample_id}")
    if not all(kinds.values()):
        raise RuntimeError("confidence candidate bank lacks call/refuse coverage")
    return {
        "rows": len(rows),
        "expected_kind_counts": kinds,
        "unique_candidate_tools": len(tools),
        "embedded_labels": 0,
    }


def _epoch_exposure(
    rows: list[dict[str, Any]],
    steps: int,
    order_fn: Callable[[Sequence[dict[str, Any]], int], list[int]],
) -> dict[str, Any]:
    used: set[int] = set()
    cache: dict[int, list[int]] = {}
    for step in range(steps):
        epoch = step // len(rows)
        order = cache.setdefault(epoch, order_fn(rows, epoch))
        used.add(order[step % len(rows)])
    return {
        "rows": len(rows),
        "steps": steps,
        "unique_rows_exposed": len(used),
        "all_rows_exposed": len(used) == len(rows),
    }


def _retrieval_exposure(
    rows: list[dict[str, Any]], steps: int, batch_size: int = 8
) -> dict[str, Any]:
    """Audit the exact CPU-only schedule used by ``train_retrieval_v3``."""

    by_tool: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_tool[str(row.get("gold_tool") or "")].append(index)
    tool_names = sorted(by_tool)
    if not tool_names or "" in by_tool:
        raise RuntimeError("retrieval exposure requires a complete tool partition")
    schedule = training.retrieval_training_schedule(rows, steps, batch_size)
    used = {index for batch in schedule for index in batch}
    selected_per_tool: dict[str, int] = defaultdict(int)
    for batch in schedule:
        for index in batch:
            selected_per_tool[str(rows[index].get("gold_tool") or "")] += 1
    unexposed = [
        str(row.get("sample_id") or index)
        for index, row in enumerate(rows)
        if index not in used
    ]
    return {
        "rows": len(rows),
        "steps": steps,
        "batch_size": batch_size,
        "unique_rows_exposed": len(used),
        "all_rows_exposed": not unexposed,
        "unexposed_sample_ids": unexposed[:100],
        "rows_per_tool": {
            "min": min(len(values) for values in by_tool.values()),
            "max": max(len(values) for values in by_tool.values()),
        },
        "selections_per_tool": {
            "min": min(selected_per_tool.values()),
            "max": max(selected_per_tool.values()),
        },
        "bank_exposures": {
            bank: sum(
                1
                for index in (value for batch in schedule for value in batch)
                if str(rows[index].get("_training_bank") or "") == bank
            )
            for bank in sorted(
                {str(row.get("_training_bank") or "") for row in rows} - {""}
            )
        },
    }


def _fullcall_exposure(rows: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    schedule = training.fullcall_training_schedule(rows, steps)
    used = set(schedule)
    return {
        "rows": len(rows),
        "steps": steps,
        "unique_rows_exposed": len(used),
        "all_rows_exposed": used == set(range(len(rows))),
        "bank_exposures": {
            bank: sum(
                str(rows[index].get("_training_bank") or "") == bank
                for index in schedule
            )
            for bank in sorted(training.SFT_V4_BANK_WEIGHTS)
        },
        "sampler": training.BANKED_FULLCALL_SAMPLER_ID,
    }


def _narration_release(path: Path) -> dict[str, Any]:
    manifest = contract.load_json(path / "manifest.json")
    files: dict[str, Any] = {}
    for name, spec in (manifest.get("files") or {}).items():
        artifact = path / name
        if not artifact.is_file() or contract.sha_file(artifact) != spec.get("sha256"):
            raise RuntimeError(f"narration artifact hash drift: {artifact}")
        rows = _rows(artifact)
        if len(rows) != int(spec.get("rows") or -1):
            raise RuntimeError(f"narration row count drift: {artifact}")
        files[name] = {"rows": len(rows), "sha256": spec["sha256"]}
    isolation = contract.load_json(path / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("narration isolation did not pass")
    return {
        "release_id": manifest.get("release_id"),
        "adapter": manifest.get("adapter"),
        "coverage": manifest.get("coverage"),
        "files": files,
        "manifest_sha256": contract.sha_file(path / "manifest.json"),
    }


def build_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    product_args = productizer.parse_args(
        [
            "--base-release",
            str(args.base_release),
            "--base-weights",
            str(args.base_weights),
            "--qat-import-receipt",
            str(args.qat_import_receipt),
            "--data-release",
            str(args.data_release),
            "--linguistic-augmentation",
            str(args.linguistic_augmentation),
            "--eval-lock",
            str(args.eval_lock),
            "--narration-release",
            str(args.narration_release),
        ]
    )
    plan = productizer.build_plan(product_args)
    release = training.verify_release_contract(args.data_release)
    linguistic = contract.load_json(args.linguistic_augmentation / "manifest.json")
    linguistic_quality = contract.load_json(
        args.linguistic_augmentation / "quality-audit-receipt.json"
    )
    linguistic_budget = contract.load_json(
        args.linguistic_augmentation / "token-budget-receipt.json"
    )
    lock = longitudinal.verify_lock(args.eval_lock)
    current_sha = contract.sha_file(CURRENT_PATH)
    if current_sha != CURRENT_BASELINE_SHA256:
        raise RuntimeError("CURRENT.json drifted from the protected baseline")

    universe = contract.load_json(args.data_release / "tool-universe.json")
    deploy_tools = [contract.compact_tool(tool) for tool in universe.get("tools") or []]
    deploy_by_name = {str(tool["name"]): tool for tool in deploy_tools}
    training_universe = contract.load_json(
        args.data_release / "training-tool-universe.json"
    )
    training_tools = [
        contract.compact_tool(tool) for tool in training_universe.get("tools") or []
    ]
    training_by_name = {str(tool["name"]): tool for tool in training_tools}
    holdout_universe = contract.load_json(
        args.eval_lock / "schema-holdout-tool-universe.json"
    )
    holdout_tools = [
        contract.compact_tool(tool) for tool in holdout_universe.get("tools") or []
    ]
    schema_eval_tools = [*deploy_tools, *holdout_tools]
    schema_eval_by_name = {str(tool["name"]): tool for tool in schema_eval_tools}
    if (
        len(deploy_by_name) != 147
        or len(training_by_name) != 211
        or len(schema_eval_by_name) != 179
    ):
        raise RuntimeError("preflight tool universe counts drifted")
    tokenizer = freeze_sft._load_tokenizer()

    structural_fullcall_train = productizer._with_training_bank(
        _rows(args.data_release / "full-call.train.jsonl"), "structural"
    )
    schema_fullcall_train = productizer._with_training_bank(
        _rows(args.data_release / "schema-full-call.train.jsonl"), "schema"
    )
    linguistic_fullcall_train = productizer._with_training_bank(
        _rows(args.linguistic_augmentation / "linguistic-full-call.train.jsonl")
    )
    structural_retrieval_train = productizer._with_training_bank(
        _rows(args.data_release / "retrieval.train.jsonl"), "structural"
    )
    schema_retrieval_train = productizer._with_training_bank(
        _rows(args.data_release / "schema-retrieval.train.jsonl"), "schema"
    )
    linguistic_retrieval_train = productizer._with_training_bank(
        _rows(args.linguistic_augmentation / "linguistic-retrieval.train.jsonl")
    )
    fullcall_banks = {
        "train": [
            *structural_fullcall_train,
            *schema_fullcall_train,
            *linguistic_fullcall_train,
        ],
        "structural_train": structural_fullcall_train,
        "schema_train": schema_fullcall_train,
        "linguistic_train": linguistic_fullcall_train,
        "valid": [
            *_rows(args.data_release / "full-call.valid.jsonl"),
            *_rows(args.data_release / "schema-full-call.valid.jsonl"),
            *_rows(args.linguistic_augmentation / "linguistic-full-call.valid.jsonl"),
        ],
        "structural_dev": _rows(args.eval_lock / "fullcall.dev.jsonl"),
        "structural_test": _rows(args.eval_lock / "fullcall.test.jsonl"),
        "natural_dev": _rows(args.eval_lock / "natural-fullcall.dev.jsonl"),
        "natural_test": _rows(args.eval_lock / "natural-fullcall.test.jsonl"),
        "schema_dev": _rows(args.eval_lock / "schema-fullcall.dev.jsonl"),
        "schema_test": _rows(args.eval_lock / "schema-fullcall.test.jsonl"),
    }
    agent_banks = {
        "train": _rows(args.data_release / "agent-continuation.train.jsonl"),
        "valid": _rows(args.data_release / "agent-continuation.valid.jsonl"),
        "dev": _rows(args.eval_lock / "multistep.dev.jsonl"),
        "test": _rows(args.eval_lock / "multistep.test.jsonl"),
    }
    retrieval_banks = {
        "train": [
            *structural_retrieval_train,
            *schema_retrieval_train,
            *linguistic_retrieval_train,
        ],
        "structural_train": structural_retrieval_train,
        "schema_train": schema_retrieval_train,
        "linguistic_train": linguistic_retrieval_train,
        "valid": [
            *_rows(args.data_release / "retrieval.valid.jsonl"),
            *_rows(args.data_release / "schema-retrieval.valid.jsonl"),
            *_rows(args.linguistic_augmentation / "linguistic-retrieval.valid.jsonl"),
        ],
        "structural_dev": _rows(args.eval_lock / "retrieval.dev.jsonl"),
        "structural_test": _rows(args.eval_lock / "retrieval.test.jsonl"),
        "natural_dev": _rows(args.eval_lock / "natural-retrieval.dev.jsonl"),
        "natural_test": _rows(args.eval_lock / "natural-retrieval.test.jsonl"),
        "schema_dev": _rows(args.eval_lock / "schema-retrieval.dev.jsonl"),
        "schema_test": _rows(args.eval_lock / "schema-retrieval.test.jsonl"),
    }
    mw_banks = {
        "train": _rows(args.data_release / "mw-disposition.train.jsonl"),
        "valid": _rows(args.data_release / "mw-disposition.valid.jsonl"),
        "dev": _rows(args.eval_lock / "mw.dev.jsonl"),
        "test": _rows(args.eval_lock / "mw.test.jsonl"),
    }
    confidence_banks = {
        "train": _rows(args.data_release / "confidence-harvest.train.jsonl"),
        "valid": _rows(args.data_release / "confidence-harvest.valid.jsonl"),
        "dev": [
            *_rows(args.eval_lock / "confidence.dev.jsonl"),
            *_rows(args.eval_lock / "schema-confidence.dev.jsonl"),
        ],
        "test": [
            *_rows(args.eval_lock / "confidence.test.jsonl"),
            *_rows(args.eval_lock / "schema-confidence.test.jsonl"),
        ],
    }

    fullcall_encoding = {}
    for split, rows in fullcall_banks.items():
        mapping = (
            schema_eval_by_name
            if split.startswith("schema_") and split not in {"schema_train"}
            else training_by_name
            if split in {"train", "valid", "schema_train"}
            else deploy_by_name
        )
        fullcall_encoding[split] = _encode_fullcall_bank(rows, tokenizer, mapping)
    agent_encoding = {
        split: _encode_fullcall_bank(rows, tokenizer, deploy_by_name)
        for split, rows in agent_banks.items()
    }
    retrieval = {}
    for split, rows in retrieval_banks.items():
        mapping = (
            schema_eval_by_name
            if split.startswith("schema_") and split not in {"schema_train"}
            else training_by_name
            if split in {"train", "valid", "schema_train"}
            else deploy_by_name
        )
        retrieval[split] = _retrieval_bank(rows, tokenizer, mapping)
    mw = {
        split: _mw_bank(rows, tokenizer, deploy_by_name)
        for split, rows in mw_banks.items()
    }
    confidence = {
        split: _confidence_bank(
            rows,
            training_by_name if split in {"train", "valid"} else schema_eval_by_name,
        )
        for split, rows in confidence_banks.items()
    }

    for group in (fullcall_encoding, agent_encoding, mw):
        for report in group.values():
            if report.get("stable_prefix_over_budget") or report.get(
                "rolling_window_over_budget"
            ) or report.get("answer_over_budget"):
                raise RuntimeError("an encoded SFT bank exceeds a frozen token budget")
    expected_retrieval_tools = {
        "train": 211,
        "structural_train": 147,
        "schema_train": 64,
        "linguistic_train": 147,
        "valid": 211,
        "structural_dev": 147,
        "structural_test": 147,
        "schema_dev": 32,
        "schema_test": 32,
    }
    for split, report in retrieval.items():
        if report["query_over_retrieval_budget"]:
            raise RuntimeError("a retrieval query exceeds the frozen retrieval budget")
        expected = expected_retrieval_tools.get(split)
        if expected is not None and report["unique_gold_tools"] != expected:
            raise RuntimeError(
                f"retrieval split {split} covers {report['unique_gold_tools']} tools, expected {expected}"
            )
    expected_confidence_tools = {"train": 211, "valid": 211, "dev": 179, "test": 179}
    for split, report in confidence.items():
        if report["unique_candidate_tools"] != expected_confidence_tools[split]:
            raise RuntimeError(f"confidence split {split} tool coverage drifted")
    if set(mw["train"]["classes"]) != set(range(20)) or set(mw["valid"]["classes"]) != set(
        range(20)
    ):
        raise RuntimeError("MW train/valid does not cover the 20-class codebook")

    sampler_exposure = {
        "fullcall": _fullcall_exposure(
            fullcall_banks["train"], product_args.fullcall_steps
        ),
        "retrieval_r0": _retrieval_exposure(
            retrieval_banks["train"],
            product_args.retrieval_r0_steps,
        ),
        "retrieval_r1": _retrieval_exposure(
            retrieval_banks["train"],
            product_args.retrieval_r1_steps,
        ),
        "agent": _epoch_exposure(
            agent_banks["train"],
            product_args.agent_steps,
            training.agent_epoch_order,
        ),
    }
    if not all(value["all_rows_exposed"] for value in sampler_exposure.values()):
        raise RuntimeError("fixed SFT step budget does not expose every training row")

    script_sha = contract.sha_file(Path(__file__))
    body = {
        "schema": "mei-51m-sft-v4-exhaustive-preflight-receipt-v1",
        "status": "passed",
        "preflight_id": PREFLIGHT_ID,
        "product": contract.PRODUCT_ID,
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "inputs": {
            "base_release_sha256": contract.sha_file(args.base_release),
            "base_weights_sha256": contract.sha_file(args.base_weights),
            "qat_import_receipt_sha256": contract.sha_file(
                args.qat_import_receipt
            ),
            "data_release_id": release["release_id"],
            "data_manifest_sha256": contract.sha_file(args.data_release / "manifest.json"),
            "data_release_fingerprint": release["release_fingerprint"],
            "linguistic_augmentation_id": linguistic["release_id"],
            "linguistic_augmentation_manifest_sha256": contract.sha_file(
                args.linguistic_augmentation / "manifest.json"
            ),
            "linguistic_augmentation_fingerprint": linguistic[
                "release_fingerprint"
            ],
            "linguistic_quality_receipt_sha256": contract.sha_file(
                args.linguistic_augmentation / "quality-audit-receipt.json"
            ),
            "linguistic_token_budget_receipt_sha256": contract.sha_file(
                args.linguistic_augmentation / "token-budget-receipt.json"
            ),
            "eval_lock_id": lock["id"],
            "eval_lock_sha256": contract.sha_file(args.eval_lock / "lock.json"),
            "evaluation_fingerprint": lock["evaluation_fingerprint"],
            "current_sha256": current_sha,
        },
        "contracts": plan["immutable"]["contracts"],
        "source_sha256": script_sha,
        "fullcall_encoding": fullcall_encoding,
        "agent_encoding": agent_encoding,
        "retrieval": retrieval,
        "linguistic_augmentation": {
            "manifest": plan["immutable"]["linguistic_augmentation"],
            "quality": linguistic_quality,
            "token_budget": linguistic_budget,
        },
        "mw_disposition": mw,
        "mw_prompt_id": training.MW_PROMPT_ID,
        "confidence_candidates": confidence,
        "narration": _narration_release(args.narration_release),
        "sampler_exposure": sampler_exposure,
        "invariants": {
            "mlx_imported": False,
            "metal_claimed": False,
            "current_mutated": False,
            "frozen_release_mutated": False,
            "all_hashes_verified": True,
            "all_lm_rows_encoded": True,
            "all_fixed_step_lm_rows_exposed": True,
        },
    }
    fingerprint = contract.sha_bytes(contract.canonical_bytes(body))
    return {**body, "preflight_fingerprint_sha256": fingerprint}, fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, default=productizer.DEFAULT_BASE_RELEASE)
    parser.add_argument("--base-weights", type=Path, default=productizer.DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        "--qat-import-receipt",
        type=Path,
        default=productizer.DEFAULT_QAT_IMPORT_RECEIPT,
    )
    parser.add_argument("--data-release", type=Path, default=productizer.DEFAULT_DATA_RELEASE)
    parser.add_argument(
        "--linguistic-augmentation",
        "--natural-augmentation",
        dest="linguistic_augmentation",
        type=Path,
        default=productizer.DEFAULT_LINGUISTIC_AUGMENTATION,
    )
    parser.add_argument("--eval-lock", type=Path, default=productizer.DEFAULT_EVAL_LOCK)
    parser.add_argument(
        "--narration-release", type=Path, default=productizer.DEFAULT_NARRATION_RELEASE
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt, fingerprint = build_receipt(args)
    output = args.out or (
        DEFAULT_OUT_ROOT
        / f"sft-v4-v4-eval-v7-preflight-{fingerprint[:12]}.json"
    )
    contract.write_json_once(output, receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(output),
                "preflight_fingerprint_sha256": fingerprint,
                "run_fingerprint_sha256": receipt["run_fingerprint_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
