#!/usr/bin/env python3
"""Freeze the exposure-independent mei-51m longitudinal evaluation lock.

The lock is created before SFT-v3 training and is reused unchanged for the
300M, 600M, 900M, 1.2B, 1.5B and 2.1B Base comparisons.  It contains complete
147-tool retrieval/full-call coverage plus a fixed confidence-candidate bank
and the already isolated MW, multi-step, narration and Base-language banks.
The script never mutates a historical lock or an existing non-identical
target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import sft_v3_contract_51m as contract


METRIC_CONTRACT = {
    "schema": "mei-51m-longitudinal-metric-contract-v3",
    "id": "mei-51m-longitudinal-metrics-v3",
    "comparison_unit": {
        "base": "immutable Base weights SHA-256 plus cumulative exposure",
        "product": "Base SHA-256 plus SFT release SHA-256 plus recipe fingerprint",
        "required_pairing": "same eval lock, SFT release, sampler and training budget",
    },
    "base_language": {
        "metrics": [
            "validation_nll",
            "domain_nll_hq",
            "domain_nll_colloquial",
            "domain_nll_structure",
            "probe_mean_nll",
            "probe_exact_match",
            "utf8_copy_number_nll",
            "date_json_instruction_nll",
        ],
        "direction": "lower_nll_higher_exact",
    },
    "retrieval": {
        "encoding_id": contract.RETRIEVAL_ENCODING_ID,
        "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        "training_objective": "tool_uniform_inbatch_infonce_temperature_0.07",
        "hard_negative_sampler": contract.SAMPLER_ID,
        "metrics": [
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "ndcg_at_5",
            "lexical_recall_at_5",
            "learned_minus_lexical_recall_at_5",
        ],
        "slices": ["family", "tool", "similar_name", "catalog_size"],
        "release_floor": {"recall_at_5": 0.60, "mrr": 0.35},
    },
    "fullcall": {
        "prompt_framing_id": contract.PROMPT_FRAMING_ID,
        "serializer_id": contract.SERIALIZER_ID,
        "stable_prefix_tokens_max": contract.STABLE_PREFIX_TOKENS_MAX,
        "rolling_window_tokens": contract.ROLLING_WINDOW_TOKENS,
        "modes": ["oracle_top5", "learned_top5", "no_retrieval"],
        "metrics": [
            "execute_tool_name_exact",
            "execute_arguments_exact",
            "execute_schema_valid",
            "refusal_accuracy",
            "false_refuse_rate",
            "false_execute_rate",
            "balanced_accuracy",
        ],
        "required_baselines": ["all_refuse", "always_first_tool", "lexical_top1"],
        "release_floor": {
            "execute_tool_name_exact": 0.50,
            "execute_arguments_exact": 0.40,
            "refusal_accuracy": 0.70,
            "balanced_accuracy": 0.60,
        },
    },
    "mw_disposition": {
        "semantic_boundary": "independent_20_class_sidecar_not_mw_deviation_gate",
        "metrics": ["accuracy", "macro_f1", "per_class_precision_recall", "confusion"],
        "safety_metric": "class_0_false_continue_rate",
        "release_floor": {"macro_f1": 0.50, "class_0_false_continue_rate_max": 0.10},
    },
    "confidence": {
        "label_source": "actual_frozen_runtime_outcome_only",
        "candidate_bank": "confidence.{split}.jsonl",
        "candidate_identity_policy": "exact_sample_id_coverage_no_extra_no_missing",
        "score_contract": {
            "id": "mei-confidence-combined-score-v2",
            "decode_probability": "exp(min(0, logprob_sum / max(output_tokens, 1)))",
            "raw_score": "min(sigmoid(head_logit), decode_probability)",
            "calibration_input": "raw_score",
        },
        "minimum_class_rows": {"positive": 100, "negative": 100},
        "metrics": ["auroc", "auprc", "ece_10", "brier", "risk_coverage"],
        "single_class_policy": "hard_fail",
        "release_floor": {"auroc": 0.60, "ece_10_max": 0.15},
    },
    "multi_step": {
        "candidate_bank": "multistep.{split}.jsonl",
        "minimum_unique_call_tools": 147,
        "terminal_success_tool_coverage": 147,
        "cross_tool_chain_source": "reviewed_historical_bank_only",
        "metrics": [
            "trajectory_success",
            "step_accuracy",
            "call_id_integrity",
            "terminal_response_accuracy",
        ],
        "slices": ["trajectory_length", "family"],
        "release_floor": {"trajectory_success": 0.40, "call_id_integrity": 1.0},
    },
    "narration": {
        "metrics": [
            "required_fact_recall",
            "number_preservation",
            "polarity_accuracy",
            "fallback_exact",
            "delivered_answer_correctness",
        ],
        "release_floor": {
            "required_fact_recall": 0.90,
            "number_preservation": 0.95,
            "polarity_accuracy": 0.90,
            "fallback_exact": 1.0,
        },
    },
    "completion_policy": {
        "process_complete": "all banks and metrics have terminal receipts",
        "release_eligible": "all hard safety/resource gates and preregistered floors pass",
        "quality_failure": "retain candidate and report release_ineligible without unbounded tuning",
    },
}


def _load_tokenizer() -> Any:
    architecture_dir = contract.ROOT / "architecture/mei-1.0-51m-arch-v1"
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


def _source_file_spec(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(contract.ROOT)),
        "sha256": contract.sha_file(path),
        "bytes": path.stat().st_size,
    }


def _artifact_spec(payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": contract.sha_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _jsonl_payload(rows: list[dict[str, Any]]) -> bytes:
    return contract.jsonl_bytes(rows)


def _confidence_candidates(
    rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    """Freeze candidate identities; correctness labels remain model outcomes."""

    output: list[dict[str, Any]] = []
    for row in rows:
        answers = row.get("answers") or []
        expected_call = answers[0] if row.get("kind") == "execute" and answers else None
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


def _query_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> int:
    return len(contract.query_hashes(left) & contract.query_hashes(right))


def _read_source_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    rows = contract.load_jsonl(path)
    return rows, path.read_bytes()


def _normalized_agent_rows(
    path: Path,
    split: str,
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    rows = contract.load_jsonl(path)
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["source_sample_id"] = raw.get("sample_id")
        row["source_generator_version"] = raw.get("generator_version")
        row["permissions"] = {}
        row["slot_provenance"] = []
        row["split"] = split
        row["prompt_framing"] = contract.PROMPT_FRAMING_ID
        row["generator_version"] = "mei-agent-continuation-portable-input-v2"
        row["target_text"] = contract.serialize_tool_target(row.get("answers") or [])
        output.append(row)
    output.extend(
        contract.build_agent_terminal_rows(
            tools,
            split=split,
            variants_per_tool=contract.EVAL_AGENT_TERMINAL_PER_TOOL,
        )
    )
    audit = contract.audit_agent_rows(
        output,
        tools,
        expected_split=split,
        minimum_unique_call_tools=len(tools),
        minimum_terminal_tools=len(tools),
    )
    if audit["status"] != "passed":
        raise RuntimeError("longitudinal Agent bank failed audit: " + audit["errors"][0])
    return output, contract.jsonl_bytes(output), audit


def _balanced_mw_rows(
    historical: Path, tools_by_name: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repartition the historical bank because its dev/test queries overlap.

    The old 20-class lock has 1,500 identical queries across its nominal dev
    and test files.  We retain its reviewed examples but deduplicate by query,
    strip old-model learned retrieval fields, and hash-partition exactly 50
    rows per class into each new split.
    """

    codebook_path = (
        contract.ROOT
        / "notebook/sft/mei-1.0-51m/recipes/mw-disposition-codebook-v1.json"
    )
    codebook = contract.load_json(codebook_path)
    class_ids = {
        str(row["reason_code"]): int(row["class_id"])
        for row in codebook.get("classes") or []
    }
    if len(class_ids) != 20 or set(class_ids.values()) != set(range(20)):
        raise RuntimeError("MW longitudinal bank requires the frozen 20-class codebook")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in ("eval-mw-dev.jsonl", "eval-mw-test.jsonl"):
        for raw in contract.load_jsonl(historical / filename):
            reason = str(raw.get("reason_code") or "")
            query = str(raw.get("query") or "").strip()
            if reason and query:
                unique.setdefault((reason, query), raw)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (reason, query), raw in unique.items():
        row = dict(raw)
        row.pop("learned_top5", None)
        row.pop("retrieval_hit_learned", None)
        row.pop("learned_reason_code", None)
        row.pop("universe_fp", None)
        row.pop("universe_id", None)
        if reason not in class_ids:
            raise RuntimeError(f"MW reason is absent from codebook: {reason}")
        raw_oracle = raw.get("oracle_top5") or []
        names = [
            str(item.get("name") if isinstance(item, dict) else item)
            for item in raw_oracle
        ]
        projected_oracle = [
            contract.compact_tool(tools_by_name[name])
            for name in names
            if name in tools_by_name
        ]
        if len(projected_oracle) != 5 or len({tool["name"] for tool in projected_oracle}) != 5:
            raise RuntimeError(f"MW row lacks five portable oracle tools: {raw.get('sample_id')}")
        row["oracle_top5"] = projected_oracle
        row["reason_class_id"] = class_ids[reason]
        row["source_sample_id"] = raw.get("sample_id")
        row["source_generator_version"] = raw.get("generator_version")
        row["generator_version"] = "mei-longitudinal-mw-repartition-v2"
        by_reason[reason].append(row)
    if len(by_reason) != 20:
        raise RuntimeError(f"MW longitudinal bank requires 20 classes, got {len(by_reason)}")
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for reason in sorted(by_reason):
        candidates = sorted(
            by_reason[reason],
            key=lambda row: contract.sha_bytes(
                str(row.get("query") or "").encode("utf-8")
            ),
        )
        if len(candidates) < 100:
            raise RuntimeError(f"MW class {reason} has only {len(candidates)} unique queries")
        for split, selected in (("dev", candidates[:50]), ("test", candidates[50:100])):
            output = dev if split == "dev" else test
            for index, raw in enumerate(selected):
                row = dict(raw)
                sample_id = contract.stable_id(
                    "MWL", split, reason, index, row.get("query")
                )
                row["sample_id"] = sample_id
                row["item_id"] = sample_id
                row["split"] = split
                output.append(row)
    return dev, test


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    projected_universe, projection_receipt = contract.portable_universe_document(
        args.universe
    )
    tools = list(projected_universe["tools"])
    dev = contract.build_single_step_rows(
        tools,
        split="dev",
        retrieval_per_tool=contract.EVAL_RETRIEVAL_PER_TOOL,
        execute_per_tool=contract.EVAL_EXECUTE_PER_TOOL,
        refuse_per_tool=contract.EVAL_REFUSE_PER_TOOL,
    )
    test = contract.build_single_step_rows(
        tools,
        split="test",
        retrieval_per_tool=contract.EVAL_RETRIEVAL_PER_TOOL,
        execute_per_tool=contract.EVAL_EXECUTE_PER_TOOL,
        refuse_per_tool=contract.EVAL_REFUSE_PER_TOOL,
    )
    dev_audit = contract.audit_single_step_rows(
        dev,
        tools,
        expected_split="dev",
        min_retrieval_per_tool=contract.EVAL_RETRIEVAL_PER_TOOL,
        min_execute_per_tool=contract.EVAL_EXECUTE_PER_TOOL,
        min_refuse_per_tool=contract.EVAL_REFUSE_PER_TOOL,
    )
    test_audit = contract.audit_single_step_rows(
        test,
        tools,
        expected_split="test",
        min_retrieval_per_tool=contract.EVAL_RETRIEVAL_PER_TOOL,
        min_execute_per_tool=contract.EVAL_EXECUTE_PER_TOOL,
        min_refuse_per_tool=contract.EVAL_REFUSE_PER_TOOL,
        forbidden_query_hashes=set(dev_audit["query_hashes"]),
    )
    if dev_audit["status"] != "passed" or test_audit["status"] != "passed":
        raise RuntimeError("generated longitudinal single-step bank failed audit")

    historical = args.historical_eval
    parent = args.agent_release
    narration = args.narration_release
    source_paths = {
        "multistep.dev.jsonl": parent / "agent-continuation.valid.jsonl",
        "multistep.test.jsonl": parent / "agent-continuation.eval.jsonl",
        "narration.dev.jsonl": narration / "narration.valid.jsonl",
        "narration.test.jsonl": narration / "narration.eval.jsonl",
        "base-probes.jsonl": (
            contract.ROOT
            / "notebook/evaluation/banks/needle-pretrain-probes-v0/probes-v0.jsonl"
        ),
    }
    source_rows: dict[str, list[dict[str, Any]]] = {}
    agent_audits: dict[str, dict[str, Any]] = {}
    token_budget = contract.token_budget_audit(
        {
            "retrieval": dev["retrieval"] + test["retrieval"],
            "fullcall": dev["fullcall"] + test["fullcall"],
        },
        tools,
        _load_tokenizer(),
    )
    if token_budget["status"] != "passed":
        raise RuntimeError("longitudinal token budget failed: " + token_budget["errors"][0])
    payloads: dict[str, bytes] = {
        "tool-universe.json": contract.canonical_bytes(projected_universe) + b"\n",
        "tool-universe-projection-receipt.json": (
            contract.canonical_bytes(projection_receipt) + b"\n"
        ),
        "token-budget-receipt.json": contract.canonical_bytes(token_budget) + b"\n",
        "retrieval.dev.jsonl": _jsonl_payload(dev["retrieval"]),
        "retrieval.test.jsonl": _jsonl_payload(test["retrieval"]),
        "fullcall.dev.jsonl": _jsonl_payload(dev["fullcall"]),
        "fullcall.test.jsonl": _jsonl_payload(test["fullcall"]),
        "confidence.dev.jsonl": _jsonl_payload(
            _confidence_candidates(dev["fullcall"], "dev")
        ),
        "confidence.test.jsonl": _jsonl_payload(
            _confidence_candidates(test["fullcall"], "test")
        ),
        "metric-contract.json": contract.canonical_bytes(METRIC_CONTRACT) + b"\n",
    }
    source_rows["confidence.dev.jsonl"] = _confidence_candidates(
        dev["fullcall"], "dev"
    )
    source_rows["confidence.test.jsonl"] = _confidence_candidates(
        test["fullcall"], "test"
    )
    mw_dev, mw_test = _balanced_mw_rows(
        historical, {str(tool["name"]): tool for tool in tools}
    )
    source_rows["mw.dev.jsonl"] = mw_dev
    source_rows["mw.test.jsonl"] = mw_test
    payloads["mw.dev.jsonl"] = _jsonl_payload(mw_dev)
    payloads["mw.test.jsonl"] = _jsonl_payload(mw_test)
    for name, path in source_paths.items():
        if name.startswith("multistep."):
            split = name.split(".")[1]
            rows, payload, audit = _normalized_agent_rows(path, split, tools)
            agent_audits[split] = audit
        else:
            rows, payload = _read_source_rows(path)
        source_rows[name] = rows
        payloads[name] = payload

    isolation_errors: list[str] = []
    if _query_overlap(dev["retrieval"], test["retrieval"]):
        isolation_errors.append("retrieval dev/test query overlap")
    if _query_overlap(dev["fullcall"], test["fullcall"]):
        isolation_errors.append("fullcall dev/test query overlap")
    for stem in ("mw", "multistep", "narration", "confidence"):
        dev_name = f"{stem}.dev.jsonl"
        test_name = f"{stem}.test.jsonl"
        if _query_overlap(source_rows[dev_name], source_rows[test_name]):
            isolation_errors.append(f"{stem} dev/test query overlap")

    training_sources = [
        parent / "retrieval.train.jsonl",
        parent / "full-call.train.jsonl",
        parent / "mw-disposition.train.jsonl",
        parent / "agent-continuation.train.jsonl",
        narration / "narration.train.jsonl",
    ]
    training_query_hashes: set[str] = set()
    for path in training_sources:
        training_query_hashes |= contract.query_hashes(contract.load_jsonl(path))
    for split, variants_per_tool in (
        ("train", contract.TRAIN_AGENT_TERMINAL_PER_TOOL),
        ("valid", contract.VALID_AGENT_TERMINAL_PER_TOOL),
    ):
        training_query_hashes |= contract.query_hashes(
            contract.build_agent_terminal_rows(
                tools,
                split=split,
                variants_per_tool=variants_per_tool,
            )
        )
    eval_query_hashes = set(dev_audit["query_hashes"]) | set(test_audit["query_hashes"])
    for rows in source_rows.values():
        eval_query_hashes |= contract.query_hashes(rows)
    historical_overlap = len(training_query_hashes & eval_query_hashes)
    if historical_overlap:
        isolation_errors.append(f"historical train/eval query overlap: {historical_overlap}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        rows = None
        if name.endswith(".jsonl"):
            rows = sum(1 for line in payload.splitlines() if line)
        artifacts[name] = _artifact_spec(payload, rows=rows)

    source_specs = {
        "tool_universe": _source_file_spec(args.universe),
        "historical_eval_lock": _source_file_spec(historical / "lock.json"),
        "agent_release_manifest": _source_file_spec(parent / "manifest.json"),
        "narration_release_manifest": _source_file_spec(narration / "manifest.json"),
        "base_probe_source": _source_file_spec(source_paths["base-probes.jsonl"]),
        "generator_source": _source_file_spec(
            Path(__file__).with_name("sft_v3_contract_51m.py")
        ),
        "freezer_source": _source_file_spec(Path(__file__)),
    }
    isolation = {
        "schema": "mei-51m-longitudinal-isolation-receipt-v3",
        "status": "passed" if not isolation_errors else "failed",
        "errors": isolation_errors,
        "generated_dev_test_query_overlap": 0,
        "historical_train_eval_query_overlap": historical_overlap,
        "generated_dev_audit": {
            key: value for key, value in dev_audit.items() if key != "query_hashes"
        },
        "generated_test_audit": {
            key: value for key, value in test_audit.items() if key != "query_hashes"
        },
        "eval_query_hash_count": len(eval_query_hashes),
        "model_visible_gold_slot_provenance_rows": sum(
            1
            for row in dev["fullcall"] + test["fullcall"]
            if row.get("slot_provenance")
        ),
        "portable_tool_universe": projection_receipt["output"],
        "token_budget_status": token_budget["status"],
        "multistep_audits": agent_audits,
        "multistep_unique_call_tools": {
            split: audit["unique_call_tools"]
            for split, audit in sorted(agent_audits.items())
        },
        "multistep_unique_terminal_tools": {
            split: audit["unique_terminal_tools"]
            for split, audit in sorted(agent_audits.items())
        },
    }
    if isolation_errors:
        raise RuntimeError("longitudinal isolation failed: " + isolation_errors[0])
    payloads["isolation-receipt.json"] = contract.canonical_bytes(isolation) + b"\n"
    artifacts["isolation-receipt.json"] = _artifact_spec(
        payloads["isolation-receipt.json"]
    )

    fingerprint_material = {
        "contract": METRIC_CONTRACT,
        "sources": source_specs,
        "artifacts": artifacts,
        "generator": contract.GENERATOR_ID,
    }
    lock = {
        "schema": "mei-51m-longitudinal-eval-lock-v3",
        "id": args.eval_id,
        "status": "frozen",
        "product": contract.PRODUCT_ID,
        "weight_contract_id": contract.WEIGHT_CONTRACT_ID,
        "tokenizer_id": contract.TOKENIZER_ID,
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
        "sources": source_specs,
        "artifacts": artifacts,
        "evaluation_fingerprint": contract.sha_bytes(
            contract.canonical_bytes(fingerprint_material)
        ),
        "exposure_policy": {
            "same_lock_for_cumulative_exposure_tokens": [
                300_000_000,
                600_000_000,
                900_000_000,
                1_200_000_000,
                1_500_000_000,
                2_100_000_000,
            ],
            "arbitrary_future_exposure_supported": True,
            "main_scale_curve": "same SFT data, sampler and budget across Base exposures",
            "capacity_adaptation": "record separately; never replace the main scale curve",
        },
        "required_result_identity": [
            "base_weights_sha256",
            "base_exposure_tokens",
            "sft_release_manifest_sha256",
            "training_recipe_fingerprint",
            "package_sha256",
            "evaluation_fingerprint",
        ],
        "isolation_receipt": {
            "file": "isolation-receipt.json",
            "sha256": artifacts["isolation-receipt.json"]["sha256"],
        },
        "projection_receipt": {
            "file": "tool-universe-projection-receipt.json",
            "sha256": artifacts["tool-universe-projection-receipt.json"]["sha256"],
        },
        "token_budget_receipt": {
            "file": "token-budget-receipt.json",
            "sha256": artifacts["token-budget-receipt.json"]["sha256"],
        },
    }
    payloads["lock.json"] = contract.canonical_bytes(lock) + b"\n"
    return payloads, lock


def _verify_existing(target: Path, payloads: dict[str, bytes]) -> None:
    actual_names = {path.name for path in target.iterdir() if path.is_file()}
    if actual_names != set(payloads):
        raise RuntimeError(f"existing lock file set differs: {target}")
    for name, payload in payloads.items():
        path = target / name
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing lock artifact differs: {path}")


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    payloads, lock = build_payloads(args)
    args.eval_root.mkdir(parents=True, exist_ok=True)
    target = args.eval_root / args.eval_id
    if target.exists():
        _verify_existing(target, payloads)
        return {
            "ok": True,
            "reused": True,
            "path": str(target),
            "evaluation_fingerprint": lock["evaluation_fingerprint"],
        }
    with tempfile.TemporaryDirectory(prefix=f".{args.eval_id}-", dir=args.eval_root) as name:
        temporary = Path(name) / args.eval_id
        temporary.mkdir()
        for filename, payload in payloads.items():
            (temporary / filename).write_bytes(payload)
        temporary.replace(target)
    return {
        "ok": True,
        "reused": False,
        "path": str(target),
        "evaluation_fingerprint": lock["evaluation_fingerprint"],
        "artifacts": len(payloads),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=contract.DEFAULT_EVAL_ROOT)
    parser.add_argument("--eval-id", default=contract.EVAL_ID)
    parser.add_argument("--universe", type=Path, default=contract.TOOL_UNIVERSE_PATH)
    parser.add_argument("--historical-eval", type=Path, default=contract.HISTORICAL_EVAL_DIR)
    parser.add_argument("--agent-release", type=Path, default=contract.PARENT_RELEASE_DIR)
    parser.add_argument("--narration-release", type=Path, default=contract.NARRATION_RELEASE_DIR)
    return parser.parse_args()


def main() -> int:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
