#!/usr/bin/env python3
"""Freeze the quality-first longitudinal eval-v7 for mei-1.0-51m.

The v6 lock remains immutable and supplies the already isolated 147-tool,
Agent, narration, confidence-candidate, and Base-language banks.  V7 replaces
the contaminated MW bank, adds admitted cross-generator Chinese evaluation,
and adds a whole-tool/schema holdout that is disjoint from SFT-v4 training.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import freeze_longitudinal_eval_51m as v6
import freeze_sft_natural_augmentation_51m as natural
import freeze_sft_v3_release_51m as v3_freezer
import sft_v4_contract_51m as contract


PARENT_EVAL = contract.DEFAULT_EVAL_ROOT / "mei-51m-longitudinal-eval-v6"
PARENT_SFT = contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-tool-sft-v3-300m-v7"
NATURAL_RELEASE = (
    contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-tool-sft-natural-aug300m-v1"
)
NARRATION_RELEASE = (
    contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-narration-sft-agent300m-v3"
)

COPIED_PARENT_ARTIFACTS = (
    "tool-universe.json",
    "tool-universe-projection-receipt.json",
    "token-budget-receipt.json",
    "retrieval.dev.jsonl",
    "retrieval.test.jsonl",
    "fullcall.dev.jsonl",
    "fullcall.test.jsonl",
    "confidence.dev.jsonl",
    "confidence.test.jsonl",
    "multistep.dev.jsonl",
    "multistep.test.jsonl",
    "narration.dev.jsonl",
    "narration.test.jsonl",
    "base-probes.jsonl",
)


def _metric_contract() -> dict[str, Any]:
    result = copy.deepcopy(v6.METRIC_CONTRACT)
    result["schema"] = "mei-51m-longitudinal-metric-contract-v4"
    result["id"] = "mei-51m-longitudinal-metrics-v4-quality-schema"
    result["retrieval"]["banks"] = {
        "structural_seen": "retrieval.{split}.jsonl",
        "natural_cross_generator": "natural-retrieval.{split}.jsonl",
        "whole_schema_holdout": "schema-retrieval.{split}.jsonl",
    }
    result["retrieval"]["slices"] = [
        "bank",
        "family",
        "tool",
        "seen_schema",
        "schema_feature",
        "teacher",
    ]
    result["fullcall"]["banks"] = {
        "structural_seen": "fullcall.{split}.jsonl",
        "natural_cross_generator": "natural-fullcall.{split}.jsonl",
        "whole_schema_holdout": "schema-fullcall.{split}.jsonl",
    }
    result["fullcall"]["slices"] = [
        "bank",
        "kind",
        "family",
        "seen_schema",
        "schema_feature",
        "teacher",
    ]
    result["mw_disposition"].update(
        {
            "prompt_id": contract.MW_PROMPT_ID,
            "structured_inputs": [
                "context",
                "evidence",
                "permissions",
                "state",
                "history",
                "tool_results",
            ],
            "synthetic_shortcut_rows_required": 0,
            "bank": "mw.{split}.jsonl",
        }
    )
    result["confidence"]["banks"] = [
        "confidence.{split}.jsonl",
        "schema-confidence.{split}.jsonl",
    ]
    result["multi_step"]["terminal_outcome_coverage"] = {
        "success": "multistep.{split}.jsonl",
        "error_cancelled_partial": "narration.{split}.jsonl plus runtime state-machine gates",
        "note": "narration is terminal-only and never feeds tool decisions",
    }
    result["completion_policy"]["quality_axis"] = (
        "report structural-seen, natural-crossgen and whole-schema-holdout separately"
    )
    return result


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


def _payload_rows(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _verify_parent_lock(parent: Path) -> dict[str, Any]:
    lock = contract.load_json(parent / "lock.json")
    if (
        lock.get("schema") != "mei-51m-longitudinal-eval-lock-v3"
        or lock.get("id") != "mei-51m-longitudinal-eval-v6"
        or lock.get("status") != "frozen"
    ):
        raise RuntimeError("eval-v7 requires the frozen eval-v6 parent")
    for name, spec in (lock.get("artifacts") or {}).items():
        path = parent / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"eval-v6 artifact drift: {path}")
    isolation = contract.load_json(parent / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("eval-v6 parent isolation is not passed")
    return lock


def _schema_universe(tools: Sequence[dict[str, Any]], partition: str) -> dict[str, Any]:
    material = {
        "schema": "mei-sft-v4-schema-tool-universe-v1",
        "partition": partition,
        "schema_subset_id": contract.SCHEMA_SUBSET_ID,
        "tools": list(tools),
    }
    material["fingerprint"] = contract.sha_bytes(contract.canonical_bytes(material))
    return material


def _confidence_candidates(
    rows: Sequence[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        answers = list(row.get("answers") or [])
        expected_call = answers[0] if row.get("kind") == "execute" and answers else None
        output.append(
            {
                "sample_id": contract.stable_id("CONF4S", split, row["sample_id"]),
                "source_sample_id": row["sample_id"],
                "split": split,
                "task": "confidence_outcome_harvest",
                "bank": "whole_schema_holdout",
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


def _natural_rows(
    path: Path,
    *,
    split: str,
    task: str,
    tools_by_name: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    ids: set[str] = set()
    queries: set[str] = set()
    for raw in contract.load_jsonl(path):
        row = dict(raw)
        source_id = str(raw.get("sample_id") or "")
        row["source_sample_id"] = source_id
        row["sample_id"] = contract.stable_id("NATE7", split, task, source_id)
        row["case_id"] = row["sample_id"]
        row["split"] = split
        row["eval_bank"] = "admitted_cross_generator_natural"
        query = str(row.get("query") or "").strip()
        if not source_id or row["sample_id"] in ids:
            errors.append(f"empty or duplicate natural ID: {source_id}")
        ids.add(row["sample_id"])
        if not query or query in queries or contract.query_has_synthetic_shortcut(query):
            errors.append(f"invalid natural query: {source_id}")
        queries.add(query)
        visible = row.get("catalog_tools") if task == "retrieval" else row.get("oracle_top5")
        names = [
            str(item.get("name") if isinstance(item, dict) else item)
            for item in visible or []
        ]
        if len(names) != 5 or len(set(names)) != 5 or not set(names).issubset(tools_by_name):
            errors.append(f"invalid natural top5: {source_id}")
        if task == "retrieval":
            if str(row.get("gold_tool") or "") not in names:
                errors.append(f"natural retrieval gold absent: {source_id}")
        elif row.get("kind") == "execute":
            answers = list(row.get("answers") or [])
            if len(answers) != 1:
                errors.append(f"natural execute answer count: {source_id}")
            else:
                answer = answers[0]
                name = str(answer.get("name") or "")
                if name not in tools_by_name or not contract.arguments_match_schema(
                    answer.get("arguments") or {}, tools_by_name[name]["parameters"]
                ):
                    errors.append(f"natural execute schema mismatch: {source_id}")
                elif not natural.grounding_ok(row, answer.get("arguments") or {}):
                    errors.append(f"natural execute arguments ungrounded: {source_id}")
        output.append(row)
    teachers = Counter(
        str((row.get("natural_source") or {}).get("teacher_model") or "unknown")
        for row in output
    )
    audit = {
        "schema": "mei-sft-v4-natural-eval-audit-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors[:200],
        "split": split,
        "task": task,
        "rows": len(output),
        "tools": len(
            {
                str(row.get("gold_tool") or row.get("candidate_tool") or "")
                for row in output
                if row.get("gold_tool") or row.get("candidate_tool")
            }
        ),
        "teachers": dict(sorted(teachers.items())),
        "synthetic_shortcut_rows": sum(
            contract.query_has_synthetic_shortcut(str(row.get("query") or ""))
            for row in output
        ),
    }
    return output, audit


def _clean_mw_eval(
    parent_rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    split: str,
    rows_per_class: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in parent_rows:
        row = contract.normalize_mw_row(raw, split=split)
        if row is not None:
            by_class[int(row["reason_class_id"])].append(row)
    supplement = contract.generate_clean_mw_rows(
        tools, split=split, variants_per_tool=1
    )
    for row in supplement:
        by_class[int(row["reason_class_id"])].append(row)
    output: list[dict[str, Any]] = []
    for class_id in range(20):
        candidates = sorted(
            by_class[class_id],
            key=lambda row: contract.sha_bytes(
                str(row.get("query") or "").encode("utf-8")
            ),
        )
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in candidates:
            query = str(row.get("query") or "")
            if query in seen:
                continue
            seen.add(query)
            unique.append(row)
        if len(unique) < rows_per_class:
            raise RuntimeError(
                f"clean MW eval class {class_id} has only {len(unique)} rows"
            )
        output.extend(unique[:rows_per_class])
    audit = contract.audit_mw_rows(output, split=split)
    return output, audit


def _all_queries(rows_by_name: dict[str, Sequence[dict[str, Any]]]) -> set[str]:
    result: set[str] = set()
    for rows in rows_by_name.values():
        result |= contract.query_hashes_normalized(rows)
    return result


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    parent_lock = _verify_parent_lock(args.parent_eval)
    payloads = {
        name: (args.parent_eval / name).read_bytes()
        for name in COPIED_PARENT_ARTIFACTS
    }
    deploy_universe = json.loads(payloads["tool-universe.json"])
    deploy_tools = list(deploy_universe.get("tools") or [])
    deploy_by_name = {str(tool["name"]): tool for tool in deploy_tools}
    historical_training_rows: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "retrieval.train.jsonl",
        "full-call.train.jsonl",
        "mw-disposition.train.jsonl",
        "agent-continuation.train.jsonl",
    ):
        historical_training_rows["parent-" + name] = contract.load_jsonl(
            args.parent_sft / name
        )
    for name in ("natural-retrieval.train.jsonl", "natural-full-call.train.jsonl"):
        historical_training_rows[name] = contract.load_jsonl(args.natural_release / name)
    historical_training_rows["narration.train.jsonl"] = contract.load_jsonl(
        args.narration_release / "narration.train.jsonl"
    )
    historical_training_hashes = _all_queries(historical_training_rows)

    holdout_tools = contract.schema_feature_tools("holdout")
    train_schema_tools = contract.schema_feature_tools("train")
    holdout_document = _schema_universe(holdout_tools, "eval_only_whole_schema_holdout")
    payloads["schema-holdout-tool-universe.json"] = (
        contract.canonical_bytes(holdout_document) + b"\n"
    )

    schema_rows: dict[str, list[dict[str, Any]]] = {}
    schema_audits: dict[str, dict[str, Any]] = {}
    for split in ("dev", "test"):
        bank = contract.build_schema_feature_rows(
            holdout_tools,
            deploy_tools,
            split=split,
            retrieval_per_tool=4,
            execute_per_tool=2,
            refuse_per_tool=2,
        )
        audit = contract.audit_schema_feature_rows(
            bank, holdout_tools, expected_split=split
        )
        if audit["status"] != "passed":
            raise RuntimeError("schema holdout audit failed: " + audit["errors"][0])
        schema_audits[split] = audit
        schema_rows[f"schema-retrieval.{split}.jsonl"] = bank["retrieval"]
        schema_rows[f"schema-fullcall.{split}.jsonl"] = bank["fullcall"]
        schema_rows[f"schema-confidence.{split}.jsonl"] = _confidence_candidates(
            bank["fullcall"], split
        )
    for name, rows in schema_rows.items():
        payloads[name] = contract.jsonl_bytes(rows)

    schema_budget = contract.token_budget_audit(
        {
            "retrieval": schema_rows["schema-retrieval.dev.jsonl"]
            + schema_rows["schema-retrieval.test.jsonl"],
            "fullcall": schema_rows["schema-fullcall.dev.jsonl"]
            + schema_rows["schema-fullcall.test.jsonl"],
        },
        [*deploy_tools, *holdout_tools],
        v3_freezer._load_tokenizer(),
    )
    if schema_budget["status"] != "passed":
        raise RuntimeError("schema holdout token budget failed: " + schema_budget["errors"][0])

    natural_rows: dict[str, list[dict[str, Any]]] = {}
    natural_audits: dict[str, dict[str, Any]] = {}
    for task, source_stem, output_stem in (
        ("retrieval", "crossgen-retrieval", "natural-retrieval"),
        ("fullcall", "crossgen-full-call", "natural-fullcall"),
    ):
        for split in ("dev", "test"):
            rows, audit = _natural_rows(
                args.natural_release / f"{source_stem}.{split}.jsonl",
                split=split,
                task=task,
                tools_by_name=deploy_by_name,
            )
            before = len(rows)
            rows = [
                row
                for row in rows
                if not (
                    contract.query_hashes_normalized([row])
                    & historical_training_hashes
                )
            ]
            audit["historical_training_overlap_excluded"] = before - len(rows)
            audit["rows"] = len(rows)
            if audit["status"] != "passed":
                raise RuntimeError("natural eval audit failed: " + audit["errors"][0])
            name = f"{output_stem}.{split}.jsonl"
            natural_rows[name] = rows
            natural_audits[name] = audit
            payloads[name] = contract.jsonl_bytes(rows)

    mw_rows: dict[str, list[dict[str, Any]]] = {}
    mw_audits: dict[str, dict[str, Any]] = {}
    for split in ("dev", "test"):
        parent_rows = contract.load_jsonl(args.parent_eval / f"mw.{split}.jsonl")
        rows, audit = _clean_mw_eval(parent_rows, deploy_tools, split=split)
        if audit["status"] != "passed":
            raise RuntimeError("clean MW eval audit failed: " + audit["errors"][0])
        mw_rows[f"mw.{split}.jsonl"] = rows
        mw_audits[split] = audit
        payloads[f"mw.{split}.jsonl"] = contract.jsonl_bytes(rows)

    metric_contract = _metric_contract()
    payloads["metric-contract.json"] = contract.canonical_bytes(metric_contract) + b"\n"

    eval_rows: dict[str, list[dict[str, Any]]] = {
        name: _payload_rows(payload)
        for name, payload in payloads.items()
        if name.endswith(".jsonl")
    }
    eval_hashes = _all_queries(eval_rows)
    errors: list[str] = []
    for stem in (
        "retrieval",
        "fullcall",
        "confidence",
        "multistep",
        "narration",
        "mw",
        "schema-retrieval",
        "schema-fullcall",
        "schema-confidence",
        "natural-retrieval",
        "natural-fullcall",
    ):
        left = eval_rows.get(f"{stem}.dev.jsonl", [])
        right = eval_rows.get(f"{stem}.test.jsonl", [])
        overlap = contract.query_hashes_normalized(left) & contract.query_hashes_normalized(right)
        if overlap:
            errors.append(f"{stem} dev/test query overlap: {len(overlap)}")

    training_rows: dict[str, list[dict[str, Any]]] = dict(historical_training_rows)
    for split in ("train", "valid"):
        bank = contract.build_schema_feature_rows(
            train_schema_tools,
            deploy_tools,
            split=split,
            retrieval_per_tool=4,
            execute_per_tool=2,
            refuse_per_tool=2,
        )
        training_rows[f"schema-retrieval-{split}"] = bank["retrieval"]
        training_rows[f"schema-fullcall-{split}"] = bank["fullcall"]
        training_rows[f"mw-clean-{split}"] = contract.generate_clean_mw_rows(
            deploy_tools, split=split, variants_per_tool=2 if split == "train" else 1
        )
    training_overlap = len(_all_queries(training_rows) & eval_hashes)
    if training_overlap:
        errors.append(f"known train/eval query overlap: {training_overlap}")

    train_names = {str(tool["name"]) for tool in train_schema_tools}
    holdout_names = {str(tool["name"]) for tool in holdout_tools}
    if train_names & holdout_names:
        errors.append("schema train/holdout tool identity overlap")
    if any(audit["synthetic_shortcut_rows"] for audit in mw_audits.values()):
        errors.append("MW eval contains synthetic shortcuts")

    isolation = {
        "schema": "mei-51m-longitudinal-isolation-receipt-v4",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "parent_eval_id": parent_lock["id"],
        "parent_evaluation_fingerprint": parent_lock["evaluation_fingerprint"],
        "known_train_eval_query_overlap": training_overlap,
        "dev_test_query_overlap": 0 if not errors else None,
        "schema_train_tool_count": len(train_names),
        "schema_holdout_tool_count": len(holdout_names),
        "schema_train_holdout_tool_overlap": len(train_names & holdout_names),
        "schema_audits": schema_audits,
        "schema_token_budget": schema_budget,
        "natural_audits": natural_audits,
        "mw_audits": mw_audits,
        "mw_synthetic_shortcut_rows": sum(
            audit["synthetic_shortcut_rows"] for audit in mw_audits.values()
        ),
        "eval_query_hash_count": len(eval_hashes),
        "group_aware": True,
        "family_aware": True,
        "whole_tool_schema_holdout": True,
    }
    if errors:
        raise RuntimeError("eval-v7 isolation failed: " + errors[0])
    payloads["isolation-receipt.json"] = contract.canonical_bytes(isolation) + b"\n"

    artifacts: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        rows = sum(1 for line in payload.splitlines() if line) if name.endswith(".jsonl") else None
        artifacts[name] = _artifact(payload, rows=rows)
    sources = {
        "parent_eval_lock": _source(args.parent_eval / "lock.json"),
        "parent_sft_manifest": _source(args.parent_sft / "manifest.json"),
        "natural_release_manifest": _source(args.natural_release / "manifest.json"),
        "narration_release_manifest": _source(args.narration_release / "manifest.json"),
        "generator_source": _source(Path(__file__).with_name("sft_v4_contract_51m.py")),
        "freezer_source": _source(Path(__file__)),
    }
    fingerprint_material = {
        "metric_contract": metric_contract,
        "sources": sources,
        "artifacts": artifacts,
        "generator": contract.GENERATOR_ID,
    }
    lock = {
        "schema": "mei-51m-longitudinal-eval-lock-v4",
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
        "parents": {
            "eval_id": parent_lock["id"],
            "evaluation_fingerprint": parent_lock["evaluation_fingerprint"],
            "lock_sha256": sources["parent_eval_lock"]["sha256"],
        },
        "sources": sources,
        "artifacts": artifacts,
        "evaluation_fingerprint": contract.sha_bytes(
            contract.canonical_bytes(fingerprint_material)
        ),
        "banks": {
            "structural_seen": ["retrieval", "fullcall", "confidence"],
            "natural_cross_generator": ["natural-retrieval", "natural-fullcall"],
            "whole_schema_holdout": [
                "schema-retrieval",
                "schema-fullcall",
                "schema-confidence",
            ],
            "clean_structured_mw": ["mw"],
            "agent": ["multistep"],
            "narration": ["narration"],
            "base_language": ["base-probes"],
        },
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
            "main_scale_curve": "same SFT-v4 data, sampler, budget and eval-v7 lock",
            "capacity_adaptation": "separate release and scorecard only",
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
    }
    payloads["lock.json"] = contract.canonical_bytes(lock) + b"\n"
    return payloads, lock


def _verify_existing(target: Path, payloads: dict[str, bytes]) -> None:
    actual = {path.name for path in target.iterdir() if path.is_file()}
    if actual != set(payloads):
        raise RuntimeError(f"existing eval-v7 file set differs: {target}")
    for name, payload in payloads.items():
        if (target / name).read_bytes() != payload:
            raise RuntimeError(f"existing eval-v7 artifact differs: {target / name}")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=contract.DEFAULT_EVAL_ROOT)
    parser.add_argument("--eval-id", default=contract.EVAL_ID)
    parser.add_argument("--parent-eval", type=Path, default=PARENT_EVAL)
    parser.add_argument("--parent-sft", type=Path, default=PARENT_SFT)
    parser.add_argument("--natural-release", type=Path, default=NATURAL_RELEASE)
    parser.add_argument("--narration-release", type=Path, default=NARRATION_RELEASE)
    return parser.parse_args(argv)


def main() -> int:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
