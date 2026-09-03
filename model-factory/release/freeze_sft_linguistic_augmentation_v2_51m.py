#!/usr/bin/env python3
"""Freeze SFT-v4 linguistic coverage for all deployed mei-51m tools.

This release keeps the admitted historical multi-teacher rows, and adds a
deterministic offline linguistic program for coverage.  The latter is never
described as natural-user data.  Teachers may vary wording only; tool, schema,
arguments, reason codes, splits and all labels remain locally verified.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import release.freeze_sft_natural_augmentation_51m as natural
import release.freeze_sft_v3_release_51m as v3_freezer
import contracts.sft_v4_contract_51m as contract


PARENT_SFT = contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-tool-sft-v3-300m-v7"
PARENT_NATURAL = (
    contract.DEFAULT_RELEASE_ROOT / "mei-1.0-51m-tool-sft-natural-aug300m-v1"
)
EVAL_DIR = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID

TRAIN_COUNTS = {"retrieval": 6, "execute": 4, "refuse": 4}
VALID_COUNTS = {"retrieval": 2, "execute": 2, "refuse": 2}


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


def _eval_hashes(eval_dir: Path) -> set[str]:
    lock = contract.load_json(eval_dir / "lock.json")
    if (
        lock.get("schema") != "mei-51m-longitudinal-eval-lock-v4"
        or lock.get("id") != contract.EVAL_ID
        or lock.get("status") != "frozen"
    ):
        raise RuntimeError("linguistic-v2 requires frozen eval-v7")
    for name, spec in (lock.get("artifacts") or {}).items():
        path = eval_dir / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"eval-v7 artifact drift: {path}")
    hashes: set[str] = set()
    for path in sorted(eval_dir.glob("*.jsonl")):
        hashes |= contract.query_hashes_normalized(contract.load_jsonl(path))
    return hashes


def _adopt_natural_rows(
    rows: Sequence[dict[str, Any]], *, task: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        source_id = str(raw.get("sample_id") or "")
        row["source_sample_id"] = source_id
        row["sample_id"] = contract.stable_id("LINGNAT4", task, source_id)
        row["case_id"] = row["sample_id"]
        row["split"] = "train"
        row["source_role"] = "admitted-multiteacher-natural-query"
        row["linguistic_layer"] = "historical_admitted_natural"
        output.append(row)
    return output


def _audit_rows(
    rows: dict[str, list[dict[str, Any]]],
    tools: Sequence[dict[str, Any]],
    *,
    eval_hashes: set[str],
) -> dict[str, Any]:
    tools_by_name = {str(tool["name"]): tool for tool in tools}
    errors: list[str] = []
    split_hashes: dict[str, set[str]] = {}
    task_stats: dict[str, dict[str, Any]] = {}
    for name, values in sorted(rows.items()):
        split = "train" if ".train." in name else "valid"
        task = "retrieval" if name.startswith("linguistic-retrieval") else "fullcall"
        ids: set[str] = set()
        queries: set[str] = set()
        tools_seen: set[str] = set()
        sources: Counter[str] = Counter()
        teachers: Counter[str] = Counter()
        execute = 0
        refuse = 0
        for row in values:
            sample_id = str(row.get("sample_id") or "")
            query = str(row.get("query") or "").strip()
            if not sample_id or sample_id in ids:
                errors.append(f"empty or duplicate linguistic ID: {sample_id}")
            ids.add(sample_id)
            if not query or query in queries or contract.query_has_synthetic_shortcut(query):
                errors.append(f"invalid or duplicate linguistic query: {sample_id}")
            queries.add(query)
            if row.get("split") != split:
                errors.append(f"linguistic split mismatch: {sample_id}")
            source_role = str(row.get("source_role") or "unknown")
            sources[source_role] += 1
            teacher = str((row.get("natural_source") or {}).get("teacher_model") or "")
            if teacher:
                teachers[teacher] += 1
            visible = row.get("catalog_tools") if task == "retrieval" else row.get("oracle_top5")
            names = [
                str(item.get("name") if isinstance(item, dict) else item)
                for item in visible or []
            ]
            if len(names) != 5 or len(set(names)) != 5 or not set(names).issubset(tools_by_name):
                errors.append(f"invalid linguistic top5: {sample_id}")
            if task == "retrieval":
                gold = str(row.get("gold_tool") or "")
                tools_seen.add(gold)
                if gold not in names:
                    errors.append(f"linguistic retrieval gold absent: {sample_id}")
            else:
                candidate = str(row.get("candidate_tool") or "")
                if candidate:
                    tools_seen.add(candidate)
                if row.get("kind") == "execute":
                    execute += 1
                    answers = list(row.get("answers") or [])
                    if len(answers) != 1:
                        errors.append(f"linguistic execute answer count: {sample_id}")
                    else:
                        answer = answers[0]
                        tool_name = str(answer.get("name") or "")
                        arguments = answer.get("arguments") or {}
                        if tool_name not in tools_by_name or not contract.arguments_match_schema(
                            arguments, tools_by_name[tool_name]["parameters"]
                        ):
                            errors.append(f"linguistic execute schema mismatch: {sample_id}")
                        elif source_role == "admitted-multiteacher-natural-query":
                            if not natural.grounding_ok(row, arguments):
                                errors.append(f"admitted natural arguments ungrounded: {sample_id}")
                        elif not contract.arguments_grounded(arguments, query):
                            errors.append(f"offline linguistic arguments ungrounded: {sample_id}")
                else:
                    refuse += 1
                    if row.get("answers"):
                        errors.append(f"linguistic refusal has call target: {sample_id}")
        hashes = contract.query_hashes_normalized(values)
        split_hashes.setdefault(split, set()).update(hashes)
        overlap = len(hashes & eval_hashes)
        if overlap:
            errors.append(f"{name} overlaps frozen eval-v7: {overlap}")
        task_stats[name] = {
            "rows": len(values),
            "tools": len(tools_seen),
            "execute": execute,
            "refuse": refuse,
            "source_roles": dict(sorted(sources.items())),
            "teachers": dict(sorted(teachers.items())),
            "query_eval_overlap": overlap,
            "synthetic_shortcut_rows": sum(
                contract.query_has_synthetic_shortcut(str(row.get("query") or ""))
                for row in values
            ),
        }
    split_overlap = len(split_hashes.get("train", set()) & split_hashes.get("valid", set()))
    if split_overlap:
        errors.append(f"linguistic train/valid overlap: {split_overlap}")
    offline_train_tools = {
        str(row.get("gold_tool") or row.get("candidate_tool") or "")
        for name, values in rows.items()
        if ".train." in name
        for row in values
        if row.get("source_role") == "offline-linguistic-program-not-natural-user"
    }
    if offline_train_tools != set(tools_by_name):
        errors.append("offline linguistic program does not cover all deployed tools")
    return {
        "schema": "mei-sft-v4-linguistic-quality-audit-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors[:200],
        "tasks": task_stats,
        "train_valid_query_overlap": split_overlap,
        "eval_query_overlap": 0,
        "offline_train_tool_coverage": len(offline_train_tools),
        "offline_is_natural_user_data": False,
    }


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    eval_hashes = _eval_hashes(args.eval_dir)
    parent_manifest = contract.load_json(args.parent_sft / "manifest.json")
    natural_manifest = contract.load_json(args.parent_natural / "manifest.json")
    if parent_manifest.get("release_id") != "mei-1.0-51m-tool-sft-v3-300m-v7":
        raise RuntimeError("linguistic-v2 parent SFT identity drifted")
    if natural_manifest.get("release_id") != "mei-1.0-51m-tool-sft-natural-aug300m-v1":
        raise RuntimeError("linguistic-v2 natural parent identity drifted")
    universe = contract.load_json(args.parent_sft / "tool-universe.json")
    tools = list(universe.get("tools") or [])
    if len(tools) != 147:
        raise RuntimeError("linguistic-v2 requires the 147-tool deployed universe")

    generated: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, counts in (("train", TRAIN_COUNTS), ("valid", VALID_COUNTS)):
        bank = contract.build_linguistic_rows(
            tools,
            split=split,
            retrieval_per_tool=counts["retrieval"],
            execute_per_tool=counts["execute"],
            refuse_per_tool=counts["refuse"],
        )
        audit = contract.audit_single_step_rows(
            bank,
            tools,
            expected_split=split,
            min_retrieval_per_tool=counts["retrieval"],
            min_execute_per_tool=counts["execute"],
            min_refuse_per_tool=counts["refuse"],
            forbidden_query_hashes=eval_hashes,
        )
        if audit["status"] != "passed":
            raise RuntimeError("offline linguistic audit failed: " + audit["errors"][0])
        generated[split] = bank

    rows = {
        "linguistic-retrieval.train.jsonl": [
            *_adopt_natural_rows(
                contract.load_jsonl(args.parent_natural / "natural-retrieval.train.jsonl"),
                task="retrieval",
            ),
            *generated["train"]["retrieval"],
        ],
        "linguistic-full-call.train.jsonl": [
            *_adopt_natural_rows(
                contract.load_jsonl(args.parent_natural / "natural-full-call.train.jsonl"),
                task="fullcall",
            ),
            *generated["train"]["fullcall"],
        ],
        "linguistic-retrieval.valid.jsonl": generated["valid"]["retrieval"],
        "linguistic-full-call.valid.jsonl": generated["valid"]["fullcall"],
    }
    quality = _audit_rows(rows, tools, eval_hashes=eval_hashes)
    if quality["status"] != "passed":
        raise RuntimeError("combined linguistic audit failed: " + quality["errors"][0])

    token_budget = contract.token_budget_audit(
        {
            "retrieval": rows["linguistic-retrieval.train.jsonl"]
            + rows["linguistic-retrieval.valid.jsonl"],
            "fullcall": rows["linguistic-full-call.train.jsonl"]
            + rows["linguistic-full-call.valid.jsonl"],
        },
        tools,
        v3_freezer._load_tokenizer(),
    )
    if token_budget["status"] != "passed":
        raise RuntimeError("linguistic token budget failed: " + token_budget["errors"][0])

    payloads = {name: contract.jsonl_bytes(values) for name, values in rows.items()}
    payloads["quality-audit-receipt.json"] = contract.canonical_bytes(quality) + b"\n"
    payloads["token-budget-receipt.json"] = contract.canonical_bytes(token_budget) + b"\n"
    contract_document = {
        "schema": "mei-sft-v4-linguistic-data-contract-v1",
        "id": "mei-sft-v4-linguistic-coverage-v1",
        "layers": {
            "historical_admitted_natural": {
                "source": natural_manifest["release_id"],
                "teacher_changes_wording_only": True,
                "model_visible_teacher_metadata": False,
            },
            "offline_linguistic_program": {
                "natural_user_claim": False,
                "tool_coverage": "147_of_147",
                "dimensions": [
                    "register",
                    "word_order",
                    "discourse_wrapper",
                    "slot_realization",
                    "execute_refuse_counterfactual",
                ],
            },
        },
        "label_authority": "local schema and deterministic contract only",
        "evaluation": "all cross-generator and whole-schema rows remain eval-only in eval-v7",
        "expansion_policy": (
            "admit consented anonymized user queries or new teacher wording only after "
            "grounding, provenance, group-isolation and frozen-eval checks"
        ),
    }
    payloads["data-contract.json"] = contract.canonical_bytes(contract_document) + b"\n"

    artifacts: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        count = sum(1 for line in payload.splitlines() if line) if name.endswith(".jsonl") else None
        artifacts[name] = _artifact(payload, rows=count)
    sources = {
        "parent_sft_manifest": _source(args.parent_sft / "manifest.json"),
        "parent_natural_manifest": _source(args.parent_natural / "manifest.json"),
        "eval_lock": _source(args.eval_dir / "lock.json"),
        "generator_source": _source(contract.ROOT / "model-factory/contracts/sft_v4_contract_51m.py"),
        "freezer_source": _source(Path(__file__)),
    }
    fingerprint = contract.sha_bytes(
        contract.canonical_bytes(
            {
                "contract": contract_document,
                "sources": sources,
                "artifacts": artifacts,
            }
        )
    )
    manifest = {
        "schema": "mei-sft-linguistic-augmentation-v2",
        "release_id": args.release_id,
        "status": "frozen",
        "product": contract.PRODUCT_ID,
        "release_fingerprint": fingerprint,
        "parent": {
            "release_id": parent_manifest["release_id"],
            "manifest_sha256": sources["parent_sft_manifest"]["sha256"],
        },
        "historical_natural_parent": {
            "release_id": natural_manifest["release_id"],
            "manifest_sha256": sources["parent_natural_manifest"]["sha256"],
        },
        "evaluation": {
            "id": contract.EVAL_ID,
            "lock_sha256": sources["eval_lock"]["sha256"],
        },
        "base_compatibility": {
            "weight_contract_id": contract.WEIGHT_CONTRACT_ID,
            "tokenizer_id": contract.TOKENIZER_ID,
            "exposure_allowlist": None,
            "arbitrary_cumulative_exposure_supported": True,
        },
        "claims": {
            "all_147_tools_have_offline_linguistic_coverage": True,
            "offline_rows_are_natural_user_data": False,
            "historical_natural_rows_remain_provenance_bound": True,
        },
        "sources": sources,
        "artifacts": artifacts,
        "quality_audit": {
            "file": "quality-audit-receipt.json",
            "sha256": artifacts["quality-audit-receipt.json"]["sha256"],
        },
        "token_budget": {
            "file": "token-budget-receipt.json",
            "sha256": artifacts["token-budget-receipt.json"]["sha256"],
        },
        "data_contract": {
            "file": "data-contract.json",
            "sha256": artifacts["data-contract.json"]["sha256"],
        },
    }
    payloads["manifest.json"] = contract.canonical_bytes(manifest) + b"\n"
    return payloads, manifest


def _verify_existing(target: Path, payloads: dict[str, bytes]) -> None:
    actual = {path.name for path in target.iterdir() if path.is_file()}
    if actual != set(payloads):
        raise RuntimeError(f"existing linguistic-v2 file set differs: {target}")
    for name, payload in payloads.items():
        if (target / name).read_bytes() != payload:
            raise RuntimeError(f"existing linguistic-v2 artifact differs: {target / name}")


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
    parser.add_argument("--release-id", default=contract.LINGUISTIC_AUGMENTATION_ID)
    parser.add_argument("--parent-sft", type=Path, default=PARENT_SFT)
    parser.add_argument("--parent-natural", type=Path, default=PARENT_NATURAL)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    return parser.parse_args(argv)


def main() -> int:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
