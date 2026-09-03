#!/usr/bin/env python3
"""Freeze validation-calibrated retrieval and batched model-visible views.

This module never mutates the source SFT release.  Every output is a derived,
model-bound artifact whose identity includes the LM, retrieval head, index,
tokenizer, calibration, packer and source rows.  MW training consumes only the
materialized prompts written here and therefore cannot discover a new
over-budget schema combination halfway through optimization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import contracts.sft_v4_contract_51m as contract

_SDK_PYTHON = str(contract.ROOT / "platform/python-sdk")
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)

from mei_sdk.protocol import render_budgeted_request
from mei_sdk.shared import (
    CONTEXT_PACKER_ID,
    RETRIEVAL_BATCH_POLICY_ID,
    RETRIEVAL_CALIBRATION_ID,
    RankedCandidate,
    fit_platt_calibrator,
    plan_candidate_batches,
    platt_relevance,
    select_retrieval_thresholds,
)


CALIBRATION_VALIDATION_ID = "mei-retrieval-10-20-50-validation-v1"
MW_VISIBLE_BATCH_SCHEMA = "mei-mw-visible-batch-v1"
MW_VISIBLE_BATCH_RELEASE_SCHEMA = "mei-mw-visible-batches-release-v1"
MW_BATCH_PROMPT_ID = "mei-mw-disposition-batched-prompt-v1"
MW_BATCH_TASK = contract.TASK_CONTRACT
ALIGNMENT_VIEW_SCHEMA = "mei-schema-budget-alignment-view-v1"
ALIGNMENT_RELEASE_SCHEMA = "mei-schema-budget-alignment-release-v1"

_NO_MATCH_QUERIES = (
    "请用一句话解释什么是耐心。",
    "帮我把这句话改得更礼貌一些。",
    "给孩子讲一个关于月亮的短故事。",
    "这段文字的中心思想是什么？",
    "请比较两个观点的共同点。",
    "写一首四行的秋天小诗。",
    "把下面这句话翻译成法语。",
    "我今天有点紧张，陪我聊两句。",
    "解释一下为什么天空看起来是蓝色的。",
    "请总结这段没有提供的文章。",
    "给我三个学习新概念的方法。",
    "把这句话改写成疑问句。",
    "什么是比喻，举一个简单例子。",
    "请判断这句话的语气是正式还是口语。",
    "帮我想一个不涉及设备操作的昵称。",
    "描述一下雨后空气的感觉。",
    "请把一到十写成中文大写。",
    "为什么人需要睡眠？",
    "给出一个适合课堂讨论的话题。",
    "请解释成语画蛇添足。",
)


def _canonical(value: Any) -> bytes:
    return contract.canonical_bytes(value)


def _sha(value: Any) -> str:
    return contract.sha_bytes(_canonical(value))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical(row) + b"\n")
            count += 1
    return count


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": contract.sha_file(path),
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def deterministic_catalog_subset(
    *,
    sample_id: str,
    catalog: Sequence[dict[str, Any]],
    size: int,
    required_tool_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if size not in {10, 20, 50}:
        raise ValueError("calibration catalog size must be 10, 20 or 50")
    by_name = {str(tool.get("name") or ""): tool for tool in catalog}
    required = []
    for name in required_tool_ids:
        key = str(name)
        if key and key not in required:
            if key not in by_name:
                raise ValueError(f"calibration tool is outside catalog: {key}")
            required.append(key)
    if len(required) > size:
        required = required[:size]
    rest = sorted(
        (name for name in by_name if name not in required),
        key=lambda name: (hashlib.sha256(f"{sample_id}\0{name}".encode()).hexdigest(), name),
    )
    names = required + rest[: size - len(required)]
    return [by_name[name] for name in names]


def no_match_validation_rows(catalog: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in (10, 20, 50):
        for index, query in enumerate(_NO_MATCH_QUERIES):
            sample_id = f"NOMATCH-{size}-{index:03d}"
            subset = deterministic_catalog_subset(
                sample_id=sample_id, catalog=catalog, size=size
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "query": query,
                    "catalog_size": size,
                    "catalog_tool_ids": [str(tool["name"]) for tool in subset],
                    "gold_tool": None,
                    "kind": "no_match",
                }
            )
    return rows


def _global_raw_ranking(
    runtime: Any, query: str, catalog: Sequence[dict[str, Any]]
) -> list[RankedCandidate]:
    ranked = runtime.search_ranked(query, list(catalog))
    return [
        RankedCandidate(
            tool_id=row.tool_id,
            schema=row.schema,
            raw_score=float(row.raw_score),
            relevance=0.0,
            rank=index + 1,
        )
        for index, row in enumerate(ranked)
    ]


def calibrate_retrieval_v1(
    runtime: Any,
    positive_rows: Sequence[dict[str, Any]],
    catalog: Sequence[dict[str, Any]],
    *,
    model_sha256: str,
    retrieval_head_sha256: str,
    tokenizer_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit Platt parameters and both thresholds on 10/20/50 validation only."""

    validation_specs: list[dict[str, Any]] = []
    raw_cache: dict[str, list[RankedCandidate]] = {}
    by_name = {str(tool.get("name") or ""): tool for tool in catalog}
    for source in positive_rows:
        sample_id = str(source.get("sample_id") or "")
        gold = str(source.get("gold_tool") or "")
        if not sample_id or gold not in by_name:
            raise ValueError("retrieval validation row lacks an in-catalog gold tool")
        required = [gold, *[str(value) for value in source.get("hard_negatives") or []]]
        raw_cache[sample_id] = _global_raw_ranking(
            runtime, str(source.get("query") or ""), catalog
        )
        for size in (10, 20, 50):
            subset = deterministic_catalog_subset(
                sample_id=f"{sample_id}:{size}",
                catalog=catalog,
                size=size,
                required_tool_ids=required,
            )
            validation_specs.append(
                {
                    "sample_id": sample_id,
                    "query": str(source.get("query") or ""),
                    "catalog_size": size,
                    "catalog_tool_ids": [str(tool["name"]) for tool in subset],
                    "gold_tool": gold,
                    "kind": "positive",
                }
            )
    no_match = no_match_validation_rows(catalog)
    for row in no_match:
        raw_cache[row["sample_id"]] = _global_raw_ranking(
            runtime, str(row["query"]), catalog
        )
    validation_specs.extend(no_match)

    scored_labels: list[tuple[float, int]] = []
    seen_pairs: set[tuple[str, str, int]] = set()
    filtered_raw: list[tuple[dict[str, Any], list[RankedCandidate]]] = []
    for spec in validation_specs:
        allowed = set(spec["catalog_tool_ids"])
        ranked = [row for row in raw_cache[spec["sample_id"]] if row.tool_id in allowed]
        gold = spec.get("gold_tool")
        for row in ranked:
            key = (str(spec["sample_id"]), row.tool_id, int(spec["catalog_size"]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            scored_labels.append((row.raw_score, int(gold is not None and row.tool_id == gold)))
        filtered_raw.append((spec, ranked))

    platt = fit_platt_calibrator(scored_labels)
    threshold_examples: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for spec, raw_rows in filtered_raw:
        ranked = [
            RankedCandidate(
                tool_id=row.tool_id,
                schema=row.schema,
                raw_score=row.raw_score,
                relevance=platt_relevance(
                    row.raw_score, scale=platt["scale"], bias=platt["bias"]
                ),
                rank=index + 1,
            )
            for index, row in enumerate(raw_rows)
        ]
        gold = spec.get("gold_tool")
        gold_index = next(
            (index for index, row in enumerate(ranked) if row.tool_id == gold), None
        )
        threshold_examples.append(
            {
                "catalog_size": int(spec["catalog_size"]),
                "gold_index": gold_index,
                "relevances": [row.relevance for row in ranked],
            }
        )
        evidence_rows.append(
            {
                **spec,
                "ranking": [row.as_dict() for row in ranked],
                "gold_rank": None if gold_index is None else gold_index + 1,
            }
        )
    thresholds = select_retrieval_thresholds(threshold_examples)
    validation_identity = {
        "id": CALIBRATION_VALIDATION_ID,
        "model_sha256": model_sha256,
        "retrieval_head_sha256": retrieval_head_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "rows": [
            {
                "sample_id": row["sample_id"],
                "catalog_size": row["catalog_size"],
                "catalog_tool_ids": row["catalog_tool_ids"],
                "gold_tool": row.get("gold_tool"),
            }
            for row in evidence_rows
        ],
    }
    calibration = {
        "calibration_id": RETRIEVAL_CALIBRATION_ID,
        "scale": float(platt["scale"]),
        "bias": float(platt["bias"]),
        "discard_threshold": float(thresholds["discard_threshold"]),
        "expand_threshold": float(thresholds["expand_threshold"]),
        "validated": bool(thresholds["validated"]),
        "fallback_scan_all_eligible": bool(thresholds["fallback_scan_all_eligible"]),
        "validation_manifest_sha256": _sha(validation_identity),
        "catalog_sizes": [10, 20, 50],
        "metrics": {
            **thresholds["metrics"],
            "platt_brier": platt["brier"],
            "platt_rows": platt["rows"],
            "positive_validation_rows": len(positive_rows) * 3,
            "no_match_validation_rows": len(no_match),
        },
    }
    return calibration, evidence_rows


def calibrated_ranking(
    runtime: Any,
    query: str,
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
) -> list[RankedCandidate]:
    raw = _global_raw_ranking(runtime, query, catalog)
    return [
        RankedCandidate(
            tool_id=row.tool_id,
            schema=row.schema,
            raw_score=row.raw_score,
            relevance=platt_relevance(
                row.raw_score,
                scale=float(calibration["scale"]),
                bias=float(calibration["bias"]),
            ),
            rank=index + 1,
        )
        for index, row in enumerate(raw)
    ]


def _oracle_ranking(
    row: Mapping[str, Any], tools_by_name: Mapping[str, dict[str, Any]]
) -> list[RankedCandidate]:
    names = [str(value) for value in row.get("retrieved_tools") or []][:5]
    if not names or any(name not in tools_by_name for name in names):
        raise ValueError(f"MW oracle view cannot resolve tools: {row.get('sample_id')}")
    return [
        RankedCandidate(
            tool_id=name,
            schema=tools_by_name[name],
            raw_score=1.0 - index * 1e-6,
            relevance=1.0,
            rank=index + 1,
        )
        for index, name in enumerate(names)
    ]


def _runtime_request(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "wire_version": "mei-runtime-wire-v2",
        "query": str(row.get("query") or ""),
        "context": dict(row.get("context") or {}),
        "evidence": list(row.get("evidence") or []),
        "permissions": dict(row.get("permissions") or {}),
        "state": dict(row.get("state") or {}),
        "history": list(row.get("history") or []),
        "tool_results": list(
            row.get("tool_results") or row.get("prior_tool_results") or []
        ),
    }


def iter_mw_visible_batches(
    runtime: Any,
    rows: Sequence[dict[str, Any]],
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    *,
    retrieval_mode: str,
    profile: str,
    learned_ranking_cache: dict[str, list[RankedCandidate]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield frozen prompt rows; no retrieval occurs in the MW trainer."""

    if retrieval_mode not in {"oracle", "learned"}:
        raise ValueError("retrieval_mode must be oracle or learned")
    tools_by_name = {str(tool.get("name") or ""): tool for tool in catalog}
    discard = 0.0 if retrieval_mode == "oracle" else float(calibration["discard_threshold"])
    expand = 0.0 if retrieval_mode == "oracle" else float(calibration["expand_threshold"])
    for source in rows:
        if retrieval_mode == "oracle":
            ranking = _oracle_ranking(source, tools_by_name)
        else:
            cache_key = str(source.get("sample_id") or "")
            if learned_ranking_cache is not None and cache_key in learned_ranking_cache:
                ranking = learned_ranking_cache[cache_key]
            else:
                ranking = calibrated_ranking(
                    runtime, str(source.get("query") or ""), catalog, calibration
                )
                if learned_ranking_cache is not None:
                    learned_ranking_cache[cache_key] = ranking
        plan = plan_candidate_batches(
            ranking,
            discard_threshold=discard,
            expand_threshold=expand,
        )
        ranking_payload = [candidate.as_dict() for candidate in ranking]
        ranking_sha = _sha(ranking_payload)
        if not plan.batches:
            yield {
                "schema": MW_VISIBLE_BATCH_SCHEMA,
                "view_id": _sha([source.get("sample_id"), retrieval_mode, profile, "terminal"]),
                "source_sample_id": source.get("sample_id"),
                "split": source.get("split"),
                "retrieval_mode": retrieval_mode,
                "runtime_profile": profile,
                "retrieval_terminal": "retrieval_no_match",
                "mw_eligible": False,
                "full_ranking": ranking_payload,
                "full_ranking_sha256": ranking_sha,
                "threshold_filtered_tools": [],
                "original_reason_class_id": int(source["reason_class_id"]),
                "effective_reason_class_id": None,
                "original_reason_code": str(source.get("reason_code") or ""),
                "effective_reason_code": None,
            }
            continue
        queue = [candidate for batch in plan.batches for candidate in batch]
        batch_index = 0
        while queue:
            requested = list(queue[:5])
            del queue[: len(requested)]
            rendered = render_budgeted_request(
                _runtime_request(source),
                [candidate.schema for candidate in requested],
                runtime.tokenizer,
                relevances=[candidate.relevance for candidate in requested],
                runtime_profile=profile,
                output_reserve=128,
                already_normalized=True,
                task_contract=MW_BATCH_TASK,
                prompt_suffix=contract.ASSISTANT_SUFFIX,
            )
            selected_names = list(rendered.get("selected_tools") or [])
            selected_count = len(selected_names)
            if selected_count < len(requested):
                queue = requested[selected_count:] + queue
            if rendered.get("error") or selected_count == 0:
                raise RuntimeError(
                    f"MW view is context-unrepresentable: {source.get('sample_id')} {profile}"
                )
            batch_index += 1
            original_label = int(source["reason_class_id"])
            target_tool = str(
                source.get("candidate_tool")
                or source.get("gold_name")
                or (
                    (source.get("retrieved_tools") or [""])[0]
                    if original_label == 0
                    else ""
                )
            )
            if original_label == 0 and target_tool and target_tool not in selected_names:
                effective_label = 10
                effective_reason = "capability_insufficient"
            else:
                effective_label = original_label
                effective_reason = str(source.get("reason_code") or "")
            view_identity = [
                source.get("sample_id"),
                retrieval_mode,
                profile,
                batch_index,
                rendered["schema_projection_sha256"],
                effective_label,
            ]
            yield {
                "schema": MW_VISIBLE_BATCH_SCHEMA,
                "view_id": _sha(view_identity),
                "source_sample_id": source.get("sample_id"),
                "split": source.get("split"),
                "family": source.get("family"),
                "retrieval_mode": retrieval_mode,
                "runtime_profile": profile,
                "retrieval_terminal": None,
                "mw_eligible": True,
                "batch_index": batch_index,
                "full_ranking": ranking_payload,
                "full_ranking_sha256": ranking_sha,
                "threshold_filtered_tools": [row.tool_id for row in plan.candidates],
                "batch_tools_before_budget": [row.tool_id for row in requested],
                "visible_tools": selected_names,
                "remaining_tools": [row.tool_id for row in queue],
                "raw_scores": [row.raw_score for row in requested[:selected_count]],
                "retrieval_relevance": [row.relevance for row in requested[:selected_count]],
                "schema_budget": rendered["schema_budget"],
                "input_budget": rendered["input_budget"],
                "schema_projection_sha256": rendered["schema_projection_sha256"],
                "prompt_tokens": rendered["prompt_tokens"],
                "prompt": rendered["prompt"],
                "sink": rendered["sink"],
                "ordinary": rendered["ordinary"],
                "prompt_id": MW_BATCH_PROMPT_ID,
                "original_reason_class_id": original_label,
                "effective_reason_class_id": effective_label,
                "original_reason_code": str(source.get("reason_code") or ""),
                "effective_reason_code": effective_reason,
                "target_tool": target_tool or None,
                "context_packer_id": CONTEXT_PACKER_ID,
                "retrieval_batch_policy_id": RETRIEVAL_BATCH_POLICY_ID,
            }
            # Runtime scans another candidate batch only for the explicit
            # capability-insufficient disposition.  Safety, permission,
            # state, evidence and ready-to-execute outcomes are terminal for
            # this retrieval pass, so unreachable later prompts are not
            # admitted to the frozen MW training artifact.
            if effective_label != 10:
                break


def freeze_mw_visible_batches(
    runtime: Any,
    splits: Mapping[str, Sequence[dict[str, Any]]],
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    out_dir: Path,
    *,
    input_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite MW visible views: {out_dir}")
    out_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split, rows in sorted(splits.items()):
        learned_ranking_cache: dict[str, list[RankedCandidate]] = {}
        for mode in ("oracle", "learned"):
            for profile in ("compact", "standard"):
                path = out_dir / f"mw-visible.{split}.{mode}.{profile}.jsonl"
                counts = {str(index): 0 for index in range(20)}

                def generated() -> Iterable[dict[str, Any]]:
                    for row in iter_mw_visible_batches(
                        runtime,
                        rows,
                        catalog,
                        calibration,
                        retrieval_mode=mode,
                        profile=profile,
                        learned_ranking_cache=learned_ranking_cache,
                    ):
                        label = row.get("effective_reason_class_id")
                        if row.get("mw_eligible") and label is not None:
                            counts[str(label)] += 1
                        yield row

                count = _write_jsonl(path, generated())
                artifacts[path.name] = _artifact(path, rows=count)
                class_counts[path.name] = counts
    manifest = {
        "schema": MW_VISIBLE_BATCH_RELEASE_SCHEMA,
        "status": "frozen",
        "prompt_id": MW_BATCH_PROMPT_ID,
        "context_packer_id": CONTEXT_PACKER_ID,
        "retrieval_batch_policy_id": RETRIEVAL_BATCH_POLICY_ID,
        "calibration": dict(calibration),
        "input_artifacts": dict(input_artifacts),
        "artifacts": artifacts,
        "effective_class_counts": class_counts,
        "source_release_mutated": False,
        "trainer_live_retrieval_allowed": False,
        "runtime_shared_prompt_framing": True,
    }
    manifest["release_fingerprint_sha256"] = _sha(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _alignment_mode(sample_id: str) -> tuple[str, str]:
    """Freeze the declared 25% anchor / 25% standard / 50% compact mix."""

    bucket = int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:8], 16) % 4
    if bucket == 0:
        return "anchor_top5", "standard"
    if bucket == 1:
        return "adaptive", "standard"
    return "adaptive", "compact"


def iter_alignment_views(
    runtime: Any,
    rows: Sequence[dict[str, Any]],
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    *,
    task: str,
) -> Iterable[dict[str, Any]]:
    """Materialize quant-aware LM replay prompts including later-batch gold."""

    if task not in {"full_call", "agent_continuation"}:
        raise ValueError("alignment task must be full_call or agent_continuation")
    tools_by_name = {str(tool.get("name") or ""): tool for tool in catalog}
    for source in rows:
        sample_id = str(source.get("sample_id") or "")
        mode, profile = _alignment_mode(sample_id)
        if mode == "anchor_top5":
            ranking = _oracle_ranking(source, tools_by_name)
            plan = plan_candidate_batches(ranking)
        else:
            ranking = calibrated_ranking(
                runtime, str(source.get("query") or ""), catalog, calibration
            )
            plan = plan_candidate_batches(
                ranking,
                discard_threshold=float(calibration["discard_threshold"]),
                expand_threshold=float(calibration["expand_threshold"]),
            )
        queue = [candidate for batch in plan.batches for candidate in batch]
        expected = list(source.get("answers") or [])
        gold_tool = str(
            source.get("gold_name")
            or ((expected[0] or {}).get("name") if expected and isinstance(expected[0], dict) else "")
        )
        terminal_refusal = not expected
        if not queue:
            continue
        batch_index = 0
        while queue:
            requested = list(queue[:5])
            del queue[: len(requested)]
            rendered = render_budgeted_request(
                _runtime_request(source),
                [candidate.schema for candidate in requested],
                runtime.tokenizer,
                relevances=[candidate.relevance for candidate in requested],
                runtime_profile=profile,
                output_reserve=128,
                already_normalized=True,
                task_contract=contract.TASK_CONTRACT,
                prompt_suffix=contract.ASSISTANT_SUFFIX,
            )
            selected_names = list(rendered.get("selected_tools") or [])
            selected_count = len(selected_names)
            if selected_count < len(requested):
                queue = requested[selected_count:] + queue
            if rendered.get("error") or selected_count == 0:
                raise RuntimeError(
                    f"alignment view is context-unrepresentable: {sample_id} {profile}"
                )
            batch_index += 1
            gold_visible = bool(gold_tool and gold_tool in selected_names)
            answers = expected if gold_visible else []
            if terminal_refusal:
                outcome = "terminal_refusal"
            elif gold_visible:
                outcome = "call"
            else:
                outcome = "capability_insufficient"
            prompt = str(rendered["prompt"])
            prompt_tokens = len(
                runtime.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            )
            if prompt_tokens + 128 > 2048:
                raise RuntimeError(f"alignment prompt exceeds joint budget: {sample_id}")
            view = dict(source)
            view.update(
                {
                    "schema": ALIGNMENT_VIEW_SCHEMA,
                    "view_id": _sha(
                        [
                            sample_id,
                            task,
                            mode,
                            profile,
                            batch_index,
                            rendered["schema_projection_sha256"],
                            answers,
                        ]
                    ),
                    "source_sample_id": sample_id,
                    "alignment_task": task,
                    "retrieval_mode": mode,
                    "runtime_profile": profile,
                    "batch_index": batch_index,
                    "batch_outcome": outcome,
                    "retrieved_tools": selected_names,
                    "answers": answers,
                    "target_text": contract.serialize_tool_target(answers),
                    "_budgeted_prompt": prompt,
                    "_budgeted_prompt_tokens": prompt_tokens,
                    "_schema_budget": rendered["schema_budget"],
                    "_input_budget": rendered["input_budget"],
                    "_schema_projection_sha256": rendered[
                        "schema_projection_sha256"
                    ],
                    "_full_ranking_sha256": _sha(
                        [candidate.as_dict() for candidate in ranking]
                    ),
                    "_target_retrieval_miss": bool(
                        gold_tool
                        and gold_tool
                        not in {candidate.tool_id for candidate in queue + requested}
                        and not gold_visible
                    ),
                }
            )
            yield view
            if outcome in {"call", "terminal_refusal"} or mode == "anchor_top5":
                break


def select_alignment_replay_views(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select the exact frozen 25/25/50 replay mix without live resampling.

    Selection happens after multi-batch materialization so later-batch
    capability-insufficient and success prompts remain eligible.  Within each
    profile cell a deterministic round-robin spans training bank, target kind,
    batch outcome and batch number before repeating a cell.
    """

    if int(limit) < 4:
        raise ValueError("alignment replay limit must be at least four")
    target = {
        "anchor_top5:standard": int(limit) // 4,
        "adaptive:standard": int(limit) // 4,
        "adaptive:compact": int(limit) - 2 * (int(limit) // 4),
    }
    by_mode: dict[str, list[dict[str, Any]]] = {key: [] for key in target}
    for row in rows:
        key = f"{row.get('retrieval_mode')}:{row.get('runtime_profile')}"
        if key in by_mode:
            by_mode[key].append(dict(row))

    selected: list[dict[str, Any]] = []
    for mode in target:
        required = target[mode]
        candidates = by_mode[mode]
        if len(candidates) < required:
            raise RuntimeError(
                f"alignment replay cell {mode} has {len(candidates)} rows, needs {required}"
            )
        groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
        for row in candidates:
            cell = (
                str(row.get("_training_bank") or "unbanked"),
                str(row.get("kind") or row.get("alignment_task") or "unknown"),
                str(row.get("batch_outcome") or "unknown"),
                int(row.get("batch_index") or 0),
            )
            groups.setdefault(cell, []).append(row)
        for cell, values in groups.items():
            values.sort(key=lambda row: (_sha([cell, row.get("view_id")]), str(row.get("view_id"))))
        cells = sorted(groups)
        offsets = {cell: 0 for cell in cells}
        mode_rows: list[dict[str, Any]] = []
        while len(mode_rows) < required:
            progressed = False
            for cell in cells:
                offset = offsets[cell]
                values = groups[cell]
                if offset >= len(values):
                    continue
                mode_rows.append(values[offset])
                offsets[cell] += 1
                progressed = True
                if len(mode_rows) == required:
                    break
            if not progressed:
                raise RuntimeError(f"alignment replay cell exhausted unexpectedly: {mode}")
        selected.extend(mode_rows)
    selected.sort(key=lambda row: (_sha(row.get("view_id")), str(row.get("view_id"))))
    if len(selected) != int(limit) or len({row["view_id"] for row in selected}) != int(limit):
        raise RuntimeError("alignment replay selector produced a duplicate or incomplete sample")
    return selected


def select_alignment_source_rows(
    rows: Sequence[dict[str, Any]],
    *,
    view_limit: int,
    adaptive_oversample: int = 2,
) -> list[dict[str, Any]]:
    """Preselect a deterministic source pool before expensive LM retrieval.

    Anchor sources yield exactly one view.  Adaptive sources can yield several
    candidate batches or no view after threshold filtering, so each adaptive
    cell is oversampled and the exact 25/25/50 contract is still enforced by
    :func:`select_alignment_replay_views` after materialization.
    """

    if int(view_limit) < 4 or int(adaptive_oversample) < 1:
        raise ValueError("alignment source selection requires limit>=4 and oversample>=1")
    target = {
        "anchor_top5:standard": int(view_limit) // 4,
        "adaptive:standard": int(view_limit) // 4,
        "adaptive:compact": int(view_limit) - 2 * (int(view_limit) // 4),
    }
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in target}
    for row in rows:
        mode, profile = _alignment_mode(str(row.get("sample_id") or ""))
        pools[f"{mode}:{profile}"].append(dict(row))
    selected: list[dict[str, Any]] = []
    for cell, required in target.items():
        pool = sorted(
            pools[cell],
            key=lambda row: (_sha([cell, row.get("sample_id")]), str(row.get("sample_id"))),
        )
        wanted = required if cell.startswith("anchor_top5:") else required * int(adaptive_oversample)
        if len(pool) < required:
            raise RuntimeError(
                f"alignment source cell {cell} has {len(pool)} rows, needs {required}"
            )
        selected.extend(pool[: min(len(pool), wanted)])
    selected.sort(key=lambda row: (_sha(row.get("sample_id")), str(row.get("sample_id"))))
    return selected


def freeze_alignment_views(
    runtime: Any,
    task_splits: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    out_dir: Path,
    *,
    input_artifacts: Mapping[str, Any],
    row_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite alignment views: {out_dir}")
    out_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for task, splits in sorted(task_splits.items()):
        for split, rows in sorted(splits.items()):
            path = out_dir / f"{task}.{split}.jsonl"
            counts: dict[str, int] = {}
            batch_gold: dict[str, int] = {}
            source_rows = list(rows)
            limit_key = f"{task}.{split}"
            if row_limits is not None and limit_key in row_limits:
                source_rows = select_alignment_source_rows(
                    source_rows, view_limit=int(row_limits[limit_key])
                )
            materialized = list(
                iter_alignment_views(runtime, source_rows, catalog, calibration, task=task)
            )
            if row_limits is not None and limit_key in row_limits:
                materialized = select_alignment_replay_views(
                    materialized, limit=int(row_limits[limit_key])
                )
            for row in materialized:
                key = f"{row['retrieval_mode']}:{row['runtime_profile']}"
                counts[key] = counts.get(key, 0) + 1
                if row["batch_outcome"] == "call":
                    batch = str(row["batch_index"])
                    batch_gold[batch] = batch_gold.get(batch, 0) + 1
            count = _write_jsonl(path, materialized)
            artifacts[path.name] = _artifact(path, rows=count)
            coverage[path.name] = {
                "source_rows_total": len(rows),
                "source_rows_materialized": len(source_rows),
                "view_counts": counts,
                "gold_batch_counts": batch_gold,
            }
    manifest = {
        "schema": ALIGNMENT_RELEASE_SCHEMA,
        "status": "frozen",
        "mix_by_source": {
            "anchor_top5_standard": 0.25,
            "adaptive_standard": 0.25,
            "adaptive_compact": 0.50,
        },
        "context_packer_id": CONTEXT_PACKER_ID,
        "retrieval_batch_policy_id": RETRIEVAL_BATCH_POLICY_ID,
        "calibration": dict(calibration),
        "input_artifacts": dict(input_artifacts),
        "row_limits": dict(row_limits or {}),
        "artifacts": artifacts,
        "coverage": coverage,
        "source_release_mutated": False,
    }
    manifest["release_fingerprint_sha256"] = _sha(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    return manifest
