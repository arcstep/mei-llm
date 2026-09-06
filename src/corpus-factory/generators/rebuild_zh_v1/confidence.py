#!/usr/bin/env python3
"""Confidence-calibration candidate freezer (label=null).

Per the task boundary, no SFT/QAT training happens in this cycle, so no real
runtime outcome exists yet. This module ONLY freezes: which scenarios will be
harvested, their candidate IDs, the frozen deterministic host simulator ID,
and the outcome schema -- exactly matching the existing v4 release's
`confidence-harvest.jsonl` convention (`label: null,
label_state: "pending_actual_final_runtime_outcome"`). A later cycle must
harvest real success/failure/refuse/error outcomes from the actually-trained
model before any calibration training happens.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import common as C

FAMILY = "confidence"
HOST_SIMULATOR_ID = "mei-51m-deterministic-host-simulator-rebuild-v1"


def from_fullcall_row(row: Mapping[str, Any], *, seed: str) -> dict[str, Any]:
    expected_kind = "execute" if row["kind"] == "execute" else "refuse"
    expected_call = {"name": row["gold_name"], "arguments": row["gold_args"]} if expected_kind == "execute" else None
    return {
        "case_id": C.case_id(FAMILY, "fullcall", seed),
        "cf_group": row["cf_group"],
        "family": FAMILY,
        "task": "confidence_harvest",
        "generator_version": C.GENERATOR_VERSION,
        "source_family": "full_call",
        "source_case_id": row["case_id"],
        "query": row["query"],
        "context": row["context"],
        "evidence": row["evidence"],
        "history": row["history"],
        "oracle_top5": row["oracle_top5"],
        "candidate_tool": row["gold_name"] or (row["oracle_top5"][0] if row["oracle_top5"] else None),
        "expected_kind": expected_kind,
        "expected_call": expected_call,
        "label": None,
        "label_contract": "exact_call_or_correct_refusal_after_r1",
        "label_state": "pending_actual_final_runtime_outcome",
        "host_simulator_id": HOST_SIMULATOR_ID,
    }


def from_agent_row(row: Mapping[str, Any], *, seed: str) -> dict[str, Any]:
    last = row["steps"][-1]
    expected_kind = "error" if last["tool_result"].get("status") == "error" else "execute"
    return {
        "case_id": C.case_id(FAMILY, "agent", seed),
        "cf_group": row["cf_group"],
        "family": FAMILY,
        "task": "confidence_harvest",
        "generator_version": C.GENERATOR_VERSION,
        "source_family": "agent",
        "source_case_id": row["case_id"],
        "query": row["query"],
        "context": row["context"],
        "evidence": row["evidence"],
        "history": row["history"],
        "oracle_top5": row["catalog_tool_names"],
        "candidate_tool": last["call"]["name"],
        "expected_kind": expected_kind,
        "expected_call": last["call"] if expected_kind == "execute" else None,
        "label": None,
        "label_contract": "exact_call_or_correct_refusal_after_r1",
        "label_state": "pending_actual_final_runtime_outcome",
        "host_simulator_id": HOST_SIMULATOR_ID,
    }


def from_retrieval_row(row: Mapping[str, Any], *, seed: str) -> dict[str, Any]:
    expected_kind = "no_match" if row["gold_name"] is None else ("execute" if row["retrieval_terminal_reason"] == "found" else "capability_insufficient")
    return {
        "case_id": C.case_id(FAMILY, "retrieval", seed),
        "cf_group": row["cf_group"],
        "family": FAMILY,
        "task": "confidence_harvest",
        "generator_version": C.GENERATOR_VERSION,
        "source_family": "retrieval",
        "source_case_id": row["case_id"],
        "query": row["query"],
        "context": row["context"],
        "evidence": row["evidence"],
        "history": row["history"],
        "oracle_top5": row["catalog_tool_names"][:5],
        "candidate_tool": row["gold_name"],
        "expected_kind": expected_kind,
        "expected_call": None,
        "label": None,
        "label_contract": "exact_call_or_correct_refusal_after_r1",
        "label_state": "pending_actual_final_runtime_outcome",
        "host_simulator_id": HOST_SIMULATOR_ID,
    }


def generate(
    fullcall_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    *,
    fullcall_n: int, agent_n: int, retrieval_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(C.stratified_sample(fullcall_rows, "kind", fullcall_n)):
        rows.append(from_fullcall_row(r, seed=f"confidence:fc:{i}"))
    for i, r in enumerate(C.stratified_sample(agent_rows, "trajectory_kind", agent_n)):
        rows.append(from_agent_row(r, seed=f"confidence:agent:{i}"))
    for i, r in enumerate(C.stratified_sample(retrieval_rows, "scenario", retrieval_n)):
        rows.append(from_retrieval_row(r, seed=f"confidence:retrieval:{i}"))

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_expected_kind": _count_by(rows, "expected_kind"),
        "by_source_family": _count_by(rows, "source_family"),
        "by_split": _count_by(rows, "split"),
        "positive_class_present": any(r["expected_kind"] == "execute" for r in rows),
        "negative_class_present": any(r["expected_kind"] != "execute" for r in rows),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
