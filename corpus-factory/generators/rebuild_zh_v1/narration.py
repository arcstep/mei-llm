#!/usr/bin/env python3
"""Terminal-only Chinese narration generator.

Consumes ONLY already-verified terminal ToolResult objects produced by
fullcall.py (single execute/refuse) and agent.py (final trajectory step) --
never fabricates a number, status, subject or causal claim absent from the
result. One narration target per case, generated after the terminal state
only (never after intermediate ToolResults in a multi-step trajectory).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import common as C

FAMILY = "narration"


def _field_desc(tool: Mapping[str, Any], field: str) -> str:
    props = (tool.get("parameters") or {}).get("properties") or {}
    return str(props.get(field, {}).get("description") or field)


def _narrate_execute(tool: Mapping[str, Any], args: Mapping[str, Any]) -> tuple[str, str]:
    """Returns (narration, outcome_kind)."""
    for field, value in args.items():
        spec = ((tool.get("parameters") or {}).get("properties") or {}).get(field, {})
        if spec.get("type") == "boolean":
            state_word = "打开" if value else "关闭"
            return f"已{state_word}{_field_desc(tool, field)}。", "success"
    for field, value in args.items():
        spec = ((tool.get("parameters") or {}).get("properties") or {}).get(field, {})
        if spec.get("type") in ("integer", "number") and ("温度" in _field_desc(tool, field) or "摄氏" in _field_desc(tool, field)):
            return f"已调到{value}度。", "success"
    base = str(tool.get("description") or "").rstrip("。")
    return f"已完成：{base}。", "success"


def _narrate_error(tool: Mapping[str, Any], error_code: str) -> str:
    base = str(tool.get("description") or "").rstrip("。")
    return f"抱歉，{base}未能完成。"


def _narrate_refusal(kind: str, missing_fields: Sequence[str], tool: Mapping[str, Any]) -> str:
    if kind == "refuse_missing_slot" and missing_fields:
        return f"还需要{_field_desc(tool, missing_fields[0])}，请补充后我再继续。"
    if kind == "refuse_conflicting_params" and missing_fields:
        return f"{_field_desc(tool, missing_fields[0])}前后说法不一致，请确认一下具体是哪个。"
    if kind == "refuse_unsupported_param":
        return "这部分超出当前可以处理的范围，我先只帮你完成能做的部分。"
    return "这个请求暂时无法处理。"


def from_fullcall_row(deploy: C.ToolRegistry, row: Mapping[str, Any], *, seed: str) -> dict[str, Any] | None:
    if row["kind"] == "execute":
        tool = deploy.by_name.get(row["gold_name"])
        if tool is None:
            return None
        result = C.simulate_tool_result(tool, row["gold_args"], call_id="call-1")
        narration, outcome = _narrate_execute(tool, row["gold_args"])
    else:
        tool_name = row["oracle_top5"][0] if row.get("oracle_top5") else None
        tool = deploy.by_name.get(tool_name) if tool_name else None
        if tool is None:
            return None
        result = None
        narration = _narrate_refusal(row["kind"], row.get("missing_or_conflicting_fields") or [], tool)
        outcome = "refused"

    case_id = C.case_id(FAMILY, "fullcall", seed)
    return {
        "case_id": case_id,
        "cf_group": C.cf_group(FAMILY, row["kind"], row.get("gold_name") or "none", seed[-8:]),
        "family": FAMILY,
        "task": "terminal_narration_zh",
        "generator_version": C.GENERATOR_VERSION,
        "source_family": "full_call",
        "outcome_kind": outcome,
        "verified_terminal_result": result,
        "query": row["query"],
        "narration_target": narration,
        "forbidden": ["intermediate_tool_loop", "execution_authority"],
        "input_contract": "verified_terminal_result_view_only",
    }


def from_agent_row(deploy: C.ToolRegistry, row: Mapping[str, Any], *, seed: str) -> dict[str, Any] | None:
    steps = row.get("steps") or []
    if not steps:
        return None
    last = steps[-1]
    tool = deploy.by_name.get(last["call"]["name"])
    if tool is None:
        return None
    result = last["tool_result"]
    any_error = any(s["tool_result"].get("status") == "error" for s in steps[:-1])
    if result.get("status") == "error":
        narration = _narrate_error(tool, result.get("error_code", "unavailable"))
        outcome = "failure"
    elif any_error:
        narration, _ = _narrate_execute(tool, last["call"]["arguments"])
        narration = "第一步未能完成，已改用其他方式处理并完成。" if row.get("trajectory_kind") == "failure_then_alternate" else narration
        outcome = "partial_then_success"
    else:
        narration, outcome = _narrate_execute(tool, last["call"]["arguments"])
        if row.get("step_count", 1) > 1:
            outcome = "multi_step_success"

    case_id = C.case_id(FAMILY, "agent", seed)
    return {
        "case_id": case_id,
        "cf_group": C.cf_group(FAMILY, row["trajectory_kind"], tool.get("name"), seed[-8:]),
        "family": FAMILY,
        "task": "terminal_narration_zh",
        "generator_version": C.GENERATOR_VERSION,
        "source_family": "agent",
        "outcome_kind": outcome,
        "verified_terminal_result": result,
        "query": row["query"],
        "narration_target": narration,
        "forbidden": ["intermediate_tool_loop", "execution_authority"],
        "input_contract": "verified_terminal_result_view_only",
    }


def generate(deploy: C.ToolRegistry, fullcall_rows: Sequence[Mapping[str, Any]], agent_rows: Sequence[Mapping[str, Any]], *, max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fc_sample = C.stratified_sample(fullcall_rows, "kind", max_rows // 2)
    ag_sample = C.stratified_sample(agent_rows, "trajectory_kind", max_rows - max_rows // 2)
    for i, r in enumerate(fc_sample):
        out = from_fullcall_row(deploy, r, seed=f"narration:fc:{i}")
        if out:
            rows.append(out)
    for i, r in enumerate(ag_sample):
        out = from_agent_row(deploy, r, seed=f"narration:agent:{i}")
        if out:
            rows.append(out)

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_outcome_kind": _count_by(rows, "outcome_kind"),
        "by_source_family": _count_by(rows, "source_family"),
        "by_split": _count_by(rows, "split"),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
