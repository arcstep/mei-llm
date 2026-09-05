#!/usr/bin/env python3
"""联动轨迹 family: one interaction flows through the collaborating heads.

The six heads no longer train on isolated rows only -- a trajectory row is a
single interaction where the MW head acts as the ROUTER (ask the user / fill
via a query tool / execute), the agent head follows the route, retrieval and
full-call skills execute each call, and the terminal pairs confidence with
success (high/mid + grounded narration) or an MW reason with failure
(low/mid confidence + explanatory narration). Every label is compiled
locally and deterministically from the seed; nothing is model-generated.

Kinds (user-specified flow):
- direct_execute:   conditions ready -> route=execute -> call ok -> success
- ask_then_execute: missing slot -> route=ask -> user replies -> execute ok
- tool_fill_execute: missing external fact -> route=fill -> query tool ->
                     execute main call ok
- unfillable_fail:  missing slot and user cannot provide -> failure + reason
- execute_fail:     route=execute but the tool errors -> failure + error_code
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from . import common as C
from . import retrieval as R
from . import text_variants as tv

FAMILY = "trajectory"

KINDS = ("direct_execute", "ask_then_execute", "tool_fill_execute", "unfillable_fail", "execute_fail")
CONFIDENCE_LEVELS = ("high", "mid", "low")
_QUERY_MARKERS = ("查询", "获取", "查看", "lookup", "track", "status")


def _rand(seed: str, lo: int, hi: int) -> int:
    seed_int = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
    return lo + seed_int % (hi - lo + 1)


def _required_props(tool: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    params = tool.get("parameters") or {}
    return list(params.get("required") or []), dict(params.get("properties") or {})


def _phrase(props: Mapping[str, Any], name: str) -> str:
    return str(props.get(name, {}).get("description") or name)


def _seeded_args(tool: Mapping[str, Any], seed: str) -> dict[str, Any]:
    required, props = _required_props(tool)
    return {f: C.seeded_value(dict(props.get(f, {"type": "string"})), f"{seed}:{f}") for f in required}


def _clause_for(tool: Mapping[str, Any], args: Mapping[str, Any]) -> str:
    _, props = _required_props(tool)
    return "，".join(f"{_phrase(props, k)}是{v}" for k, v in args.items())


def _query_tools(deploy: C.ToolRegistry) -> list[Mapping[str, Any]]:
    out = []
    for t in deploy.tools:
        text = f"{t.get('description') or ''} {t.get('name') or ''}".lower()
        if any(m in text for m in _QUERY_MARKERS):
            out.append(t)
    return out


def _fill_pair(deploy: C.ToolRegistry, seed: str) -> tuple[Mapping[str, Any], str, Mapping[str, Any], str] | None:
    """(query_tool, shared_field, main_tool, main_missing_field) pair where the
    query tool can supply the main tool's missing required field."""
    query_tools = sorted(_query_tools(deploy), key=lambda t: t.get("name", ""))
    mains = sorted(deploy.tools, key=lambda t: t.get("name", ""))
    offset = _rand(seed, 0, len(mains) - 1)
    for i in range(len(mains)):
        main = mains[(offset + i) % len(mains)]
        required, _ = _required_props(main)
        if not required:
            continue
        for missing in required:
            for qidx in range(len(query_tools)):
                tq = query_tools[(qidx + _rand(f"{seed}:q", 0, len(query_tools) - 1)) % len(query_tools)]
                if tq.get("name") == main.get("name"):
                    continue
                tq_props = (tq.get("parameters") or {}).get("properties") or {}
                if missing in tq_props:
                    return tq, missing, main, missing
    return None


def _mw_step(step_id: str, route: str, reason_code: str, candidate_tool: str, missing_fields: list[str] | None = None) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "role": "mw",
        "route": route,
        "reason_code": reason_code,
        "candidate_tool": candidate_tool,
        "missing_fields": missing_fields or [],
    }


def _ask_step(step_id: str, question: str, user_reply: str) -> dict[str, Any]:
    return {"step_id": step_id, "role": "ask_user", "question": question, "user_reply": user_reply}


def _call_step(call_id: str, tool: Mapping[str, Any], args: Mapping[str, Any], *, status: str = "ok", error_code: str | None = None) -> dict[str, Any]:
    if status == "ok":
        result = C.simulate_tool_result(tool, args, call_id=call_id)
    else:
        result = C.simulate_tool_error(tool, call_id=call_id, error_code=error_code or "tool_unavailable")
    return {"step_id": call_id, "role": "call", "call": {"name": tool.get("name"), "arguments": dict(args)}, "tool_result": result}


def _narration_success(tool: Mapping[str, Any], args: Mapping[str, Any]) -> str:
    _, props = _required_props(tool)
    detail = "，".join(f"{_phrase(props, k)}{v}" for k, v in args.items())
    desc = str(tool.get("description") or "").rstrip("。")
    return f"好的，已完成{desc}（{detail}）。"


def _narration_fail(reason_text: str, missing: Sequence[str]) -> str:
    if missing:
        return f"抱歉，这个请求暂时办不了：还缺{'、'.join(missing)}。你补充一下我再继续。"
    return f"抱歉，这次没能完成：{reason_text}。"


def _terminal(step_id: str, *, outcome: str, confidence: str, narration: str, mw_reason: str, error_code: str | None = None) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "role": "terminal",
        "outcome": outcome,
        "confidence": confidence,
        "narration": narration,
        "mw_reason": mw_reason,
        "error_code": error_code,
    }


def build_trajectory(deploy: C.ToolRegistry, *, kind: str, seed: str) -> dict[str, Any] | None:
    mains = sorted([t for t in deploy.tools if _required_props(t)[0]], key=lambda t: t.get("name", ""))
    if not mains:
        return None

    steps: list[dict[str, Any]] = []
    query_parts: list[str] = []

    if kind == "direct_execute":
        tool = mains[_rand(f"{seed}:t", 0, len(mains) - 1)]
        args = _seeded_args(tool, f"{seed}:args")
        steps.append(_mw_step("mw-1", "execute", "ready_to_execute", tool.get("name")))
        steps.append(_call_step("call-1", tool, args))
        steps.append(_terminal("terminal", outcome="success", confidence=CONFIDENCE_LEVELS[_rand(f"{seed}:c", 0, 1)],
                               narration=_narration_success(tool, args), mw_reason="ready_to_execute"))
        query_parts = [f"请帮我{str(tool.get('description') or '').rstrip('。')}，{_clause_for(tool, args)}"]

    elif kind == "ask_then_execute":
        tool = mains[_rand(f"{seed}:t", 0, len(mains) - 1)]
        required, props = _required_props(tool)
        missing = required[_rand(f"{seed}:m", 0, len(required) - 1)]
        args = _seeded_args(tool, f"{seed}:args")
        reply_value = args[missing]
        question = f"请问{_phrase(props, missing)}是多少？"
        user_reply = f"{_phrase(props, missing)}是{reply_value}"
        steps.append(_mw_step("mw-1", "ask", "missing_slot", tool.get("name"), [missing]))
        steps.append(_ask_step("ask-1", question, user_reply))
        steps.append(_mw_step("mw-2", "execute", "ready_to_execute", tool.get("name")))
        steps.append(_call_step("call-1", tool, args))
        steps.append(_terminal("terminal", outcome="success", confidence=CONFIDENCE_LEVELS[_rand(f"{seed}:c", 0, 1)],
                               narration=_narration_success(tool, args), mw_reason="ready_to_execute"))
        stated = {k: v for k, v in args.items() if k != missing}
        query_parts = [f"请帮我{str(tool.get('description') or '').rstrip('。')}" + (f"，{_clause_for(tool, stated)}" if stated else "")]

    elif kind == "tool_fill_execute":
        pair = _fill_pair(deploy, f"{seed}:pair")
        if pair is None:
            return None
        tq, _, main, missing = pair
        query_args = _seeded_args(tq, f"{seed}:qargs")
        main_args = _seeded_args(main, f"{seed}:margs")
        main_args[missing] = C.seeded_value(
            dict(((main.get("parameters") or {}).get("properties") or {}).get(missing, {"type": "string"})),
            f"{seed}:qargs:{missing}",
        )
        steps.append(_mw_step("mw-1", "fill", "missing_external_fact", tq.get("name"), [missing]))
        steps.append(_call_step("call-1", tq, query_args))
        steps.append(_mw_step("mw-2", "execute", "ready_to_execute", main.get("name")))
        steps.append(_call_step("call-2", main, main_args))
        steps.append(_terminal("terminal", outcome="success", confidence=CONFIDENCE_LEVELS[_rand(f"{seed}:c", 0, 1)],
                               narration=_narration_success(main, main_args), mw_reason="ready_to_execute"))
        stated = {k: v for k, v in main_args.items() if k != missing}
        query_parts = [f"先帮我{str(tq.get('description') or '').rstrip('。')}，然后{str(main.get('description') or '').rstrip('。')}" + (f"，{_clause_for(main, stated)}" if stated else "")]

    elif kind == "unfillable_fail":
        tool = mains[_rand(f"{seed}:t", 0, len(mains) - 1)]
        required, props = _required_props(tool)
        missing = required[_rand(f"{seed}:m", 0, len(required) - 1)]
        question = f"请问{_phrase(props, missing)}是多少？"
        user_reply = "这个我也不清楚，查不到"
        steps.append(_mw_step("mw-1", "ask", "missing_slot", tool.get("name"), [missing]))
        steps.append(_ask_step("ask-1", question, user_reply))
        steps.append(_terminal("terminal", outcome="failure", confidence="low",
                               narration=_narration_fail("缺少必要信息", [_phrase(props, missing)]),
                               mw_reason="missing_slot"))
        stated = {k: v for k, v in _seeded_args(tool, f"{seed}:args").items() if k != missing}
        query_parts = [f"请帮我{str(tool.get('description') or '').rstrip('。')}" + (f"，{_clause_for(tool, stated)}" if stated else "")]

    elif kind == "execute_fail":
        tool = mains[_rand(f"{seed}:t", 0, len(mains) - 1)]
        args = _seeded_args(tool, f"{seed}:args")
        error_code = "tool_unavailable"
        steps.append(_mw_step("mw-1", "execute", "ready_to_execute", tool.get("name")))
        steps.append(_call_step("call-1", tool, args, status="error", error_code=error_code))
        steps.append(_terminal("terminal", outcome="failure", confidence="low",
                               narration=_narration_fail("工具暂时不可用，稍后再试", []),
                               mw_reason="ready_to_execute", error_code=error_code))
        query_parts = [f"请帮我{str(tool.get('description') or '').rstrip('。')}，{_clause_for(tool, args)}"]

    else:
        return None

    base_query = "，".join(query_parts) + "。"
    variant_kind = tv.VARIANT_KINDS[int(C.sha256_text(seed)[:6], 16) % len(tv.VARIANT_KINDS)]
    query = tv.build_variant(base_query, variant_kind, seed=seed)
    involved = sorted({s["call"]["name"] for s in steps if s.get("role") == "call"})
    pool = [t for t in deploy.tools if t.get("name") not in involved]
    first = next((deploy.by_name[n] for n in involved if n in deploy.by_name), deploy.tools[0])
    distractors = R._candidates_sorted_by_similarity(first, pool)[:3]
    catalog = list(dict.fromkeys(involved + [t.get("name") for t in distractors]))[:5]

    _, receipt = C.fit_batch_to_budget(
        tools_batch=[deploy.by_name[n] for n in catalog if n in deploy.by_name],
        query=query, context={"locale": "zh-CN"}, evidence=[],
        history=[s["user_reply"] for s in steps if s.get("role") == "ask_user"],
        tool_results=[s["tool_result"] for s in steps if s.get("role") == "call"][:-1],
        profile="standard",
    )

    success = steps[-1].get("outcome") == "success"
    return {
        "case_id": C.case_id(FAMILY, kind, seed),
        "cf_group": C.cf_group(FAMILY, kind, seed[-8:]),
        "family": FAMILY,
        "task": "trajectory",
        "generator_version": C.GENERATOR_VERSION,
        "trajectory_kind": kind,
        "step_count": len(steps),
        "query": query,
        "context": {"locale": "zh-CN"},
        "evidence": [],
        "history": [],
        "permissions": {"principal": "user", "grants": ["standard"]},
        "state": {"running": True},
        "steps": steps,
        "catalog_tool_names": catalog,
        "final_outcome": "success" if success else "failure",
        "final_confidence": steps[-1].get("confidence"),
        "final_mw_reason": steps[-1].get("mw_reason"),
        "narration": steps[-1].get("narration"),
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
        "source_role": "deterministic-host-simulator",
    }


def generate(deploy: C.ToolRegistry, *, per_kind_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        made = 0
        i = 0
        while made < per_kind_count and i < per_kind_count * 8:
            seed = f"trajectory:{kind}:{i}"
            row = build_trajectory(deploy, kind=kind, seed=seed)
            i += 1
            if row is None:
                continue
            row["split"] = C.assign_split(row["cf_group"])
            rows.append(row)
            made += 1
    coverage = {
        "total_rows": len(rows),
        "by_kind": {k: sum(1 for r in rows if r["trajectory_kind"] == k) for k in KINDS},
        "by_outcome": {k: sum(1 for r in rows if r["final_outcome"] == k) for k in ("success", "failure")},
        "success_confidence": {c: sum(1 for r in rows if r["final_outcome"] == "success" and r["final_confidence"] == c) for c in CONFIDENCE_LEVELS},
        "routes_seen": {k: sum(1 for r in rows for s in r["steps"] if s.get("role") == "mw" and s.get("route") == k) for k in ("ask", "fill", "execute")},
    }
    return rows, coverage
