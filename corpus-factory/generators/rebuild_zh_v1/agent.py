#!/usr/bin/env python3
"""1-4 step Agent trajectory generator.

Every trajectory is `call -> verified non-empty ToolResult -> call|respond|
refuse|error`, produced by the frozen deterministic host simulator in
common.py. Step N+1 arguments are bound to step N's real simulated result
values (never fabricated facts). Covers: query-then-act, act-then-verify,
failure-then-alternate-tool, and state-change multi-tool collaboration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import common as C
from . import retrieval as R
from . import text_variants as tv

FAMILY = "agent"


def _rotate(tools: list[Mapping[str, Any]], seed: str, n: int) -> list[Mapping[str, Any]]:
    if len(tools) <= n:
        return list(tools)
    offset = int(C.sha256_text(seed)[:6], 16) % len(tools)
    return [tools[(offset + i) % len(tools)] for i in range(n)]


def _required_props(tool: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    params = tool.get("parameters") or {}
    return list(params.get("required") or []), dict(params.get("properties") or {})


def _build_args(tool: Mapping[str, Any], seed: str, prior_result: Mapping[str, Any] | None) -> dict[str, Any]:
    required, props = _required_props(tool)
    args: dict[str, Any] = {}
    prior_payload = (prior_result or {}).get("payload") or {}
    for f in required:
        spec = dict(props.get(f, {"type": "string"}))
        bound = None
        for prefix in ("confirmed_", "observed_"):
            key = prefix + f
            if key in prior_payload:
                bound = prior_payload[key]
                break
        args[f] = bound if bound is not None else C.seeded_value(spec, f"{seed}:{f}")
    return args


def _phrase(props: Mapping[str, Any], name: str) -> str:
    return str(props.get(name, {}).get("description") or name)


def _clause_for(tool: Mapping[str, Any], args: Mapping[str, Any]) -> str:
    _, props = _required_props(tool)
    parts = [f"{_phrase(props, k)}是{v}" for k, v in args.items()]
    return "，".join(parts)


def _family_tools(deploy: C.ToolRegistry, family: str, min_count: int = 2) -> list[Mapping[str, Any]]:
    names = deploy.families.get(family, [])
    return [deploy.by_name[n] for n in names] if len(names) >= min_count else []


def _step(call_id: str, tool: Mapping[str, Any], args: Mapping[str, Any], *, status: str = "ok", error_code: str | None = None) -> dict[str, Any]:
    if status == "ok":
        result = C.simulate_tool_result(tool, args, call_id=call_id)
    else:
        result = C.simulate_tool_error(tool, call_id=call_id, error_code=error_code or "unavailable")
    return {
        "call_id": call_id,
        "call": {"name": tool.get("name"), "arguments": dict(args)},
        "tool_result": result,
    }


def build_trajectory(deploy: C.ToolRegistry, family: str, *, kind: str, seed: str) -> dict[str, Any] | None:
    tools = _family_tools(deploy, family, min_count=2)
    if not tools:
        return None
    tools = sorted(tools, key=lambda t: t.get("name", ""))

    steps: list[dict[str, Any]] = []
    query_parts: list[str] = []
    final_disposition = "respond"

    if kind == "query_then_act" and len(tools) >= 2:
        t1, t2 = _rotate(tools, seed, 2)
        a1 = _build_args(t1, seed + ":1", None)
        s1 = _step("call-1", t1, a1)
        steps.append(s1)
        a2 = _build_args(t2, seed + ":2", s1["tool_result"])
        s2 = _step("call-2", t2, a2)
        steps.append(s2)
        query_parts = [f"先帮我{t1.get('description','').rstrip('。')}", f"再根据结果{t2.get('description','').rstrip('。')}"]

    elif kind == "act_then_verify" and len(tools) >= 2:
        t1, t2 = _rotate(tools, seed, 2)
        a1 = _build_args(t1, seed + ":1", None)
        s1 = _step("call-1", t1, a1)
        steps.append(s1)
        a2 = _build_args(t2, seed + ":2", s1["tool_result"])
        s2 = _step("call-2", t2, a2)
        steps.append(s2)
        query_parts = [f"帮我{t1.get('description','').rstrip('。')}", "然后帮我确认一下是否生效"]

    elif kind == "failure_then_alternate" and len(tools) >= 2:
        t1, t2 = _rotate(tools, seed, 2)
        a1 = _build_args(t1, seed + ":1", None)
        s1 = _step("call-1", t1, a1, status="error", error_code="tool_unavailable")
        steps.append(s1)
        a2 = _build_args(t2, seed + ":2", None)
        s2 = _step("call-2", t2, a2)
        steps.append(s2)
        query_parts = [f"帮我{t1.get('description','').rstrip('。')}", "如果不行就换个方式帮我搞定"]

    elif kind == "single_step_direct" and len(tools) >= 1:
        t1 = _rotate(tools, seed, 1)[0]
        a1 = _build_args(t1, seed + ":1", None)
        s1 = _step("call-1", t1, a1)
        steps.append(s1)
        query_parts = [f"帮我{t1.get('description','').rstrip('。')}"]

    elif kind == "state_change_3step" and len(tools) >= 3:
        chosen = _rotate(tools, seed, 3)
        prior_result = None
        for idx, t in enumerate(chosen):
            args = _build_args(t, f"{seed}:{idx}", prior_result)
            s = _step(f"call-{idx+1}", t, args)
            steps.append(s)
            prior_result = s["tool_result"]
        query_parts = [f"依次帮我把这几件事办完：{'，然后'.join(t.get('description','').rstrip('。') for t in chosen)}"]

    elif kind == "state_change_multistep" and len(tools) >= 4:
        chosen = _rotate(tools, seed, 4)
        prior_result = None
        for idx, t in enumerate(chosen):
            args = _build_args(t, f"{seed}:{idx}", prior_result)
            s = _step(f"call-{idx+1}", t, args)
            steps.append(s)
            prior_result = s["tool_result"]
        query_parts = [f"依次帮我把这几件事办完：{'，然后'.join(t.get('description','').rstrip('。') for t in chosen)}"]
    else:
        return None

    base_query = "，".join(query_parts) + "。"
    variant_kind = tv.VARIANT_KINDS[int(C.sha256_text(seed)[:6], 16) % len(tv.VARIANT_KINDS)]
    query = tv.build_variant(base_query, variant_kind, seed=seed)
    tool_names_involved = [s["call"]["name"] for s in steps]
    oracle_pool = [t for t in deploy.tools if t.get("name") not in tool_names_involved]
    distractors = R._candidates_sorted_by_similarity(tools[0], oracle_pool)[:2]
    catalog = list(dict.fromkeys(tool_names_involved))[:5] + [t.get("name") for t in distractors]
    catalog = catalog[:5]

    _, receipt = C.fit_batch_to_budget(
        tools_batch=[deploy.by_name[n] for n in catalog if n in deploy.by_name],
        query=query, context={"locale": "zh-CN"}, evidence=[], history=[],
        tool_results=[s["tool_result"] for s in steps[:-1]], profile="standard",
    )

    row = {
        "case_id": C.case_id(FAMILY, kind, seed),
        "cf_group": C.cf_group(FAMILY, kind, family, seed[-8:]),
        "family": FAMILY,
        "task": "agent_continuation",
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
        "final_disposition": final_disposition,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
        "source_role": "deterministic-host-simulator",
    }
    return row


TRAJECTORY_KINDS = (
    "single_step_direct", "query_then_act", "act_then_verify",
    "failure_then_alternate", "state_change_3step", "state_change_multistep",
)


def generate(deploy: C.ToolRegistry, *, per_kind_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families = [f for f in deploy.family_names if len(deploy.families[f]) >= 2]
    rows: list[dict[str, Any]] = []
    for kind in TRAJECTORY_KINDS:
        made = 0
        i = 0
        while made < per_kind_count and i < per_kind_count * 6:
            family = families[(hash(kind) + i) % len(families)]
            seed = f"agent:{kind}:{i}"
            row = build_trajectory(deploy, family, kind=kind, seed=seed)
            i += 1
            if row is None:
                continue
            rows.append(row)
            made += 1

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_trajectory_kind": _count_by(rows, "trajectory_kind"),
        "by_step_count": _count_by(rows, "step_count"),
        "by_split": _count_by(rows, "split"),
        "non_empty_verified_results": sum(1 for r in rows if all(s["tool_result"].get("verified") for s in r["steps"])),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
