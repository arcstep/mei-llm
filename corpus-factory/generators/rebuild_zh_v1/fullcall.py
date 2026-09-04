#!/usr/bin/env python3
"""Full-call / argument-filling generator.

Every emitted `gold_args` is validated against the tool's complete registered
schema (compiler_v1.validate_instance), never a trimmed projection. Covers
required/optional, all scalar+array types, enum/const, numeric bounds,
multipleOf, string length/pattern constraints, missing/conflicting/
unsupported parameters, unit conversion, Chinese numerals and
context-supplied (not query-stated) required values.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import common as C
from . import retrieval as R

FAMILY = "full_call"

_CN_DIGITS = {0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def chinese_numeral(n: int) -> str:
    if 0 <= n <= 10:
        return _CN_DIGITS[n]
    if 11 <= n <= 19:
        return "十" + _CN_DIGITS[n - 10] if n != 10 else "十"
    if 20 <= n <= 99:
        tens, rem = divmod(n, 10)
        out = _CN_DIGITS[tens] + "十"
        if rem:
            out += _CN_DIGITS[rem]
        return out
    return str(n)


def _is_temperature_field(name: str, spec: Mapping[str, Any]) -> bool:
    desc = str(spec.get("description") or "") + name
    return "摄氏" in desc or "温度" in desc


def _required_and_props(tool: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    params = tool.get("parameters") or {}
    return list(params.get("required") or []), dict(params.get("properties") or {})


def _phrase(props: Mapping[str, Any], name: str) -> str:
    return str(props.get(name, {}).get("description") or name)


def _pick_oracle_top5(deploy: C.ToolRegistry, gold: Mapping[str, Any], seed: str) -> list[str]:
    pool = [t for t in deploy.tools if t.get("name") != gold.get("name")]
    ranked = R._candidates_sorted_by_similarity(gold, pool)
    return [gold.get("name")] + [t.get("name") for t in ranked[:4]]


def _value_and_phrase(field: str, spec: Mapping[str, Any], seed: str, *, use_cn_numeral: bool, use_unit_conversion: bool) -> tuple[Any, str]:
    value = C.seeded_value(spec, seed)
    t = spec.get("type")
    if use_unit_conversion and t in ("integer", "number") and _is_temperature_field(field, spec):
        fahrenheit = round(value * 9 / 5 + 32, 1)
        return value, f"{_phrase({field: spec}, field)}调到华氏{fahrenheit}度"
    if use_cn_numeral and t == "integer" and isinstance(value, int) and 0 <= value <= 99:
        return value, f"{_phrase({field: spec}, field)}是{chinese_numeral(value)}"
    return value, f"{_phrase({field: spec}, field)}是{value}"


def _validate(tool: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    return C.validate_instance(tool.get("parameters") or {}, args)


def build_execute_row(deploy: C.ToolRegistry, tool: Mapping[str, Any], *, scenario: str, seed: str) -> dict[str, Any] | None:
    required, props = _required_and_props(tool)
    include_optional = scenario in ("required_plus_optional", "boundary_values")
    optional = [k for k in props if k not in required]
    fields = list(required)
    if include_optional and optional:
        fields += optional[: min(2, len(optional))]

    use_cn = scenario == "chinese_numeral"
    use_unit = scenario == "unit_conversion"
    use_boundary = scenario == "boundary_values"

    args: dict[str, Any] = {}
    phrases: list[str] = []
    for f in fields:
        spec = dict(props.get(f, {"type": "string"}))
        if use_boundary and spec.get("type") in ("integer", "number") and ("minimum" in spec or "maximum" in spec):
            value = spec.get("maximum", spec.get("minimum"))
            phrase = f"{_phrase(props, f)}是{value}"
        else:
            value, phrase = _value_and_phrase(f, spec, f"{seed}:{f}", use_cn_numeral=use_cn, use_unit_conversion=use_unit)
        args[f] = value
        phrases.append(phrase)

    errors = _validate(tool, args)
    if errors:
        return None

    context: dict[str, Any] = {"locale": "zh-CN"}
    history: list[dict[str, Any]] = []
    context_completion = scenario == "context_completion"
    query_fields = list(fields)
    if context_completion and required:
        deferred = required[0]
        query_fields = [f for f in fields if f != deferred]
        phrases = [p for f, p in zip(fields, phrases) if f != deferred]
        context["prior_established"] = {deferred: args[deferred]}
        history = [{"role": "user", "content": f"{_phrase(props, deferred)}是{args[deferred]}，记一下。"}, {"role": "assistant", "content": "好的，已记录。"}]

    base_desc = str(tool.get("description") or "").rstrip("。")
    clause = "，".join(phrases)
    query = f"请帮我{base_desc}" + (f"，{clause}" if clause else "") + "。"

    oracle_top5 = _pick_oracle_top5(deploy, tool, seed)
    tools_batch = [deploy.by_name[n] for n in oracle_top5]
    _, receipt = C.fit_batch_to_budget(
        tools_batch=tools_batch, query=query, context=context, evidence=[], history=history, tool_results=[], profile="standard",
    )
    row = {
        "case_id": C.case_id(FAMILY, scenario, seed),
        "cf_group": C.cf_group(FAMILY, scenario, tool.get("name"), seed[-8:]),
        "family": FAMILY,
        "task": "full_call",
        "generator_version": C.GENERATOR_VERSION,
        "kind": "execute",
        "scenario": scenario,
        "query": query,
        "context": context,
        "evidence": [],
        "history": history,
        "oracle_top5": oracle_top5,
        "gold_name": tool.get("name"),
        "gold_args": args,
        "answers": [{"name": tool.get("name"), "arguments": args}],
        "hard_negatives": oracle_top5[1:],
        "schema_valid": True,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
    }
    return row


def build_refusal_row(deploy: C.ToolRegistry, tool: Mapping[str, Any], *, kind: str, seed: str) -> dict[str, Any] | None:
    required, props = _required_and_props(tool)
    base_desc = str(tool.get("description") or "").rstrip("。")

    if kind == "refuse_missing_slot":
        if not required:
            return None
        missing = required[0]
        rest = required[1:]
        args = {}
        phrases = []
        for f in rest:
            spec = dict(props.get(f, {"type": "string"}))
            v = C.seeded_value(spec, f"{seed}:{f}")
            args[f] = v
            phrases.append(f"{_phrase(props, f)}是{v}")
        clause = "，".join(phrases)
        query = f"我想{base_desc}" + (f"，{clause}" if clause else "") + f"，{_phrase(props, missing)}我再想想。"
        expected_missing = [missing]
    elif kind == "refuse_conflicting_params":
        target_field = required[0] if required else (list(props)[0] if props else None)
        if not target_field:
            return None
        spec = dict(props.get(target_field, {"type": "string"}))
        v1 = C.seeded_value(spec, f"{seed}:a")
        v2 = C.seeded_value(spec, f"{seed}:b")
        query = f"请{base_desc}，{_phrase(props, target_field)}先是{v1}，不对，其实是{v2}，你自己定吧。"
        expected_missing = [target_field]
    elif kind == "refuse_unsupported_param":
        query = f"请{base_desc}，另外把这次的操作日志导出成PDF发我邮箱。"
        expected_missing = []
    else:
        return None

    oracle_top5 = _pick_oracle_top5(deploy, tool, seed)
    tools_batch = [deploy.by_name[n] for n in oracle_top5]
    _, receipt = C.fit_batch_to_budget(
        tools_batch=tools_batch, query=query, context={"locale": "zh-CN"}, evidence=[], history=[], tool_results=[], profile="standard",
    )
    row = {
        "case_id": C.case_id(FAMILY, kind, seed),
        "cf_group": C.cf_group(FAMILY, kind, tool.get("name"), seed[-8:]),
        "family": FAMILY,
        "task": "full_call",
        "generator_version": C.GENERATOR_VERSION,
        "kind": kind,
        "scenario": kind,
        "query": query,
        "context": {"locale": "zh-CN"},
        "evidence": [],
        "history": [],
        "oracle_top5": oracle_top5,
        "gold_name": None,
        "gold_args": None,
        "answers": [],
        "hard_negatives": oracle_top5[1:],
        "missing_or_conflicting_fields": expected_missing,
        "schema_valid": True,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
    }
    return row


EXECUTE_SCENARIOS = (
    "required_only", "required_plus_optional", "boundary_values",
    "unit_conversion", "chinese_numeral", "context_completion",
)
REFUSAL_KINDS = ("refuse_missing_slot", "refuse_conflicting_params", "refuse_unsupported_param")


def _tools_with_temperature_field(tools: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for t in tools:
        props = (t.get("parameters") or {}).get("properties") or {}
        if any(_is_temperature_field(name, spec) and spec.get("type") in ("integer", "number") for name, spec in props.items()):
            out.append(t)
    return out


def _tools_with_small_integer_field(tools: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for t in tools:
        props = (t.get("parameters") or {}).get("properties") or {}
        for spec in props.values():
            if spec.get("type") == "integer":
                hi = spec.get("maximum", 99)
                if isinstance(hi, (int, float)) and hi <= 99:
                    out.append(t)
                    break
    return out


def _tool_pool_for_scenario(scenario: str, all_tools: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    if scenario == "unit_conversion":
        pool = _tools_with_temperature_field(all_tools)
        return pool or all_tools
    if scenario == "chinese_numeral":
        pool = _tools_with_small_integer_field(all_tools)
        return pool or all_tools
    return all_tools


def generate(deploy: C.ToolRegistry, *, per_scenario_count: int, per_refusal_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tools = [t for t in deploy.tools if (t.get("parameters") or {}).get("properties")]
    rows: list[dict[str, Any]] = []
    for scenario in EXECUTE_SCENARIOS:
        made = 0
        i = 0
        pool = list(_tool_pool_for_scenario(scenario, tools))
        while made < per_scenario_count and i < per_scenario_count * 6:
            tool = pool[(hash(scenario) + i) % len(pool)]
            seed = f"fullcall:{scenario}:{i}"
            row = build_execute_row(deploy, tool, scenario=scenario, seed=seed)
            i += 1
            if row is None:
                continue
            rows.append(row)
            made += 1
    for kind in REFUSAL_KINDS:
        made = 0
        i = 0
        while made < per_refusal_count and i < per_refusal_count * 4:
            tool = tools[(hash(kind) + i) % len(tools)]
            seed = f"fullcall:{kind}:{i}"
            row = build_refusal_row(deploy, tool, kind=kind, seed=seed)
            i += 1
            if row is None:
                continue
            rows.append(row)
            made += 1

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_scenario": _count_by(rows, "scenario"),
        "by_split": _count_by(rows, "split"),
        "distinct_gold_tools": len({r["gold_name"] for r in rows if r["gold_name"]}),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
