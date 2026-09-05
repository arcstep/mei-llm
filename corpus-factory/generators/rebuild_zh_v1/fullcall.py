#!/usr/bin/env python3
"""Full-call / argument-filling generator.

Every emitted `gold_args` is validated against the tool's complete registered
schema (compiler_v1.validate_instance), never a trimmed projection. Covers
required/optional, all scalar+array types, enum/const, numeric bounds,
multipleOf, string length/pattern constraints, missing/conflicting/
unsupported parameters, unit conversion, Chinese numerals and
context-supplied (not query-stated) required values.

arg_norm_* scenarios (v3 addition) train argument NORMALIZATION: the query
expresses a value colloquially and gold is the canonical value a caller must
submit -- clock time ("上午十点半" -> "10:30", pattern-constrained), ISO
dates ("明天下午3点" -> "2026-09-09T15:00:00"), enum aliases ("调成自动" ->
"auto"), duration conversion ("一个半小时" -> 90 分钟), large/decimal
Chinese numerals ("两百五" -> 250, "四十七块四" -> 47.4) and entity
resolution ("客厅的灯" -> catalog id from context). Gold stays locally
compiled: every normalization is a deterministic function of the seed.
"""

from __future__ import annotations

import hashlib
import re
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
    return _assemble_row(
        deploy, tool, args=args, query=query, scenario=scenario, seed=seed,
        context=context, history=history,
    )


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
NORM_SCENARIOS = (
    "arg_norm_clock", "arg_norm_iso", "arg_norm_enum",
    "arg_norm_duration", "arg_norm_numeral", "arg_norm_entity",
)
REFUSAL_KINDS = ("refuse_missing_slot", "refuse_conflicting_params", "refuse_unsupported_param")


# ---------------------------------------------------------------------------
# arg_norm_* helpers: every normalization is a deterministic function of seed.
# ---------------------------------------------------------------------------
def _rand(seed: str, lo: int, hi: int) -> int:
    seed_int = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
    return lo + seed_int % (hi - lo + 1)


_CN_BIG = {0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
_CN_NUM = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}


def _cn_numeral_big(n: int) -> str:
    """口语大数字：2 只在最高位用「两」（两/两百/两千），组内用「二」（十二/二十）。"""
    if n <= 10:
        return _CN_BIG[n]
    if 11 <= n <= 19:
        return "十" + (_CN_NUM[n - 10] if n % 10 else "")
    if n < 100:
        tens, rem = divmod(n, 10)
        return _CN_NUM[tens] + "十" + (_CN_NUM[rem] if rem else "")
    if n < 1000:
        hun, rest = divmod(n, 100)
        return _CN_BIG[hun] + "百" + (_cn_numeral_big(rest) if rest else "")
    if n < 10000:
        tho, rest = divmod(n, 1000)
        return _CN_BIG[tho] + "千" + (_cn_numeral_big(rest) if rest else "")
    return str(n)


def _cn_decimal_money(x: float) -> str:
    """Money-style decimal: 47.4 -> 四十七块四; 7.5 -> 七块五."""
    whole, frac = int(x), int(round((x - int(x)) * 10))
    return f"{_cn_numeral_big(whole)}块{_CN_BIG[frac]}"


def _clock_band(hour: int) -> str:
    if 0 <= hour <= 5:
        return "凌晨"
    if hour <= 11:
        return "上午"
    if hour == 12:
        return "中午"
    if hour <= 18:
        return "下午"
    return "晚上"


def _cn_hour(hour: int) -> str:
    display = hour % 12 or 12
    if display == 11:
        return "十一"
    if display == 12:
        return "十二"
    return _CN_BIG[display]


def _clock_pair(seed: str) -> tuple[str, str]:
    """Colloquial clock expression -> HH:MM (24h)."""
    hour = _rand(f"{seed}:h", 6, 21)
    half = _rand(f"{seed}:m", 0, 1) == 1
    expr = f"{_clock_band(hour)}{_cn_hour(hour)}点"
    if half:
        expr += "半"
    return expr, f"{hour:02d}:{'30' if half else '00'}"


_ISO_KINDS = {"date", "iso", "time"}


def _iso_pair(seed: str, kind: str) -> tuple[str, str]:
    """Colloquial date/time -> canonical value (date | iso | HH:MM)."""
    from datetime import date, timedelta

    ref = date(2026, 9, 5) + timedelta(days=_rand(f"{seed}:ref", 0, 14))
    style = _rand(f"{seed}:style", 0, 4)
    hour = _rand(f"{seed}:h", 8, 20)
    half = _rand(f"{seed}:m", 0, 1) == 1
    clock_text = f"{_clock_band(hour)}{_cn_hour(hour)}点" + ("半" if half else "")
    if style == 0:  # 明天
        target, day_text = ref + timedelta(days=1), "明天"
    elif style == 1:  # 后天
        target, day_text = ref + timedelta(days=2), "后天"
    elif style == 2:  # 大后天
        target, day_text = ref + timedelta(days=3), "大后天"
    elif style == 3:  # 下周X
        wd = (_rand(f"{seed}:wd", 1, 7))  # 1=周一
        days = (wd - ref.weekday() - 1) % 7 + 7
        target = ref + timedelta(days=days)
        day_text = "下周" + "一二三四五六日"[wd - 1]
    else:  # 本周X（未来）
        wd = _rand(f"{seed}:wd2", 1, 7)
        days = (wd - ref.weekday() - 1) % 7
        if days == 0:
            days = 7
        target = ref + timedelta(days=days)
        day_text = "周" + "一二三四五六日"[wd - 1]
    date_text = f"{target.isoformat()}"
    hour_value = f"{hour:02d}:{'30' if half else '00'}"
    if kind == "date":
        return f"{day_text}", date_text
    if kind == "time":
        return f"{day_text}{clock_text}", hour_value
    return f"{day_text}{clock_text}", f"{date_text}T{hour_value}:00"


_ENUM_ALIASES: dict[str, tuple[str, ...]] = {
    "CNY": ("人民币", "人民币结算"),
    "USD": ("美元", "美金"),
    "front": ("前门", "前面那个门"),
    "back": ("后门", "后面那个门"),
    "alpha": ("阿尔法",),
    "beta": ("贝塔",),
    "gamma": ("伽马",),
    "kitchen": ("厨房",),
    "living": ("客厅",),
    "entry": ("门口", "入口"),
    "trash": ("垃圾房", "垃圾桶那"),
    "left": ("左边", "左侧"),
    "right": ("右边", "右侧"),
    "user": ("用户所在位置",),
    "auto": ("自动", "自动模式"),
    "manual": ("手动", "手动模式"),
    "kitchen_light": ("厨房的灯",),
    "living_light": ("客厅的灯",),
}


def _enum_candidates(tools: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], str]]:
    """(tool, field) pairs whose enum values all have colloquial aliases."""
    out: list[tuple[Mapping[str, Any], str]] = []
    for tool in tools:
        props = (tool.get("parameters") or {}).get("properties") or {}
        for name, spec in props.items():
            values = spec.get("enum") or []
            if values and all(v in _ENUM_ALIASES for v in values):
                out.append((tool, name))
    return out


def _duration_pair(seed: str, lo: int, hi: int, *, unit: str) -> tuple[str, int]:
    """Colloquial duration -> canonical value (minutes or hours by unit)."""
    if unit == "hours":
        target = _rand(seed, lo, hi)
        return f"{_cn_numeral_big(target)}个小时", target
    # minutes: target snapped to 5-min steps within [lo, hi]
    target = lo + (_rand(seed, 0, max(0, (hi - lo) // 5)) * 5)
    hours, minutes = divmod(target, 60)
    style = _rand(f"{seed}:st", 0, 2)
    if minutes == 0:
        return f"{_cn_numeral_big(hours)}个小时", target
    if minutes == 30 and style == 0:
        return f"{_cn_numeral_big(hours)}个半小时", target
    if style == 1 and hours:
        return f"{_cn_numeral_big(hours)}小时{_cn_numeral_big(minutes)}分钟", target
    if hours:
        return f"{_cn_numeral_big(hours)}小时零{_cn_numeral_big(minutes)}分钟", target
    return f"{_cn_numeral_big(minutes)}分钟", target


_ENTITY_ALIASES = ("客厅的", "实验室的", "办公室的", "会议室的", "仓库的")
_REFERRAL_ALIASES = ("昨天那个", "上午那个", "刚才那个")


def _entity_id(seed: str) -> str:
    letters = "".join(chr(65 + _rand(f"{seed}:{i}", 0, 25)) for i in range(2))
    return f"{letters}{_rand(seed, 1000, 9999)}"


# ---------------------------------------------------------------------------
# norm row builder: shared tail with build_execute_row.
# ---------------------------------------------------------------------------
def _assemble_row(
    deploy: C.ToolRegistry,
    tool: Mapping[str, Any],
    *,
    args: Mapping[str, Any],
    query: str,
    scenario: str,
    seed: str,
    context: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    oracle_top5 = _pick_oracle_top5(deploy, tool, seed)
    tools_batch = [deploy.by_name[n] for n in oracle_top5]
    _, receipt = C.fit_batch_to_budget(
        tools_batch=tools_batch, query=query, context=context, evidence=[],
        history=history, tool_results=[], profile="standard",
    )
    return {
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
        "gold_args": dict(args),
        "answers": [{"name": tool.get("name"), "arguments": dict(args)}],
        "hard_negatives": oracle_top5[1:],
        "schema_valid": True,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
    }


def build_norm_row(
    deploy: C.ToolRegistry,
    tool: Mapping[str, Any],
    *,
    scenario: str,
    field: str,
    seed: str,
) -> dict[str, Any] | None:
    """One arg_norm_* row: colloquial expression -> canonical gold value."""
    required, props = _required_and_props(tool)
    base_desc = str(tool.get("description") or "").rstrip("。")
    spec = dict(props.get(field, {"type": "string"}))
    # other required fields stay literal (standard style); context = None
    args: dict[str, Any] = {}
    clauses: list[str] = []
    for f in required:
        if f == field:
            continue
        f_spec = dict(props.get(f, {"type": "string"}))
        v = C.seeded_value(f_spec, f"{seed}:{f}")
        args[f] = v
        clauses.append(f"{_phrase(props, f)}是{v}")
    context: dict[str, Any] = {"locale": "zh-CN"}
    if scenario == "arg_norm_clock":
        expr, value = _clock_pair(f"{seed}:{field}")
        args[field] = value
        clauses.append(f"{_phrase(props, field)}设成{expr}")
    elif scenario == "arg_norm_iso":
        kind = "iso"
        if "日期" in (str(spec.get("description") or "") + field):
            kind = "date"
        elif "时间" in (str(spec.get("description") or "") + field):
            kind = "time"
        expr, value = _iso_pair(f"{seed}:{field}", kind)
        args[field] = value
        clauses.append(f"{_phrase(props, field)}是{expr}")
    elif scenario == "arg_norm_enum":
        values = spec.get("enum") or []
        value = values[_rand(f"{seed}:v", 0, len(values) - 1)]
        alias = _ENUM_ALIASES[value][_rand(f"{seed}:a", 0, len(_ENUM_ALIASES[value]) - 1)]
        args[field] = value
        clauses.append(f"{_phrase(props, field)}选{alias}")
    elif scenario == "arg_norm_duration":
        lo = int(spec.get("minimum", 0))
        hi = int(spec.get("maximum", lo + 120))
        name_desc = str(spec.get("description") or "") + field
        if "小时" in name_desc and "分钟" not in name_desc and hi <= 23:
            # hour-of-day field: clock expression, not a duration
            expr, hhmm = _clock_pair(f"{seed}:{field}")
            value = int(hhmm[:2])
            args[field] = value
            clauses.append(f"{_phrase(props, field)}设在{expr}")
        else:
            unit = "hours" if "小时" in name_desc and "分钟" not in name_desc else "minutes"
            expr, value = _duration_pair(f"{seed}:{field}", lo, hi, unit=unit)
            args[field] = value
            clauses.append(f"{_phrase(props, field)}{expr}")
    elif scenario == "arg_norm_numeral":
        t = spec.get("type")
        desc_text = str(spec.get("description") or "") + field
        moneyish = any(k in desc_text for k in ("金额", "钱", "费用", "价格", "amount", "限额"))
        if t == "number" and moneyish:
            value = round(_rand(f"{seed}:v", 10, 990) + _rand(f"{seed}:d", 1, 9) / 10, 1)
            expr = _cn_decimal_money(value)
        elif t == "number":
            value = round(_rand(f"{seed}:v", 10, 990) + _rand(f"{seed}:d", 1, 9) / 10, 1)
            expr = f"{_cn_numeral_big(int(value))}点{_CN_BIG[int(round((value - int(value)) * 10))]}"
        else:
            hi = int(spec.get("maximum", 9999)) if spec.get("maximum") is not None else 9999
            hi = max(100, min(hi, 9999))
            value = _rand(f"{seed}:v", 100, hi)
            expr = _cn_numeral_big(value)
        args[field] = value
        clauses.append(f"{_phrase(props, field)}是{expr}")
    elif scenario == "arg_norm_entity":
        name_desc = str(spec.get("description") or "") + field
        aliases = _ENTITY_ALIASES if any(k in name_desc for k in ("设备", "柜", "橱", "箱", "桶", "unit", "hood")) else _REFERRAL_ALIASES
        alias = aliases[_rand(f"{seed}:a", 0, len(aliases) - 1)]
        entity_id = _entity_id(f"{seed}:id")
        args[field] = entity_id
        context["entities"] = {alias: entity_id}
        clauses.append(f"{_phrase(props, field)}用{alias}的")
    else:
        return None
    errors = _validate(tool, args)
    if errors:
        return None
    clause = "，".join(clauses)
    query = f"请帮我{base_desc}" + (f"，{clause}" if clause else "") + "。"
    return _assemble_row(
        deploy, tool, args=args, query=query, scenario=scenario, seed=seed,
        context=context, history=[],
    )


def _norm_candidates(scenario: str, tools: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], str]]:
    if scenario == "arg_norm_clock":
        return [
            (t, name)
            for t in tools
            for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if str(spec.get("pattern") or "").startswith("^[0-9]{2}:[0-9]{2}")
        ]
    if scenario == "arg_norm_iso":
        return [
            (t, name)
            for t in tools
            for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if spec.get("type") == "string" and ("日期" in str(spec.get("description") or "") + name or "时间" in str(spec.get("description") or "") + name)
        ]
    if scenario == "arg_norm_enum":
        return _enum_candidates(tools)
    if scenario == "arg_norm_duration":
        return [
            (t, name)
            for t in tools
            for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if spec.get("type") == "integer" and ("分钟" in str(spec.get("description") or "") + name or "小时" in str(spec.get("description") or "") + name)
        ]
    if scenario == "arg_norm_numeral":
        return [
            (t, name)
            for t in tools
            for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if spec.get("type") in ("integer", "number")
            and (spec.get("maximum") is None or float(spec.get("maximum")) >= 100)
        ]
    if scenario == "arg_norm_entity":
        return [
            (t, name)
            for t in tools
            for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if spec.get("type") == "string" and not spec.get("enum")
            and ("id" in name.lower() or "号" in str(spec.get("description") or "") + name)
        ]
    return []


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
    for scenario in NORM_SCENARIOS:
        made = 0
        i = 0
        pool = _norm_candidates(scenario, tools)
        if not pool:
            raise RuntimeError(f"no tool/field candidates for {scenario}")
        while made < per_scenario_count and i < per_scenario_count * 8:
            tool, field = pool[(hash(scenario) + i) % len(pool)]
            seed = f"fullcall:{scenario}:{field}:{i}"
            row = build_norm_row(deploy, tool, scenario=scenario, field=field, seed=seed)
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
