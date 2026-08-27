"""Post-generation provenance / permission / state validator. Fail closed."""

from __future__ import annotations

import re
from typing import Any

from normalizers import coerce_schema_value, find_booleans, find_currency, find_numbers, parse_chinese_int
from schema_render import dumps_canonical
from tool_call_protocol_v2 import parse_v2_text

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日")
UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)(度|厘米|米|公斤|千克|%|％)")


def _span_in(text: str, value: Any) -> tuple[int, int] | None:
    needle = str(value)
    if not needle:
        return None
    i = (text or "").find(needle)
    if i < 0:
        return None
    return i, i + len(needle)


def _number_evidence(query: str, value: Any) -> dict[str, Any] | None:
    for hit in find_numbers(query or ""):
        if hit.value == value or (isinstance(value, (int, float)) and hit.value == value):
            return {
                "evidence_source": "query",
                "evidence_span_or_locator": [hit.start, hit.end],
                "canonical_value": value,
                "resolution_source": "allowed_normalizer",
                "normalizer_id": hit.normalizer_id,
                "normalizer_version": "mei-normalizer-v1",
            }
        if isinstance(value, (int, float)) and hit.value == int(value):
            return {
                "evidence_source": "query",
                "evidence_span_or_locator": [hit.start, hit.end],
                "canonical_value": int(value),
                "resolution_source": "allowed_normalizer",
                "normalizer_id": hit.normalizer_id,
                "normalizer_version": "mei-normalizer-v1",
            }
    return None


def evidence_for_value(
    *,
    value: Any,
    schema: dict[str, Any],
    query: str,
    system_facts: str = "",
    prior_tool_results: list[str] | None = None,
    entities: list[dict[str, Any]] | None = None,
    permissions: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    texts = [("query", query or "")]
    if system_facts:
        texts.append(("system_facts", system_facts))
    for item in prior_tool_results or []:
        texts.append(("prior_tool_result", str(item)))
    typ = str(schema.get("type") or "string")
    for source, text in texts:
        span = _span_in(text, value)
        if span:
            return {
                "evidence_source": source,
                "evidence_span_or_locator": list(span),
                "canonical_value": value,
                "resolution_source": source,
                "normalizer_id": None,
                "normalizer_version": None,
            }
    if typ in {"integer", "number"}:
        ev = _number_evidence(query, value)
        if ev:
            return ev
        for source, text in texts[1:]:
            ev = _number_evidence(text, value)
            if ev:
                ev["evidence_source"] = source
                return ev
    if typ == "boolean":
        for hit in find_booleans(query or ""):
            if hit.value == value:
                return {
                    "evidence_source": "query",
                    "evidence_span_or_locator": [hit.start, hit.end],
                    "canonical_value": value,
                    "resolution_source": "allowed_normalizer",
                    "normalizer_id": hit.normalizer_id,
                    "normalizer_version": "mei-normalizer-v1",
                }
    if typ == "string":
        for source, text in texts:
            for m in DATE_RE.finditer(text or ""):
                if str(value) in {m.group(), m.group().replace("-", "")} or str(value) == m.group():
                    return {
                        "evidence_source": source,
                        "evidence_span_or_locator": [m.start(), m.end()],
                        "canonical_value": value,
                        "resolution_source": "allowed_normalizer",
                        "normalizer_id": "date_span",
                        "normalizer_version": "mei-normalizer-v1",
                    }
            for m in UNIT_RE.finditer(text or ""):
                if str(value) == m.group(1) or str(value) == m.group():
                    return {
                        "evidence_source": source,
                        "evidence_span_or_locator": [m.start(), m.end()],
                        "canonical_value": value,
                        "resolution_source": "allowed_normalizer",
                        "normalizer_id": "unit_span",
                        "normalizer_version": "mei-normalizer-v1",
                    }
    for ent in entities or []:
        for key in ("name", "id", "canonical"):
            if str(ent.get(key) or "") == str(value):
                return {
                    "evidence_source": "registered_entity",
                    "evidence_span_or_locator": str(ent.get("id") or ent.get("name")),
                    "canonical_value": value,
                    "resolution_source": "registered_entity",
                    "entity_id": ent.get("id"),
                    "normalizer_id": None,
                    "normalizer_version": None,
                }
    return None


def _arg_schema(tool: dict[str, Any], key: str) -> dict[str, Any]:
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    return dict(props.get(key) or {})


def validate_generated_call(
    text: str,
    *,
    tools: list[dict[str, Any]],
    query: str,
    system_facts: str = "",
    prior_tool_results: list[str] | None = None,
    entities: list[dict[str, Any]] | None = None,
    permissions: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_v2_text(text)
    if not parsed.get("ok"):
        return {
            "ok": False,
            "function_calls": [],
            "error": parsed.get("error") or "parse",
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    if parsed.get("refuse"):
        return {
            "ok": True,
            "function_calls": [],
            "error": None,
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    call = parsed["function_calls"][0]
    by_name = {str(t.get("name")): t for t in tools}
    tool = by_name.get(call["name"])
    if tool is None:
        return {
            "ok": False,
            "function_calls": [],
            "error": "no_matching_tool",
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    params = tool.get("parameters") or {}
    required = [str(x) for x in (params.get("required") or [])]
    props = params.get("properties") or {}
    args = dict(call.get("arguments") or {})
    missing = [k for k in required if k not in args]
    if missing:
        return {
            "ok": True,
            "function_calls": [],
            "error": None,
            "refuse": True,
            "reason": "missing_required",
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    provenance = {}
    errors: list[str] = []
    if permissions and call["name"] in set(permissions.get("denied_tools") or []):
        errors.append("permission_denied")
    if state and state.get("invalid"):
        errors.append("state_invalid")
    if state and state.get("conflict"):
        errors.append("source_conflict")
    named_ents = [e for e in (entities or []) if str(e.get("name") or "")]
    for key, value in args.items():
        if key not in props:
            errors.append(f"unsupported:{key}")
            continue
        schema = _arg_schema(tool, key)
        try:
            coerced = coerce_schema_value(value, str(schema.get("type") or "string")) if schema.get("type") in {
                "boolean",
                "integer",
                "number",
            } else value
        except ValueError:
            errors.append(f"type_or_range_mismatch:{key}")
            continue
        if schema.get("enum") is not None and coerced not in schema.get("enum"):
            errors.append(f"type_or_range_mismatch:{key}")
            continue
        if schema.get("minimum") is not None and isinstance(coerced, (int, float)) and coerced < schema["minimum"]:
            errors.append(f"type_or_range_mismatch:{key}")
            continue
        if schema.get("maximum") is not None and isinstance(coerced, (int, float)) and coerced > schema["maximum"]:
            errors.append(f"type_or_range_mismatch:{key}")
            continue
        ev = evidence_for_value(
            value=coerced,
            schema=schema,
            query=query,
            system_facts=system_facts,
            prior_tool_results=prior_tool_results,
            entities=entities,
            permissions=permissions,
        )
        if ev is None:
            errors.append(f"unprovenanced:{key}")
            continue
        ent_hits = [
            e
            for e in named_ents
            if str(e.get("name")) in (query or "")
            or str(e.get("name")) == str(value)
        ]
        name_hits = [e for e in ent_hits if str(e.get("name")) == str(value)]
        if str(value) and len([e for e in named_ents if str(e.get("name")) in (query or "")]) > 1 and name_hits:
            aliases = [e for e in named_ents if str(e.get("name")) in (query or "")]
            if len(aliases) > 1:
                errors.append("ambiguous_entity")
                continue
        fact_span = _span_in(system_facts, value)
        other_fact = None
        if system_facts and fact_span is None:
            for e in named_ents:
                nm = str(e.get("name") or "")
                if nm and nm != str(value) and nm in system_facts and nm in (query or ""):
                    other_fact = nm
        if other_fact:
            errors.append("source_conflict")
            continue
        provenance[key] = ev
    unsupported = sum(1 for e in errors if e.startswith("unsupported:"))
    unprov = sum(1 for e in errors if e.startswith("unprovenanced:"))
    if errors:
        return {
            "ok": False,
            "function_calls": [],
            "error": ";".join(errors),
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
            "would_have_accepted_unsupported": unsupported,
            "would_have_accepted_unprovenanced": unprov,
        }
    return {
        "ok": True,
        "function_calls": [call],
        "error": None,
        "refuse": False,
        "provenance": provenance,
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
        "canonical": dumps_canonical(call),
    }
