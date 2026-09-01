"""Deterministic post-decode gates for MEI Runtime v2.

The order in :func:`validate_generated_call` is part of the wire semantics:
grammar -> schema -> provenance -> permission -> state -> MW -> confidence.
Learned scores are evaluated last and can never override a deterministic deny.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

try:
    from .byte_grammar import parse_call_text
    from .canonical_json import MAX_SAFE_INTEGER, dumps_canonical
    from .normalizers import find_booleans, find_currency, find_numbers
except ImportError:
    from byte_grammar import parse_call_text
    from canonical_json import MAX_SAFE_INTEGER, dumps_canonical
    from normalizers import find_booleans, find_currency, find_numbers

VALIDATION_ORDER = (
    "grammar",
    "schema",
    "provenance",
    "permission",
    "state",
    "mw",
    "confidence",
)
EXECUTE_HIGH = 0.70
ESCALATE_LOW = 0.35


def _gate(name: str, ok: bool, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    row = {"gate": name, "ok": bool(ok), "detail": detail}
    row.update(extra)
    return row


def _blocked(
    gates: list[dict[str, Any]],
    error: str,
    *,
    detail: str | None = None,
    parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "refuse": True,
        "execution": "refuse",
        "function_calls": [],
        "error": error,
        "detail": detail,
        "gates": gates,
        "provenance": {},
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
        "parsed": parsed,
    }


def _tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((tool for tool in tools if str(tool.get("name")) == name), None)


def _same_value(left: Any, right: Any) -> bool:
    return _json_equal(left, right)


def _portable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    return (
        math.isfinite(value)
        and (not value.is_integer() or abs(value) <= MAX_SAFE_INTEGER)
    )


def _json_equal(left: Any, right: Any) -> bool:
    """Cross-runtime JSON equality (no approximate authorization)."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _portable_number(left) and _portable_number(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _explicit_evidence(
    *, request: dict[str, Any], tool_name: str, argument: str, value: Any
) -> dict[str, Any] | None:
    evidence_rows = list(request.get("evidence") or [])
    context = request.get("context") or {}
    if isinstance(context, dict):
        evidence_rows.extend(context.get("facts") or [])
    for index, item in enumerate(evidence_rows):
        if (
            not isinstance(item, dict)
            or item.get("verified") is not True
            or not isinstance(item.get("source"), str)
            or not item.get("source")
        ):
            continue
        binding_matches = (
            item.get("tool") == tool_name and item.get("argument") == argument
        ) or (
            item.get("subject") == tool_name and item.get("predicate") == argument
        )
        if not binding_matches:
            continue
        candidate = item.get("value", item.get("canonical_value"))
        if _json_equal(candidate, value):
            return {
                "source": str(item.get("source") or "request_evidence"),
                "locator": item.get("locator", index),
                "canonical_value": value,
                "verified": True,
            }
    verified_result_map = request.get("_verified_result_map") or {}
    for index, result in enumerate(request.get("tool_results") or request.get("prior_tool_results") or []):
        if not isinstance(result, dict) or str(result.get("status") or "ok") not in {"ok", "success"}:
            continue
        call_id = str(result.get("call_id") or "")
        trusted_result = (
            verified_result_map.get(call_id)
            if isinstance(verified_result_map, dict)
            else None
        )
        if not isinstance(trusted_result, dict) or not _json_equal(trusted_result, result):
            continue
        result_provenance = result.get("provenance")
        nested_verified = (
            result_provenance.get("verified") is True
            if isinstance(result_provenance, dict)
            else False
        )
        # v2 results carry verification inside their provenance object.  The
        # top-level spelling is accepted only as the read-only v1 adapter.
        compat_verified = (
            result.get("verified") is True
            and (request.get("_compat") or {}).get("source_wire") == "mei-runtime-wire-v1"
        )
        if not (nested_verified or compat_verified):
            continue
        payload = result.get("payload", result.get("result"))
        if isinstance(payload, dict) and argument in payload and _json_equal(payload[argument], value):
            return {
                "source": "verified_tool_result",
                "locator": call_id or index,
                "canonical_value": value,
                "verified": True,
            }
    is_v1_adapter = (request.get("_compat") or {}).get("source_wire") == "mei-runtime-wire-v1"
    for index, entity in enumerate(request.get("entities") or []):
        if not isinstance(entity, dict):
            continue
        if entity.get("verified") is not True or not is_v1_adapter:
            continue
        for key in ("canonical", "value", "id", "name"):
            if key in entity and _same_value(entity[key], value):
                return {
                    "source": "registered_entity",
                    "locator": entity.get("id", index),
                    "canonical_value": value,
                    "verified": True,
                }
    return None


def _legacy_text_evidence(request: dict[str, Any], value: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    compat = request.get("_compat") or {}
    if compat.get("source_wire") != "mei-runtime-wire-v1":
        return None
    sources: list[tuple[str, str]] = [("query", str(request.get("query") or ""))]
    context = request.get("context") or {}
    if isinstance(context, dict):
        sources.append(("system_facts", str(context.get("system_facts") or "")))
    sources.append(("system_facts", str(request.get("system_facts") or "")))
    for source, text in sources:
        needle = str(value)
        if needle and needle in text:
            start = text.index(needle)
            return {
                "source": source,
                "locator": [start, start + len(needle)],
                "canonical_value": value,
                "verified": True,
                "degraded": True,
            }
        if schema.get("type") in {"integer", "number"}:
            for hit in find_numbers(text):
                if _same_value(hit.value, value):
                    return {
                        "source": source,
                        "locator": [hit.start, hit.end],
                        "canonical_value": value,
                        "normalizer": hit.normalizer_id,
                        "verified": True,
                        "degraded": True,
                    }
        if schema.get("type") == "boolean":
            for hit in find_booleans(text):
                if hit.value is value:
                    return {
                        "source": source,
                        "locator": [hit.start, hit.end],
                        "canonical_value": value,
                        "normalizer": hit.normalizer_id,
                        "verified": True,
                        "degraded": True,
                    }
        if schema.get("type") == "string":
            for hit in find_currency(text):
                if hit.value == value:
                    return {
                        "source": source,
                        "locator": [hit.start, hit.end],
                        "canonical_value": value,
                        "normalizer": hit.normalizer_id,
                        "verified": True,
                        "degraded": True,
                    }
    return None


def _provenance_gate(
    call: dict[str, Any], tool: dict[str, Any], request: dict[str, Any]
) -> tuple[bool, dict[str, Any], list[str]]:
    props = ((tool.get("parameters") or {}).get("properties") or {})
    provenance: dict[str, Any] = {}
    missing: list[str] = []
    for argument, value in call.get("arguments", {}).items():
        schema = props[argument]
        if "const" in schema and _json_equal(schema["const"], value):
            provenance[argument] = {
                "source": "schema_const",
                "locator": f"parameters.properties.{argument}.const",
                "canonical_value": value,
                "verified": True,
            }
            continue
        evidence = _explicit_evidence(
            request=request,
            tool_name=str(call["name"]),
            argument=argument,
            value=value,
        ) or _legacy_text_evidence(request, value, schema)
        if evidence is None:
            missing.append(argument)
        else:
            provenance[argument] = evidence
    return not missing, provenance, missing


def _permission_gate(call: dict[str, Any], tool: dict[str, Any], request: dict[str, Any]) -> tuple[bool, str | None]:
    raw = request.get("permissions") or {}
    if isinstance(raw, list):
        if (
            not all(isinstance(value, str) and value for value in raw)
            or len(set(raw)) != len(raw)
        ):
            return False, "permissions_invalid"
        allowed_tools: set[str] | None = None
        denied_tools: set[str] = set()
        scopes = set(raw)
        denied_scopes: set[str] = set()
    elif isinstance(raw, dict):
        allowed_keys = {
            "scopes",
            "grants",
            "denies",
            "allowed_tools",
            "denied_tools",
        }
        if not set(raw) <= allowed_keys:
            return False, "permissions_invalid"
        if "scopes" in raw and "grants" in raw:
            return False, "permissions_invalid"
        for key in allowed_keys:
            if key not in raw:
                continue
            values = raw[key]
            if (
                not isinstance(values, list)
                or not all(isinstance(value, str) and value for value in values)
                or len(set(values)) != len(values)
            ):
                return False, "permissions_invalid"
        allowed_tools = set(raw["allowed_tools"]) if "allowed_tools" in raw else None
        denied_tools = set(raw.get("denied_tools") or [])
        scopes = set(raw.get("scopes") or raw.get("grants") or [])
        denied_scopes = set(raw.get("denies") or [])
    else:
        return False, "permissions_invalid"
    name = str(call["name"])
    if name in denied_tools:
        return False, "permission_denied"
    if allowed_tools is not None and name not in allowed_tools:
        return False, "permission_not_granted"
    required_raw = (
        tool.get("required_permissions")
        if "required_permissions" in tool
        else tool.get("x-mei-permissions")
    )
    required = {str(value) for value in (required_raw or [])}
    if required & denied_scopes:
        return False, "permission_denied"
    if required - scopes:
        return False, "permission_scope_missing"
    return True, None


def _state_gate(tool: dict[str, Any], request: dict[str, Any]) -> tuple[bool, str | None]:
    state = request.get("state") or {}
    if not isinstance(state, dict):
        return False, "state_invalid"
    if state.get("invalid"):
        return False, "state_invalid"
    if state.get("conflict"):
        return False, "state_conflict"
    required = tool.get("required_state") if "required_state" in tool else tool.get("x-mei-state")
    required = required or {}
    if not isinstance(required, dict):
        return False, "tool_state_contract_invalid"
    for key, expected in required.items():
        if key not in state or not _json_equal(state[key], expected):
            return False, f"state_requirement_missing:{key}"
    return True, None


def _mw_gate(call: dict[str, Any], request: dict[str, Any]) -> tuple[bool, str | None]:
    if request.get("mw") is not None and request.get("mw_disposition") is not None:
        return False, "mw_invalid"
    is_v1 = (request.get("_compat") or {}).get("source_wire") == "mei-runtime-wire-v1"
    if request.get("mw") is not None:
        mw = request["mw"]
        expected_sources = {"protocol-test"}
    elif request.get("mw_disposition") is not None:
        mw = request["mw_disposition"]
        expected_sources = {"mw-head", "deterministic-policy"}
    else:
        mw = {"decision": "continue", "source": "deterministic-policy"}
        expected_sources = {"deterministic-policy"}
    if isinstance(mw, str):
        if not is_v1:
            return False, "mw_invalid"
        decision = mw
        allowed_tools = None
    elif isinstance(mw, dict):
        allowed_keys = {"decision", "allowed_tools", "source", "receipt_sha256"}
        if not is_v1 and (
            not set(mw) <= allowed_keys
            or "decision" not in mw
            or "source" not in mw
        ):
            return False, "mw_invalid"
        decision = str(mw.get("decision") or mw.get("disposition") or "continue")
        source = mw.get("source")
        if not is_v1 and source not in expected_sources:
            return False, "mw_invalid"
        allowed_raw = mw.get("allowed_tools")
        if allowed_raw is not None and (
            not isinstance(allowed_raw, list)
            or not allowed_raw
            or not all(isinstance(value, str) for value in allowed_raw)
            or len(set(allowed_raw)) != len(allowed_raw)
        ):
            return False, "mw_invalid"
        receipt = mw.get("receipt_sha256")
        if receipt is not None and (
            not isinstance(receipt, str) or re.fullmatch(r"[0-9a-f]{64}", receipt) is None
        ):
            return False, "mw_invalid"
        if not is_v1 and source == "mw-head" and receipt is None:
            return False, "mw_invalid"
        allowed_tools = {str(value) for value in allowed_raw} if isinstance(allowed_raw, list) else None
    else:
        return False, "mw_invalid"
    if decision in {"stop", "block", "refuse"}:
        return False, "mw_stop"
    if decision == "constrain" and allowed_tools is None:
        return False, "mw_invalid"
    if decision == "constrain" and allowed_tools is not None and str(call["name"]) not in allowed_tools:
        return False, "mw_constrained"
    if decision not in {"continue", "constrain"}:
        return False, "mw_invalid"
    return True, None


def _confidence_gate(
    confidence: float | dict[str, Any] | None, *, enforce: bool
) -> tuple[str, float | None, str | None]:
    if isinstance(confidence, dict):
        if not set(confidence) <= {"value", "execute_high", "escalate_low", "source"}:
            return "refuse", None, "confidence_invalid"
        source = confidence.get("source")
        if source not in {"protocol-test", "confidence-head"}:
            return "refuse", None, "confidence_invalid"
        raw_value = confidence.get("value")
        execute_raw = confidence.get("execute_high", EXECUTE_HIGH)
        escalate_raw = confidence.get("escalate_low", ESCALATE_LOW)
    else:
        raw_value = confidence
        execute_raw = EXECUTE_HIGH
        escalate_raw = ESCALATE_LOW
    if raw_value is None:
        return ("refuse", None, "confidence_unavailable") if enforce else ("execute", None, None)
    if not _portable_number(raw_value):
        return "refuse", None, "confidence_invalid"
    value = float(raw_value)
    if any(not _portable_number(raw) for raw in (execute_raw, escalate_raw)):
        return "refuse", value, "confidence_invalid"
    execute_high = float(execute_raw)
    escalate_low = float(escalate_raw)
    if (
        not all(math.isfinite(item) for item in (value, execute_high, escalate_low))
        or not 0.0 <= value <= 1.0
        or not 0.0 <= escalate_low <= execute_high <= 1.0
    ):
        return "refuse", value, "confidence_invalid"
    if value >= execute_high:
        return "execute", value, None
    if value >= escalate_low:
        return "escalate", value, None
    return "refuse", value, None


def validate_generated_call(
    text: str,
    *,
    tools: list[dict[str, Any]],
    request: dict[str, Any],
    confidence: float | dict[str, Any] | None = None,
    enforce_confidence: bool = True,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    parsed = parse_call_text(text, tools)
    grammar_ok = parsed.get("error") not in {"grammar", "json", "illegal_shape"}
    gates.append(_gate("grammar", grammar_ok, parsed.get("error") if not grammar_ok else None))
    if not grammar_ok:
        return _blocked(gates, "grammar_violation", detail=parsed.get("error"), parsed=parsed)
    schema_ok = bool(parsed.get("ok"))
    gates.append(_gate("schema", schema_ok, parsed.get("error") if not schema_ok else None))
    if not schema_ok:
        return _blocked(gates, "schema_validation", detail=parsed.get("error"), parsed=parsed)
    if parsed.get("refuse"):
        # A model refusal is a valid terminal turn. Remaining gates are not applicable.
        return {
            "ok": True,
            "refuse": True,
            "execution": "refuse",
            "function_calls": [],
            "error": None,
            "gates": gates,
            "provenance": {},
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
            "canonical": "[]",
        }
    call = parsed["function_calls"][0]
    tool = _tool_by_name(tools, str(call["name"]))
    assert tool is not None
    provenance_ok, provenance, missing = _provenance_gate(call, tool, request)
    gates.append(_gate("provenance", provenance_ok, None if provenance_ok else "missing", missing=missing))
    if not provenance_ok:
        return _blocked(gates, "provenance_missing", detail=",".join(missing), parsed=parsed)
    permission_ok, permission_error = _permission_gate(call, tool, request)
    gates.append(_gate("permission", permission_ok, permission_error))
    if not permission_ok:
        return _blocked(gates, permission_error or "permission_denied", parsed=parsed)
    state_ok, state_error = _state_gate(tool, request)
    gates.append(_gate("state", state_ok, state_error))
    if not state_ok:
        return _blocked(gates, state_error or "state_invalid", parsed=parsed)
    mw_ok, mw_error = _mw_gate(call, request)
    gates.append(_gate("mw", mw_ok, mw_error))
    if not mw_ok:
        return _blocked(gates, mw_error or "mw_stop", parsed=parsed)
    execution, confidence_value, confidence_error = _confidence_gate(confidence, enforce=enforce_confidence)
    confidence_ok = confidence_error is None and execution == "execute"
    gates.append(
        _gate(
            "confidence",
            confidence_ok,
            confidence_error or (None if confidence_ok else execution),
            value=confidence_value,
            execution=execution,
        )
    )
    if confidence_error:
        return _blocked(gates, confidence_error, parsed=parsed)
    if execution != "execute":
        return {
            "ok": True,
            "refuse": True,
            "execution": execution,
            "function_calls": [],
            "error": None,
            "gates": gates,
            "provenance": provenance,
            "confidence_value": confidence_value,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    return {
        "ok": True,
        "refuse": False,
        "execution": "execute",
        "function_calls": [call],
        "error": None,
        "gates": gates,
        "provenance": provenance,
        "confidence_value": confidence_value,
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
        "canonical": dumps_canonical(call),
    }


def verified_tool_result(
    *, call_id: str, status: str, payload: Any, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    if status not in {"ok", "error", "cancelled"}:
        raise ValueError("tool result status must be ok, error or cancelled")
    # Ensure payload is JSON-serializable at the trust boundary.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    source = str((provenance or {}).get("source") or "host_executor")
    verified_provenance: dict[str, Any] = {"source": source, "verified": True}
    receipt = (provenance or {}).get("receipt_sha256")
    if isinstance(receipt, str) and re.fullmatch(r"[0-9a-f]{64}", receipt):
        verified_provenance["receipt_sha256"] = receipt
    error = None
    if status != "ok":
        if isinstance(payload, dict):
            error = {
                "code": str(payload.get("code") or status),
                "message": str(payload.get("message") or payload),
            }
        else:
            error = {"code": status, "message": str(payload)}
    return {
        "wire_version": "mei-runtime-wire-v2",
        "call_id": call_id,
        "status": status,
        "payload": payload,
        "error": error,
        "provenance": verified_provenance,
    }
