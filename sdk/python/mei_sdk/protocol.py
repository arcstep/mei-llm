from __future__ import annotations

import json
from typing import Any

from .canonical import compact_tools, dumps_canonical, schema_fingerprint
from .json_schema_lite import ContractSchemaError, loads_strict, validate_instance
from .shared import UnsupportedSchemaError, validate_tools
from .version import SPEC_DIR, load_json

_PROTO = load_json("protocol.json")
PROTOCOL_ID = str(_PROTO["protocol_id"])
SERIALIZER_ID = str(_PROTO["serializer_id"])
MAX_SELECTED_TOOLS = int(_PROTO["max_selected_tools"])
MAX_CALLS = int(_PROTO["max_calls"])
TASK_CONTRACT = str(_PROTO["task_contract"])
FORBIDDEN_MARKERS = tuple(_PROTO["forbidden_markers"])
WIRE_V1 = "mei-runtime-wire-v1"
WIRE_V2 = "mei-runtime-wire-v2"
DEFAULT_MAX_STEPS = 4
HARD_MAX_STEPS = 8
DEFAULT_MAX_OUTPUT_TOKENS = 128
MAX_TOOL_RESULT_BYTES = 64 * 1024
_REQUEST_V2_SCHEMA: dict[str, Any] | None = None
_REQUEST_V2_REGISTRY: dict[str, dict[str, Any]] | None = None
_TOOL_RESULT_V2_SCHEMA: dict[str, Any] | None = None


def _validate_request_v2(request: dict[str, Any]) -> None:
    global _REQUEST_V2_SCHEMA, _REQUEST_V2_REGISTRY
    if _REQUEST_V2_SCHEMA is None or _REQUEST_V2_REGISTRY is None:
        try:
            request_schema = loads_strict(
                (SPEC_DIR / "request-v2.schema.json").read_text(encoding="utf-8")
            )
            tool_result_schema = loads_strict(
                (SPEC_DIR / "tool-result-v2.schema.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load CompleteRequestV2 schema: {exc}") from exc
        if not isinstance(request_schema, dict) or not isinstance(tool_result_schema, dict):
            raise ValueError("wire v2 schemas must be JSON objects")
        _REQUEST_V2_SCHEMA = request_schema
        _REQUEST_V2_REGISTRY = {"tool-result-v2.schema.json": tool_result_schema}
    try:
        validate_instance(request, _REQUEST_V2_SCHEMA, registry=_REQUEST_V2_REGISTRY)
    except ContractSchemaError as exc:
        raise ValueError(f"CompleteRequestV2 JSON Schema: {exc}") from exc


def validate_tool_result_v2(result: dict[str, Any]) -> None:
    """Validate the exact public ``ToolResultV2`` contract fail closed.

    This deliberately uses the checked-in schema rather than a second set of
    hand-written conditionals.  In particular, an ``error`` result must carry
    ``{code,message}``, while an ``ok`` result may not smuggle a non-null error.
    """

    global _TOOL_RESULT_V2_SCHEMA
    if not isinstance(result, dict):
        raise ValueError("ToolResultV2 must be an object")
    if _TOOL_RESULT_V2_SCHEMA is None:
        try:
            loaded = loads_strict(
                (SPEC_DIR / "tool-result-v2.schema.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load ToolResultV2 schema: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("ToolResultV2 schema root must be an object")
        _TOOL_RESULT_V2_SCHEMA = loaded
    try:
        validate_instance(result, _TOOL_RESULT_V2_SCHEMA)
    except ContractSchemaError as exc:
        raise ValueError(f"ToolResultV2 JSON Schema: {exc}") from exc


def leak_markers(text: str) -> list[str]:
    return [m for m in FORBIDDEN_MARKERS if m in (text or "")]


def request_leaks(request: dict[str, Any]) -> list[str]:
    chunks = [
        str(request.get("query") or ""),
        str(request.get("system_facts") or ""),
        str(request.get("selected_entity") or ""),
        str(request.get("candidate_text") or ""),
        dumps_canonical(request.get("history") or []),
        dumps_canonical(request.get("prior_tool_results") or []),
        dumps_canonical(request.get("catalog") or []),
        dumps_canonical(request.get("oracle_tools") or []),
        dumps_canonical(request.get("permissions") or []),
        dumps_canonical(request.get("context") or {}),
        dumps_canonical(request.get("evidence") or []),
        dumps_canonical(request.get("entities") or []),
        dumps_canonical(request.get("tool_results") or []),
        dumps_canonical(request.get("state") or {}),
        dumps_canonical(request.get("mw") or {}),
        dumps_canonical(request.get("mw_disposition") or {}),
        dumps_canonical(request.get("confidence") or {}),
    ]
    found: list[str] = []
    blob = "\n".join(chunks)
    for marker in FORBIDDEN_MARKERS:
        if marker in blob and marker not in found:
            found.append(marker)
    return found


def parse_v2_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "function_calls": [], "error": "empty", "refuse": True}
    try:
        obj = loads_strict(raw)
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "function_calls": [], "error": "json", "refuse": True}
    if obj == []:
        return {"ok": True, "function_calls": [], "error": None, "refuse": True}
    if not isinstance(obj, list) or len(obj) > MAX_CALLS:
        return {"ok": False, "function_calls": [], "error": "illegal_shape", "refuse": True}
    if len(obj) != 1 or not isinstance(obj[0], dict):
        return {"ok": False, "function_calls": [], "error": "illegal_shape", "refuse": True}
    name = obj[0].get("name")
    args = obj[0].get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return {"ok": False, "function_calls": [], "error": "illegal_item", "refuse": True}
    return {
        "ok": True,
        "function_calls": [{"name": name, "arguments": args}],
        "error": None,
        "refuse": False,
    }


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    return "<tools>" + dumps_canonical(compact_tools(tools)) + "</tools>"


def adapt_v1_request(request: dict[str, Any]) -> dict[str, Any]:
    """Map the read-only v1 request shape into a degraded in-memory v2 view."""
    # Caller-supplied underscore fields are internal trust-domain state, not a
    # legacy extension surface.  In particular, accepting a forged verified
    # result map would let an untrusted ToolResult manufacture provenance.
    out = {key: value for key, value in request.items() if not str(key).startswith("_")}
    context = dict(out.get("context") or {})
    if out.get("system_facts") is not None:
        context.setdefault("system_facts", str(out.get("system_facts") or ""))
    if out.get("selected_entity") is not None:
        entities = list(out.get("entities") or [])
        entities.append(
            {
                "id": "v1:selected_entity",
                "canonical": out.get("selected_entity"),
                "verified": True,
            }
        )
        out["entities"] = entities
    out["context"] = context
    if "tool_results" not in out and "prior_tool_results" in out:
        out["tool_results"] = list(out.get("prior_tool_results") or [])
    out["wire_version"] = WIRE_V2
    out["_compat"] = {
        "source_wire": WIRE_V1,
        "mode": "read_only_degraded_adapter",
        "capability_complete": False,
    }
    return out


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    wire = request.get("wire_version")
    if wire == WIRE_V2:
        _validate_request_v2(request)
    out = adapt_v1_request(request) if wire in (None, "", WIRE_V1) else dict(request)
    if out.get("wire_version") != WIRE_V2:
        raise ValueError(f"unsupported wire_version: {out.get('wire_version')}")
    if not isinstance(out.get("context") or {}, dict):
        raise ValueError("context must be an object")
    for key in ("history", "evidence", "entities", "tool_results"):
        if not isinstance(out.get(key) or [], list):
            raise ValueError(f"{key} must be an array")
    if not isinstance(out.get("state") or {}, dict):
        raise ValueError("state must be an object")
    permissions = out.get("permissions") or {}
    if (
        wire == WIRE_V2
        and isinstance(permissions, dict)
        and "scopes" in permissions
        and "grants" in permissions
    ):
        raise ValueError("permissions cannot contain both scopes and grants")
    # Python callers do not necessarily cross a JSON parser.  Audit the same
    # finite, string-keyed value domain used by native wire implementations.
    dumps_canonical(out)
    return out


def render_request(
    request: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    already_normalized: bool = False,
) -> dict[str, Any]:
    if len(tools) > MAX_SELECTED_TOOLS:
        raise ValueError("complete() accepts at most 5 selected tools; retrieval must run first")
    validate_tools(tools)
    normalized = dict(request) if already_normalized else normalize_request(request)
    context = normalized.get("context") or {}
    is_v1_adapter = (normalized.get("_compat") or {}).get("source_wire") == WIRE_V1
    sink = TASK_CONTRACT
    if is_v1_adapter:
        facts = str(context.get("system_facts") or normalized.get("system_facts") or "").strip()
        if facts:
            sink += "\n系统事实：" + facts
        permissions = normalized.get("permissions") or []
        if permissions:
            if isinstance(permissions, dict):
                permission_text = dumps_canonical(permissions)
            else:
                permission_text = "、".join(str(p) for p in permissions)
            sink += "\n权限：" + permission_text
        selected_entity = normalized.get("selected_entity")
        sink += "\n点选实体：" + (str(selected_entity) if selected_entity else "无")
    sink += "\n" + render_tools_block(tools)
    ordinary_parts: list[str] = []
    for turn in normalized.get("history") or []:
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            ordinary_parts.append(f"{role}：{content}")
    for result in normalized.get("tool_results") or normalized.get("prior_tool_results") or []:
        ordinary_parts.append("tool：" + (dumps_canonical(result) if isinstance(result, dict) else str(result)))
    if not is_v1_adapter:
        for tag, value in (
            ("context", normalized.get("context") or {}),
            ("evidence", normalized.get("evidence") or []),
            ("permissions", normalized.get("permissions") or {}),
            ("state", normalized.get("state") or {}),
            ("mw", normalized.get("mw") or normalized.get("mw_disposition") or {}),
        ):
            ordinary_parts.append(f"<{tag}>" + dumps_canonical(value) + f"</{tag}>")
    ordinary_parts.append("user：" + str(normalized.get("query") or "").strip())
    prompt = sink + "\n" + "\n".join(ordinary_parts)
    return {
        "prompt": prompt,
        "sink": sink,
        "ordinary": "\n".join(ordinary_parts),
        "schema_fingerprint": schema_fingerprint(tools),
        "serializer_id": SERIALIZER_ID,
        "protocol_id": PROTOCOL_ID,
        "selected_tools": [str(t.get("name") or "") for t in tools],
        "request": normalized,
        "compatibility": normalized.get("_compat"),
    }
