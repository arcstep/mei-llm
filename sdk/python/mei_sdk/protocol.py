from __future__ import annotations

import json
from typing import Any

from .canonical import compact_tools, dumps_canonical, schema_fingerprint
from .version import load_json

_PROTO = load_json("protocol.json")
PROTOCOL_ID = str(_PROTO["protocol_id"])
SERIALIZER_ID = str(_PROTO["serializer_id"])
MAX_SELECTED_TOOLS = int(_PROTO["max_selected_tools"])
MAX_CALLS = int(_PROTO["max_calls"])
TASK_CONTRACT = str(_PROTO["task_contract"])
FORBIDDEN_MARKERS = tuple(_PROTO["forbidden_markers"])


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
        obj = json.loads(raw)
    except json.JSONDecodeError:
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


def render_request(request: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tools) > MAX_SELECTED_TOOLS:
        raise ValueError("complete() accepts at most 5 selected tools; retrieval must run first")
    facts = str(request.get("system_facts") or "").strip()
    sink = TASK_CONTRACT
    if facts:
        sink += "\n系统事实：" + facts
    permissions = request.get("permissions") or []
    if permissions:
        sink += "\n权限：" + "、".join(str(p) for p in permissions)
    sink += "\n点选实体：" + (str(request["selected_entity"]) if request.get("selected_entity") else "无")
    sink += "\n" + render_tools_block(tools)
    ordinary_parts: list[str] = []
    for turn in request.get("history") or []:
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            ordinary_parts.append(f"{role}：{content}")
    for result in request.get("prior_tool_results") or []:
        ordinary_parts.append("tool：" + str(result))
    ordinary_parts.append("user：" + str(request.get("query") or "").strip())
    prompt = sink + "\n" + "\n".join(ordinary_parts)
    return {
        "prompt": prompt,
        "schema_fingerprint": schema_fingerprint(tools),
        "serializer_id": SERIALIZER_ID,
        "protocol_id": PROTOCOL_ID,
        "selected_tools": [str(t.get("name") or "") for t in tools],
    }
