"""v2 full-call envelope: [] or one {name, arguments} object."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_ID = "mei-tool-call-protocol-v2"
MAX_CALLS = 1


def dump_empty() -> str:
    return "[]"


def dump_call(name: str, arguments: dict[str, Any]) -> str:
    payload = [{"name": name, "arguments": arguments}]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_v2_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "function_calls": [], "error": "empty", "refuse": True}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "function_calls": [], "error": f"json:{exc}", "refuse": True}
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
