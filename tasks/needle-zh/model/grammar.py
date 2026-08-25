"""Phase-1 constrained decode: [] or one schema-legal tool call. No libneedle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from repo_paths import EVAL_SHARED_ROOT  # noqa: E402


@dataclass
class ToolSpec:
    name: str
    required: list[str]
    properties: dict[str, dict]


def load_toolset(toolset_id: str = "needle-vrm-agent-v0") -> dict[str, ToolSpec]:
    path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ToolSpec] = {}
    for tool in raw.get("tools") or []:
        params = tool.get("parameters") or {}
        out[str(tool["name"])] = ToolSpec(
            name=str(tool["name"]),
            required=list(params.get("required") or []),
            properties=dict(params.get("properties") or {}),
        )
    return out


LEGAL_FOOD = {("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")}


def _type_ok(value: Any, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_calls(calls: Any, toolset: dict[str, ToolSpec] | None = None) -> list[str]:
    toolset = toolset or load_toolset()
    if calls == [] or calls is None:
        return []
    if not isinstance(calls, list):
        return ["not a list"]
    if len(calls) > 1:
        return ["phase1 allows at most one call"]
    errors: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            errors.append("call not object")
            continue
        name = str(call.get("name") or "")
        spec = toolset.get(name)
        if not spec:
            errors.append(f"unknown tool {name}")
            continue
        args = call.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            errors.append("arguments not object")
            continue
        for key in spec.required:
            if key not in args:
                errors.append(f"{name} missing {key}")
        for key, val in args.items():
            prop = spec.properties.get(key)
            if not prop:
                errors.append(f"{name} unexpected {key}")
                continue
            declared = str(prop.get("type") or "")
            if declared and not _type_ok(val, declared):
                errors.append(f"{name}.{key} type")
            enum = prop.get("enum")
            if enum is not None and val not in enum:
                errors.append(f"{name}.{key} not in enum")
        if name == "order_food":
            pair = (args.get("shop"), args.get("dish"))
            if pair not in LEGAL_FOOD:
                errors.append("illegal shop/dish pair")
    return errors


def parse_phase1_text(text: str, toolset: dict[str, ToolSpec] | None = None) -> dict:
    """Parse model text into function_calls or reject."""
    raw = (text or "").strip()
    raw = re.sub(r"</?act_(execute|refuse)>", "", raw)
    raw = raw.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    if not raw:
        return {"ok": False, "function_calls": [], "error": "empty"}
    # Phase-1: first non-space must start JSON array.
    if not raw.startswith("["):
        return {"ok": False, "function_calls": [], "error": "must start with ["}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "function_calls": [], "error": f"json: {exc}"}
    errors = validate_calls(obj, toolset)
    if errors:
        return {"ok": False, "function_calls": [], "error": ";".join(errors)}
    calls = obj if isinstance(obj, list) else []
    return {"ok": True, "function_calls": calls, "function_calls": calls, "error": None}


def is_legal_prefix(prefix: str) -> bool:
    s = prefix.lstrip()
    if not s:
        return True
    if s[0] != "[":
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError as exc:
        msg = str(exc)
        return "Unterminated" in msg or "Expecting" in msg or "delimiter" in msg


def allowed_first_chars() -> str:
    return "["


def dump_calls(calls: list[dict]) -> str:
    return json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
