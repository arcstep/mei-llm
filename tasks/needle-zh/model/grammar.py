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

from schema_mask import compile_schema, is_schema_legal_prefix, validate_schema_calls
from schema_render import load_toolset_json

try:
    from route_protocol import is_legal_internal_prefix, parse_internal
except ImportError:
    is_legal_internal_prefix = None
    parse_internal = None

LEGAL_FOOD = {("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")}


@dataclass
class ToolSpec:
    name: str
    required: list[str]
    properties: dict[str, dict]


def load_toolset(toolset_id: str) -> dict[str, ToolSpec]:
    if not toolset_id:
        raise ValueError("toolset_id is required; no default VRM")
    raw = load_toolset_json(toolset_id, root=EVAL_SHARED_ROOT)
    out: dict[str, ToolSpec] = {}
    for tool in raw.get("tools") or []:
        params = tool.get("parameters") or {}
        out[str(tool["name"])] = ToolSpec(
            name=str(tool["name"]),
            required=list(params.get("required") or []),
            properties=dict(params.get("properties") or {}),
        )
    return out


def load_toolset_raw(toolset_id: str) -> dict[str, Any]:
    return load_toolset_json(toolset_id, root=EVAL_SHARED_ROOT)


def _as_schema(toolset: Any) -> Any:
    if toolset is None:
        raise ValueError("toolset is required; no default VRM")
    return toolset


def product_rule_errors(calls: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(calls, list):
        return errors
    for call in calls:
        if not isinstance(call, dict):
            continue
        if call.get("name") != "order_food":
            continue
        args = call.get("arguments") or {}
        pair = (args.get("shop"), args.get("dish"))
        if pair not in LEGAL_FOOD:
            errors.append("illegal shop/dish pair")
    return errors


def validate_calls(
    calls: Any,
    toolset: Any = None,
    *,
    product_rules: bool = False,
) -> list[str]:
    compiled = compile_schema(_as_schema(toolset))
    errors = validate_schema_calls(calls, compiled)
    if product_rules:
        errors.extend(product_rule_errors(calls))
    return errors


def parse_phase1_text(
    text: str,
    toolset: Any = None,
    *,
    product_rules: bool = False,
) -> dict:
    """Parse model text into function_calls or reject."""
    raw = (text or "").strip()
    raw = re.sub(r"</?act_(execute|refuse)>", "", raw)
    raw = raw.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    if not raw:
        return {"ok": False, "function_calls": [], "error": "empty"}
    if not raw.startswith("["):
        return {"ok": False, "function_calls": [], "error": "must start with ["}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "function_calls": [], "error": f"json: {exc}"}
    errors = validate_calls(obj, toolset, product_rules=product_rules)
    if errors:
        return {"ok": False, "function_calls": [], "error": ";".join(errors)}
    calls = obj if isinstance(obj, list) else []
    return {"ok": True, "function_calls": calls, "error": None}


def is_legal_prefix(prefix: str, toolset: Any = None) -> bool:
    return is_schema_legal_prefix(prefix, _as_schema(toolset))


def allowed_first_chars() -> str:
    return "["


def dump_calls(calls: list[dict]) -> str:
    return json.dumps(calls, ensure_ascii=False, separators=(",", ":"))


def parse_route_text(text: str, n_routes: int) -> dict:
    if parse_internal is None:
        raise RuntimeError("route_protocol unavailable")
    return parse_internal(text, n_routes)


def is_legal_route_prefix(prefix: str, n_routes: int) -> bool:
    if is_legal_internal_prefix is None:
        raise RuntimeError("route_protocol unavailable")
    return is_legal_internal_prefix(prefix, n_routes)
