from __future__ import annotations

import hashlib
import json
from typing import Any


def dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        out.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    payload = dumps_canonical(compact_tools(tools))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
