from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any

MAX_SAFE_INTEGER = 9_007_199_254_740_991


def portable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    return (
        math.isfinite(value)
        and (not value.is_integer() or abs(value) <= MAX_SAFE_INTEGER)
    )


def _audit_json(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not portable_number(value):
            raise ValueError(
                f"non-portable JSON number outside the cross-runtime safe domain at {path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _audit_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string")
            _audit_json(item, path=f"{path}.{key}")
        return
    raise TypeError(f"value at {path} is not JSON-compatible")


def _number_to_string(value: int | float) -> str:
    if value == 0:
        return "0"
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    negative = value < 0
    absolute = -value if negative else value
    decimal = Decimal(repr(absolute))
    digits = "".join(str(digit) for digit in decimal.as_tuple().digits)
    exponent = decimal.as_tuple().exponent
    while len(digits) > 1 and digits.endswith("0"):
        digits = digits[:-1]
        exponent += 1
    decimal_position = len(digits) + exponent
    if 1e-6 <= absolute < 1e21:
        if decimal_position <= 0:
            rendered = "0." + ("0" * -decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + ("0" * (decimal_position - len(digits)))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
    else:
        coefficient = digits[0]
        if len(digits) > 1:
            coefficient += "." + digits[1:]
        scientific_exponent = decimal_position - 1
        exponent_text = (
            f"+{scientific_exponent}"
            if scientific_exponent >= 0
            else str(scientific_exponent)
        )
        rendered = coefficient + "e" + exponent_text
    return ("-" if negative else "") + rendered


def _render(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number_to_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_render(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-8"))
        return "{" + ",".join(
            _render(key) + ":" + _render(value[key]) for key in keys
        ) + "}"
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def dumps_canonical(obj: Any) -> str:
    """Compact canonical JSON: recursively UTF-8-key-sorted, arrays stable."""

    _audit_json(obj)
    text = _render(obj)
    text.encode("utf-8")
    return text


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
