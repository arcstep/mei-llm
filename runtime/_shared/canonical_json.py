"""Canonical JSON used by the portable runtime-v2 semantic contracts.

The historical SFT renderer in :mod:`schema_render` intentionally remains a
v1 training artefact.  Wire-v2 fingerprints, call IDs, tool indexes and trust
receipts use this insertion-order-independent representation instead.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _portable_number(value: Any) -> bool:
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
        if not _portable_number(value):
            raise ValueError(f"non-portable JSON number at {path}")
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
    """Render one safe JSON number with ECMAScript/JCS thresholds.

    Python and ECMAScript use the same shortest-roundtrip binary64 digits but
    select different plain/scientific notation thresholds.  Re-positioning
    Python's shortest digits gives the portable wire-v2 representation.
    """

    if value == 0:
        return "0"
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    negative = value < 0
    absolute = -value if negative else value
    decimal = Decimal(repr(absolute))
    digits_tuple = decimal.as_tuple().digits
    digits = "".join(str(digit) for digit in digits_tuple)
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


def dumps_canonical(value: Any) -> str:
    """Return compact UTF-8 JSON with recursively sorted object keys.

    Python's string ordering and UTF-8 byte ordering agree for valid Unicode
    scalar strings, so ``sort_keys=True`` implements the cross-runtime key
    rule. Arrays retain their input order. NaN, infinities and non-string
    object keys are rejected rather than coerced.
    """

    _audit_json(value)
    text = _render(value)
    # Reject lone surrogates at the boundary where every runtime hashes bytes.
    text.encode("utf-8")
    return text
