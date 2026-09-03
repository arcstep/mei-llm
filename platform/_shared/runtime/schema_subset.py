"""Fail-closed JSON Schema subset used by every MEI tool runtime.

The product grammar deliberately supports a small, portable subset.  A tool is
rejected at registration time when its schema cannot be represented by the
Python, Rust and browser decoders; generation is never allowed to silently
fall back to unconstrained JSON.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

try:
    from .canonical_json import MAX_SAFE_INTEGER, dumps_canonical
except ImportError:
    from canonical_json import MAX_SAFE_INTEGER, dumps_canonical

SCHEMA_SUBSET_ID = "mei-tool-schema-subset-v2"
SCALAR_TYPES = frozenset({"string", "boolean", "integer", "number", "null"})
SUPPORTED_FORMATS = frozenset(
    {"date", "date-time", "time", "email", "uuid", "uri", "ipv4", "ipv6"}
)
_TIME_RE = re.compile(
    r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?$"
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:.+$")
FORBIDDEN_KEYWORDS = frozenset(
    {
        "$ref",
        "$dynamicRef",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "patternProperties",
        "propertyNames",
        "prefixItems",
        "contains",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
ANNOTATION_KEYWORDS = frozenset({"title", "description", "examples", "$comment"})
TOOL_KEYS = frozenset(
    {
        "name",
        "description",
        "parameters",
        "required_permissions",
        "required_state",
        "x-mei-permissions",
        "x-mei-state",
    }
)
SCALAR_KEYWORDS = frozenset(
    {
        "type",
        "enum",
        "const",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
    }
) | ANNOTATION_KEYWORDS


class UnsupportedSchemaError(ValueError):
    """A schema uses semantics that the portable constrained decoder lacks."""

    def __init__(self, path: str, keyword: str, detail: str | None = None):
        self.path = path
        self.keyword = keyword
        self.detail = detail or "unsupported schema keyword or value"
        super().__init__(f"unsupported_schema at {path}: {keyword}: {self.detail}")

    def as_dict(self) -> dict[str, str]:
        return {
            "id": "unsupported_schema",
            "path": self.path,
            "keyword": self.keyword,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    keyword: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "keyword": self.keyword, "message": self.message}


def _fail_forbidden(schema: dict[str, Any], path: str) -> None:
    for keyword in sorted(FORBIDDEN_KEYWORDS):
        if keyword in schema:
            raise UnsupportedSchemaError(path, keyword)


def _reject_unknown(schema: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(schema) - set(allowed))
    if unknown:
        raise UnsupportedSchemaError(path, unknown[0], "keyword is outside the portable subset")


def _types(schema: dict[str, Any], path: str) -> tuple[str, ...]:
    raw = schema.get("type")
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, list) and raw and all(isinstance(v, str) for v in raw):
        values = tuple(dict.fromkeys(raw))
        non_null = [v for v in values if v != "null"]
        if len(non_null) != 1 or len(values) > 2:
            raise UnsupportedSchemaError(path, "type", "only T or [T, null] is supported")
    elif raw is None and ("enum" in schema or "const" in schema):
        if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
            raise UnsupportedSchemaError(path, "enum", "enum must be a non-empty list")
        values = (_infer_literal_type(schema.get("const") if "const" in schema else schema["enum"][0]),)
    else:
        raise UnsupportedSchemaError(path, "type", "an explicit supported type is required")
    if any(v not in SCALAR_TYPES | {"array"} for v in values):
        raise UnsupportedSchemaError(path, "type", f"unsupported type {values!r}")
    return values


def _infer_literal_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise UnsupportedSchemaError("$", "enum/const", "only scalar literals are supported")


def _portable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    return (
        math.isfinite(value)
        and (not value.is_integer() or abs(value) <= MAX_SAFE_INTEGER)
    )


def _decimal(value: int | float) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid portable number") from exc


def _validate_scalar_schema(schema: dict[str, Any], path: str) -> None:
    _fail_forbidden(schema, path)
    _reject_unknown(schema, SCALAR_KEYWORDS, path)
    types = _types(schema, path)
    if "array" in types:
        raise UnsupportedSchemaError(path, "type", "nested arrays are not supported")
    if "format" in schema:
        if "string" not in types or schema["format"] not in SUPPORTED_FORMATS:
            raise UnsupportedSchemaError(path, "format", str(schema["format"]))
    if "pattern" in schema:
        if "string" not in types or not isinstance(schema["pattern"], str):
            raise UnsupportedSchemaError(path, "pattern", "pattern requires string")
        _validate_portable_pattern(schema["pattern"], path)
    for key in ("minLength", "maxLength"):
        if key in schema and ("string" not in types or not _non_negative_int(schema[key])):
            raise UnsupportedSchemaError(path, key)
    if int(schema.get("minLength", 0)) > int(schema.get("maxLength", 2**31 - 1)):
        raise UnsupportedSchemaError(path, "minLength", "minLength exceeds maxLength")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema:
            if not ({"integer", "number"} & set(types)) or not _portable_number(schema[key]):
                raise UnsupportedSchemaError(path, key)
            if key == "multipleOf" and _decimal(schema[key]) <= 0:
                raise UnsupportedSchemaError(path, key, "multipleOf must be positive")
    lower_candidates = [
        (_decimal(schema[key]), key == "exclusiveMinimum")
        for key in ("minimum", "exclusiveMinimum")
        if key in schema
    ]
    upper_candidates = [
        (_decimal(schema[key]), key == "exclusiveMaximum")
        for key in ("maximum", "exclusiveMaximum")
        if key in schema
    ]
    if lower_candidates and upper_candidates:
        lower, lower_exclusive = max(lower_candidates, key=lambda item: (item[0], item[1]))
        upper, upper_exclusive = min(upper_candidates, key=lambda item: (item[0], not item[1]))
        if lower > upper or (lower == upper and (lower_exclusive or upper_exclusive)):
            raise UnsupportedSchemaError(path, "minimum", "numeric bounds are unsatisfiable")
    for key in ("minItems", "maxItems", "items"):
        if key in schema:
            raise UnsupportedSchemaError(path, key, "array keyword on scalar")
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise UnsupportedSchemaError(path, "enum", "enum must be a non-empty list")
        for item in enum:
            if isinstance(item, (dict, list)):
                raise UnsupportedSchemaError(path, "enum", "only scalar enum values are supported")
            if not any(_type_ok(item, type_name) for type_name in types):
                raise UnsupportedSchemaError(path, "enum", "enum value does not match declared type")
    if "const" in schema and not any(_type_ok(schema["const"], type_name) for type_name in types):
        raise UnsupportedSchemaError(path, "const", "const value does not match declared type")
    if "default" in schema:
        issues = validate_value(schema["default"], schema, path=path)
        if issues:
            raise UnsupportedSchemaError(path, "default", issues[0].message)


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_portable_pattern(pattern: str, path: str) -> None:
    """Reject regex features that differ across Python/Rust/ECMAScript.

    The intentionally small common subset excludes group extensions,
    alphanumeric escapes (classes, Unicode properties and backreferences),
    and nested/set-operation character classes.
    """

    escaped = False
    in_class = False
    for index, char in enumerate(pattern):
        if escaped:
            if char.isascii() and char.isalnum():
                raise UnsupportedSchemaError(
                    path, "pattern", f"non-portable escape \\{char}"
                )
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "(" and pattern[index + 1 : index + 2] == "?":
            raise UnsupportedSchemaError(path, "pattern", "non-portable group extension")
        if in_class and (
            (char == "&" and pattern[index + 1 : index + 2] == "&")
            or (char == "-" and pattern[index + 1 : index + 2] == "-")
            or char == "["
        ):
            raise UnsupportedSchemaError(
                path, "pattern", "non-portable character class syntax"
            )
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
    if escaped:
        raise UnsupportedSchemaError(path, "pattern", "pattern ends with an escape")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise UnsupportedSchemaError(path, "pattern", str(exc)) from exc


def validate_parameter_schema(schema: dict[str, Any], *, path: str = "$.parameters") -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise UnsupportedSchemaError(path, "schema", "schema must be an object")
    _fail_forbidden(schema, path)
    _reject_unknown(
        schema,
        frozenset({"type", "properties", "required", "additionalProperties", "$schema", "$id"})
        | ANNOTATION_KEYWORDS,
        path,
    )
    if schema.get("type") != "object":
        raise UnsupportedSchemaError(path, "type", "tool parameters must be a root object")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise UnsupportedSchemaError(path, "properties", "properties must be an object")
    additional = schema.get("additionalProperties", False)
    if additional is not False:
        raise UnsupportedSchemaError(path, "additionalProperties", "open objects are not portable")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
        raise UnsupportedSchemaError(path, "required", "required must be a string array")
    if len(set(required)) != len(required):
        raise UnsupportedSchemaError(path, "required", "duplicate required property")
    missing = sorted(set(required) - set(props))
    if missing:
        raise UnsupportedSchemaError(path, "required", f"unknown required properties: {missing}")
    for name, child in props.items():
        child_path = f"{path}.properties.{name}"
        if not isinstance(name, str) or not name:
            raise UnsupportedSchemaError(path, "properties", "property names must be non-empty strings")
        if not isinstance(child, dict):
            raise UnsupportedSchemaError(child_path, "schema", "property schema must be an object")
        _fail_forbidden(child, child_path)
        types = _types(child, child_path)
        if "object" in types:
            raise UnsupportedSchemaError(child_path, "type", "nested objects are not supported")
        if "array" in types:
            if len(types) != 1:
                raise UnsupportedSchemaError(child_path, "type", "nullable arrays are not supported")
            items = child.get("items")
            if not isinstance(items, dict):
                raise UnsupportedSchemaError(child_path, "items", "scalar array items are required")
            _validate_scalar_schema(items, child_path + ".items")
            _reject_unknown(
                child,
                frozenset({"type", "items", "minItems", "maxItems", "enum", "const", "default"})
                | ANNOTATION_KEYWORDS,
                child_path,
            )
            for key in ("minItems", "maxItems"):
                if key in child and not _non_negative_int(child[key]):
                    raise UnsupportedSchemaError(child_path, key)
            if int(child.get("minItems", 0)) > int(child.get("maxItems", 2**31 - 1)):
                raise UnsupportedSchemaError(child_path, "minItems", "minItems exceeds maxItems")
            for key in ("minLength", "maxLength", "pattern", "format"):
                if key in child:
                    raise UnsupportedSchemaError(child_path, key, "scalar keyword on array")
            for key in ("const", "default"):
                if key in child and validate_value(child[key], child, path=child_path):
                    raise UnsupportedSchemaError(child_path, key, "array literal violates schema")
            if "enum" in child:
                if not isinstance(child["enum"], list) or not child["enum"]:
                    raise UnsupportedSchemaError(child_path, "enum", "enum must be non-empty")
                for item in child["enum"]:
                    if validate_value(item, child, path=child_path):
                        raise UnsupportedSchemaError(child_path, "enum", "array enum violates schema")
        else:
            _validate_scalar_schema(child, child_path)
    return schema


def validate_tool(tool: dict[str, Any], *, path: str = "$.tool") -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise UnsupportedSchemaError(path, "tool", "tool must be an object")
    _reject_unknown(tool, TOOL_KEYS, path)
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedSchemaError(path, "name", "tool name is required")
    if "description" in tool and not isinstance(tool["description"], str):
        raise UnsupportedSchemaError(path, "description", "description must be a string")
    if "parameters" not in tool:
        raise UnsupportedSchemaError(path, "parameters", "parameters schema is required")
    params = tool["parameters"]
    validate_parameter_schema(params, path=path + ".parameters")
    for key in ("required_permissions", "x-mei-permissions"):
        if key not in tool:
            continue
        values = tool[key]
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) and value for value in values)
            or len(set(values)) != len(values)
        ):
            raise UnsupportedSchemaError(
                path, key, "permission contract must be a unique non-empty string array"
            )
    if "required_permissions" in tool and "x-mei-permissions" in tool:
        raise UnsupportedSchemaError(
            path,
            "required_permissions",
            "canonical and compatibility permission contracts cannot both be present",
        )
    for key in ("required_state", "x-mei-state"):
        if key not in tool:
            continue
        value = tool[key]
        if not isinstance(value, dict):
            raise UnsupportedSchemaError(path, key, "state contract must be an object")
        try:
            dumps_canonical(value)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise UnsupportedSchemaError(path, key, "state contract must be JSON-safe") from exc
    if "required_state" in tool and "x-mei-state" in tool:
        raise UnsupportedSchemaError(
            path,
            "required_state",
            "canonical and compatibility state contracts cannot both be present",
        )
    return tool


def validate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise UnsupportedSchemaError("$.tools", "type", "tools must be an array")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        validate_tool(tool, path=f"$.tools[{index}]")
        name = str(tool["name"])
        if name in names:
            raise UnsupportedSchemaError(f"$.tools[{index}]", "name", "duplicate tool name")
        names.add(name)
    return tools


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and abs(value) <= MAX_SAFE_INTEGER
        )
    if type_name == "number":
        return _portable_number(value)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _format_ok(value: str, format_name: str) -> bool:
    try:
        if format_name == "date":
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
                return False
            _dt.date.fromisoformat(value)
        elif format_name == "date-time":
            if "T" not in value:
                return False
            date, time = value.split("T", 1)
            return _format_ok(date, "date") and _format_ok(time, "time")
        elif format_name == "time":
            return _TIME_RE.fullmatch(value) is not None
        elif format_name == "email":
            return _EMAIL_RE.fullmatch(value) is not None
        elif format_name == "uuid":
            return _UUID_RE.fullmatch(value) is not None
        elif format_name == "uri":
            return _URI_RE.fullmatch(value) is not None
        elif format_name == "ipv4":
            return ipaddress.ip_address(value).version == 4
        elif format_name == "ipv6":
            return ipaddress.ip_address(value).version == 6
        else:
            return False
    except (ValueError, OverflowError):
        return False
    return True


def validate_value(value: Any, schema: dict[str, Any], *, path: str = "$") -> list[ValidationIssue]:
    types = _types(schema, path)
    if not any(_type_ok(value, typ) for typ in types):
        return [ValidationIssue(path, "type", f"expected {types}, got {type(value).__name__}")]
    if "const" in schema and value != schema["const"]:
        return [ValidationIssue(path, "const", "value does not equal const")]
    if "enum" in schema and value not in schema["enum"]:
        return [ValidationIssue(path, "enum", "value is not in enum")]
    if value is None:
        return []
    issues: list[ValidationIssue] = []
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            issues.append(ValidationIssue(path, "minLength", "string is too short"))
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            issues.append(ValidationIssue(path, "maxLength", "string is too long"))
        # JSON Schema ``pattern`` is a search, not an implicit full match.
        if schema.get("pattern") and re.search(str(schema["pattern"]), value) is None:
            issues.append(ValidationIssue(path, "pattern", "string does not match pattern"))
        if schema.get("format") and not _format_ok(value, str(schema["format"])):
            issues.append(ValidationIssue(path, "format", f"invalid {schema['format']}"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _decimal(value)
        checks = (
            ("minimum", lambda x, y: x >= y),
            ("maximum", lambda x, y: x <= y),
            ("exclusiveMinimum", lambda x, y: x > y),
            ("exclusiveMaximum", lambda x, y: x < y),
        )
        for key, predicate in checks:
            if key in schema and not predicate(number, _decimal(schema[key])):
                issues.append(ValidationIssue(path, key, f"number violates {key}"))
        if "multipleOf" in schema:
            divisor = _decimal(schema["multipleOf"])
            if number % divisor != 0:
                issues.append(ValidationIssue(path, "multipleOf", "number is not a multiple"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            issues.append(ValidationIssue(path, "minItems", "array is too short"))
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            issues.append(ValidationIssue(path, "maxItems", "array is too long"))
        item_schema = schema["items"]
        for index, item in enumerate(value):
            issues.extend(validate_value(item, item_schema, path=f"{path}[{index}]"))
    return issues


def validate_arguments(arguments: Any, parameters: dict[str, Any]) -> list[ValidationIssue]:
    validate_parameter_schema(parameters)
    if not isinstance(arguments, dict):
        return [ValidationIssue("$.arguments", "type", "arguments must be an object")]
    props = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    issues: list[ValidationIssue] = []
    for name in sorted(required - set(arguments)):
        issues.append(ValidationIssue(f"$.arguments.{name}", "required", "required argument is missing"))
    for name, value in arguments.items():
        if name not in props:
            issues.append(ValidationIssue(f"$.arguments.{name}", "additionalProperties", "unknown argument"))
            continue
        issues.extend(validate_value(value, props[name], path=f"$.arguments.{name}"))
    return issues
