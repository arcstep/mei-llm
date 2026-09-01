"""Small fail-closed JSON Schema validator for bundled SDK contracts.

This is not a general-purpose replacement for ``jsonschema``.  It implements
exactly the 2020-12 keywords used by the checked-in MEI package schemas and
raises when a future schema introduces an unknown assertion keyword, avoiding
a silent validation downgrade in the dependency-free SDK.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

from .canonical import MAX_SAFE_INTEGER, portable_number


class ContractSchemaError(ValueError):
    pass


_KNOWN = {
    "$id",
    "$ref",
    "$defs",
    "$schema",
    "title",
    "description",
    "default",
    "deprecated",
    "type",
    "const",
    "enum",
    "required",
    "additionalProperties",
    "properties",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
}


def loads_strict(text: str) -> Any:
    """Parse standard JSON while rejecting duplicate keys and non-finite literals."""

    import json

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in values:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    def integer(value: str) -> int:
        parsed = int(value)
        if abs(parsed) > MAX_SAFE_INTEGER:
            raise ValueError("JSON integer exceeds the wire-v2 safe domain")
        return parsed

    def number(value: str) -> float:
        parsed = float(value)
        if not portable_number(parsed):
            raise ValueError("JSON number exceeds the wire-v2 binary64 domain")
        return parsed

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=constant,
        parse_int=integer,
        parse_float=number,
    )


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    registry: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not isinstance(schema, dict):
        raise ContractSchemaError("contract schema root must be an object")
    schemas = registry or {}
    _audit_schema(schema, root=schema, registry=schemas, seen=set())
    _validate(instance, schema, root=schema, registry=schemas, path="$")


def _audit_schema(
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    seen: set[int],
) -> None:
    identity = id(schema)
    if identity in seen:
        return
    seen.add(identity)
    unknown = sorted(set(schema) - _KNOWN)
    if unknown:
        raise ContractSchemaError(f"unsupported schema keyword {unknown[0]}")
    if "$ref" in schema:
        resolved, resolved_root = _resolve(root, str(schema["$ref"]), registry)
        _audit_schema(resolved, root=resolved_root, registry=registry, seen=seen)
    for key in ("$defs", "properties"):
        children = schema.get(key)
        if isinstance(children, dict):
            for child in children.values():
                if not isinstance(child, dict):
                    raise ContractSchemaError(f"invalid child schema under {key}")
                _audit_schema(child, root=root, registry=registry, seen=seen)
    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        child = schema.get(key)
        if isinstance(child, dict):
            _audit_schema(child, root=root, registry=registry, seen=seen)
    for key in ("allOf", "anyOf", "oneOf"):
        children = schema.get(key)
        if children is not None:
            if not isinstance(children, list):
                raise ContractSchemaError(f"{key} must be an array")
            for child in children:
                if not isinstance(child, dict):
                    raise ContractSchemaError(f"invalid child schema under {key}")
                _audit_schema(child, root=root, registry=registry, seen=seen)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return portable_number(left) and portable_number(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    return left == right


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and portable_number(value)
            and (isinstance(value, int) or value.is_integer())
        )
    if expected == "number":
        return portable_number(value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ContractSchemaError(f"unsupported contract schema type: {expected}")


def _pointer(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContractSchemaError(f"unsupported non-local schema reference: {ref}")
    value: Any = root
    for raw in ref[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            raise ContractSchemaError(f"unresolved schema reference: {ref}")
        value = value[key]
    if not isinstance(value, dict):
        raise ContractSchemaError(f"schema reference is not an object: {ref}")
    return value


def _resolve(
    root: dict[str, Any], ref: str, registry: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if ref.startswith("#/"):
        return _pointer(root, ref), root
    name, separator, fragment = ref.partition("#")
    target_root = registry.get(name)
    if target_root is None:
        raise ContractSchemaError(f"unresolved external schema reference: {ref}")
    if separator and fragment:
        return _pointer(target_root, "#" + fragment), target_root
    return target_root, target_root


def _matches(
    instance: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    path: str,
) -> bool:
    try:
        _validate(instance, schema, root=root, registry=registry, path=path)
    except ContractSchemaError:
        return False
    return True


def _validate(
    instance: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    path: str,
) -> None:
    unknown = sorted(set(schema) - _KNOWN)
    if unknown:
        raise ContractSchemaError(f"unsupported schema keyword {unknown[0]} at {path}")
    if "$ref" in schema:
        resolved, resolved_root = _resolve(root, str(schema["$ref"]), registry)
        _validate(instance, resolved, root=resolved_root, registry=registry, path=path)

    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else expected
        if not isinstance(types, list) or not types or not all(isinstance(item, str) for item in types):
            raise ContractSchemaError(f"invalid type assertion in contract schema at {path}")
        if not any(_type_ok(instance, item) for item in types):
            raise ContractSchemaError(f"{path}: expected {types}, got {type(instance).__name__}")
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ContractSchemaError(f"{path}: value does not equal const")
    if "enum" in schema and not any(_json_equal(instance, item) for item in schema["enum"]):
        raise ContractSchemaError(f"{path}: value is outside enum")

    for child in schema.get("allOf") or []:
        _validate(instance, child, root=root, registry=registry, path=path)
    any_of = schema.get("anyOf")
    if any_of is not None and not any(
        _matches(instance, child, root=root, registry=registry, path=path) for child in any_of
    ):
        raise ContractSchemaError(f"{path}: no anyOf branch matched")
    one_of = schema.get("oneOf")
    if one_of is not None and sum(
        _matches(instance, child, root=root, registry=registry, path=path)
        for child in one_of
    ) != 1:
        raise ContractSchemaError(f"{path}: expected exactly one oneOf branch")
    if "not" in schema and _matches(
        instance, schema["not"], root=root, registry=registry, path=path
    ):
        raise ContractSchemaError(f"{path}: prohibited schema branch matched")
    if "if" in schema:
        branch = (
            schema.get("then")
            if _matches(instance, schema["if"], root=root, registry=registry, path=path)
            else schema.get("else")
        )
        if branch is not None:
            _validate(instance, branch, root=root, registry=registry, path=path)

    if isinstance(instance, dict):
        required = schema.get("required") or []
        missing = [key for key in required if key not in instance]
        if missing:
            raise ContractSchemaError(f"{path}: missing required field {missing[0]}")
        properties = schema.get("properties") or {}
        for key, value in instance.items():
            if key in properties:
                _validate(
                    value,
                    properties[key],
                    root=root,
                    registry=registry,
                    path=f"{path}.{key}",
                )
            elif schema.get("additionalProperties") is False:
                raise ContractSchemaError(f"{path}: unexpected field {key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(
                    value,
                    schema["additionalProperties"],
                    root=root,
                    registry=registry,
                    path=f"{path}.{key}",
                )
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            raise ContractSchemaError(f"{path}: array is too short")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise ContractSchemaError(f"{path}: array is too long")
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(instance):
                if any(_json_equal(item, previous) for previous in instance[:index]):
                    raise ContractSchemaError(f"{path}: array items are not unique")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(instance):
                _validate(
                    item,
                    schema["items"],
                    root=root,
                    registry=registry,
                    path=f"{path}[{index}]",
                )
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            raise ContractSchemaError(f"{path}: string is too short")
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            raise ContractSchemaError(f"{path}: string is too long")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            raise ContractSchemaError(f"{path}: string does not match pattern")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not portable_number(instance):
            raise ContractSchemaError(f"{path}: number is outside the portable domain")
        number = Decimal(str(instance))
        if "minimum" in schema and number < Decimal(str(schema["minimum"])):
            raise ContractSchemaError(f"{path}: number is below minimum")
        if "maximum" in schema and number > Decimal(str(schema["maximum"])):
            raise ContractSchemaError(f"{path}: number is above maximum")
