#!/usr/bin/env python3
"""Offline control plane for mei-1.0-51m CPT and tool-SFT corpora.

The factory is deliberately provider-free.  Semantic worlds, task gold and tool
schemas are frozen before any visible Chinese realization is applied.  A
realizer/teacher may change only the ``query`` field; all executable semantics
are compiled again from the frozen task object.

The first release is a small, deterministic control-plane fixture.  It proves
contracts, isolation, hashing, recovery and audit behavior; it is not a claim
that a scale-ready training corpus has been produced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "mei-51m-corpus-factory-contract-v1"
BUILD_SCHEMA = "mei-51m-corpus-factory-build-v1"
AUDIT_SCHEMA = "mei-51m-corpus-factory-audit-v1"
RELEASE_SCHEMA = "mei-51m-corpus-factory-release-v1"
EVAL_LOCK_SCHEMA = "mei-51m-independent-eval-lock-v1"
EXPERIMENT_SCHEMA = "mei-51m-corpus-experiment-plan-v1"
SOURCE_REGISTRY_SCHEMA = "mei-51m-corpus-source-registry-v1"
SCALE_INTAKE_SCHEMA = "mei-51m-corpus-scale-intake-v1"
SCALE_PLAN_SCHEMA = "mei-51m-corpus-scale-plan-v1"
MAX_SCALE_WORK_UNITS = 100_000

PRODUCT = "mei-1.0-51m"
SERIALIZER = "mei-tool-call-serializer-v2"
SPLITS = ("train", "dev", "test")
LANES = (
    "cpt_colloquial",
    "retrieval",
    "full_call",
    "agent",
    "mw_disposition",
    "confidence",
    "narration",
)
ISOLATION_AXES = (
    "semantic_family",
    "schema_family",
    "realizer_id",
    "teacher_id",
    "scene_id",
    "template_id",
)
TERMINAL_STATUSES = {"success", "error", "cancelled", "partial"}

SCALE_AUDIT_STRATEGIES = {
    "byte_exact": "partitioned_sha256_external_sort",
    "normalized_template": "partitioned_normalized_hash_external_sort",
    "near_duplicate": "minhash_lsh_candidates_then_exact_jaccard",
    "semantic_consistency": "local_schema_recompile",
    "identifier_padding": "streaming_regex_and_distribution",
    "eval_contamination": "locked_eval_hash_and_lsh_join",
    "provenance_license": "manifest_join_fail_closed",
}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
LONG_HEX_RE = re.compile(r"\b[0-9a-f]{20,}\b", re.I)
IDENTIFIER_PADDING_RE = re.compile(
    r"(?:编号(?:口|号)?\s*\d{5,}|\b[a-z][a-z0-9_-]*[-_]\d{6,}\b)", re.I
)
EVAL_MARKER_RE = re.compile(r"\bEVAL[-_][A-Z0-9_-]+\b", re.I)
SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


class FactoryError(RuntimeError):
    """Fail-closed corpus factory error."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{sha256_json(list(parts))[:32]}"


def _reject_json_constant(raw: str) -> None:
    raise ValueError(f"non-finite JSON number: {raw}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise FactoryError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FactoryError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FactoryError(f"cannot read JSONL {path}: {error}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise FactoryError(f"invalid JSONL {path}:{number}: {error}") from error
        if not isinstance(value, dict):
            raise FactoryError(f"JSONL object required at {path}:{number}")
        rows.append(value)
    return rows


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(dict(row)) for row in rows)


def _atomic_write(path: Path, payload: bytes, *, write_once: bool = False) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        if write_once:
            raise FactoryError(
                f"refusing to overwrite different immutable file: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as h:
        temporary = Path(h.name)
        h.write(payload)
        h.flush()
    temporary.replace(path)


def _safe_relative(raw: str) -> Path:
    if not raw or not SAFE_RELATIVE_RE.fullmatch(raw):
        raise FactoryError(f"unsafe artifact path: {raw!r}")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise FactoryError(f"artifact path must be relative and contained: {raw!r}")
    return path


def artifact_spec(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def artifact_spec_bytes(payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"sha256": sha256_bytes(payload), "bytes": len(payload)}
    if rows is not None:
        result["rows"] = rows
    return result


def merkle_root(artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a path-bound binary Merkle root over immutable artifacts."""

    leaves = [
        hashlib.sha256(
            b"leaf\0"
            + name.encode("utf-8")
            + b"\0"
            + str(spec["sha256"]).encode("ascii")
            + b"\0"
            + str(int(spec["bytes"])).encode("ascii")
        ).digest()
        for name, spec in sorted(artifacts.items())
    ]
    if not leaves:
        return hashlib.sha256(b"empty-merkle-v1").hexdigest()
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level = [*level, level[-1]]
        level = [
            hashlib.sha256(b"node\0" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def verify_artifacts(
    root: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    errors: list[str] = []
    for raw_name, spec in sorted(artifacts.items()):
        rel = _safe_relative(raw_name)
        path = root / rel
        if not path.is_file():
            errors.append(f"missing artifact: {raw_name}")
            continue
        if path.stat().st_size != int(spec.get("bytes") or -1):
            errors.append(f"artifact byte size drift: {raw_name}")
            continue
        if sha256_file(path) != spec.get("sha256"):
            errors.append(f"artifact hash drift: {raw_name}")
            continue
        if "rows" in spec and path.suffix == ".jsonl":
            actual_rows = sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if actual_rows != int(spec["rows"]):
                errors.append(f"artifact row-count drift: {raw_name}")
    return errors


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def validate_instance(
    schema: Mapping[str, Any], value: Any, *, where: str = "$"
) -> list[str]:
    """Validate the JSON-Schema subset used by the offline fixture."""

    errors: list[str] = []
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or [])
    if types and not any(_type_ok(value, item) for item in types):
        return [f"{where}: expected type {types}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: value does not equal const")
    if "enum" in schema and value not in schema.get("enum", []):
        errors.append(f"{where}: value not in enum")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                errors.append(f"{where}: missing required property {name}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{where}: unsupported additional property {name}")
        for name, item in value.items():
            if name in properties:
                errors.extend(
                    validate_instance(properties[name], item, where=f"{where}.{name}")
                )

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < int(minimum):
            errors.append(f"{where}: fewer than minItems")
        if maximum is not None and len(value) > int(maximum):
            errors.append(f"{where}: more than maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_instance(item_schema, item, where=f"{where}[{index}]")
                )

    if isinstance(value, str):
        if schema.get("minLength") is not None and len(value) < int(
            schema["minLength"]
        ):
            errors.append(f"{where}: shorter than minLength")
        if schema.get("maxLength") is not None and len(value) > int(
            schema["maxLength"]
        ):
            errors.append(f"{where}: longer than maxLength")
        if schema.get("pattern") is not None:
            try:
                if re.search(str(schema["pattern"]), value) is None:
                    errors.append(f"{where}: pattern mismatch")
            except re.error:
                errors.append(f"{where}: invalid schema pattern")
        if schema.get("format") == "date":
            try:
                date.fromisoformat(value)
            except ValueError:
                errors.append(f"{where}: invalid date format")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.get("minimum") is not None and value < schema["minimum"]:
            errors.append(f"{where}: below minimum")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            errors.append(f"{where}: above maximum")
        multiple = schema.get("multipleOf")
        if multiple is not None and not math.isclose(
            value / multiple, round(value / multiple)
        ):
            errors.append(f"{where}: not a multipleOf {multiple}")
    return errors


def validate_schema_definition(
    schema: Mapping[str, Any], *, where: str = "$"
) -> list[str]:
    """Validate the bounded JSON-Schema dialect accepted by this factory."""

    errors: list[str] = []
    allowed_keywords = {
        "type",
        "title",
        "description",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
    }
    unsupported = sorted(set(schema) - allowed_keywords)
    if unsupported:
        errors.append(f"{where}: unsupported schema keywords {unsupported}")

    declared = schema.get("type")
    if isinstance(declared, str):
        types = [declared]
    elif (
        isinstance(declared, list)
        and declared
        and all(isinstance(item, str) for item in declared)
    ):
        types = list(declared)
    else:
        errors.append(f"{where}: schema type must be a string or non-empty string list")
        types = []
    supported_types = {
        "string",
        "boolean",
        "integer",
        "number",
        "null",
        "array",
        "object",
    }
    unknown_types = sorted(set(types) - supported_types)
    if unknown_types:
        errors.append(f"{where}: unsupported schema types {unknown_types}")
    if len(types) != len(set(types)):
        errors.append(f"{where}: schema type list contains duplicates")

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            errors.append(f"{where}: enum must be a non-empty list")
        elif len({canonical_bytes(item) for item in enum}) != len(enum):
            errors.append(f"{where}: enum contains duplicate values")
    if "const" in schema and "enum" in schema and schema["const"] not in schema["enum"]:
        errors.append(f"{where}: const must be present in enum")

    if "object" in types:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            errors.append(f"{where}: object schema requires properties")
            properties = {}
        if schema.get("additionalProperties") is not False:
            errors.append(f"{where}: object schema must set additionalProperties=false")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            errors.append(f"{where}: required must be a string list")
            required = []
        if len(required) != len(set(required)):
            errors.append(f"{where}: required contains duplicates")
        missing = sorted(set(required) - set(properties))
        if missing:
            errors.append(f"{where}: required names absent from properties {missing}")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                errors.append(f"{where}: property names must be non-empty strings")
                continue
            if not isinstance(child, dict):
                errors.append(f"{where}.{name}: property schema must be an object")
                continue
            errors.extend(validate_schema_definition(child, where=f"{where}.{name}"))

    if "array" in types:
        items = schema.get("items")
        if not isinstance(items, dict):
            errors.append(f"{where}: array schema requires one item schema")
        else:
            errors.extend(validate_schema_definition(items, where=f"{where}[]"))
        for key in ("minItems", "maxItems"):
            if key in schema and (
                not isinstance(schema[key], int)
                or isinstance(schema[key], bool)
                or schema[key] < 0
            ):
                errors.append(f"{where}: {key} must be a non-negative integer")
        if (
            isinstance(schema.get("minItems"), int)
            and isinstance(schema.get("maxItems"), int)
            and schema["minItems"] > schema["maxItems"]
        ):
            errors.append(f"{where}: minItems exceeds maxItems")

    if "string" in types:
        for key in ("minLength", "maxLength"):
            if key in schema and (
                not isinstance(schema[key], int)
                or isinstance(schema[key], bool)
                or schema[key] < 0
            ):
                errors.append(f"{where}: {key} must be a non-negative integer")
        if (
            isinstance(schema.get("minLength"), int)
            and isinstance(schema.get("maxLength"), int)
            and schema["minLength"] > schema["maxLength"]
        ):
            errors.append(f"{where}: minLength exceeds maxLength")
        if "pattern" in schema:
            if not isinstance(schema["pattern"], str):
                errors.append(f"{where}: pattern must be a string")
            else:
                try:
                    re.compile(schema["pattern"])
                except re.error:
                    errors.append(f"{where}: invalid schema pattern")
        if "format" in schema and schema["format"] != "date":
            errors.append(f"{where}: unsupported string format {schema['format']!r}")

    if {"integer", "number"} & set(types):
        for key in ("minimum", "maximum", "multipleOf"):
            if key in schema and (
                not isinstance(schema[key], (int, float))
                or isinstance(schema[key], bool)
            ):
                errors.append(f"{where}: {key} must be numeric")
        if (
            isinstance(schema.get("multipleOf"), (int, float))
            and schema["multipleOf"] <= 0
        ):
            errors.append(f"{where}: multipleOf must be greater than zero")
        if (
            isinstance(schema.get("minimum"), (int, float))
            and isinstance(schema.get("maximum"), (int, float))
            and schema["minimum"] > schema["maximum"]
        ):
            errors.append(f"{where}: minimum exceeds maximum")

    return errors


def _required_string(row: Mapping[str, Any], key: str, *, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FactoryError(f"{where}.{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ContractIndex:
    contract: dict[str, Any]
    tools: dict[str, dict[str, Any]]
    codebook: dict[str, dict[str, Any]]
    allowed_licenses: frozenset[str]


def validate_contract(contract: dict[str, Any]) -> ContractIndex:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise FactoryError(f"contract schema must be {CONTRACT_SCHEMA}")
    if contract.get("product") != PRODUCT:
        raise FactoryError(f"contract product must be {PRODUCT}")
    _required_string(contract, "factory_id", where="contract")
    if contract.get("purpose") != "offline_control_plane_fixture":
        raise FactoryError(
            "phase-1 factory accepts only offline_control_plane_fixture purpose"
        )

    teacher = contract.get("teacher_contract") or {}
    if teacher.get("mutable_fields") != ["query"]:
        raise FactoryError("teacher mutable_fields must be exactly ['query']")
    if teacher.get("provider_calls_allowed") is not False:
        raise FactoryError("provider calls must be disabled in the phase-1 factory")
    if teacher.get("gold_compiled_locally") is not True:
        raise FactoryError("teacher contract must recompile gold locally")

    registry = contract.get("split_registry") or {}
    for axis in ISOLATION_AXES:
        mapping = registry.get(axis)
        if not isinstance(mapping, dict) or not mapping:
            raise FactoryError(f"split_registry.{axis} must be a non-empty mapping")
        invalid = {value for value in mapping.values() if value not in SPLITS}
        if invalid:
            raise FactoryError(
                f"split_registry.{axis} contains invalid splits: {sorted(invalid)}"
            )

    allowed = frozenset(
        str(item) for item in (contract.get("license_policy") or {}).get("allowed", [])
    )
    if not allowed:
        raise FactoryError("license_policy.allowed cannot be empty")
    if (contract.get("license_policy") or {}).get(
        "public_distribution_clearance_asserted"
    ) is not False:
        raise FactoryError("factory must not assert public distribution clearance")

    quality = contract.get("quality_gates") or {}
    if quality.get("fixture_only") is not True:
        raise FactoryError("phase-1 contract must remain fixture_only")
    threshold = quality.get("near_duplicate_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < threshold <= 1
    ):
        raise FactoryError("near_duplicate_threshold must be in (0, 1]")
    maximum_rows = quality.get("max_fixture_rows")
    if (
        not isinstance(maximum_rows, int)
        or isinstance(maximum_rows, bool)
        or not 1 <= maximum_rows <= 100_000
    ):
        raise FactoryError("max_fixture_rows must be between 1 and 100,000")
    budget = contract.get("budget") or {}
    maximum_cost = budget.get("max_cost_cny")
    if (
        not isinstance(maximum_cost, (int, float))
        or isinstance(maximum_cost, bool)
        or maximum_cost != 0
    ):
        raise FactoryError("phase-1 budget.max_cost_cny must be numeric zero")
    if not isinstance(contract.get("cost_events"), list):
        raise FactoryError("phase-1 cost_events must be a list")

    raw_tools = contract.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise FactoryError("at least one tool schema is required")
    tools: dict[str, dict[str, Any]] = {}
    for number, tool in enumerate(raw_tools):
        where = f"tools[{number}]"
        if not isinstance(tool, dict):
            raise FactoryError(f"{where} must be an object")
        name = _required_string(tool, "name", where=where)
        if name in tools:
            raise FactoryError(f"duplicate tool name: {name}")
        schema_family = _required_string(tool, "schema_family", where=where)
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise FactoryError(f"{where}.parameters must be an object schema")
        schema_errors = validate_schema_definition(
            parameters, where=f"{where}.parameters"
        )
        if schema_errors:
            raise FactoryError(
                f"{where} has invalid schema: {'; '.join(schema_errors)}"
            )
        if schema_family not in registry["schema_family"]:
            raise FactoryError(
                f"tool {name} uses unregistered schema_family {schema_family}"
            )
        provenance = tool.get("provenance") or {}
        if provenance.get("license") not in allowed:
            raise FactoryError(f"tool {name} has unapproved license")
        _required_string(provenance, "source_id", where=f"{where}.provenance")
        _safe_relative(
            _required_string(provenance, "source_path", where=f"{where}.provenance")
        )
        if provenance.get("public_distribution_clearance_asserted") is True:
            raise FactoryError(f"tool {name} infers public distribution clearance")
        tools[name] = dict(tool)
    raw_codebook = contract.get("mw_codebook")
    if not isinstance(raw_codebook, list):
        raise FactoryError("mw_codebook must be a list")
    codebook: dict[str, dict[str, Any]] = {}
    class_ids: set[int] = set()
    for number, item in enumerate(raw_codebook):
        if not isinstance(item, dict):
            raise FactoryError(f"mw_codebook[{number}] must be an object")
        reason = _required_string(item, "reason_code", where=f"mw_codebook[{number}]")
        class_id = item.get("class_id")
        if not isinstance(class_id, int) or isinstance(class_id, bool):
            raise FactoryError("MW disposition class_id must be an integer")
        if reason in codebook or class_id in class_ids or class_id < 0:
            raise FactoryError("MW disposition reason_code and class_id must be unique")
        class_ids.add(class_id)
        codebook[reason] = dict(item)
    if len(codebook) < 2:
        raise FactoryError("MW disposition codebook must be a non-trivial closed set")

    worlds = contract.get("worlds") or []
    if not isinstance(worlds, list) or not worlds:
        raise FactoryError("worlds cannot be empty")
    world_ids: set[str] = set()
    for number, world in enumerate(worlds):
        where = f"worlds[{number}]"
        if not isinstance(world, dict):
            raise FactoryError(f"{where} must be an object")
        world_id = _required_string(world, "world_id", where=where)
        if world_id in world_ids:
            raise FactoryError(f"duplicate world_id: {world_id}")
        world_ids.add(world_id)
        for key in ("semantic_family", "schema_family", "realizer_id", "scene_id"):
            value = _required_string(world, key, where=where)
            if value not in registry[key]:
                raise FactoryError(f"{where}.{key} is not registered: {value}")
        world_provenance = world.get("provenance") or {}
        if world_provenance.get("license") not in allowed:
            raise FactoryError(f"{where} has unapproved provenance license")
        _required_string(world_provenance, "source_id", where=f"{where}.provenance")
        _safe_relative(
            _required_string(
                world_provenance, "source_path", where=f"{where}.provenance"
            )
        )
        if world_provenance.get("public_distribution_clearance_asserted") is True:
            raise FactoryError(f"{where} infers public distribution clearance")
        tool_names = world.get("tool_names")
        if not isinstance(tool_names, list) or not tool_names:
            raise FactoryError(f"{where}.tool_names must be a non-empty list")
        if len(tool_names) != len(set(tool_names)):
            raise FactoryError(f"{where}.tool_names contains duplicates")
        for name in tool_names:
            if name not in tools:
                raise FactoryError(f"{where} references unknown tool {name}")
            if tools[name]["schema_family"] != world["schema_family"]:
                raise FactoryError(f"{where} mixes schema families through tool {name}")
    index = ContractIndex(contract, tools, codebook, allowed)
    _cost_ledger(index)
    return index


def _split_for(index: ContractIndex, attrs: Mapping[str, Any], *, where: str) -> str:
    registry = index.contract["split_registry"]
    assignments: dict[str, str] = {}
    for axis in ISOLATION_AXES:
        value = attrs.get(axis)
        if value is None:
            continue
        mapping = registry[axis]
        if value not in mapping:
            raise FactoryError(f"{where}: {axis} value is not registered: {value}")
        assignments[axis] = mapping[value]
    if not assignments:
        raise FactoryError(f"{where}: no split-bearing isolation attributes")
    splits = set(assignments.values())
    if len(splits) != 1:
        raise FactoryError(f"{where}: isolation axes disagree on split: {assignments}")
    return next(iter(splits))


def _base_attrs(
    world: Mapping[str, Any], *, template_id: str, teacher_id: str | None = None
) -> dict[str, Any]:
    attrs = {
        "semantic_family": world["semantic_family"],
        "schema_family": world["schema_family"],
        "realizer_id": world["realizer_id"],
        "scene_id": world["scene_id"],
        "template_id": template_id,
    }
    if teacher_id is not None:
        attrs["teacher_id"] = teacher_id
    return attrs


def _provenance(
    world: Mapping[str, Any], *, generator_id: str, teacher_id: str | None = None
) -> dict[str, Any]:
    base = world.get("provenance") or {}
    return {
        "source_id": base.get("source_id"),
        "source_path": base.get("source_path"),
        "license": base.get("license"),
        "generator_id": generator_id,
        "teacher_id": teacher_id,
        "provider_called": False,
        "public_distribution_clearance_asserted": False,
    }


def _compact_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description") or "",
        "parameters": tool["parameters"],
    }


def _semantic_task(world: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "world_id": world["world_id"],
        "semantic_family": world["semantic_family"],
        "schema_family": world["schema_family"],
        "task_id": task.get("task_id"),
        "kind": task.get("kind"),
        "tool_name": task.get("tool_name"),
        "arguments": task.get("arguments") or {},
        "protected_values": task.get("protected_values") or [],
        "mw_case": task.get("mw_case") or {},
    }


def _teacher_envelope(
    *,
    teacher_id: str,
    base_query: str,
    query: str,
    semantic_payload: Mapping[str, Any],
    protected_values: Sequence[Any],
) -> dict[str, Any]:
    semantic_hash = sha256_json(semantic_payload)
    return {
        "contract": "visible-chinese-query-only-v1",
        "teacher_id": teacher_id,
        "provider_called": False,
        "input": {"query": base_query},
        "output": {"query": query},
        "mutable_fields": ["query"],
        "protected_values": list(protected_values),
        "semantic_payload_sha256_before": semantic_hash,
        "semantic_payload_sha256_after": semantic_hash,
    }


def _query_visible_values_ok(query: str, protected: Sequence[Any]) -> bool:
    return all(str(value) in query for value in protected if value is not None)


def _compile_target(tool_name: str | None, arguments: Mapping[str, Any]) -> str:
    if not tool_name:
        return "[]"
    return json.dumps(
        [{"name": tool_name, "arguments": dict(arguments)}],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _catalog(index: ContractIndex, world: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_compact_tool(index.tools[name]) for name in world.get("tool_names") or []]


def _validate_task(
    index: ContractIndex, world: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    task_id = _required_string(task, "task_id", where=f"world {world['world_id']} task")
    kind = task.get("kind")
    if kind not in {"execute", "refuse", "no_match"}:
        raise FactoryError(f"task {task_id}: unsupported kind {kind!r}")
    tool_name = task.get("tool_name")
    arguments = task.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise FactoryError(f"task {task_id}: arguments must be an object")
    protected_values = task.get("protected_values") or []
    if not isinstance(protected_values, list):
        raise FactoryError(f"task {task_id}: protected_values must be a list")
    if kind == "execute":
        if tool_name not in world.get("tool_names", []):
            raise FactoryError(f"task {task_id}: execute tool not in world catalog")
        errors = validate_instance(index.tools[tool_name]["parameters"], arguments)
        if errors:
            raise FactoryError(
                f"task {task_id}: invalid gold arguments: {'; '.join(errors)}"
            )
    elif tool_name is not None or arguments:
        raise FactoryError(
            f"task {task_id}: refusal/no-match must not carry executable gold"
        )
    if not isinstance(task.get("mw_case"), dict):
        raise FactoryError(f"task {task_id}: mw_case must be an object")
    realizations = task.get("realizations") or []
    if not isinstance(realizations, list) or not realizations:
        raise FactoryError(
            f"task {task_id}: at least one visible realization is required"
        )
    realization_ids: set[str] = set()
    for realization in realizations:
        if not isinstance(realization, dict):
            raise FactoryError(f"task {task_id}: realization must be an object")
        realization_id = _required_string(
            realization, "realization_id", where=f"task {task_id} realization"
        )
        if realization_id in realization_ids:
            raise FactoryError(
                f"task {task_id}: duplicate realization_id {realization_id}"
            )
        realization_ids.add(realization_id)
        teacher_id = _required_string(
            realization, "teacher_id", where=f"task {task_id} realization"
        )
        query = _required_string(
            realization, "query", where=f"task {task_id} realization"
        )
        template_id = _required_string(task, "template_id", where=f"task {task_id}")
        attrs = _base_attrs(world, template_id=template_id, teacher_id=teacher_id)
        _split_for(index, attrs, where=f"task {task_id} realization {teacher_id}")
        if not _query_visible_values_ok(query, protected_values):
            raise FactoryError(
                f"task {task_id}: teacher rewrite dropped a protected visible value"
            )
        if EVAL_MARKER_RE.search(query):
            raise FactoryError(
                f"task {task_id}: visible query contains a locked-eval marker"
            )


def _row_common(
    index: ContractIndex,
    world: Mapping[str, Any],
    *,
    sample_id: str,
    split: str,
    attrs: Mapping[str, Any],
    generator_id: str,
    teacher_id: str | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "product": PRODUCT,
        "world_id": world["world_id"],
        "split": split,
        "semantic_family": attrs.get("semantic_family"),
        "schema_family": attrs.get("schema_family"),
        "realizer_id": attrs.get("realizer_id"),
        "teacher_id": attrs.get("teacher_id"),
        "scene_id": attrs.get("scene_id"),
        "template_id": attrs.get("template_id"),
        "provenance": _provenance(
            world, generator_id=generator_id, teacher_id=teacher_id
        ),
    }


def _compile_task_rows(
    index: ContractIndex,
    world: Mapping[str, Any],
    task: Mapping[str, Any],
    rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    _validate_task(index, world, task)
    semantic = _semantic_task(world, task)
    semantic_hash = sha256_json(semantic)
    base_query = str((task.get("realizations") or [])[0]["query"])
    template_id = str(task["template_id"])
    catalog = _catalog(index, world)
    kind = str(task["kind"])
    tool_name = task.get("tool_name") if kind == "execute" else None
    arguments = dict(task.get("arguments") or {}) if kind == "execute" else {}
    target = _compile_target(tool_name, arguments)
    for realization in task["realizations"]:
        teacher_id = str(realization["teacher_id"])
        query = str(realization["query"])
        attrs = _base_attrs(world, template_id=template_id, teacher_id=teacher_id)
        split = _split_for(index, attrs, where=f"task {task['task_id']}")
        envelope = _teacher_envelope(
            teacher_id=teacher_id,
            base_query=base_query,
            query=query,
            semantic_payload=semantic,
            protected_values=task.get("protected_values") or [],
        )
        suffix = (task["task_id"], realization.get("realization_id"), split)

        retrieval_id = stable_id("RET", *suffix)
        retrieval = _row_common(
            index,
            world,
            sample_id=retrieval_id,
            split=split,
            attrs=attrs,
            generator_id="mei-51m-semantic-world-compiler-v1",
            teacher_id=teacher_id,
        )
        retrieval.update(
            {
                "lane": "retrieval",
                "query": query,
                "catalog_tools": catalog,
                "gold_tool": tool_name,
                "kind": kind,
                "semantic_payload_sha256": semantic_hash,
                "teacher_envelope": envelope,
            }
        )
        rows["retrieval"][split].append(retrieval)

        full_id = stable_id("CALL", *suffix)
        full = _row_common(
            index,
            world,
            sample_id=full_id,
            split=split,
            attrs=attrs,
            generator_id="mei-51m-semantic-world-compiler-v1",
            teacher_id=teacher_id,
        )
        full.update(
            {
                "lane": "full_call",
                "query": query,
                "catalog_tools": catalog,
                "kind": kind,
                "expected_call": (
                    {"name": tool_name, "arguments": arguments} if tool_name else None
                ),
                "target_text": target,
                "serializer": SERIALIZER,
                "protected_values": list(task.get("protected_values") or []),
                "semantic_payload_sha256": semantic_hash,
                "teacher_envelope": envelope,
            }
        )
        rows["full_call"][split].append(full)

        confidence_id = stable_id("CONF", full_id)
        confidence = _row_common(
            index,
            world,
            sample_id=confidence_id,
            split=split,
            attrs=attrs,
            generator_id="mei-51m-confidence-candidate-compiler-v1",
            teacher_id=teacher_id,
        )
        confidence.update(
            {
                "lane": "confidence",
                "query": query,
                "source_sample_id": full_id,
                "expected_kind": "call" if tool_name else "refuse",
                "expected_call": (
                    {"name": tool_name, "arguments": arguments} if tool_name else None
                ),
                "label": None,
                "label_state": "pending_actual_frozen_runtime_outcome",
                "label_contract": "exact_call_or_correct_refusal_after_final_runtime",
                "semantic_payload_sha256": semantic_hash,
                "teacher_envelope": envelope,
            }
        )
        rows["confidence"][split].append(confidence)

        mw_case = task.get("mw_case") or {}
        for field in ("context", "permissions", "state"):
            if not isinstance(mw_case.get(field, {}), dict):
                raise FactoryError(
                    f"task {task['task_id']}: MW {field} must be an object"
                )
        for field in ("evidence", "history", "tool_results"):
            if not isinstance(mw_case.get(field, []), list):
                raise FactoryError(f"task {task['task_id']}: MW {field} must be a list")
        reason = mw_case.get("reason_code")
        if reason not in index.codebook:
            raise FactoryError(
                f"task {task['task_id']}: MW reason is not in frozen codebook"
            )
        mw_id = stable_id("MW", *suffix)
        mw = _row_common(
            index,
            world,
            sample_id=mw_id,
            split=split,
            attrs=attrs,
            generator_id="mei-51m-mw-disposition-compiler-v1",
            teacher_id=teacher_id,
        )
        mw.update(
            {
                "lane": "mw_disposition",
                "query": query,
                "context": mw_case.get("context") or {},
                "evidence": mw_case.get("evidence") or [],
                "permissions": mw_case.get("permissions") or {},
                "state": mw_case.get("state") or {},
                "history": mw_case.get("history") or [],
                "tool_results": mw_case.get("tool_results") or [],
                "reason_code": reason,
                "reason_class_id": index.codebook[reason]["class_id"],
                "semantic_boundary": "independent_mw_disposition_not_mw_deviation",
                "semantic_payload_sha256": semantic_hash,
                "teacher_envelope": envelope,
            }
        )
        rows["mw_disposition"][split].append(mw)


def _tool_result(
    *, trajectory_id: str, call_id: str, tool_name: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    status = raw.get("status")
    if status not in {"ok", "error", "cancelled", "partial"}:
        raise FactoryError(
            f"trajectory {trajectory_id}: invalid ToolResult status {status!r}"
        )
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise FactoryError(
            f"trajectory {trajectory_id}: ToolResult payload must be an object"
        )
    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status,
        "payload": dict(payload),
        "provenance": {
            "source": "trusted-offline-fixture",
            "verified": raw.get("verified") is True,
        },
    }


def _compile_trajectory(
    index: ContractIndex,
    world: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    trajectory_id = _required_string(trajectory, "trajectory_id", where="trajectory")
    template_id = _required_string(
        trajectory, "template_id", where=f"trajectory {trajectory_id}"
    )
    teacher_id = _required_string(
        trajectory, "teacher_id", where=f"trajectory {trajectory_id}"
    )
    query = _required_string(trajectory, "query", where=f"trajectory {trajectory_id}")
    attrs = _base_attrs(world, template_id=template_id, teacher_id=teacher_id)
    split = _split_for(index, attrs, where=f"trajectory {trajectory_id}")
    protected_values = trajectory.get("protected_values") or []
    if not isinstance(protected_values, list):
        raise FactoryError(
            f"trajectory {trajectory_id}: protected_values must be a list"
        )
    if not _query_visible_values_ok(query, protected_values):
        raise FactoryError(
            f"trajectory {trajectory_id}: query dropped protected values"
        )
    if EVAL_MARKER_RE.search(query):
        raise FactoryError(f"trajectory {trajectory_id}: query contains eval marker")
    terminal = trajectory.get("terminal_status")
    if terminal not in TERMINAL_STATUSES:
        raise FactoryError(f"trajectory {trajectory_id}: invalid terminal status")
    steps = trajectory.get("steps") or []
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(step, dict) for step in steps)
        or steps[-1].get("kind") != "respond"
    ):
        raise FactoryError(f"trajectory {trajectory_id}: final step must be respond")

    semantic = {
        "world_id": world["world_id"],
        "trajectory_id": trajectory_id,
        "steps": steps,
        "terminal_status": terminal,
        "protected_values": list(protected_values),
    }
    semantic_hash = sha256_json(semantic)
    envelope = _teacher_envelope(
        teacher_id=teacher_id,
        base_query=str(trajectory.get("base_query") or query),
        query=query,
        semantic_payload=semantic,
        protected_values=protected_values,
    )
    catalog = _catalog(index, world)
    prior_calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    pending: tuple[str, str] | None = None
    decision_index = 0
    for step_index, step in enumerate(steps, 1):
        kind = step.get("kind")
        if kind == "call":
            if pending is not None:
                raise FactoryError(
                    f"trajectory {trajectory_id}: call before prior ToolResult"
                )
            tool_name = step.get("tool_name")
            if tool_name not in world.get("tool_names", []):
                raise FactoryError(
                    f"trajectory {trajectory_id}: unknown call tool {tool_name}"
                )
            arguments = step.get("arguments") or {}
            errors = validate_instance(index.tools[tool_name]["parameters"], arguments)
            if errors:
                raise FactoryError(
                    f"trajectory {trajectory_id}: invalid call arguments: {'; '.join(errors)}"
                )
            decision_index += 1
            call_id = stable_id("call", trajectory_id, decision_index)
            target_call = {"name": tool_name, "arguments": arguments}
            sample_id = stable_id("AGENT", trajectory_id, decision_index, "call")
            row = _row_common(
                index,
                world,
                sample_id=sample_id,
                split=split,
                attrs=attrs,
                generator_id="mei-51m-agent-trajectory-compiler-v1",
                teacher_id=teacher_id,
            )
            row.update(
                {
                    "lane": "agent",
                    "trajectory_id": trajectory_id,
                    "trajectory_step": decision_index,
                    "query": query,
                    "catalog_tools": catalog,
                    "prior_calls": list(prior_calls),
                    "tool_results": list(results),
                    "target_action": {
                        "kind": "call",
                        "call_id": call_id,
                        **target_call,
                    },
                    "target_text": _compile_target(tool_name, arguments),
                    "serializer": SERIALIZER,
                    "semantic_payload_sha256": semantic_hash,
                    "teacher_envelope": envelope,
                }
            )
            rows["agent"][split].append(row)
            confidence = dict(row)
            confidence.update(
                {
                    "sample_id": stable_id("CONFAG", sample_id),
                    "lane": "confidence",
                    "source_sample_id": sample_id,
                    "expected_kind": "call",
                    "expected_call": target_call,
                    "label": None,
                    "label_state": "pending_actual_frozen_runtime_outcome",
                    "label_contract": "exact_call_after_final_runtime",
                }
            )
            confidence.pop("target_action", None)
            rows["confidence"][split].append(confidence)
            prior_calls.append({"call_id": call_id, **target_call})
            pending = (call_id, str(tool_name))
        elif kind == "tool_result":
            if pending is None:
                raise FactoryError(f"trajectory {trajectory_id}: orphan ToolResult")
            result = _tool_result(
                trajectory_id=trajectory_id,
                call_id=pending[0],
                tool_name=pending[1],
                raw=step,
            )
            if result["provenance"]["verified"] is not True:
                raise FactoryError(f"trajectory {trajectory_id}: unverified ToolResult")
            results.append(result)
            pending = None
        elif kind == "respond":
            if pending is not None:
                raise FactoryError(
                    f"trajectory {trajectory_id}: terminal response before ToolResult"
                )
            if step_index != len(steps):
                raise FactoryError(
                    f"trajectory {trajectory_id}: respond must be terminal"
                )
            response = _required_string(
                step, "response", where=f"trajectory {trajectory_id} respond"
            )
            decision_index += 1
            sample_id = stable_id("AGENT", trajectory_id, decision_index, "respond")
            row = _row_common(
                index,
                world,
                sample_id=sample_id,
                split=split,
                attrs=attrs,
                generator_id="mei-51m-agent-trajectory-compiler-v1",
                teacher_id=teacher_id,
            )
            row.update(
                {
                    "lane": "agent",
                    "trajectory_id": trajectory_id,
                    "trajectory_step": decision_index,
                    "query": query,
                    "catalog_tools": catalog,
                    "prior_calls": list(prior_calls),
                    "tool_results": list(results),
                    "target_action": {"kind": "respond", "response": response},
                    "target_text": response,
                    "semantic_payload_sha256": semantic_hash,
                    "teacher_envelope": envelope,
                }
            )
            rows["agent"][split].append(row)
        else:
            raise FactoryError(
                f"trajectory {trajectory_id}: invalid step kind {kind!r}"
            )
    if pending is not None:
        raise FactoryError(f"trajectory {trajectory_id}: missing terminal ToolResult")
    if not results:
        raise FactoryError(
            f"trajectory {trajectory_id}: narration requires verified results"
        )
    expected_terminal = {
        "ok": "success",
        "error": "error",
        "cancelled": "cancelled",
        "partial": "partial",
    }[str(results[-1]["status"])]
    if terminal != expected_terminal:
        raise FactoryError(
            f"trajectory {trajectory_id}: terminal status {terminal} does not match final ToolResult {results[-1]['status']}"
        )
    terminal_narration = _required_string(
        trajectory, "terminal_narration", where=f"trajectory {trajectory_id}"
    )
    raw_required_facts = trajectory.get("required_facts") or []
    if not isinstance(raw_required_facts, list):
        raise FactoryError(f"trajectory {trajectory_id}: required_facts must be a list")
    required_facts = list(raw_required_facts)
    missing_facts = [
        fact for fact in required_facts if str(fact) not in terminal_narration
    ]
    if missing_facts:
        raise FactoryError(
            f"trajectory {trajectory_id}: terminal narration omits required facts {missing_facts}"
        )

    narration = _row_common(
        index,
        world,
        sample_id=stable_id("NARR", trajectory_id),
        split=split,
        attrs=attrs,
        generator_id="mei-51m-terminal-narration-compiler-v1",
        teacher_id=teacher_id,
    )
    narration.update(
        {
            "lane": "narration",
            "trajectory_id": trajectory_id,
            "query": query,
            "terminal_status": terminal,
            "terminal_only": True,
            "can_execute_tools": False,
            "verified_results": results,
            "required_facts": required_facts,
            "target": terminal_narration,
            "semantic_payload_sha256": semantic_hash,
            "teacher_envelope": envelope,
        }
    )
    rows["narration"][split].append(narration)


def compile_contract(
    index: ContractIndex,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = {lane: {split: [] for split in SPLITS} for lane in LANES}
    document_ids: set[str] = set()
    task_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    for world in index.contract["worlds"]:
        for document in world.get("colloquial_documents") or []:
            document_id = _required_string(
                document, "document_id", where="colloquial document"
            )
            if document_id in document_ids:
                raise FactoryError(
                    f"duplicate document_id across worlds: {document_id}"
                )
            document_ids.add(document_id)
            template_id = _required_string(document, "template_id", where=document_id)
            text = _required_string(document, "text", where=document_id)
            attrs = _base_attrs(world, template_id=template_id)
            split = _split_for(index, attrs, where=f"colloquial document {document_id}")
            if IDENTIFIER_PADDING_RE.search(text) or EVAL_MARKER_RE.search(text):
                raise FactoryError(
                    f"colloquial document {document_id}: padding/eval marker rejected"
                )
            row = _row_common(
                index,
                world,
                sample_id=stable_id("CPTZH", document_id, split),
                split=split,
                attrs=attrs,
                generator_id="mei-51m-colloquial-document-compiler-v1",
                teacher_id=None,
            )
            row.update(
                {
                    "lane": "cpt_colloquial",
                    "document_id": document_id,
                    "text": text,
                    "semantic_payload_sha256": sha256_json(
                        {
                            "world_id": world["world_id"],
                            "document_id": document_id,
                            "text": text,
                        }
                    ),
                }
            )
            rows["cpt_colloquial"][split].append(row)
        for task in world.get("tasks") or []:
            task_id = _required_string(task, "task_id", where="task")
            if task_id in task_ids:
                raise FactoryError(f"duplicate task_id across worlds: {task_id}")
            task_ids.add(task_id)
            _compile_task_rows(index, world, task, rows)
        for trajectory in world.get("trajectories") or []:
            trajectory_id = _required_string(
                trajectory, "trajectory_id", where="trajectory"
            )
            if trajectory_id in trajectory_ids:
                raise FactoryError(
                    f"duplicate trajectory_id across worlds: {trajectory_id}"
                )
            trajectory_ids.add(trajectory_id)
            _compile_trajectory(index, world, trajectory, rows)

    for lane in LANES:
        for split in SPLITS:
            rows[lane][split].sort(
                key=lambda item: str(item["sample_id"]).encode("utf-8")
            )
            if not rows[lane][split]:
                raise FactoryError(
                    f"fixture must exercise every lane/split: {lane}.{split}"
                )
    return rows


def _cost_ledger(index: ContractIndex) -> dict[str, Any]:
    events = []
    total = 0.0
    for number, raw in enumerate(index.contract.get("cost_events") or []):
        if not isinstance(raw, dict):
            raise FactoryError(f"cost event {number} must be an object")
        event = dict(raw)
        raw_amount = event.get("amount", 0.0)
        if not isinstance(raw_amount, (int, float)) or isinstance(raw_amount, bool):
            raise FactoryError("cost amount must be numeric")
        amount = float(raw_amount)
        if amount < 0:
            raise FactoryError("cost amount cannot be negative")
        if event.get("provider_called") is not False:
            raise FactoryError("phase-1 cost ledger cannot record a provider call")
        if event.get("paid") is not False or amount != 0:
            raise FactoryError("phase-1 fixture must have zero paid cost")
        event["event_index"] = number
        events.append(event)
        total += amount
    return {
        "schema": "mei-51m-corpus-cost-ledger-v1",
        "currency": "CNY",
        "events": events,
        "total": round(total, 6),
        "provider_calls": 0,
        "paid_external_teacher_used": False,
        "budget_ceiling": float(
            (index.contract.get("budget") or {}).get("max_cost_cny", 0)
        ),
        "status": "passed" if total == 0 else "blocked",
    }


def _build_payloads(
    index: ContractIndex, descriptor: dict[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if descriptor.get("schema") != "mei-51m-independent-eval-descriptor-v1":
        raise FactoryError("build accepts only the public independent eval descriptor")
    if descriptor.get("payload_paths_exposed") is not False:
        raise FactoryError("eval descriptor must not expose holdout payload paths")
    compiled = compile_contract(index)
    total_rows = sum(len(compiled[lane][split]) for lane in LANES for split in SPLITS)
    maximum_rows = int(index.contract["quality_gates"]["max_fixture_rows"])
    if total_rows > maximum_rows:
        raise FactoryError(
            f"fixture row count {total_rows} exceeds max_fixture_rows={maximum_rows}; use plan-scale"
        )
    payloads: dict[str, bytes] = {
        "contract.snapshot.json": pretty_bytes(index.contract),
        "eval-descriptor.snapshot.json": pretty_bytes(descriptor),
        "cost-ledger.json": pretty_bytes(_cost_ledger(index)),
    }
    row_counts: dict[str, int] = {}
    for lane in LANES:
        for split in SPLITS:
            name = f"data/{lane}.{split}.jsonl"
            payloads[name] = jsonl_bytes(compiled[lane][split])
            row_counts[name] = len(compiled[lane][split])
    artifacts = {
        name: artifact_spec_bytes(payload, rows=row_counts.get(name))
        for name, payload in sorted(payloads.items())
    }
    manifest = {
        "schema": BUILD_SCHEMA,
        "factory_id": index.contract["factory_id"],
        "product": PRODUCT,
        "purpose": index.contract["purpose"],
        "contract_sha256": sha256_json(index.contract),
        "eval_descriptor_sha256": sha256_json(descriptor),
        "evaluation_fingerprint": descriptor["evaluation_fingerprint"],
        "seed": int(index.contract.get("seed") or 0),
        "generator": {
            "id": "mei-51m-semantic-world-compiler-v1",
            "implementation_sha256": implementation_sha256(),
        },
        "lanes": list(LANES),
        "splits": list(SPLITS),
        "artifacts": artifacts,
        "artifact_merkle_root": merkle_root(artifacts),
        "generator_access": {
            "eval_descriptor_read": True,
            "eval_payload_read": False,
            "provider_calls": False,
        },
    }
    return payloads, manifest


def verify_build(build_dir: Path) -> dict[str, Any]:
    manifest = load_json(build_dir / "build-manifest.json")
    if manifest.get("schema") != BUILD_SCHEMA:
        raise FactoryError("invalid build manifest schema")
    errors = verify_artifacts(build_dir, manifest.get("artifacts") or {})
    expected = merkle_root(manifest.get("artifacts") or {})
    if expected != manifest.get("artifact_merkle_root"):
        errors.append("build artifact Merkle root mismatch")
    generator = manifest.get("generator") or {}
    if not _valid_sha256(generator.get("implementation_sha256")):
        errors.append("build manifest lacks a valid generator implementation hash")
    if manifest.get("product") != PRODUCT:
        errors.append("build manifest product mismatch")
    if manifest.get("lanes") != list(LANES) or manifest.get("splits") != list(SPLITS):
        errors.append("build manifest lane/split contract mismatch")
    if manifest.get("generator_access") != {
        "eval_descriptor_read": True,
        "eval_payload_read": False,
        "provider_calls": False,
    }:
        errors.append("build manifest generator-access boundary mismatch")

    contract_path = build_dir / "contract.snapshot.json"
    if contract_path.is_file():
        contract = load_json(contract_path)
        if sha256_json(contract) != manifest.get("contract_sha256"):
            errors.append("build manifest does not bind contract snapshot")
        if contract.get("factory_id") != manifest.get("factory_id"):
            errors.append("build manifest factory_id mismatch")
    descriptor_path = build_dir / "eval-descriptor.snapshot.json"
    if descriptor_path.is_file():
        descriptor = load_json(descriptor_path)
        if sha256_json(descriptor) != manifest.get("eval_descriptor_sha256"):
            errors.append("build manifest does not bind eval descriptor snapshot")
        if descriptor.get("evaluation_fingerprint") != manifest.get(
            "evaluation_fingerprint"
        ):
            errors.append("build/eval fingerprint mismatch")
        if descriptor.get("payload_paths_exposed") is not False:
            errors.append("build snapshot exposes eval payload paths")
    receipt_path = build_dir / "generation-receipt.json"
    if not receipt_path.is_file():
        errors.append("missing generation receipt")
    else:
        receipt = load_json(receipt_path)
        if receipt.get("schema") != "mei-51m-corpus-generation-receipt-v1":
            errors.append("invalid generation receipt schema")
        if (
            receipt.get("status") != "passed"
            or receipt.get("process_complete") is not True
        ):
            errors.append("generation receipt is not terminal passed")
        if receipt.get("current_mutated") is not False:
            errors.append("generation receipt reports CURRENT mutation")
        if receipt.get("provider_calls") != 0:
            errors.append("generation receipt reports provider calls")
        if receipt.get("build_manifest_sha256") != sha256_file(
            build_dir / "build-manifest.json"
        ):
            errors.append("generation receipt does not bind build manifest")
        if receipt.get("artifact_merkle_root") != expected:
            errors.append("generation receipt artifact Merkle root mismatch")
        if receipt.get("generator_implementation_sha256") != generator.get(
            "implementation_sha256"
        ):
            errors.append("generation receipt generator hash mismatch")
    return {
        "schema": "mei-51m-corpus-build-verification-v1",
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "artifact_merkle_root": expected,
        "manifest_sha256": sha256_file(build_dir / "build-manifest.json"),
    }


def build_contract(
    contract_path: Path, descriptor_path: Path, out_dir: Path
) -> dict[str, Any]:
    index = validate_contract(load_json(contract_path))
    descriptor = load_json(descriptor_path)
    payloads, manifest = _build_payloads(index, descriptor)
    parent = out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.build.", dir=parent))
    try:
        for name, payload in payloads.items():
            _atomic_write(temporary / _safe_relative(name), payload, write_once=True)
        _atomic_write(
            temporary / "build-manifest.json", pretty_bytes(manifest), write_once=True
        )
        receipt = {
            "schema": "mei-51m-corpus-generation-receipt-v1",
            "status": "passed",
            "process_complete": True,
            "release_eligible": False,
            "release_ineligible_reason": "unfrozen_control_plane_fixture",
            "build_manifest_sha256": sha256_file(temporary / "build-manifest.json"),
            "artifact_merkle_root": manifest["artifact_merkle_root"],
            "generator_implementation_sha256": implementation_sha256(),
            "current_mutated": False,
            "provider_calls": 0,
        }
        _atomic_write(
            temporary / "generation-receipt.json",
            pretty_bytes(receipt),
            write_once=True,
        )
        if out_dir.exists():
            existing = load_json(out_dir / "build-manifest.json")
            if existing != manifest:
                raise FactoryError(f"refusing to overwrite different build: {out_dir}")
            shutil.rmtree(temporary)
            return {**verify_build(out_dir), "reused": True, "path": str(out_dir)}
        temporary.replace(out_dir)
        return {**verify_build(out_dir), "reused": False, "path": str(out_dir)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def normalize_visible(text: str, protected: Sequence[Any] = ()) -> str:
    value = text.casefold()
    for item in sorted(
        (str(item) for item in protected if item is not None), key=len, reverse=True
    ):
        value = value.replace(item.casefold(), "<slot>")
    value = UUID_RE.sub("<uuid>", value)
    value = LONG_HEX_RE.sub("<hex>", value)
    value = NUMBER_RE.sub("<n>", value)
    value = re.sub(r"[\s，。！？；：、,.!?;:'\"“”‘’（）()]+", "", value)
    return value


def char_trigrams(text: str) -> set[str]:
    value = normalize_visible(text)
    if len(value) < 3:
        return {value} if value else set()
    return {value[index : index + 3] for index in range(len(value) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _model_visible_parts(row: Mapping[str, Any]) -> list[str]:
    """Return every model-visible text surface without audit-only metadata."""

    parts: list[str] = []
    for field in ("text", "query", "target", "target_text"):
        value = row.get(field)
        if isinstance(value, str) and value:
            parts.append(value)

    lane = row.get("lane")
    structured_fields = {
        "retrieval": ("catalog_tools", "gold_tool"),
        "full_call": ("catalog_tools", "expected_call"),
        "agent": ("catalog_tools", "prior_calls", "tool_results"),
        "mw_disposition": (
            "context",
            "evidence",
            "permissions",
            "state",
            "history",
            "tool_results",
            "reason_code",
        ),
        "confidence": ("expected_call",),
        "narration": ("verified_results",),
    }.get(str(lane), ())
    for field in structured_fields:
        value = row.get(field)
        if value in (None, [], {}):
            continue
        if isinstance(value, str):
            if value not in parts:
                parts.append(value)
            continue
        parts.append(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return parts


def _visible_text(row: Mapping[str, Any]) -> str:
    return "\n<mei-visible-boundary>\n".join(_model_visible_parts(row))


def _load_build_rows(build_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        lane: {
            split: load_jsonl(build_dir / f"data/{lane}.{split}.jsonl")
            for split in SPLITS
        }
        for lane in LANES
    }


def _audit_split_isolation(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[str, Any], list[str]]:
    observed: dict[str, dict[str, set[str]]] = {
        axis: defaultdict(set) for axis in ISOLATION_AXES
    }
    row_split_mismatches: list[str] = []
    sample_locations: dict[str, list[str]] = defaultdict(list)
    for lane in LANES:
        for split in SPLITS:
            for row in rows[lane][split]:
                if row.get("split") != split:
                    row_split_mismatches.append(
                        f"{row.get('sample_id')}:{row.get('split')}!=container:{split}"
                    )
                sample_locations[str(row.get("sample_id"))].append(f"{lane}.{split}")
                for axis in ISOLATION_AXES:
                    value = row.get(axis)
                    if value is not None:
                        observed[axis][str(value)].add(split)
    leaks = {
        axis: {
            value: sorted(splits) for value, splits in values.items() if len(splits) > 1
        }
        for axis, values in observed.items()
    }
    errors = [
        f"split isolation leak on {axis}: {sorted(values)[:8]}"
        for axis, values in leaks.items()
        if values
    ]
    if row_split_mismatches:
        errors.append(f"row/container split mismatch: {row_split_mismatches[:8]}")
    duplicate_sample_ids = {
        sample_id: locations
        for sample_id, locations in sample_locations.items()
        if sample_id and sample_id != "None" and len(locations) > 1
    }
    if duplicate_sample_ids:
        errors.append(
            f"duplicate sample IDs across lane/split locations: {list(duplicate_sample_ids)[:8]}"
        )
    report = {
        "status": "passed" if not errors else "blocked",
        "row_split_mismatches": row_split_mismatches[:20],
        "duplicate_sample_ids": dict(list(duplicate_sample_ids.items())[:20]),
        "axes": {
            axis: {
                "values": len(values),
                "leaks": leaks[axis],
            }
            for axis, values in observed.items()
        },
    }
    return report, errors


def _audit_duplicates(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    near_threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    report: dict[str, Any] = {}
    for lane in LANES:
        exact: dict[str, set[str]] = defaultdict(set)
        normalized: dict[str, set[str]] = defaultdict(set)
        texts: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for split in SPLITS:
            for row in rows[lane][split]:
                text = _visible_text(row)
                if not text:
                    continue
                protected = row.get("protected_values") or []
                exact[sha256_bytes(text.encode("utf-8"))].add(split)
                normalized[normalize_visible(text, protected)].add(split)
                texts[split].append((str(row.get("sample_id")), text))
        exact_cross = sum(1 for splits in exact.values() if len(splits) > 1)
        template_cross = sum(1 for splits in normalized.values() if len(splits) > 1)
        near_hits: list[dict[str, Any]] = []
        trigram_cache = {
            sample_id: char_trigrams(text)
            for split in SPLITS
            for sample_id, text in texts[split]
        }
        for left_index, left_split in enumerate(SPLITS):
            for right_split in SPLITS[left_index + 1 :]:
                for left_id, _ in texts[left_split]:
                    for right_id, _ in texts[right_split]:
                        score = jaccard(trigram_cache[left_id], trigram_cache[right_id])
                        if score >= near_threshold:
                            near_hits.append(
                                {
                                    "left": left_id,
                                    "left_split": left_split,
                                    "right": right_id,
                                    "right_split": right_split,
                                    "score": round(score, 6),
                                }
                            )
        if exact_cross:
            errors.append(f"{lane}: {exact_cross} byte-exact visible rows cross splits")
        if template_cross:
            errors.append(f"{lane}: {template_cross} normalized templates cross splits")
        if near_hits:
            errors.append(
                f"{lane}: {len(near_hits)} near-duplicate visible rows cross splits"
            )
        report[lane] = {
            "byte_exact_cross_split": exact_cross,
            "normalized_template_cross_split": template_cross,
            "near_duplicate_cross_split": len(near_hits),
            "near_duplicate_examples": near_hits[:12],
            "visible_rows": sum(len(values) for values in texts.values()),
        }
    return report, errors


def _tool_map_from_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(tool["name"]): dict(tool) for tool in contract.get("tools") or []}


def _audit_semantics(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    tools = _tool_map_from_contract(contract)
    codebook = {
        str(item["reason_code"]): int(item["class_id"])
        for item in contract["mw_codebook"]
    }
    counts = Counter()
    for lane in LANES:
        for split in SPLITS:
            for row in rows[lane][split]:
                sample_id = str(row.get("sample_id"))
                counts[lane] += 1
                provenance = row.get("provenance") or {}
                if not provenance.get("source_id") or not provenance.get("source_path"):
                    errors.append(f"{sample_id}: incomplete provenance source identity")
                else:
                    try:
                        _safe_relative(str(provenance["source_path"]))
                    except FactoryError:
                        errors.append(f"{sample_id}: unsafe provenance source path")
                if not provenance.get("generator_id"):
                    errors.append(f"{sample_id}: missing provenance generator identity")
                if provenance.get("teacher_id") != row.get("teacher_id"):
                    errors.append(f"{sample_id}: provenance teacher identity mismatch")
                if provenance.get("license") not in (
                    contract.get("license_policy") or {}
                ).get("allowed", []):
                    errors.append(
                        f"{sample_id}: unapproved or missing provenance license"
                    )
                if provenance.get("provider_called") is not False:
                    errors.append(
                        f"{sample_id}: provider call recorded in offline fixture"
                    )
                if (
                    provenance.get("public_distribution_clearance_asserted")
                    is not False
                ):
                    errors.append(
                        f"{sample_id}: unexpected distribution clearance assertion"
                    )
                envelope = row.get("teacher_envelope")
                if envelope:
                    if envelope.get("contract") != "visible-chinese-query-only-v1":
                        errors.append(f"{sample_id}: unknown teacher envelope contract")
                    if envelope.get("mutable_fields") != ["query"]:
                        errors.append(f"{sample_id}: teacher mutable field escape")
                    if envelope.get("provider_called") is not False:
                        errors.append(
                            f"{sample_id}: teacher provider call escaped offline policy"
                        )
                    if envelope.get("teacher_id") != row.get("teacher_id"):
                        errors.append(f"{sample_id}: teacher identity mismatch")
                    if set(envelope.get("input") or {}) != {"query"} or set(
                        envelope.get("output") or {}
                    ) != {"query"}:
                        errors.append(
                            f"{sample_id}: teacher envelope exposes non-query fields"
                        )
                    if (envelope.get("output") or {}).get("query") != row.get("query"):
                        errors.append(f"{sample_id}: teacher output/query mismatch")
                    if envelope.get("semantic_payload_sha256_before") != envelope.get(
                        "semantic_payload_sha256_after"
                    ):
                        errors.append(
                            f"{sample_id}: teacher changed executable semantics"
                        )
                    if envelope.get("semantic_payload_sha256_after") != row.get(
                        "semantic_payload_sha256"
                    ):
                        errors.append(f"{sample_id}: semantic hash does not round-trip")
                    if not _query_visible_values_ok(
                        str(row.get("query") or ""),
                        envelope.get("protected_values") or [],
                    ):
                        errors.append(
                            f"{sample_id}: teacher output dropped a protected visible value"
                        )

                if lane == "retrieval":
                    catalog = {tool["name"] for tool in row.get("catalog_tools") or []}
                    if (
                        row.get("gold_tool") is not None
                        and row.get("gold_tool") not in catalog
                    ):
                        errors.append(
                            f"{sample_id}: retrieval gold missing from catalog"
                        )
                elif lane == "full_call":
                    call = row.get("expected_call")
                    if call:
                        name = call.get("name")
                        if name not in tools:
                            errors.append(f"{sample_id}: unknown full-call tool")
                        else:
                            errors.extend(
                                f"{sample_id}: {message}"
                                for message in validate_instance(
                                    tools[name]["parameters"],
                                    call.get("arguments") or {},
                                )
                            )
                        if row.get("target_text") != _compile_target(
                            name, call.get("arguments") or {}
                        ):
                            errors.append(f"{sample_id}: target serializer drift")
                    elif row.get("target_text") != "[]":
                        errors.append(f"{sample_id}: refusal target must be []")
                elif lane == "agent":
                    action = row.get("target_action") or {}
                    if action.get("kind") == "call":
                        name = action.get("name")
                        if name not in tools:
                            errors.append(f"{sample_id}: unknown agent call tool")
                        else:
                            errors.extend(
                                f"{sample_id}: {message}"
                                for message in validate_instance(
                                    tools[name]["parameters"],
                                    action.get("arguments") or {},
                                )
                            )
                        if row.get("target_text") != _compile_target(
                            name, action.get("arguments") or {}
                        ):
                            errors.append(f"{sample_id}: agent call serializer drift")
                        if not action.get("call_id"):
                            errors.append(f"{sample_id}: agent call lacks call_id")
                    elif action.get("kind") == "respond":
                        if row.get("target_text") != action.get("response"):
                            errors.append(f"{sample_id}: agent response target drift")
                    else:
                        errors.append(
                            f"{sample_id}: agent target must be call or respond"
                        )
                    prior_calls = row.get("prior_calls") or []
                    call_ids = [str(call.get("call_id")) for call in prior_calls]
                    if len(call_ids) != len(set(call_ids)) or any(
                        call_id in {"", "None"} for call_id in call_ids
                    ):
                        errors.append(f"{sample_id}: invalid/duplicate prior call IDs")
                    call_tools = {
                        str(call.get("call_id")): str(call.get("name"))
                        for call in prior_calls
                    }
                    for result in row.get("tool_results") or []:
                        result_call_id = str(result.get("call_id"))
                        if result_call_id not in call_tools:
                            errors.append(
                                f"{sample_id}: ToolResult does not match a prior call"
                            )
                        elif str(result.get("tool_name")) != call_tools[result_call_id]:
                            errors.append(
                                f"{sample_id}: ToolResult tool does not match prior call"
                            )
                        if (result.get("provenance") or {}).get("verified") is not True:
                            errors.append(
                                f"{sample_id}: unverified ToolResult is model-visible"
                            )
                elif lane == "mw_disposition":
                    for field in ("context", "permissions", "state"):
                        if not isinstance(row.get(field), dict):
                            errors.append(f"{sample_id}: MW {field} must be an object")
                    for field in ("evidence", "history", "tool_results"):
                        if not isinstance(row.get(field), list):
                            errors.append(f"{sample_id}: MW {field} must be a list")
                    reason = row.get("reason_code")
                    if reason not in codebook or row.get(
                        "reason_class_id"
                    ) != codebook.get(reason):
                        errors.append(f"{sample_id}: MW closed-set target drift")
                    if (
                        row.get("semantic_boundary")
                        != "independent_mw_disposition_not_mw_deviation"
                    ):
                        errors.append(
                            f"{sample_id}: MW disposition/deviation boundary lost"
                        )
                elif lane == "confidence":
                    if row.get("label") is not None:
                        errors.append(
                            f"{sample_id}: synthetic confidence label is forbidden"
                        )
                    if (
                        row.get("label_state")
                        != "pending_actual_frozen_runtime_outcome"
                    ):
                        errors.append(
                            f"{sample_id}: confidence label source is not runtime outcome"
                        )
                elif lane == "narration":
                    if (
                        row.get("terminal_only") is not True
                        or row.get("can_execute_tools") is not False
                    ):
                        errors.append(
                            f"{sample_id}: narration is not terminal-only/non-executing"
                        )
                    if row.get("terminal_status") not in TERMINAL_STATUSES:
                        errors.append(f"{sample_id}: narration terminal status invalid")
                    narration_results = row.get("verified_results") or []
                    if not narration_results:
                        errors.append(
                            f"{sample_id}: narration lacks a verified ToolResult"
                        )
                    for result in narration_results:
                        if (result.get("provenance") or {}).get("verified") is not True:
                            errors.append(
                                f"{sample_id}: narration consumed unverified result"
                            )
                    if narration_results:
                        expected_terminal = {
                            "ok": "success",
                            "error": "error",
                            "cancelled": "cancelled",
                            "partial": "partial",
                        }.get(narration_results[-1].get("status"))
                        if row.get("terminal_status") != expected_terminal:
                            errors.append(
                                f"{sample_id}: narration terminal status/result mismatch"
                            )
                    target = str(row.get("target") or "")
                    if any(
                        str(fact) not in target
                        for fact in row.get("required_facts") or []
                    ):
                        errors.append(f"{sample_id}: narration omits required facts")
    return {
        "status": "passed" if not errors else "blocked",
        "rows": dict(sorted(counts.items())),
        "errors": errors[:80],
    }, errors


def verify_eval_lock(eval_dir: Path) -> dict[str, Any]:
    lock = load_json(eval_dir / "lock.json")
    descriptor = load_json(eval_dir / "descriptor.json")
    if lock.get("schema") != EVAL_LOCK_SCHEMA or lock.get("status") != "frozen":
        raise FactoryError("evaluation lock must be independently frozen")
    errors = verify_artifacts(eval_dir, lock.get("artifacts") or {})
    expected_merkle = merkle_root(lock.get("artifacts") or {})
    if lock.get("artifact_merkle_root") != expected_merkle:
        errors.append("evaluation artifact Merkle root mismatch")
    if set(lock.get("artifacts") or {}) != {"holdout.jsonl", "provenance.json"}:
        errors.append("evaluation lock artifact set mismatch")
    provenance_spec = (lock.get("artifacts") or {}).get("provenance.json") or {}
    if lock.get("provenance_sha256") != provenance_spec.get("sha256"):
        errors.append("evaluation provenance hash mismatch")
    fingerprint = sha256_json(
        {
            "lock_id": lock.get("lock_id"),
            "artifacts": lock.get("artifacts"),
            "provenance_sha256": lock.get("provenance_sha256"),
        }
    )
    if fingerprint != lock.get("evaluation_fingerprint"):
        errors.append("evaluation fingerprint mismatch")
    if descriptor.get("evaluation_fingerprint") != fingerprint:
        errors.append("public eval descriptor mismatch")
    if descriptor.get("schema") != "mei-51m-independent-eval-descriptor-v1":
        errors.append("invalid public eval descriptor schema")
    if descriptor.get("lock_sha256") != sha256_file(eval_dir / "lock.json"):
        errors.append("public eval descriptor does not bind lock.json")
    if descriptor.get("lock_id") != lock.get("lock_id"):
        errors.append("public eval descriptor lock_id mismatch")
    if descriptor.get("payload_paths_exposed") is not False:
        errors.append("public eval descriptor exposes payload paths")
    if not _valid_sha256(lock.get("freezer_implementation_sha256")):
        errors.append("evaluation lock lacks a valid freezer implementation hash")
    if lock.get("training_generator_access") != "forbidden":
        errors.append("evaluation lock does not forbid training-generator access")
    return {
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "lock": lock,
        "descriptor": descriptor,
    }


def _eval_rows(eval_dir: Path, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted((lock.get("artifacts") or {})):
        if name.endswith(".jsonl"):
            rows.extend(load_jsonl(eval_dir / _safe_relative(name)))
    return rows


def _audit_contamination(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], eval_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    verification = verify_eval_lock(eval_dir)
    errors = list(verification["errors"])
    eval_rows = _eval_rows(eval_dir, verification["lock"])
    training_axes = {
        axis: {
            str(row[axis])
            for lane in LANES
            for split in SPLITS
            for row in rows[lane][split]
            if row.get(axis) is not None
        }
        for axis in ISOLATION_AXES
    }
    eval_axes = {
        axis: {str(row[axis]) for row in eval_rows if row.get(axis) is not None}
        for axis in ISOLATION_AXES
    }
    isolation_hits = {
        axis: sorted(training_axes[axis] & eval_axes[axis])
        for axis in ISOLATION_AXES
        if training_axes[axis] & eval_axes[axis]
    }
    if isolation_hits:
        errors.append(
            "locked-eval isolation-axis contamination: "
            + "; ".join(
                f"{axis}={values[:8]}" for axis, values in isolation_hits.items()
            )
        )
    eval_texts = [
        (f"{row.get('sample_id')}#{part_index}", text)
        for row in eval_rows
        for part_index, text in enumerate(_model_visible_parts(row))
    ]
    eval_exact = {sha256_bytes(text.encode("utf-8")) for _, text in eval_texts if text}
    eval_norm = {normalize_visible(text) for _, text in eval_texts if text}
    eval_grams = [
        (sample_id, char_trigrams(text)) for sample_id, text in eval_texts if text
    ]
    exact_hits: list[str] = []
    normalized_hits: list[str] = []
    near_hits: list[dict[str, Any]] = []
    marker_hits: list[str] = []
    for lane in LANES:
        for split in SPLITS:
            for row in rows[lane][split]:
                sample_id = str(row.get("sample_id"))
                for part_index, text in enumerate(_model_visible_parts(row)):
                    location = f"{sample_id}#{part_index}"
                    if EVAL_MARKER_RE.search(text):
                        marker_hits.append(location)
                    if sha256_bytes(text.encode("utf-8")) in eval_exact:
                        exact_hits.append(location)
                    if (
                        normalize_visible(text, row.get("protected_values") or [])
                        in eval_norm
                    ):
                        normalized_hits.append(location)
                    grams = char_trigrams(text)
                    for eval_id, eval_value in eval_grams:
                        score = jaccard(grams, eval_value)
                        if score >= 0.94:
                            near_hits.append(
                                {
                                    "sample_id": location,
                                    "eval_id": eval_id,
                                    "score": round(score, 6),
                                }
                            )
    if exact_hits:
        errors.append(f"eval byte-exact contamination: {len(exact_hits)}")
    if normalized_hits:
        errors.append(f"eval normalized-template contamination: {len(normalized_hits)}")
    if near_hits:
        errors.append(f"eval near-duplicate contamination: {len(near_hits)}")
    if marker_hits:
        errors.append(f"eval marker contamination: {len(marker_hits)}")
    return {
        "status": "passed" if not errors else "blocked",
        "eval_lock_id": verification["lock"].get("lock_id"),
        "evaluation_fingerprint": verification["lock"].get("evaluation_fingerprint"),
        "eval_rows": len(eval_rows),
        "byte_exact_hits": exact_hits[:20],
        "normalized_template_hits": normalized_hits[:20],
        "near_duplicate_hits": near_hits[:20],
        "marker_hits": marker_hits[:20],
        "isolation_axis_hits": isolation_hits,
        "generator_eval_payload_read": False,
        "auditor_eval_payload_read": True,
    }, errors


def audit_source_registry(path: Path) -> dict[str, Any]:
    registry = load_json(path)
    if registry.get("schema") != SOURCE_REGISTRY_SCHEMA:
        raise FactoryError("invalid source registry schema")
    errors: list[str] = []
    degraded: list[str] = []
    if registry.get("product") != PRODUCT:
        errors.append(f"source registry product must be {PRODUCT}")
    if not isinstance(registry.get("sources"), list) or not registry.get("sources"):
        errors.append("source registry requires at least one source")
    seen_source_ids: set[str] = set()
    for source in registry.get("sources") or []:
        source_id = str(source.get("source_id") or "")
        if not source_id:
            errors.append("source registry entry missing source_id")
            continue
        if source_id in seen_source_ids:
            errors.append(f"duplicate source registry source_id: {source_id}")
            continue
        seen_source_ids.add(source_id)
        if source.get("public_distribution_clearance_asserted") is not False:
            errors.append(
                f"{source_id}: public distribution clearance must not be inferred"
            )
        if source.get("license_status") not in {
            "reviewed-internal-only",
            "unknown-not-cleared",
        }:
            errors.append(
                f"{source_id}: explicit conservative license_status is required"
            )
        if source.get("corpus_diversity_degraded") is True:
            degraded.append(source_id)
            if source.get("status") != "degraded":
                errors.append(
                    f"{source_id}: degraded diversity must set status=degraded"
                )
            policy = source.get("reuse_policy") or {}
            if policy.get("allow_900m_plus") is not False:
                errors.append(f"{source_id}: degraded source cannot roll into 900M+")
            if policy.get("verbatim_reuse") is not False:
                errors.append(f"{source_id}: degraded source cannot be reused verbatim")
            if policy.get("diagnostic_only") is not True:
                errors.append(f"{source_id}: degraded source must be diagnostic_only")
            evidence = source.get("diversity_evidence") or {}
            input_hashes = evidence.get("input_sha256") or {}
            input_paths = evidence.get("input_paths") or {}
            if (
                not _valid_sha256(evidence.get("script_sha256"))
                or not isinstance(input_hashes, dict)
                or not input_hashes
                or not all(_valid_sha256(value) for value in input_hashes.values())
            ):
                errors.append(f"{source_id}: degraded claim lacks hash-bound evidence")
            if (
                not isinstance(input_paths, dict)
                or set(input_paths) != set(input_hashes)
                or not all(isinstance(value, str) for value in input_paths.values())
            ):
                errors.append(
                    f"{source_id}: diversity input paths do not match input hashes"
                )
            else:
                for raw_path in input_paths.values():
                    try:
                        _safe_relative(raw_path)
                    except FactoryError:
                        errors.append(
                            f"{source_id}: unsafe diversity evidence input path"
                        )
        else:
            if source.get("status") != "approved":
                errors.append(
                    f"{source_id}: non-degraded source must set status=approved"
                )
            if source.get("license_reviewed") is not True:
                errors.append(
                    f"{source_id}: approved source requires completed license review"
                )
    return {
        "schema": "mei-51m-corpus-source-registry-audit-v1",
        "status": "passed" if not errors else "blocked",
        "registry_sha256": sha256_file(path),
        "degraded_sources": degraded,
        "errors": errors,
    }


def audit_build(
    build_dir: Path, eval_dir: Path, source_registry: Path
) -> dict[str, Any]:
    verification = verify_build(build_dir)
    contract = load_json(build_dir / "contract.snapshot.json")
    index = validate_contract(contract)
    rows = _load_build_rows(build_dir)
    expected_rows = compile_contract(index)
    recompile_mismatches: list[dict[str, str]] = []
    for lane in LANES:
        for split in SPLITS:
            if rows[lane][split] != expected_rows[lane][split]:
                recompile_mismatches.append(
                    {
                        "lane": lane,
                        "split": split,
                        "actual_sha256": sha256_json(rows[lane][split]),
                        "expected_sha256": sha256_json(expected_rows[lane][split]),
                    }
                )
    recompile_errors = (
        [
            "deterministic local recompile mismatch: "
            + ", ".join(
                f"{item['lane']}.{item['split']}" for item in recompile_mismatches[:12]
            )
        ]
        if recompile_mismatches
        else []
    )
    split_report, split_errors = _audit_split_isolation(rows)
    duplicate_report, duplicate_errors = _audit_duplicates(
        rows,
        near_threshold=float(
            (contract.get("quality_gates") or {}).get("near_duplicate_threshold", 0.92)
        ),
    )
    semantic_report, semantic_errors = _audit_semantics(rows, contract)
    contamination_report, contamination_errors = _audit_contamination(rows, eval_dir)
    source_report = audit_source_registry(source_registry)
    padding_hits = [
        str(row.get("sample_id"))
        for lane in LANES
        for split in SPLITS
        for row in rows[lane][split]
        if any(IDENTIFIER_PADDING_RE.search(part) for part in _model_visible_parts(row))
    ]
    errors = [
        *verification["errors"],
        *recompile_errors,
        *split_errors,
        *duplicate_errors,
        *semantic_errors,
        *contamination_errors,
        *source_report["errors"],
    ]
    if padding_hits:
        errors.append(f"identifier padding detected: {len(padding_hits)}")
    cost = load_json(build_dir / "cost-ledger.json")
    if cost.get("status") != "passed" or cost.get("provider_calls") != 0:
        errors.append("cost ledger is not zero-cost/offline")
    total_rows = sum(len(rows[lane][split]) for lane in LANES for split in SPLITS)
    receipt = {
        "schema": AUDIT_SCHEMA,
        "status": "passed" if not errors else "blocked",
        "process_complete": True,
        "release_eligible": False,
        "release_ineligible_reasons": [
            "offline_control_plane_fixture_only",
            "insufficient_rows_for_scale_quality_claim",
        ],
        "inputs": {
            "build_manifest_sha256": sha256_file(build_dir / "build-manifest.json"),
            "eval_lock_sha256": sha256_file(eval_dir / "lock.json"),
            "source_registry_sha256": sha256_file(source_registry),
        },
        "integrity": {
            "build": verification,
            "deterministic_local_recompile": {
                "status": "passed" if not recompile_mismatches else "blocked",
                "mismatches": recompile_mismatches,
            },
            "split_isolation": split_report,
            "duplicates": duplicate_report,
            "semantic_consistency": semantic_report,
            "identifier_padding_hits": padding_hits[:20],
            "contamination": contamination_report,
            "license_and_provenance": {
                "status": "passed" if not semantic_errors else "blocked",
                "public_distribution_clearance_asserted": False,
            },
            "source_registry": source_report,
            "cost_ledger": cost,
        },
        "rows": total_rows,
        "errors": errors[:120],
        "quality_scale_ready": False,
        "current_mutated": False,
        "auditor_implementation_sha256": implementation_sha256(),
    }
    _atomic_write(
        build_dir / "audit-receipt.json", pretty_bytes(receipt), write_once=True
    )
    return receipt


def freeze_eval(
    holdout: Path, provenance_path: Path, out_dir: Path, lock_id: str
) -> dict[str, Any]:
    rows = load_jsonl(holdout)
    provenance = load_json(provenance_path)
    if not lock_id.strip():
        raise FactoryError("evaluation lock_id must be non-empty")
    if not rows:
        raise FactoryError("evaluation holdout cannot be empty")
    if provenance.get("schema") != "mei-51m-independent-eval-provenance-v1":
        raise FactoryError("invalid independent eval provenance schema")
    _required_string(provenance, "source_id", where="eval provenance")
    if provenance.get("independent_authoring") is not True:
        raise FactoryError("eval holdout provenance must assert independent authoring")
    if provenance.get("used_by_training_generator") is not False:
        raise FactoryError("eval holdout must not be used by the training generator")
    if provenance.get("semantic_families_disjoint_from_fixture") is not True:
        raise FactoryError("eval provenance must assert disjoint semantic families")
    if provenance.get("teacher_or_realizer_shared_with_training") is not False:
        raise FactoryError("eval provenance must forbid shared teacher/realizer IDs")
    if provenance.get("public_distribution_clearance_asserted") is not False:
        raise FactoryError("eval provenance cannot infer distribution clearance")
    _required_string(provenance, "license", where="eval provenance")
    observed_axes: dict[str, dict[str, set[str]]] = {
        axis: defaultdict(set) for axis in ISOLATION_AXES
    }
    sample_ids: set[str] = set()
    observed_splits: set[str] = set()
    for row in rows:
        if row.get("split") not in {"dev", "test"}:
            raise FactoryError("locked evaluation can contain only dev/test rows")
        observed_splits.add(str(row["split"]))
        sample_id = _required_string(row, "sample_id", where="eval row")
        if sample_id in sample_ids:
            raise FactoryError(f"duplicate locked-eval sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if row.get("lane") not in LANES:
            raise FactoryError(
                f"locked evaluation row has invalid lane: {row.get('lane')}"
            )
        if not _visible_text(row):
            raise FactoryError("eval row requires visible query/text")
        for axis in ISOLATION_AXES:
            value = _required_string(row, axis, where="eval row")
            observed_axes[axis][value].add(str(row["split"]))
    if observed_splits != {"dev", "test"}:
        raise FactoryError("evaluation holdout must exercise both dev and test")
    for axis, values in observed_axes.items():
        leaks = sorted(value for value, splits in values.items() if len(splits) > 1)
        if leaks:
            raise FactoryError(
                f"locked evaluation {axis} crosses dev/test: {leaks[:8]}"
            )
    holdout_payload = jsonl_bytes(rows)
    provenance_payload = pretty_bytes(provenance)
    artifacts = {
        "holdout.jsonl": artifact_spec_bytes(holdout_payload, rows=len(rows)),
        "provenance.json": artifact_spec_bytes(provenance_payload),
    }
    fingerprint = sha256_json(
        {
            "lock_id": lock_id,
            "artifacts": artifacts,
            "provenance_sha256": artifacts["provenance.json"]["sha256"],
        }
    )
    lock = {
        "schema": EVAL_LOCK_SCHEMA,
        "lock_id": lock_id,
        "status": "frozen",
        "artifacts": artifacts,
        "artifact_merkle_root": merkle_root(artifacts),
        "provenance_sha256": artifacts["provenance.json"]["sha256"],
        "evaluation_fingerprint": fingerprint,
        "training_generator_access": "forbidden",
        "freezer_implementation_sha256": implementation_sha256(),
    }
    descriptor = {
        "schema": "mei-51m-independent-eval-descriptor-v1",
        "lock_id": lock_id,
        "evaluation_fingerprint": fingerprint,
        "lock_sha256": sha256_bytes(pretty_bytes(lock)),
        "payload_paths_exposed": False,
    }
    parent = out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.eval.", dir=parent))
    try:
        _atomic_write(temporary / "holdout.jsonl", holdout_payload, write_once=True)
        _atomic_write(
            temporary / "provenance.json", provenance_payload, write_once=True
        )
        _atomic_write(temporary / "lock.json", pretty_bytes(lock), write_once=True)
        _atomic_write(
            temporary / "descriptor.json", pretty_bytes(descriptor), write_once=True
        )
        if out_dir.exists():
            existing = load_json(out_dir / "lock.json")
            if existing != lock:
                raise FactoryError(
                    f"refusing to overwrite different eval lock: {out_dir}"
                )
            shutil.rmtree(temporary)
            result = verify_eval_lock(out_dir)
            return {"status": result["status"], "reused": True, "path": str(out_dir)}
        temporary.replace(out_dir)
        result = verify_eval_lock(out_dir)
        return {"status": result["status"], "reused": False, "path": str(out_dir)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("rb") as src,
        tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as dst,
    ):
        temporary = Path(dst.name)
        shutil.copyfileobj(src, dst)
        dst.flush()
    temporary.replace(target)


def freeze_release(
    build_dir: Path,
    release_dir: Path,
    release_id: str,
    *,
    object_store: Path | None = None,
) -> dict[str, Any]:
    build_verification = verify_build(build_dir)
    if build_verification["status"] != "passed":
        raise FactoryError("cannot freeze a corrupt build")
    audit = load_json(build_dir / "audit-receipt.json")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != "passed"
        or audit.get("process_complete") is not True
    ):
        raise FactoryError("cannot freeze a build whose integrity audit is blocked")
    if (audit.get("inputs") or {}).get("build_manifest_sha256") != sha256_file(
        build_dir / "build-manifest.json"
    ):
        raise FactoryError("cannot freeze an audit that does not bind this build")
    if audit.get("current_mutated") is not False or not _valid_sha256(
        audit.get("auditor_implementation_sha256")
    ):
        raise FactoryError("cannot freeze an unbound or state-mutating audit")
    build_manifest = load_json(build_dir / "build-manifest.json")
    source_files = [
        *sorted((build_manifest.get("artifacts") or {}).keys()),
        "build-manifest.json",
        "generation-receipt.json",
        "audit-receipt.json",
    ]
    parent = release_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{release_dir.name}.freeze.", dir=parent)
    )
    temporary_store = temporary / "objects/sha256"
    external_store = object_store.resolve() if object_store is not None else None
    artifacts: dict[str, dict[str, Any]] = {}
    try:
        for raw in source_files:
            rel = _safe_relative(raw)
            source = build_dir / rel
            if not source.is_file():
                raise FactoryError(f"freeze source is missing: {source}")
            target_rel = Path("artifacts") / rel
            target = temporary / target_rel
            _copy_verified(source, target)
            spec = artifact_spec(target)
            if raw.endswith(".jsonl"):
                spec["rows"] = sum(
                    1
                    for line in target.read_text(encoding="utf-8").splitlines()
                    if line
                )
            artifacts[str(target_rel)] = spec
            store_root = external_store or temporary_store
            object_path = store_root / spec["sha256"][:2] / spec["sha256"]
            if object_path.exists():
                if sha256_file(object_path) != spec["sha256"]:
                    raise FactoryError(
                        f"content-addressed object collision: {object_path}"
                    )
            else:
                _copy_verified(target, object_path)

        object_store_value = (
            str(external_store) if external_store is not None else "objects/sha256"
        )
        manifest = {
            "schema": RELEASE_SCHEMA,
            "release_id": release_id,
            "product": PRODUCT,
            "status": "frozen_fixture",
            "purpose": "offline_control_plane_fixture",
            "artifacts": artifacts,
            "artifact_merkle_root": merkle_root(artifacts),
            "object_store": object_store_value,
            "build_manifest_sha256": sha256_file(build_dir / "build-manifest.json"),
            "build_merkle_root": build_manifest["artifact_merkle_root"],
            "evaluation_fingerprint": build_manifest["evaluation_fingerprint"],
            "process_complete": True,
            "release_eligible": False,
            "release_ineligible_reasons": audit["release_ineligible_reasons"],
            "current_mutated": False,
            "freezer_implementation_sha256": implementation_sha256(),
        }
        _atomic_write(
            temporary / "release-manifest.json", pretty_bytes(manifest), write_once=True
        )
        receipt = {
            "schema": "mei-51m-corpus-freeze-receipt-v1",
            "release_id": release_id,
            "status": "passed",
            "release_manifest_sha256": sha256_file(temporary / "release-manifest.json"),
            "artifact_merkle_root": manifest["artifact_merkle_root"],
            "object_store": object_store_value,
            "process_complete": True,
            "release_eligible": False,
            "current_mutated": False,
        }
        _atomic_write(
            temporary / "receipt.json", pretty_bytes(receipt), write_once=True
        )
        if release_dir.exists():
            existing = load_json(release_dir / "release-manifest.json")
            if existing != manifest:
                raise FactoryError(
                    f"refusing to overwrite different release: {release_dir}"
                )
            shutil.rmtree(temporary)
            return {**verify_release(release_dir), "reused": True}
        temporary.replace(release_dir)
        return {**verify_release(release_dir), "reused": False}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _release_object_store(release_dir: Path, manifest: Mapping[str, Any]) -> Path:
    raw = str(manifest.get("object_store") or "")
    if not raw:
        raise FactoryError("release manifest does not name an object store")
    path = Path(raw)
    return path if path.is_absolute() else release_dir / path


def verify_release(release_dir: Path) -> dict[str, Any]:
    manifest = load_json(release_dir / "release-manifest.json")
    if manifest.get("schema") != RELEASE_SCHEMA:
        raise FactoryError("invalid corpus release schema")
    errors = verify_artifacts(release_dir, manifest.get("artifacts") or {})
    expected = merkle_root(manifest.get("artifacts") or {})
    if expected != manifest.get("artifact_merkle_root"):
        errors.append("release artifact Merkle root mismatch")
    if (
        manifest.get("product") != PRODUCT
        or manifest.get("status") != "frozen_fixture"
        or manifest.get("purpose") != "offline_control_plane_fixture"
    ):
        errors.append("release identity/purpose mismatch")
    if (
        manifest.get("process_complete") is not True
        or manifest.get("release_eligible") is not False
        or manifest.get("current_mutated") is not False
    ):
        errors.append("release terminal-state contract mismatch")
    embedded_build_path = release_dir / "artifacts/build-manifest.json"
    embedded_build_spec = (manifest.get("artifacts") or {}).get(
        "artifacts/build-manifest.json"
    ) or {}
    if manifest.get("build_manifest_sha256") != embedded_build_spec.get("sha256"):
        errors.append("release does not bind embedded build manifest")
    if embedded_build_path.is_file():
        embedded_build = load_json(embedded_build_path)
        if embedded_build.get("artifact_merkle_root") != manifest.get(
            "build_merkle_root"
        ):
            errors.append("release/build Merkle root mismatch")
        if embedded_build.get("evaluation_fingerprint") != manifest.get(
            "evaluation_fingerprint"
        ):
            errors.append("release/build evaluation fingerprint mismatch")
    store = _release_object_store(release_dir, manifest)
    for spec in (manifest.get("artifacts") or {}).values():
        digest = str(spec.get("sha256") or "")
        object_path = store / digest[:2] / digest
        if not object_path.is_file() or sha256_file(object_path) != digest:
            errors.append(f"missing/corrupt recovery object: {digest}")
    receipt = load_json(release_dir / "receipt.json")
    if receipt.get("schema") != "mei-51m-corpus-freeze-receipt-v1":
        errors.append("invalid freeze receipt schema")
    if receipt.get("release_manifest_sha256") != sha256_file(
        release_dir / "release-manifest.json"
    ):
        errors.append("freeze receipt does not bind release manifest")
    if receipt.get("release_id") != manifest.get("release_id"):
        errors.append("freeze receipt release_id mismatch")
    if receipt.get("artifact_merkle_root") != expected:
        errors.append("freeze receipt artifact Merkle root mismatch")
    if receipt.get("object_store") != manifest.get("object_store"):
        errors.append("freeze receipt object-store mismatch")
    if receipt.get("status") != "passed" or receipt.get("process_complete") is not True:
        errors.append("freeze receipt is not terminal passed")
    if (
        receipt.get("release_eligible") is not False
        or receipt.get("current_mutated") is not False
    ):
        errors.append("freeze receipt terminal-state contract mismatch")
    if not _valid_sha256(manifest.get("freezer_implementation_sha256")):
        errors.append("release manifest lacks a valid freezer implementation hash")
    return {
        "schema": "mei-51m-corpus-release-verification-v1",
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "release_id": manifest.get("release_id"),
        "artifact_merkle_root": expected,
        "process_complete": manifest.get("process_complete") is True,
        "release_eligible": manifest.get("release_eligible") is True,
    }


def restore_artifact(
    release_dir: Path, artifact: str, *, repair: bool
) -> dict[str, Any]:
    manifest = load_json(release_dir / "release-manifest.json")
    rel = str(_safe_relative(artifact))
    spec = (manifest.get("artifacts") or {}).get(rel)
    if spec is None:
        raise FactoryError(f"artifact is not registered in the release: {rel}")
    target = release_dir / rel
    if target.is_file() and sha256_file(target) == spec["sha256"]:
        return {"status": "passed", "action": "none", "artifact": rel}
    if not repair:
        raise FactoryError("artifact needs recovery; rerun with --repair")
    store = _release_object_store(release_dir, manifest)
    source = store / spec["sha256"][:2] / spec["sha256"]
    if not source.is_file() or sha256_file(source) != spec["sha256"]:
        raise FactoryError("recovery object is absent or corrupt")
    quarantined = None
    if target.exists():
        old_hash = sha256_file(target)
        quarantine = release_dir / "quarantine" / f"{rel.replace('/', '__')}.{old_hash}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            if sha256_file(quarantine) != old_hash:
                raise FactoryError("quarantine path collision")
            target.unlink()
        else:
            target.replace(quarantine)
        quarantined = str(quarantine)
    _copy_verified(source, target)
    if sha256_file(target) != spec["sha256"]:
        raise FactoryError("restored artifact hash mismatch")
    return {
        "status": "passed",
        "action": "restored",
        "artifact": rel,
        "sha256": spec["sha256"],
        "quarantined": quarantined,
    }


def validate_experiment_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    if plan.get("schema") != EXPERIMENT_SCHEMA:
        raise FactoryError("invalid corpus experiment plan schema")
    errors: list[str] = []
    tracks = plan.get("tracks") or {}
    scale = tracks.get("scale_curve") or {}
    data = tracks.get("data_innovation_curve") or {}
    if scale.get("causal_axis") != "base_exposure_tokens":
        errors.append("scale_curve causal_axis must be base_exposure_tokens")
    if data.get("causal_axis") != "data_release":
        errors.append("data_innovation_curve causal_axis must be data_release")
    expected_exposures = [
        300_000_000,
        600_000_000,
        900_000_000,
        1_200_000_000,
        1_500_000_000,
        2_100_000_000,
    ]
    exposures = [
        int(point.get("exposure_tokens") or 0) for point in scale.get("points") or []
    ]
    if exposures != expected_exposures:
        errors.append(
            "scale_curve must contain the exact ordered 300M/600M/900M/1.2B/1.5B/2.1B points"
        )
    scale_fixed = scale.get("fixed") or {}
    for key in (
        "sft_release_id",
        "sft_manifest_sha256",
        "eval_lock_id",
        "evaluation_fingerprint",
        "training_budget_sha256",
        "runtime_profile_sha256",
    ):
        if not scale_fixed.get(key):
            errors.append(f"scale_curve.fixed.{key} is required")
    for key in (
        "sft_manifest_sha256",
        "evaluation_fingerprint",
        "training_budget_sha256",
        "runtime_profile_sha256",
    ):
        if not _valid_sha256(scale_fixed.get(key)):
            errors.append(f"scale_curve.fixed.{key} must be a SHA-256")
    for point in scale.get("points") or []:
        forbidden = set(point) & {
            "sft_release_id",
            "sft_manifest_sha256",
            "evaluation_fingerprint",
            "training_budget_sha256",
        }
        if forbidden:
            errors.append(
                f"scale point overrides fixed data/eval controls: {sorted(forbidden)}"
            )
        status = point.get("status")
        if status not in {"frozen", "training", "planned"}:
            errors.append(f"scale point has invalid status: {status!r}")
        if not point.get("base_id"):
            errors.append("scale point requires base_id")
        weights = point.get("base_weights_sha256")
        if status == "frozen" and not _valid_sha256(weights):
            errors.append("frozen scale point requires a Base weights SHA-256")
        if (
            status in {"training", "planned"}
            and weights != "pending"
            and not _valid_sha256(weights)
        ):
            errors.append("unfinished scale point weights must be pending or SHA-256")
    data_fixed = data.get("fixed") or {}
    for key in (
        "base_id",
        "base_weights_sha256",
        "eval_lock_id",
        "evaluation_fingerprint",
        "training_budget_sha256",
        "runtime_profile_sha256",
    ):
        if not data_fixed.get(key):
            errors.append(f"data_innovation_curve.fixed.{key} is required")
    for key in (
        "base_weights_sha256",
        "evaluation_fingerprint",
        "training_budget_sha256",
        "runtime_profile_sha256",
    ):
        if not _valid_sha256(data_fixed.get(key)):
            errors.append(f"data_innovation_curve.fixed.{key} must be a SHA-256")
    if data_fixed.get("evaluation_fingerprint") != scale_fixed.get(
        "evaluation_fingerprint"
    ):
        errors.append("two curves must use the same locked evaluation fingerprint")
    for point in data.get("points") or []:
        forbidden = set(point) & {"base_id", "base_weights_sha256", "exposure_tokens"}
        if forbidden:
            errors.append(
                f"data innovation point overrides fixed Base: {sorted(forbidden)}"
            )
        if not point.get("data_release_id") or not point.get("data_manifest_sha256"):
            errors.append(
                "data innovation point must freeze data release ID and manifest hash"
            )
        status = point.get("status")
        manifest_hash = point.get("data_manifest_sha256")
        if status not in {"frozen", "planned"}:
            errors.append(f"data innovation point has invalid status: {status!r}")
        if status == "frozen" and not _valid_sha256(manifest_hash):
            errors.append("frozen data innovation point requires a manifest SHA-256")
        if (
            status == "planned"
            and manifest_hash != "pending"
            and not _valid_sha256(manifest_hash)
        ):
            errors.append("planned data manifest must be pending or SHA-256")
    if len(data.get("points") or []) < 2:
        errors.append(
            "data innovation curve needs a baseline and at least one innovation point"
        )
    return {
        "schema": "mei-51m-corpus-experiment-plan-audit-v1",
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "scale_points": exposures,
        "data_points": len(data.get("points") or []),
        "causal_axes_independent": not errors,
        "forbids_base_data_gain_conflation": True,
    }


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_scale_intake(intake: Mapping[str, Any]) -> None:
    if intake.get("schema") != SCALE_INTAKE_SCHEMA:
        raise FactoryError(f"scale intake schema must be {SCALE_INTAKE_SCHEMA}")
    if intake.get("product") != PRODUCT:
        raise FactoryError(f"scale intake product must be {PRODUCT}")
    _required_string(intake, "intake_id", where="scale intake")
    if intake.get("purpose") != "quality_candidate_plan":
        raise FactoryError("scale intake purpose must be quality_candidate_plan")
    if intake.get("readiness") != "planning_only":
        raise FactoryError(
            "scale intake must remain planning_only until payload execution is audited"
        )
    if intake.get("split_isolation_axes") != list(ISOLATION_AXES):
        raise FactoryError(
            "scale intake must freeze all six split isolation axes in order"
        )
    if not _valid_sha256(intake.get("split_registry_sha256")):
        raise FactoryError("scale intake requires a frozen split_registry_sha256")

    teacher = intake.get("teacher_contract") or {}
    if teacher.get("mutable_fields") != ["query"]:
        raise FactoryError("scale teacher may mutate only query")
    if teacher.get("gold_compiled_locally") is not True:
        raise FactoryError("scale gold must be compiled locally")
    if teacher.get("provider_calls_allowed") is not False:
        raise FactoryError("scale planning cannot authorize provider calls")

    evaluation = intake.get("evaluation") or {}
    if not evaluation.get("lock_id") or not _valid_sha256(
        evaluation.get("evaluation_fingerprint")
    ):
        raise FactoryError("scale intake requires a frozen evaluation identity")
    if evaluation.get("payload_paths_exposed") is not False:
        raise FactoryError("scale generator must not receive evaluation payload paths")

    budget = intake.get("budget") or {}
    if budget.get("provider_calls_allowed") is not False:
        raise FactoryError("scale intake cannot authorize provider calls")
    maximum_paid = budget.get("max_paid_cny")
    if (
        not isinstance(maximum_paid, (int, float))
        or isinstance(maximum_paid, bool)
        or not math.isfinite(float(maximum_paid))
        or float(maximum_paid) != 0
    ):
        raise FactoryError("scale intake must keep paid-provider budget at zero")

    sources = intake.get("source_inputs") or []
    if not isinstance(sources, list) or not sources:
        raise FactoryError("scale intake requires hash-bound source inputs")
    source_ids: set[str] = set()
    usable_sources = 0
    split_registry_sources = 0
    for number, source in enumerate(sources):
        where = f"source_inputs[{number}]"
        if not isinstance(source, dict):
            raise FactoryError(f"{where} must be an object")
        source_id = _required_string(source, "source_id", where=where)
        if source_id in source_ids:
            raise FactoryError(f"duplicate scale source_id: {source_id}")
        source_ids.add(source_id)
        _safe_relative(_required_string(source, "manifest_path", where=where))
        if not _valid_sha256(source.get("manifest_sha256")):
            raise FactoryError(f"{where} requires an immutable manifest_sha256")
        if source.get("provenance_complete") is not True:
            raise FactoryError(f"{where} lacks complete provenance")
        if source.get("eval_payload_access") is not False:
            raise FactoryError(f"{where} accessed or may access evaluation payload")
        if source.get("corpus_diversity_degraded") is True:
            if source.get("excluded_from_generation") is not True:
                raise FactoryError(
                    f"{where} degraded source must be excluded from generation"
                )
            if source.get("license_status") not in {
                "reviewed-internal-only",
                "unknown-not-cleared",
            }:
                raise FactoryError(
                    f"{where} degraded source lacks conservative license status"
                )
        else:
            if (
                source.get("license_reviewed") is not True
                or source.get("license_status") != "reviewed-internal-only"
            ):
                raise FactoryError(f"{where} lacks license review")
            usable_sources += 1
            if source.get("role") == "control_contract_only":
                split_registry_sources += 1
    if usable_sources == 0:
        raise FactoryError("scale intake has no non-degraded generation source")
    if split_registry_sources != 1:
        raise FactoryError(
            "scale intake requires exactly one non-degraded control_contract_only source"
        )

    targets = intake.get("targets") or {}
    for lane in LANES:
        lane_targets = targets.get(lane)
        if not isinstance(lane_targets, dict):
            raise FactoryError(f"scale target missing lane: {lane}")
        for split in SPLITS:
            value = lane_targets.get(split)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise FactoryError(
                    f"scale target {lane}.{split} must be a positive integer"
                )

    sharding = intake.get("sharding") or {}
    rows_per_shard = sharding.get("max_rows_per_shard")
    if (
        not isinstance(rows_per_shard, int)
        or isinstance(rows_per_shard, bool)
        or not 1 <= rows_per_shard <= 1_000_000
    ):
        raise FactoryError("max_rows_per_shard must be between 1 and 1,000,000")
    if sharding.get("partition_key") != "semantic_family":
        raise FactoryError(
            "scale shards must be partitioned by semantic_family before realization"
        )

    strategies = intake.get("audit_strategies") or {}
    if strategies != SCALE_AUDIT_STRATEGIES:
        raise FactoryError(
            "scale intake must use the frozen bounded-memory audit strategies"
        )
    work_unit_count = sum(
        math.ceil(int(targets[lane][split]) / rows_per_shard)
        for lane in LANES
        for split in SPLITS
    )
    if work_unit_count > MAX_SCALE_WORK_UNITS:
        raise FactoryError(
            f"scale plan would create {work_unit_count} work units; "
            f"increase shard size or split the intake (limit={MAX_SCALE_WORK_UNITS})"
        )


def _scale_plan_payload(intake: Mapping[str, Any]) -> dict[str, Any]:
    _validate_scale_intake(intake)
    shard_size = int(intake["sharding"]["max_rows_per_shard"])
    usable_sources = [
        str(source["source_id"])
        for source in intake["source_inputs"]
        if source.get("corpus_diversity_degraded") is not True
    ]
    work_units: list[dict[str, Any]] = []
    for lane in LANES:
        for split in SPLITS:
            count = int(intake["targets"][lane][split])
            shards = math.ceil(count / shard_size)
            for shard_index in range(shards):
                start = shard_index * shard_size
                stop = min(count, start + shard_size)
                unit_identity = {
                    "intake_id": intake["intake_id"],
                    "lane": lane,
                    "split": split,
                    "shard_index": shard_index,
                    "row_start": start,
                    "row_stop": stop,
                    "split_registry_sha256": intake["split_registry_sha256"],
                }
                work_units.append(
                    {
                        "work_unit_id": stable_id("SHARD", unit_identity),
                        **unit_identity,
                        "source_ids": usable_sources,
                        "status": "planned",
                    }
                )
    unit_specs = {
        str(unit["work_unit_id"]): artifact_spec_bytes(canonical_bytes(unit))
        for unit in work_units
    }
    return {
        "schema": SCALE_PLAN_SCHEMA,
        "intake_id": intake["intake_id"],
        "product": PRODUCT,
        "status": "planned",
        "readiness": "planning_only",
        "intake_sha256": sha256_json(intake),
        "split_registry_sha256": intake["split_registry_sha256"],
        "evaluation": dict(intake["evaluation"]),
        "teacher_contract": dict(intake["teacher_contract"]),
        "source_inputs": copy_json_mapping(intake["source_inputs"]),
        "audit_strategies": dict(intake["audit_strategies"]),
        "targets": copy_json_mapping(intake["targets"]),
        "sharding": dict(intake["sharding"]),
        "max_work_units": MAX_SCALE_WORK_UNITS,
        "work_units": work_units,
        "work_unit_merkle_root": merkle_root(unit_specs),
        "work_unit_count": len(work_units),
        "planning_complete": True,
        "corpus_complete": False,
        "release_eligible": False,
        "current_mutated": False,
        "planner_implementation_sha256": implementation_sha256(),
    }


def _verify_scale_source_files(intake: Mapping[str, Any], source_root: Path) -> None:
    root = source_root.resolve()
    if not root.is_dir():
        raise FactoryError(f"scale source root is not a directory: {root}")
    for source in intake.get("source_inputs") or []:
        rel = _safe_relative(str(source.get("manifest_path") or ""))
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise FactoryError(
                f"scale source manifest escapes source root: {rel}"
            ) from error
        if not path.is_file():
            raise FactoryError(f"scale source manifest is missing: {rel}")
        expected = str(source.get("manifest_sha256") or "")
        actual = sha256_file(path)
        if actual != expected:
            raise FactoryError(
                f"scale source manifest hash mismatch: {rel} expected={expected} actual={actual}"
            )
        if source.get("role") == "control_contract_only":
            contract = load_json(path)
            validate_contract(contract)
            actual_registry = sha256_json(contract["split_registry"])
            expected_registry = str(intake.get("split_registry_sha256") or "")
            if actual_registry != expected_registry:
                raise FactoryError(
                    "scale split_registry_sha256 does not bind the control contract"
                )


def copy_json_mapping(value: Any) -> Any:
    """Return a JSON-only deep copy without importing a mutable object helper."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def verify_scale_plan(out_dir: Path) -> dict[str, Any]:
    plan = load_json(out_dir / "scale-plan.json")
    receipt = load_json(out_dir / "receipt.json")
    errors: list[str] = []
    if plan.get("schema") != SCALE_PLAN_SCHEMA:
        errors.append("invalid scale plan schema")
    raw_work_units = plan.get("work_units")
    work_units = raw_work_units if isinstance(raw_work_units, list) else []
    if not isinstance(raw_work_units, list):
        errors.append("scale plan work_units must be a list")
    if any(not isinstance(unit, dict) for unit in work_units):
        errors.append("scale plan work unit must be an object")
    work_unit_ids = [
        str(unit.get("work_unit_id")) for unit in work_units if isinstance(unit, dict)
    ]
    if len(work_unit_ids) != len(set(work_unit_ids)) or any(
        item in {"", "None"} for item in work_unit_ids
    ):
        errors.append("scale plan contains invalid or duplicate work_unit_id values")
    if plan.get("work_unit_count") != len(work_units):
        errors.append("scale plan work_unit_count mismatch")
    if (
        len(work_units) > MAX_SCALE_WORK_UNITS
        or plan.get("max_work_units") != MAX_SCALE_WORK_UNITS
    ):
        errors.append("scale plan exceeds or changes the work-unit safety bound")

    expected_sources = sorted(
        str(source.get("source_id"))
        for source in plan.get("source_inputs") or []
        if isinstance(source, dict)
        if source.get("corpus_diversity_degraded") is not True
    )
    shard_size = (plan.get("sharding") or {}).get("max_rows_per_shard")
    if (
        not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or not 1 <= shard_size <= 1_000_000
    ):
        errors.append("scale plan has invalid shard-size bound")
        shard_size = 0
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in work_units:
        if not isinstance(unit, dict):
            continue
        identity = {
            "intake_id": unit.get("intake_id"),
            "lane": unit.get("lane"),
            "split": unit.get("split"),
            "shard_index": unit.get("shard_index"),
            "row_start": unit.get("row_start"),
            "row_stop": unit.get("row_stop"),
            "split_registry_sha256": unit.get("split_registry_sha256"),
        }
        if unit.get("work_unit_id") != stable_id("SHARD", identity):
            errors.append(
                f"work-unit identity hash mismatch: {unit.get('work_unit_id')}"
            )
        if unit.get("status") != "planned":
            errors.append(f"work unit is not planned: {unit.get('work_unit_id')}")
        if (
            sorted(str(item) for item in unit.get("source_ids") or [])
            != expected_sources
        ):
            errors.append(f"work-unit source set mismatch: {unit.get('work_unit_id')}")
        lane = str(unit.get("lane"))
        split = str(unit.get("split"))
        if lane not in LANES or split not in SPLITS:
            errors.append(f"work unit has invalid lane/split: {lane}.{split}")
            continue
        grouped[(lane, split)].append(unit)

    targets = plan.get("targets") or {}
    if not isinstance(targets, dict):
        targets = {}
        errors.append("scale plan targets must be an object")
    for lane in LANES:
        for split in SPLITS:
            target = (targets.get(lane) or {}).get(split)
            units = sorted(
                grouped.get((lane, split), []),
                key=lambda unit: (
                    unit.get("shard_index")
                    if isinstance(unit.get("shard_index"), int)
                    and not isinstance(unit.get("shard_index"), bool)
                    else -1
                ),
            )
            cursor = 0
            for index, unit in enumerate(units):
                if (
                    unit.get("shard_index") != index
                    or unit.get("row_start") != cursor
                    or not isinstance(unit.get("row_stop"), int)
                    or isinstance(unit.get("row_stop"), bool)
                    or int(unit.get("row_stop", -1)) <= cursor
                    or (
                        shard_size
                        and int(unit.get("row_stop", -1)) - cursor > shard_size
                    )
                ):
                    errors.append(f"non-contiguous work-unit range: {lane}.{split}")
                    break
                cursor = int(unit["row_stop"])
            if (
                not isinstance(target, int)
                or isinstance(target, bool)
                or cursor != target
            ):
                errors.append(f"work-unit coverage mismatch: {lane}.{split}")

    unit_specs = {
        str(unit.get("work_unit_id")): artifact_spec_bytes(canonical_bytes(unit))
        for unit in work_units
        if isinstance(unit, dict)
    }
    root = merkle_root(unit_specs)
    if root != plan.get("work_unit_merkle_root"):
        errors.append("scale plan work-unit Merkle root mismatch")
    if receipt.get("schema") != "mei-51m-corpus-scale-planning-receipt-v1":
        errors.append("invalid scale planning receipt schema")
    if receipt.get("plan_sha256") != sha256_file(out_dir / "scale-plan.json"):
        errors.append("scale planning receipt does not bind scale-plan.json")
    if receipt.get("work_unit_merkle_root") != root:
        errors.append("scale planning receipt work-unit root mismatch")
    if receipt.get("work_unit_count") != len(work_units):
        errors.append("scale planning receipt work-unit count mismatch")
    if receipt.get("source_manifests_verified") is not True:
        errors.append("scale planning receipt lacks source-manifest verification")
    if not _valid_sha256(plan.get("planner_implementation_sha256")):
        errors.append("scale plan lacks a valid planner implementation hash")
    if receipt.get("planner_implementation_sha256") != plan.get(
        "planner_implementation_sha256"
    ):
        errors.append("scale planning receipt planner hash mismatch")
    if (
        plan.get("status") != "planned"
        or plan.get("readiness") != "planning_only"
        or plan.get("planning_complete") is not True
        or plan.get("corpus_complete") is not False
        or plan.get("release_eligible") is not False
        or plan.get("current_mutated") is not False
    ):
        errors.append("scale plan terminal-state contract mismatch")
    if (
        receipt.get("status") != "passed"
        or receipt.get("planning_complete") is not True
        or receipt.get("corpus_complete") is not False
        or receipt.get("release_eligible") is not False
        or receipt.get("current_mutated") is not False
        or receipt.get("provider_calls") != 0
    ):
        errors.append("scale planning receipt terminal-state contract mismatch")
    return {
        "schema": "mei-51m-corpus-scale-plan-verification-v1",
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "work_unit_count": len(work_units),
        "work_unit_merkle_root": root,
        "planning_complete": plan.get("planning_complete") is True,
        "corpus_complete": plan.get("corpus_complete") is True,
        "release_eligible": plan.get("release_eligible") is True,
    }


def plan_scale(
    intake_path: Path, out_dir: Path, *, source_root: Path
) -> dict[str, Any]:
    intake = load_json(intake_path)
    plan = _scale_plan_payload(intake)
    _verify_scale_source_files(intake, source_root)
    receipt = {
        "schema": "mei-51m-corpus-scale-planning-receipt-v1",
        "status": "passed",
        "plan_sha256": sha256_bytes(pretty_bytes(plan)),
        "work_unit_merkle_root": plan["work_unit_merkle_root"],
        "work_unit_count": plan["work_unit_count"],
        "planning_complete": True,
        "corpus_complete": False,
        "release_eligible": False,
        "current_mutated": False,
        "provider_calls": 0,
        "source_manifests_verified": True,
        "planner_implementation_sha256": plan["planner_implementation_sha256"],
    }
    parent = out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.scale.", dir=parent))
    try:
        _atomic_write(
            temporary / "scale-plan.json", pretty_bytes(plan), write_once=True
        )
        _atomic_write(
            temporary / "receipt.json", pretty_bytes(receipt), write_once=True
        )
        if out_dir.exists():
            existing = load_json(out_dir / "scale-plan.json")
            if existing != plan:
                raise FactoryError(
                    f"refusing to overwrite different scale plan: {out_dir}"
                )
            shutil.rmtree(temporary)
            return {**verify_scale_plan(out_dir), "reused": True}
        temporary.replace(out_dir)
        return {**verify_scale_plan(out_dir), "reused": False}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _print(value: Any) -> None:
    print(pretty_bytes(value).decode("utf-8"), end="")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser(
        "freeze-eval", help="freeze an independently authored eval holdout"
    )
    eval_parser.add_argument("--holdout", type=Path, required=True)
    eval_parser.add_argument("--provenance", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--lock-id", required=True)

    build_parser = sub.add_parser(
        "build", help="compile semantic worlds into isolated lane data"
    )
    build_parser.add_argument("--contract", type=Path, required=True)
    build_parser.add_argument("--eval-descriptor", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)

    audit_parser = sub.add_parser("audit", help="run all integrity and quality audits")
    audit_parser.add_argument("--build", type=Path, required=True)
    audit_parser.add_argument("--eval-lock", type=Path, required=True)
    audit_parser.add_argument("--source-registry", type=Path, required=True)

    freeze_parser = sub.add_parser("freeze", help="freeze a verified fixture release")
    freeze_parser.add_argument("--build", type=Path, required=True)
    freeze_parser.add_argument("--release", type=Path, required=True)
    freeze_parser.add_argument("--release-id", required=True)
    freeze_parser.add_argument("--object-store", type=Path)

    verify_parser = sub.add_parser(
        "verify", help="verify a frozen release and recovery objects"
    )
    verify_parser.add_argument("--release", type=Path, required=True)

    restore_parser = sub.add_parser("restore", help="recover one registered artifact")
    restore_parser.add_argument("--release", type=Path, required=True)
    restore_parser.add_argument("--artifact", required=True)
    restore_parser.add_argument("--repair", action="store_true")

    experiment_parser = sub.add_parser(
        "validate-experiments", help="validate separated scale/data innovation curves"
    )
    experiment_parser.add_argument("--plan", type=Path, required=True)

    registry_parser = sub.add_parser("audit-source-registry")
    registry_parser.add_argument("--registry", type=Path, required=True)

    scale_parser = sub.add_parser(
        "plan-scale", help="freeze a deterministic bounded-memory scale work plan"
    )
    scale_parser.add_argument("--intake", type=Path, required=True)
    scale_parser.add_argument("--out", type=Path, required=True)
    scale_parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root against which every hash-bound source manifest path is resolved",
    )

    verify_scale_parser = sub.add_parser("verify-scale-plan")
    verify_scale_parser.add_argument("--plan-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze-eval":
            result = freeze_eval(args.holdout, args.provenance, args.out, args.lock_id)
        elif args.command == "build":
            result = build_contract(args.contract, args.eval_descriptor, args.out)
        elif args.command == "audit":
            result = audit_build(args.build, args.eval_lock, args.source_registry)
        elif args.command == "freeze":
            result = freeze_release(
                args.build,
                args.release,
                args.release_id,
                object_store=args.object_store,
            )
        elif args.command == "verify":
            result = verify_release(args.release)
        elif args.command == "restore":
            result = restore_artifact(args.release, args.artifact, repair=args.repair)
        elif args.command == "validate-experiments":
            result = validate_experiment_plan(args.plan)
        elif args.command == "audit-source-registry":
            result = audit_source_registry(args.registry)
        elif args.command == "plan-scale":
            result = plan_scale(args.intake, args.out, source_root=args.source_root)
        elif args.command == "verify-scale-plan":
            result = verify_scale_plan(args.plan_dir)
        else:  # pragma: no cover - argparse prevents this path.
            raise FactoryError(f"unsupported command: {args.command}")
    except FactoryError as error:
        _print({"status": "blocked", "error": str(error)})
        return 2
    _print(result)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
