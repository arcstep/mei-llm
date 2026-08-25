"""Shared provenance + schema validator. Fail closed."""

from __future__ import annotations

from typing import Any

from route_compiler import RouteManifest
from schema_mask import compile_schema, validate_schema_calls
from schema_render import dumps_canonical


def _as_schema(toolset: Any) -> Any:
    if toolset is None:
        raise ValueError("toolset is required; no default VRM")
    return toolset


def provenance_errors(route: dict[str, Any], manifest: RouteManifest) -> list[str]:
    errors: list[str] = []
    rid = route.get("route_id")
    found = manifest.by_id(int(rid)) if rid is not None else None
    if found is None:
        return ["unknown_route_id"]
    call = {"name": found.name, "arguments": found.arguments}
    if dumps_canonical(call) != dumps_canonical({"name": route.get("name"), "arguments": route.get("arguments") or found.arguments}):
        if route.get("name") and route.get("name") != found.name:
            errors.append("name_mismatch")
        if route.get("arguments") is not None and dumps_canonical(route.get("arguments")) != dumps_canonical(found.arguments):
            errors.append("arguments_mismatch")
    for param, prov in (found.provenance or {}).items():
        if param not in found.arguments:
            errors.append(f"provenance_without_arg:{param}")
            continue
        if prov.get("canonical_value") != found.arguments.get(param):
            errors.append(f"provenance_value_mismatch:{param}")
        if not prov.get("evidence_source"):
            errors.append(f"missing_evidence_source:{param}")
        if not prov.get("resolution_source"):
            errors.append(f"missing_resolution_source:{param}")
        if prov.get("evidence_source") == "schema.enum":
            errors.append(f"enum_without_evidence:{param}")
    for param in found.arguments:
        if param not in found.provenance:
            errors.append(f"unprovenanced:{param}")
    return errors


def validate_selected_route(
    route_id: int | None,
    manifest: RouteManifest,
    toolset: Any,
) -> dict[str, Any]:
    if route_id is None:
        return {"ok": True, "function_calls": [], "route_id": None, "error": None, "refuse": True}
    route = manifest.by_id(int(route_id))
    if route is None:
        return {
            "ok": False,
            "function_calls": [],
            "route_id": None,
            "error": "unknown_route_id",
            "refuse": True,
        }
    compiled = compile_schema(_as_schema(toolset))
    calls = [route.external_call()]
    schema_errs = validate_schema_calls(calls, compiled)
    prov_errs = provenance_errors(
        {"route_id": route.route_id, "name": route.name, "arguments": route.arguments},
        manifest,
    )
    errors = list(schema_errs) + prov_errs
    if manifest.overflow or manifest.refuse_reason == "manifest_overflow":
        errors.append("manifest_overflow")
    if errors:
        return {"ok": False, "function_calls": [], "route_id": None, "error": ";".join(errors), "refuse": True}
    return {"ok": True, "function_calls": calls, "route_id": route.route_id, "error": None, "refuse": False}


def validate_external_calls(calls: Any, manifest: RouteManifest, toolset: Any) -> dict[str, Any]:
    if calls in ([], None):
        return {"ok": True, "function_calls": [], "route_id": None, "error": None, "refuse": True}
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return {"ok": False, "function_calls": [], "route_id": None, "error": "illegal_shape", "refuse": True}
    want = dumps_canonical({"name": calls[0].get("name"), "arguments": calls[0].get("arguments") or {}})
    hits = [r for r in manifest.routes if dumps_canonical({"name": r.name, "arguments": r.arguments}) == want]
    if len(hits) != 1:
        return {"ok": False, "function_calls": [], "route_id": None, "error": "not_in_manifest", "refuse": True}
    return validate_selected_route(hits[0].route_id, manifest, toolset)
