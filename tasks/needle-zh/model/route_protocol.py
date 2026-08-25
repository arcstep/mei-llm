"""Internal route-ID protocol: parse, render, hash. Model never emits argument values."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from provenance_validator import validate_selected_route
from route_compiler import RouteManifest, compile_routes
from schema_render import dumps_canonical, schema_hash

PROTOCOL_ID = "mei-route-protocol-v1"
SERIALIZER_ID = "mei-route-serializer-v1"
VALIDATOR_ID = "mei-provenance-validator-v1"
SCORER_ID = "mei-grounded-scorer-v1"
INTERNAL_EMPTY = "[]"
ROUTE_RE = re.compile(r'^\s*\{\s*"route_id"\s*:\s*(\d+)\s*\}\s*$')


def dump_internal(route_id: int | None) -> str:
    if route_id is None:
        return INTERNAL_EMPTY
    return json.dumps({"route_id": int(route_id)}, ensure_ascii=False, separators=(",", ":"))


def allowed_internal_texts(n_routes: int) -> list[str]:
    return [INTERNAL_EMPTY] + [dump_internal(i) for i in range(max(0, int(n_routes)))]


def is_legal_internal_prefix(prefix: str, n_routes: int) -> bool:
    if prefix is None:
        return False
    if prefix == "":
        return True
    return any(text.startswith(prefix) for text in allowed_internal_texts(n_routes))


def parse_internal(text: str, n_routes: int) -> dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"</?act_(execute|refuse)>", "", raw).strip()
    if raw == INTERNAL_EMPTY:
        return {"ok": True, "route_id": None, "refuse": True, "error": None}
    m = ROUTE_RE.match(raw)
    if not m:
        return {"ok": False, "route_id": None, "refuse": True, "error": "not_route_id"}
    rid = int(m.group(1))
    if rid < 0 or rid >= int(n_routes):
        return {"ok": False, "route_id": None, "refuse": True, "error": "unknown_route_id"}
    return {"ok": True, "route_id": rid, "refuse": False, "error": None}


def render_external(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, separators=(",", ":"))


def manifest_hash(manifest: RouteManifest) -> str:
    payload = dumps_canonical(manifest.as_dict())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_hash(
    *,
    toolset: dict[str, Any],
    manifest: RouteManifest,
    serializer_id: str = SERIALIZER_ID,
    protocol_id: str = PROTOCOL_ID,
    validator_id: str = VALIDATOR_ID,
    normalizer_version: str = "mei-normalizer-v1",
) -> str:
    payload = dumps_canonical(
        {
            "protocol_id": protocol_id,
            "serializer_id": serializer_id,
            "validator_id": validator_id,
            "normalizer_version": normalizer_version,
            "schema_hash": schema_hash(toolset),
            "entity_snapshot_hash": manifest.entity_snapshot_hash,
            "manifest_hash": manifest_hash(manifest),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def materialize_internal(text: str, manifest: RouteManifest, toolset: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_internal(text, len(manifest.routes))
    if not parsed.get("ok") or parsed.get("route_id") is None:
        return {
            "ok": bool(parsed.get("ok")) and parsed.get("route_id") is None,
            "route_id": None,
            "function_calls": [],
            "external": INTERNAL_EMPTY,
            "raw": text,
            "error": parsed.get("error"),
            "validated": True,
        }
    checked = validate_selected_route(parsed["route_id"], manifest, toolset)
    calls = checked.get("function_calls") or []
    return {
        "ok": bool(checked.get("ok")),
        "route_id": checked.get("route_id"),
        "function_calls": calls,
        "external": render_external(calls) if calls else INTERNAL_EMPTY,
        "raw": text,
        "error": checked.get("error"),
        "validated": True,
    }


def compile_and_hash(ctx) -> tuple[RouteManifest, str]:
    manifest = compile_routes(ctx)
    return manifest, protocol_hash(toolset=ctx.toolset, manifest=manifest)
