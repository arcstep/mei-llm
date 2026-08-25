"""Compile finite schema-legal routes with per-argument provenance."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

from candidates import (
    MAX_CANDIDATES_PER_ARG,
    CandidateValue,
    ToolContext,
    candidates_for_param,
    query_scene_conflict,
    tool_intent_spans,
)
from schema_render import compact_tools, dumps_canonical, schema_hash

MAX_ROUTES = 16
LEGAL_FOOD = {("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")}


@dataclass
class Route:
    route_id: int
    name: str
    arguments: dict[str, Any]
    provenance: dict[str, dict[str, Any]]
    tool_index: int

    def external_call(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass
class RouteManifest:
    routes: list[Route] = field(default_factory=list)
    refuse_reason: str | None = None
    schema_hash: str = ""
    entity_snapshot_hash: str = ""
    overflow: bool = False

    def by_id(self, route_id: int) -> Route | None:
        for r in self.routes:
            if r.route_id == route_id:
                return r
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_hash": self.schema_hash,
            "entity_snapshot_hash": self.entity_snapshot_hash,
            "refuse_reason": self.refuse_reason,
            "overflow": self.overflow,
            "routes": [
                {
                    "route_id": r.route_id,
                    "name": r.name,
                    "arguments": r.arguments,
                    "provenance": r.provenance,
                }
                for r in self.routes
            ],
        }


def entity_snapshot_hash(entities: list[dict[str, Any]]) -> str:
    payload = dumps_canonical(
        sorted(
            [
                {
                    "entity_id": e.get("entity_id"),
                    "type": e.get("type"),
                    "canonical": e.get("canonical"),
                    "aliases": sorted(e.get("aliases") or []),
                }
                for e in entities
            ],
            key=lambda r: str(r["entity_id"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_values(cands: list[CandidateValue]) -> list[CandidateValue]:
    amb = [c for c in cands if c.ambiguous]
    if amb:
        return amb
    best: dict[str, CandidateValue] = {}
    for c in cands:
        key = dumps_canonical(c.value)
        prev = best.get(key)
        rank = {"query": 0, "scene": 1, "schema.const": 2, "schema.default": 3}
        if prev is None or rank.get(c.evidence_source, 9) < rank.get(prev.evidence_source, 9):
            best[key] = c
    return list(best.values())


def _product_illegal(name: str, arguments: dict[str, Any]) -> bool:
    if name != "order_food":
        return False
    return (arguments.get("shop"), arguments.get("dish")) not in LEGAL_FOOD


def _in_range(value: Any, prop: dict[str, Any]) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return True
    if "minimum" in prop and value < prop["minimum"]:
        return False
    if "maximum" in prop and value > prop["maximum"]:
        return False
    return True


def compile_routes(ctx: ToolContext) -> RouteManifest:
    tools = list(ctx.toolset.get("tools") or [])
    sch = schema_hash(ctx.toolset)
    seen_ent: dict[str, dict[str, Any]] = {}
    for e in ctx.entities:
        eid = str(e.get("entity_id") or "")
        if eid:
            seen_ent[eid] = e
        else:
            seen_ent[f"anon-{len(seen_ent)}"] = e
    ctx.entities = list(seen_ent.values())
    snap = entity_snapshot_hash(ctx.entities)
    reasons: list[str] = []
    drafts: list[tuple[int, str, dict[str, Any], dict[str, dict[str, Any]]]] = []
    overflow = False

    for t_i, tool in enumerate(tools):
        name = str(tool.get("name") or "")
        params = tool.get("parameters") or {"type": "object", "properties": {}}
        props: dict[str, dict[str, Any]] = dict(params.get("properties") or {})
        required = list(params.get("required") or [])
        optional = [p for p in props if p not in required]
        intent = tool_intent_spans(ctx, tool)
        if not intent and required:
            # parameterized tools still need an action cue
            continue
        if not intent and not required:
            continue

        per_req: dict[str, list[CandidateValue]] = {}
        skip = None
        for p in required:
            cands = candidates_for_param(ctx, tool, p, props.get(p) or {})
            if len(_unique_values([c for c in cands if not c.ambiguous])) > MAX_CANDIDATES_PER_ARG:
                overflow = True
                skip = "manifest_overflow"
                break
            if query_scene_conflict(cands):
                skip = "conflict"
                break
            usable = _unique_values([c for c in cands if not c.ambiguous])
            if any(c.ambiguous for c in cands) and not usable:
                skip = "ambiguous"
                break
            if not usable:
                skip = "missing"
                break
            per_req[p] = usable
        if skip:
            reasons.append(skip)
            continue
        if overflow:
            break

        req_keys = required
        req_lists = [per_req[p] for p in req_keys]
        opt_bind: dict[str, list[CandidateValue]] = {}
        for p in optional:
            cands = candidates_for_param(ctx, tool, p, props.get(p) or {})
            if query_scene_conflict(cands):
                reasons.append("conflict")
                opt_bind = {}
                req_lists = []
                break
            if any(c.ambiguous for c in cands) and not any(not c.ambiguous for c in cands):
                continue
            usable = _unique_values([c for c in cands if not c.ambiguous])
            if not usable:
                continue
            if len(usable) > MAX_CANDIDATES_PER_ARG:
                overflow = True
                break
            opt_bind[p] = usable
        if overflow:
            break
        if not req_lists and required:
            continue

        opt_keys = list(opt_bind)
        opt_lists = [opt_bind[k] for k in opt_keys]
        if not req_keys:
            combos_req = [()]
        else:
            combos_req = list(itertools.product(*req_lists))
        if opt_keys:
            combos_opt = list(itertools.product(*opt_lists))
        else:
            combos_opt = [()]

        for req_combo in combos_req:
            base_args = {k: c.value for k, c in zip(req_keys, req_combo)}
            base_prov = {k: c.provenance_dict() for k, c in zip(req_keys, req_combo)}
            range_fail = False
            for k, c in zip(req_keys, req_combo):
                if not _in_range(c.value, props.get(k) or {}):
                    range_fail = True
                    reasons.append("range")
                    break
            if range_fail:
                continue
            for opt_combo in combos_opt:
                arguments = dict(base_args)
                provenance = dict(base_prov)
                skip_combo = False
                for k, c in zip(opt_keys, opt_combo):
                    if not _in_range(c.value, props.get(k) or {}):
                        skip_combo = True
                        reasons.append("range")
                        break
                    arguments[k] = c.value
                    provenance[k] = c.provenance_dict()
                if skip_combo:
                    continue
                if _product_illegal(name, arguments):
                    reasons.append("conflict")
                    continue
                drafts.append((t_i, name, arguments, provenance))

    if overflow:
        return RouteManifest(
            routes=[],
            refuse_reason="manifest_overflow",
            schema_hash=sch,
            entity_snapshot_hash=snap,
            overflow=True,
        )

    uniq: dict[str, tuple[int, str, dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for item in drafts:
        key = dumps_canonical({"name": item[1], "arguments": item[2]})
        uniq.setdefault(key, item)
    ordered = sorted(uniq.values(), key=lambda it: (it[0], dumps_canonical(it[2])))
    if len(ordered) > MAX_ROUTES:
        return RouteManifest(
            routes=[],
            refuse_reason="manifest_overflow",
            schema_hash=sch,
            entity_snapshot_hash=snap,
            overflow=True,
        )

    routes = [
        Route(route_id=i, name=name, arguments=args, provenance=prov, tool_index=ti)
        for i, (ti, name, args, prov) in enumerate(ordered)
    ]
    refuse = None
    if not routes:
        for tag in ("conflict", "ambiguous", "range", "missing", "nomatch"):
            if tag in reasons or tag in ctx.conflict_tags:
                refuse = tag
                break
        refuse = refuse or ("nomatch" if not any(tool_intent_spans(ctx, t) for t in tools) else "missing")
    return RouteManifest(
        routes=routes,
        refuse_reason=refuse,
        schema_hash=sch,
        entity_snapshot_hash=snap,
    )


def gold_route_id(manifest: RouteManifest, call: dict[str, Any] | None) -> int | None:
    if not call:
        return None
    want = dumps_canonical({"name": call.get("name"), "arguments": call.get("arguments") or {}})
    hits = [
        r.route_id
        for r in manifest.routes
        if dumps_canonical({"name": r.name, "arguments": r.arguments}) == want
    ]
    if len(hits) != 1:
        return None
    return hits[0]
