"""Per-request candidate extraction with provenance. Enum membership is not evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from normalizers import (
    NORMALIZER_VERSION,
    coerce_schema_value,
    find_booleans,
    find_currency,
    find_numbers,
)

MAX_CANDIDATES_PER_ARG = 8
NAME_TO_TYPE = {
    "city": "city",
    "room": "room",
    "place": "place",
    "vendor": "vendor",
    "sku": "sku",
    "sku_name": "sku",
    "item": "item",
    "title": "title",
    "topic": "title",
    "currency": "currency",
    "id": "device",
    "target": "target",
    "text": "token",
    "shop": "shop",
    "dish": "dish",
}


@dataclass(frozen=True)
class CandidateValue:
    value: Any
    evidence_source: str
    evidence_span: tuple[int, int] | None
    evidence_text: str
    resolution_source: str
    normalizer_id: str | None = None
    normalizer_version: str | None = None
    entity_id: str | None = None
    ambiguous: bool = False

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "evidence_source": self.evidence_source,
            "evidence_span": list(self.evidence_span) if self.evidence_span else None,
            "evidence_text": self.evidence_text,
            "canonical_value": self.value,
            "resolution_source": self.resolution_source,
            "normalizer_id": self.normalizer_id,
            "normalizer_version": self.normalizer_version,
            "entity_id": self.entity_id,
        }


@dataclass
class ToolContext:
    query: str
    toolset: dict[str, Any]
    scene: str | None = None
    entities: list[dict[str, Any]] = field(default_factory=list)
    lexicon: dict[str, list[str]] = field(default_factory=dict)
    conflict_tags: list[str] = field(default_factory=list)
    param_types: dict[str, str] = field(default_factory=dict)

    @property
    def query_scene(self) -> tuple[str, str]:
        return self.query or "", self.scene or ""


def load_entity_catalog(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        from _repo import EVAL_SHARED_ROOT
        path = EVAL_SHARED_ROOT / "entities" / "mei-grounded-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def entities_from_mode(mode: str) -> list[dict[str, Any]]:
    cat = load_entity_catalog()
    if mode == "all":
        return list(cat["entities"])
    return [e for e in cat["entities"] if e.get("split") == "train"]


def load_lexicon(path: Path | None = None) -> dict[str, list[str]]:
    if path is None:
        from _repo import EVAL_SHARED_ROOT
        path = EVAL_SHARED_ROOT / "lexicon" / "mei-tool-intents-v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    tools = raw.get("tools") if isinstance(raw, dict) else raw
    return {str(k): [str(x) for x in (v or [])] for k, v in (tools or {}).items()}


def entity_aliases(ent: dict[str, Any]) -> list[str]:
    names = [str(ent.get("canonical") or "")]
    names.extend(str(a) for a in (ent.get("aliases") or []) if a)
    names = [n for n in names if n]
    names.sort(key=len, reverse=True)
    return names


def find_substrings(hay: str, needle: str) -> list[tuple[int, int]]:
    if not hay or not needle:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            break
        out.append((i, i + len(needle)))
        start = i + max(1, len(needle))
    return out


def semantic_type(tool_name: str, param: str, prop: dict[str, Any], param_types: dict[str, str]) -> str | None:
    key = f"{tool_name}.{param}"
    if key in param_types:
        return param_types[key]
    if param in NAME_TO_TYPE:
        return NAME_TO_TYPE[param]
    enum = prop.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(x, str) for x in enum):
        return "enum"
    return None


def _in_range(value: Any, prop: dict[str, Any]) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return True
    if "minimum" in prop and value < prop["minimum"]:
        return False
    if "maximum" in prop and value > prop["maximum"]:
        return False
    return True


def _enum_ok(value: Any, prop: dict[str, Any]) -> bool:
    enum = prop.get("enum")
    if not enum:
        return True
    return value in enum


def tool_intent_spans(ctx: ToolContext, tool: dict[str, Any]) -> list[tuple[str, tuple[int, int], str]]:
    name = str(tool.get("name") or "")
    phrases = list(ctx.lexicon.get(name) or [])
    phrases.append(name)
    desc = str(tool.get("description") or "")
    if desc:
        phrases.append(desc[:12])
    phrases = sorted({p for p in phrases if p}, key=len, reverse=True)
    hits: list[tuple[str, tuple[int, int], str]] = []
    for source, text in (("query", ctx.query or ""), ("scene", ctx.scene or "")):
        occupied: list[tuple[int, int]] = []
        for phrase in phrases:
            for span in find_substrings(text, phrase):
                if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                    continue
                occupied.append(span)
                hits.append((source, span, phrase))
    return hits


def _entity_hits(text: str, source: str, entities: list[dict[str, Any]], want_type: str) -> list[CandidateValue]:
    raw: list[CandidateValue] = []
    for ent in entities:
        if str(ent.get("type") or "") != want_type:
            continue
        for alias in entity_aliases(ent):
            for span in find_substrings(text, alias):
                raw.append(
                    CandidateValue(
                        value=ent.get("canonical"),
                        evidence_source=source,
                        evidence_span=span,
                        evidence_text=alias,
                        resolution_source="entity_alias" if alias != ent.get("canonical") else "exact_span",
                        entity_id=str(ent.get("entity_id") or ""),
                    )
                )
    by_span: dict[tuple[int, int], list[CandidateValue]] = {}
    for c in raw:
        by_span.setdefault(c.evidence_span or (-1, -1), []).append(c)
    out: list[CandidateValue] = []
    for span, group in by_span.items():
        ids = {c.entity_id for c in group}
        if len(ids) > 1:
            out.append(
                CandidateValue(
                    value=None,
                    evidence_source=group[0].evidence_source,
                    evidence_span=span,
                    evidence_text=group[0].evidence_text,
                    resolution_source="ambiguous_entity",
                    ambiguous=True,
                )
            )
            continue
        best = sorted(group, key=lambda c: -(c.evidence_span[1] - c.evidence_span[0] if c.evidence_span else 0))[0]
        out.append(best)
    return out


def _token_hits(text: str, source: str, allowed: list[str]) -> list[CandidateValue]:
    out: list[CandidateValue] = []
    for tok in sorted({str(x) for x in allowed if x}, key=len, reverse=True):
        for span in find_substrings(text, tok):
            out.append(
                CandidateValue(
                    value=tok,
                    evidence_source=source,
                    evidence_span=span,
                    evidence_text=tok,
                    resolution_source="exact_span",
                )
            )
    return out


def candidates_for_param(ctx: ToolContext, tool: dict[str, Any], param: str, prop: dict[str, Any]) -> list[CandidateValue]:
    schema_type = str(prop.get("type") or "string")
    found: list[CandidateValue] = []
    texts = (("query", ctx.query or ""), ("scene", ctx.scene or ""))
    sem = semantic_type(str(tool.get("name") or ""), param, prop, ctx.param_types)

    if schema_type == "boolean":
        for source, text in texts:
            for hit in find_booleans(text):
                found.append(
                    CandidateValue(
                        value=hit.value,
                        evidence_source=source,
                        evidence_span=(hit.start, hit.end),
                        evidence_text=hit.raw,
                        resolution_source="normalizer",
                        normalizer_id=hit.normalizer_id,
                        normalizer_version=NORMALIZER_VERSION,
                    )
                )
        extra_bool = []
        if param == "all_day":
            extra_bool.append(("全天", True))
        if param in {"outdoor", "all_day"}:
            extra_bool.append(("户外", True))
        for phrase, val in extra_bool:
            for source, text in texts:
                for span in find_substrings(text, phrase):
                    found.append(
                        CandidateValue(
                            value=val,
                            evidence_source=source,
                            evidence_span=span,
                            evidence_text=phrase,
                            resolution_source="normalizer",
                            normalizer_id="zh_bool",
                            normalizer_version=NORMALIZER_VERSION,
                        )
                    )
    elif schema_type in {"integer", "number"}:
        for source, text in texts:
            for hit in find_numbers(text):
                try:
                    val = coerce_schema_value(hit.value, schema_type)
                except ValueError:
                    continue
                if not _in_range(val, prop):
                    found.append(
                        CandidateValue(
                            value=val,
                            evidence_source=source,
                            evidence_span=(hit.start, hit.end),
                            evidence_text=hit.raw,
                            resolution_source="normalizer",
                            normalizer_id=hit.normalizer_id,
                            normalizer_version=NORMALIZER_VERSION,
                            ambiguous=False,
                        )
                    )
                    # keep out-of-range as a candidate so compiler can fail-closed on range
                    continue
                found.append(
                    CandidateValue(
                        value=val,
                        evidence_source=source,
                        evidence_span=(hit.start, hit.end),
                        evidence_text=hit.raw,
                        resolution_source="normalizer",
                        normalizer_id=hit.normalizer_id,
                        normalizer_version=NORMALIZER_VERSION,
                    )
                )
    else:
        if sem == "currency":
            for source, text in texts:
                for hit in find_currency(text):
                    found.append(
                        CandidateValue(
                            value=hit.value,
                            evidence_source=source,
                            evidence_span=(hit.start, hit.end),
                            evidence_text=hit.raw,
                            resolution_source="normalizer",
                            normalizer_id=hit.normalizer_id,
                            normalizer_version=NORMALIZER_VERSION,
                        )
                    )
        elif sem in {"city", "room", "place", "vendor", "sku", "item", "title", "device", "target", "token", "shop", "dish"}:
            want = "sku" if sem == "item" and any(e.get("type") == "sku" for e in ctx.entities) else sem
            if sem == "item":
                typed = _entity_hits(ctx.query or "", "query", ctx.entities, "item") + _entity_hits(
                    ctx.scene or "", "scene", ctx.entities, "item"
                )
                typed += _entity_hits(ctx.query or "", "query", ctx.entities, "sku") + _entity_hits(
                    ctx.scene or "", "scene", ctx.entities, "sku"
                )
            else:
                typed = _entity_hits(ctx.query or "", "query", ctx.entities, want) + _entity_hits(
                    ctx.scene or "", "scene", ctx.entities, want
                )
            found.extend(typed)
        enum = [x for x in (prop.get("enum") or []) if x is not None]
        if enum:
            for source, text in texts:
                found.extend(_token_hits(text, source, [str(x) for x in enum]))

    if "const" in prop:
        found.append(
            CandidateValue(
                value=prop["const"],
                evidence_source="schema.const",
                evidence_span=None,
                evidence_text="",
                resolution_source="const",
            )
        )
    if "default" in prop and not any(not c.ambiguous for c in found):
        found.append(
            CandidateValue(
                value=prop["default"],
                evidence_source="schema.default",
                evidence_span=None,
                evidence_text="",
                resolution_source="default",
            )
        )

    unique: dict[tuple[str, str, str], CandidateValue] = {}
    for c in found:
        if c.ambiguous:
            unique[("amb", str(c.evidence_span), str(c.evidence_text))] = c
            continue
        if not _enum_ok(c.value, prop):
            continue
        key = (str(c.value), c.evidence_source, str(c.evidence_span))
        prev = unique.get(key)
        if prev is None or (c.evidence_span and prev.evidence_span and (c.evidence_span[1] - c.evidence_span[0]) > (prev.evidence_span[1] - prev.evidence_span[0])):
            unique[key] = c
    return list(unique.values())


def query_scene_conflict(cands: list[CandidateValue]) -> bool:
    q = {c.value for c in cands if (not c.ambiguous) and c.evidence_source == "query"}
    s = {c.value for c in cands if (not c.ambiguous) and c.evidence_source == "scene"}
    return bool(q and s and q != s and not (q & s))
