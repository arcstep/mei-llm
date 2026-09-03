from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .canonical import compact_tools, dumps_canonical, schema_fingerprint
from .json_schema_lite import ContractSchemaError, loads_strict, validate_instance
from .shared import (
    DEFAULT_OUTPUT_RESERVE,
    MAX_CONTEXT,
    PROFILE_STABLE_CAPS,
    clip_head_tail,
    project_tool_batch,
    token_count,
    validate_tools,
)
from .version import SPEC_DIR, load_json

_PROTO = load_json("protocol.json")
PROTOCOL_ID = str(_PROTO["protocol_id"])
SERIALIZER_ID = str(_PROTO["serializer_id"])
MAX_SELECTED_TOOLS = int(_PROTO["max_selected_tools"])
MAX_CALLS = int(_PROTO["max_calls"])
TASK_CONTRACT = str(_PROTO["task_contract"])
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
FORBIDDEN_MARKERS = tuple(_PROTO["forbidden_markers"])
WIRE_V1 = "mei-runtime-wire-v1"
WIRE_V2 = "mei-runtime-wire-v2"
DEFAULT_MAX_STEPS = 4
HARD_MAX_STEPS = 8
DEFAULT_MAX_OUTPUT_TOKENS = 128
MAX_TOOL_RESULT_BYTES = 64 * 1024
_REQUEST_V2_SCHEMA: dict[str, Any] | None = None
_REQUEST_V2_REGISTRY: dict[str, dict[str, Any]] | None = None
_TOOL_RESULT_V2_SCHEMA: dict[str, Any] | None = None


def _validate_request_v2(request: dict[str, Any]) -> None:
    global _REQUEST_V2_SCHEMA, _REQUEST_V2_REGISTRY
    if _REQUEST_V2_SCHEMA is None or _REQUEST_V2_REGISTRY is None:
        try:
            request_schema = loads_strict(
                (SPEC_DIR / "request-v2.schema.json").read_text(encoding="utf-8")
            )
            tool_result_schema = loads_strict(
                (SPEC_DIR / "tool-result-v2.schema.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load CompleteRequestV2 schema: {exc}") from exc
        if not isinstance(request_schema, dict) or not isinstance(tool_result_schema, dict):
            raise ValueError("wire v2 schemas must be JSON objects")
        _REQUEST_V2_SCHEMA = request_schema
        _REQUEST_V2_REGISTRY = {"tool-result-v2.schema.json": tool_result_schema}
    try:
        validate_instance(request, _REQUEST_V2_SCHEMA, registry=_REQUEST_V2_REGISTRY)
    except ContractSchemaError as exc:
        raise ValueError(f"CompleteRequestV2 JSON Schema: {exc}") from exc


def validate_tool_result_v2(result: dict[str, Any]) -> None:
    """Validate the exact public ``ToolResultV2`` contract fail closed.

    This deliberately uses the checked-in schema rather than a second set of
    hand-written conditionals.  In particular, an ``error`` result must carry
    ``{code,message}``, while an ``ok`` result may not smuggle a non-null error.
    """

    global _TOOL_RESULT_V2_SCHEMA
    if not isinstance(result, dict):
        raise ValueError("ToolResultV2 must be an object")
    if _TOOL_RESULT_V2_SCHEMA is None:
        try:
            loaded = loads_strict(
                (SPEC_DIR / "tool-result-v2.schema.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load ToolResultV2 schema: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("ToolResultV2 schema root must be an object")
        _TOOL_RESULT_V2_SCHEMA = loaded
    try:
        validate_instance(result, _TOOL_RESULT_V2_SCHEMA)
    except ContractSchemaError as exc:
        raise ValueError(f"ToolResultV2 JSON Schema: {exc}") from exc


def leak_markers(text: str) -> list[str]:
    return [m for m in FORBIDDEN_MARKERS if m in (text or "")]


def request_leaks(request: dict[str, Any]) -> list[str]:
    chunks = [
        str(request.get("query") or ""),
        str(request.get("system_facts") or ""),
        str(request.get("selected_entity") or ""),
        str(request.get("candidate_text") or ""),
        dumps_canonical(request.get("history") or []),
        dumps_canonical(request.get("prior_tool_results") or []),
        dumps_canonical(request.get("catalog") or []),
        dumps_canonical(request.get("oracle_tools") or []),
        dumps_canonical(request.get("permissions") or []),
        dumps_canonical(request.get("context") or {}),
        dumps_canonical(request.get("evidence") or []),
        dumps_canonical(request.get("entities") or []),
        dumps_canonical(request.get("tool_results") or []),
        dumps_canonical(request.get("state") or {}),
        dumps_canonical(request.get("mw") or {}),
        dumps_canonical(request.get("mw_disposition") or {}),
        dumps_canonical(request.get("confidence") or {}),
    ]
    found: list[str] = []
    blob = "\n".join(chunks)
    for marker in FORBIDDEN_MARKERS:
        if marker in blob and marker not in found:
            found.append(marker)
    return found


def parse_v2_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "function_calls": [], "error": "empty", "refuse": True}
    try:
        obj = loads_strict(raw)
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "function_calls": [], "error": "json", "refuse": True}
    if obj == []:
        return {"ok": True, "function_calls": [], "error": None, "refuse": True}
    if not isinstance(obj, list) or len(obj) > MAX_CALLS:
        return {"ok": False, "function_calls": [], "error": "illegal_shape", "refuse": True}
    if len(obj) != 1 or not isinstance(obj[0], dict):
        return {"ok": False, "function_calls": [], "error": "illegal_shape", "refuse": True}
    name = obj[0].get("name")
    args = obj[0].get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return {"ok": False, "function_calls": [], "error": "illegal_item", "refuse": True}
    return {
        "ok": True,
        "function_calls": [{"name": name, "arguments": args}],
        "error": None,
        "refuse": False,
    }


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    return "<tools>" + dumps_canonical(compact_tools(tools)) + "</tools>"


def _render_ordinary(normalized: dict[str, Any], *, is_v1_adapter: bool) -> str:
    ordinary_parts: list[str] = []
    for turn in normalized.get("history") or []:
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            ordinary_parts.append(f"{role}：{content}")
    for result in normalized.get("tool_results") or normalized.get("prior_tool_results") or []:
        ordinary_parts.append(
            "tool：" + (dumps_canonical(result) if isinstance(result, dict) else str(result))
        )
    if not is_v1_adapter:
        for tag, value in (
            ("context", normalized.get("context") or {}),
            ("evidence", normalized.get("evidence") or []),
            ("permissions", normalized.get("permissions") or {}),
            ("state", normalized.get("state") or {}),
            ("mw", normalized.get("mw") or normalized.get("mw_disposition") or {}),
        ):
            ordinary_parts.append(f"<{tag}>" + dumps_canonical(value) + f"</{tag}>")
    ordinary_parts.append("user：" + str(normalized.get("query") or "").strip())
    return "\n".join(ordinary_parts)


def _minimal_tool_result(result: Any, *, keep_payload: bool) -> Any:
    if not isinstance(result, dict):
        return result if keep_payload else "…已裁剪早期工具结果…"
    kept = {
        key: deepcopy(result[key])
        for key in ("wire_version", "call_id", "status", "provenance", "error")
        if key in result
    }
    if keep_payload and "payload" in result:
        kept["payload"] = deepcopy(result["payload"])
    elif "payload" in result:
        kept["payload"] = {"_mei_omitted": "budget"}
    return kept


def _evidence_priority(value: Any, index: int) -> tuple[float, int]:
    if isinstance(value, dict):
        priority = value.get("priority", 0)
        if isinstance(priority, (int, float)) and not isinstance(priority, bool):
            return float(priority), index
    return 0.0, index


def _fit_ordinary_to_prompt(
    tokenizer: Any,
    normalized: dict[str, Any],
    *,
    sink: str,
    prompt_cap: int,
    prompt_suffix: str = "",
) -> dict[str, Any] | None:
    """Fit model-visible request data while retaining the trusted full request."""

    visible = deepcopy(normalized)
    is_v1_adapter = (normalized.get("_compat") or {}).get("source_wire") == WIRE_V1
    original_ordinary = _render_ordinary(visible, is_v1_adapter=is_v1_adapter)
    original_prompt = sink + "\n" + original_ordinary + prompt_suffix
    original_tokens = token_count(tokenizer, original_prompt, add_bos=True)
    stats: dict[str, Any] = {
        "cap": int(prompt_cap),
        "original_prompt_tokens": original_tokens,
        "dropped_history": 0,
        "compacted_tool_results": [],
        "dropped_evidence": 0,
        "dropped_context_keys": [],
        "query_clipped": False,
    }

    def snapshot() -> tuple[str, str, int]:
        ordinary = _render_ordinary(visible, is_v1_adapter=is_v1_adapter)
        prompt = sink + "\n" + ordinary + prompt_suffix
        return ordinary, prompt, token_count(tokenizer, prompt, add_bos=True)

    ordinary, prompt, used = snapshot()
    if used <= prompt_cap:
        stats.update({"used": used, "compression_level": "none"})
        return {
            "ordinary": ordinary,
            "prompt": prompt,
            "prompt_tokens": used,
            "visible_request": visible,
            "input_budget": stats,
        }

    history = list(visible.get("history") or [])
    while history and used > prompt_cap:
        history.pop(0)
        stats["dropped_history"] += 1
        visible["history"] = history
        ordinary, prompt, used = snapshot()

    results_key = "tool_results" if visible.get("tool_results") is not None else "prior_tool_results"
    results = list(visible.get(results_key) or [])
    for index in range(max(0, len(results) - 1)):
        if used <= prompt_cap:
            break
        results[index] = _minimal_tool_result(results[index], keep_payload=False)
        stats["compacted_tool_results"].append(
            str(results[index].get("call_id") or index)
            if isinstance(results[index], dict)
            else str(index)
        )
        visible[results_key] = results
        ordinary, prompt, used = snapshot()

    evidence = list(visible.get("evidence") or [])
    while evidence and used > prompt_cap:
        remove_index = min(
            range(len(evidence)), key=lambda index: _evidence_priority(evidence[index], index)
        )
        evidence.pop(remove_index)
        stats["dropped_evidence"] += 1
        visible["evidence"] = evidence
        ordinary, prompt, used = snapshot()

    context = dict(visible.get("context") or {})
    while context and used > prompt_cap:
        key = min(
            context,
            key=lambda name: (
                float(context[name].get("priority", 0))
                if isinstance(context[name], dict)
                and isinstance(context[name].get("priority", 0), (int, float))
                and not isinstance(context[name].get("priority", 0), bool)
                else 0.0,
                str(name).encode("utf-8"),
            ),
        )
        del context[key]
        stats["dropped_context_keys"].append(str(key))
        visible["context"] = context
        ordinary, prompt, used = snapshot()

    # The latest trusted result retains identity, status and provenance.  Its
    # potentially large payload is the last result payload eligible for
    # replacement, after old history/context have already been removed.
    if results and used > prompt_cap:
        results[-1] = _minimal_tool_result(results[-1], keep_payload=False)
        identifier = (
            str(results[-1].get("call_id") or len(results) - 1)
            if isinstance(results[-1], dict)
            else str(len(results) - 1)
        )
        if identifier not in stats["compacted_tool_results"]:
            stats["compacted_tool_results"].append(identifier)
        visible[results_key] = results
        ordinary, prompt, used = snapshot()

    if used > prompt_cap:
        original_query = str(visible.get("query") or "")
        query_ids = token_count(tokenizer, original_query)
        low, high = 0, query_ids
        best: tuple[str, str, str, int] | None = None
        while low <= high:
            middle = (low + high) // 2
            clipped = clip_head_tail(tokenizer, original_query, middle)
            visible["query"] = clipped
            next_ordinary, next_prompt, next_used = snapshot()
            if next_used <= prompt_cap:
                best = (clipped, next_ordinary, next_prompt, next_used)
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            visible["query"], ordinary, prompt, used = best
            stats["query_clipped"] = visible["query"] != original_query
        else:
            visible["query"] = original_query

    if used > prompt_cap:
        return None
    stats.update(
        {
            "used": used,
            "compression_level": (
                "query"
                if stats["query_clipped"]
                else "context"
                if stats["dropped_context_keys"] or stats["dropped_evidence"]
                else "tool_results"
                if stats["compacted_tool_results"]
                else "history"
            ),
            "visible_query_tokens": token_count(tokenizer, str(visible.get("query") or "")),
        }
    )
    return {
        "ordinary": ordinary,
        "prompt": prompt,
        "prompt_tokens": used,
        "visible_request": visible,
        "input_budget": stats,
    }


def render_budgeted_request(
    request: dict[str, Any],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    *,
    relevances: list[float] | None = None,
    runtime_profile: str = "standard",
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    already_normalized: bool = False,
    task_contract: str = TASK_CONTRACT,
    prompt_suffix: str = ASSISTANT_SUFFIX,
) -> dict[str, Any]:
    """Render one fixed-five batch under the unified 2048-token contract.

    A lower-ranked tool removed here is not a retrieval rejection; the caller
    must place it back in the internal candidate-scan queue.
    """

    if len(tools) > MAX_SELECTED_TOOLS:
        raise ValueError("a candidate batch may contain at most 5 selected tools")
    if runtime_profile not in PROFILE_STABLE_CAPS:
        raise ValueError("runtime_profile must be compact or standard")
    reserve = int(output_reserve)
    if reserve < 1 or reserve >= MAX_CONTEXT:
        raise ValueError("output_reserve must be in 1..2047")
    validate_tools(tools)
    normalized = dict(request) if already_normalized else normalize_request(request)
    prompt_cap = MAX_CONTEXT - reserve
    current_tools = list(tools)
    current_relevances = list(relevances or [1.0] * len(current_tools))
    if len(current_relevances) != len(current_tools):
        raise ValueError("one retrieval relevance is required per selected tool")
    all_dropped: list[str] = []

    while current_tools:
        projection = project_tool_batch(
            tokenizer,
            current_tools,
            relevances=current_relevances,
            profile=runtime_profile,
            stable_overhead=str(task_contract) + "\n",
        )
        selected_count = len(projection.full_tools)
        all_dropped.extend(projection.dropped_tool_ids)
        if projection.context_unrepresentable or selected_count == 0:
            current_tools = []
            break
        # Projection may already have removed structural-heavy tail tools.
        current_tools = current_tools[:selected_count]
        current_relevances = current_relevances[:selected_count]
        fitted = _fit_ordinary_to_prompt(
            tokenizer,
            normalized,
            sink=projection.rendered,
            prompt_cap=prompt_cap,
            prompt_suffix=prompt_suffix,
        )
        if fitted is not None:
            full_tools = list(projection.full_tools)
            return {
                **fitted,
                "sink": projection.rendered,
                "schema_fingerprint": schema_fingerprint(full_tools),
                "schema_projection_sha256": projection.projection_sha256,
                "schema_budget": {
                    **projection.as_budget_dict(),
                    "dropped_tools": all_dropped,
                },
                "serializer_id": SERIALIZER_ID,
                "protocol_id": PROTOCOL_ID,
                "selected_tools": [str(tool.get("name") or "") for tool in full_tools],
                "request": normalized,
                "compatibility": normalized.get("_compat"),
                "_validation_tools": full_tools,
                "_projected_tools": list(projection.tools),
            }
        dropped = str(current_tools[-1].get("name") or "")
        all_dropped.append(dropped)
        current_tools = current_tools[:-1]
        current_relevances = current_relevances[:-1]

    return {
        "prompt": None,
        "sink": None,
        "ordinary": None,
        "prompt_tokens": 0,
        "schema_fingerprint": None,
        "schema_projection_sha256": None,
        "schema_budget": {
            "profile": runtime_profile,
            "cap": PROFILE_STABLE_CAPS[runtime_profile],
            "used": None,
            "compression_level": "unrepresentable",
            "dropped_tools": all_dropped,
            "context_unrepresentable": True,
        },
        "input_budget": {"cap": prompt_cap, "used": None},
        "serializer_id": SERIALIZER_ID,
        "protocol_id": PROTOCOL_ID,
        "selected_tools": [],
        "request": normalized,
        "compatibility": normalized.get("_compat"),
        "_validation_tools": [],
        "_projected_tools": [],
        "error": "context_unrepresentable",
    }


def adapt_v1_request(request: dict[str, Any]) -> dict[str, Any]:
    """Map the read-only v1 request shape into a degraded in-memory v2 view."""
    # Caller-supplied underscore fields are internal trust-domain state, not a
    # legacy extension surface.  In particular, accepting a forged verified
    # result map would let an untrusted ToolResult manufacture provenance.
    out = {key: value for key, value in request.items() if not str(key).startswith("_")}
    context = dict(out.get("context") or {})
    if out.get("system_facts") is not None:
        context.setdefault("system_facts", str(out.get("system_facts") or ""))
    if out.get("selected_entity") is not None:
        entities = list(out.get("entities") or [])
        entities.append(
            {
                "id": "v1:selected_entity",
                "canonical": out.get("selected_entity"),
                "verified": True,
            }
        )
        out["entities"] = entities
    out["context"] = context
    if "tool_results" not in out and "prior_tool_results" in out:
        out["tool_results"] = list(out.get("prior_tool_results") or [])
    out["wire_version"] = WIRE_V2
    out["_compat"] = {
        "source_wire": WIRE_V1,
        "mode": "read_only_degraded_adapter",
        "capability_complete": False,
    }
    return out


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    wire = request.get("wire_version")
    if wire == WIRE_V2:
        _validate_request_v2(request)
    out = adapt_v1_request(request) if wire in (None, "", WIRE_V1) else dict(request)
    if out.get("wire_version") != WIRE_V2:
        raise ValueError(f"unsupported wire_version: {out.get('wire_version')}")
    if not isinstance(out.get("context") or {}, dict):
        raise ValueError("context must be an object")
    for key in ("history", "evidence", "entities", "tool_results"):
        if not isinstance(out.get(key) or [], list):
            raise ValueError(f"{key} must be an array")
    if not isinstance(out.get("state") or {}, dict):
        raise ValueError("state must be an object")
    permissions = out.get("permissions") or {}
    if (
        wire == WIRE_V2
        and isinstance(permissions, dict)
        and "scopes" in permissions
        and "grants" in permissions
    ):
        raise ValueError("permissions cannot contain both scopes and grants")
    # Python callers do not necessarily cross a JSON parser.  Audit the same
    # finite, string-keyed value domain used by native wire implementations.
    dumps_canonical(out)
    return out


def render_request(
    request: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    already_normalized: bool = False,
) -> dict[str, Any]:
    if len(tools) > MAX_SELECTED_TOOLS:
        raise ValueError("complete() accepts at most 5 selected tools; retrieval must run first")
    validate_tools(tools)
    normalized = dict(request) if already_normalized else normalize_request(request)
    context = normalized.get("context") or {}
    is_v1_adapter = (normalized.get("_compat") or {}).get("source_wire") == WIRE_V1
    sink = TASK_CONTRACT
    if is_v1_adapter:
        facts = str(context.get("system_facts") or normalized.get("system_facts") or "").strip()
        if facts:
            sink += "\n系统事实：" + facts
        permissions = normalized.get("permissions") or []
        if permissions:
            if isinstance(permissions, dict):
                permission_text = dumps_canonical(permissions)
            else:
                permission_text = "、".join(str(p) for p in permissions)
            sink += "\n权限：" + permission_text
        selected_entity = normalized.get("selected_entity")
        sink += "\n点选实体：" + (str(selected_entity) if selected_entity else "无")
    sink += "\n" + render_tools_block(tools)
    ordinary = _render_ordinary(normalized, is_v1_adapter=is_v1_adapter)
    prompt = sink + "\n" + ordinary
    return {
        "prompt": prompt,
        "sink": sink,
        "ordinary": ordinary,
        "schema_fingerprint": schema_fingerprint(tools),
        "serializer_id": SERIALIZER_ID,
        "protocol_id": PROTOCOL_ID,
        "selected_tools": [str(t.get("name") or "") for t in tools],
        "request": normalized,
        "compatibility": normalized.get("_compat"),
    }
