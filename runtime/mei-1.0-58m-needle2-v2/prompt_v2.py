"""v2 prompt: contract + top-5 schemas as sinks; query/history as ordinary tokens."""

from __future__ import annotations

import hashlib
from typing import Any

from schema_render import compact_tools, dumps_canonical

SERIALIZER_ID = "mei-tool-call-serializer-v2"
TASK_CONTRACT_V2 = (
    "任务：只输出一个 schema 合法的工具 JSON 数组，或 []。"
    "最多一次调用。缺少 required 证据时输出 []。禁止输出解释或 route_id。"
)
FORBIDDEN_MARKERS = (
    "<routes>",
    "gold_route_id",
    "gold_provenance",
    "compiled_call_candidates",
    "gold_entity_link",
    "holdout_only_relation",
    "scenario_id",
)
PROMPT_CHAR_SOFT_MAX = 4096  # ~2048 tokens; compress schema/facts instead of raising the window


def schema_fingerprint(toolset: dict[str, Any]) -> str:
    payload = dumps_canonical(compact_tools(toolset))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    compact = compact_tools({"tools": tools} if tools and "name" in tools[0] else {"tools": tools})
    if tools and isinstance(tools[0], dict) and "parameters" in tools[0] and "name" in tools[0]:
        compact = compact_tools({"tools": tools})
    return "<tools>" + dumps_canonical(compact) + "</tools>"


def leak_markers(text: str) -> list[str]:
    return [m for m in FORBIDDEN_MARKERS if m in (text or "")]


def _facts_compact(system_facts: str) -> str:
    text = str(system_facts).strip()
    if len(text) <= 280:
        return text
    return text[:277] + "…"


def render_v2_request(
    *,
    tools: list[dict[str, Any]],
    query: str,
    system_facts: str | None = None,
    history: list[dict[str, str]] | None = None,
    prior_tool_results: list[str] | None = None,
    entities: dict[str, str] | None = None,
    selected_entity: str | None = None,
    permissions: list[str] | None = None,
) -> dict[str, Any]:
    if len(tools) > 5:
        raise ValueError("v2 prompt accepts at most 5 tools; retrieval must run first")
    facts = str(system_facts).strip() if system_facts and str(system_facts).strip() else ""
    sink = TASK_CONTRACT_V2
    if facts:
        sink += "\n系统事实：" + facts
    if permissions:
        sink += "\n权限：" + "、".join(str(p) for p in permissions)
    if entities:
        names = "、".join(str(k) for k in entities.keys())
        sink += "\n设备目录：" + names + "（目录项不是点选证据）"
    sink += "\n点选实体：" + (str(selected_entity) if selected_entity else "无")
    sink += "\n" + render_tools_block(tools)
    ordinary_parts: list[str] = []
    for turn in history or []:
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            ordinary_parts.append(f"{role}：{content}")
    for result in prior_tool_results or []:
        ordinary_parts.append("<tool_result>" + str(result) + "</tool_result>")
    ordinary_parts.append("用户：" + (query or "").strip())
    ordinary = "\n".join(ordinary_parts)
    text = sink + "\n" + ordinary
    if len(text) > PROMPT_CHAR_SOFT_MAX and facts:
        sink = sink.replace("系统事实：" + facts, "系统事实：" + _facts_compact(facts), 1)
        text = sink + "\n" + ordinary
    leaks = leak_markers(text)
    n_chars = len(text)
    return {
        "serializer": SERIALIZER_ID,
        "sink_text": sink,
        "ordinary_text": ordinary,
        "text": text,
        "n_tools": len(tools),
        "schema_fingerprint": schema_fingerprint({"tools": tools}),
        "n_chars": n_chars,
        "prompt_tokens_est": (n_chars + 1) // 2,
        "leaks": leaks,
        "ok": not leaks,
        "window_ok": n_chars <= PROMPT_CHAR_SOFT_MAX,
    }


def encode_v2_segments(tokenizer, rendered: dict[str, Any]) -> dict[str, Any]:
    sink_ids = tokenizer.encode(rendered["sink_text"], add_bos=True, add_eos=False)
    ordinary_ids = tokenizer.encode("\n" + rendered["ordinary_text"] + "\n<|im_end|>\n<|im_start|>assistant\n", add_bos=False, add_eos=False)
    prompt_ids = sink_ids + ordinary_ids
    return {
        "prompt_ids": prompt_ids,
        "sink_ids": sink_ids,
        "ordinary_ids": ordinary_ids,
        "sink_len": len(sink_ids),
        "ordinary_len": len(ordinary_ids),
        "n_prompt": len(prompt_ids),
    }
