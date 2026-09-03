"""Compatibility facade over the canonical UTF-8 byte grammar v2."""

from __future__ import annotations

import json
from typing import Any

try:
    from .byte_grammar import (
        compile_byte_grammar_cached,
        is_accept_bytes,
        is_legal_byte_prefix,
        parse_call_text,
    )
except ImportError:
    from byte_grammar import (
        compile_byte_grammar_cached,
        is_accept_bytes,
        is_legal_byte_prefix,
        parse_call_text,
    )


def dump_calls(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, separators=(",", ":"))


def _tools(toolset: Any) -> list[dict[str, Any]]:
    if isinstance(toolset, dict):
        return list(toolset.get("tools") or [])
    if isinstance(toolset, list):
        return list(toolset)
    raise ValueError("toolset must be an object with tools or a tool list")


def is_legal_prefix(text: str, toolset: Any) -> bool:
    tools = _tools(toolset)
    grammar = compile_byte_grammar_cached(tools)
    return is_legal_byte_prefix((text or "").encode("utf-8"), grammar)


def parse_phase1_text(text: str, toolset: Any) -> dict[str, Any]:
    return parse_call_text(text, _tools(toolset))


def is_accept_text(text: str, toolset: Any) -> bool:
    tools = _tools(toolset)
    grammar = compile_byte_grammar_cached(tools)
    return is_accept_bytes((text or "").encode("utf-8"), grammar)
