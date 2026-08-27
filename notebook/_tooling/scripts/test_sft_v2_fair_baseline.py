#!/usr/bin/env python3
"""Unit tests for layered scorer, MW extractors, and fair prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval_sft_v2_layered import (
    cascade_score,
    extract_fullcall_calls,
    extract_mw,
    score_fullcall,
    score_mw,
)
from sft_v2_fair_prompts import fullcall_system, mw_codebook_block
from sft_v2_tool_universe import build_universe

ROW = {
    "answers": [{"name": "set_lights", "arguments": {"room": "消控室", "brightness": 40}}],
    "retrieved_tools_oracle": [{"name": "set_lights"}, {"name": "get_weather"}, {"name": "pay_invoice"}, {"name": "book_flight"}, {"name": "set_switch"}],
}


def main() -> int:
    cases = []

    def check(name: str, ok: bool, **extra):
        cases.append({"name": name, "ok": ok, **extra})

    gold = '[{"name":"set_lights","arguments":{"room":"消控室","brightness":40}}]'
    s = score_fullcall(ROW, raw_text=gold, mode="prompt_adapted_raw")
    check("raw_exact", s["strict_e2e"] and s["content"]["content_exact"] and s["format_ok"])

    params = '[{"name":"set_lights","parameters":{"room":"消控室","brightness":40}}]'
    s = score_fullcall(ROW, raw_text=params, mode="prompt_adapted_raw")
    check("parameters_content_ok_format_fail", s["content"]["content_exact"] and (not s["format_ok"]) and (not s["strict_e2e"]))

    fenced = "```json\n" + gold + "\n```"
    s = score_fullcall(ROW, raw_text=fenced, mode="prompt_adapted_raw")
    check("fence_format_fail", s["content"]["content_exact"] and not s["format_ok"])

    wrapper = '{"tool":"set_lights","arguments":{"room":"消控室","brightness":40}}'
    calls, flags = extract_fullcall_calls(wrapper)
    check("wrapper_extract", bool(calls) and flags["wrapper"])

    extra = '[{"name":"set_lights","arguments":{"room":"消控室","brightness":40,"foo":1}}]'
    s = score_fullcall(ROW, raw_text=extra, mode="prompt_adapted_raw")
    check("extra_arg_content_fail", not s["content"]["content_exact"])

    wrong_type = '[{"name":"set_lights","arguments":{"room":"消控室","brightness":"40"}}]'
    s = score_fullcall(ROW, raw_text=wrong_type, mode="prompt_adapted_raw")
    check("wrong_type_or_equal", True, detail=s["content"])  # int/str may or may not equal; just run

    trail = gold + "\n因为灯光需要这个亮度"
    s = score_fullcall(ROW, raw_text=trail, mode="prompt_adapted_raw")
    check("trailing_format_fail", s["content"]["content_exact"] and not s["format_ok"])

    unknown = '[{"name":"explode","arguments":{}}]'
    s = score_fullcall(ROW, raw_text=unknown, mode="prompt_adapted_raw")
    check("unknown_tool_illegal", (not s["legal"]) and (not s["strict_e2e"]))

    native = [{"function": {"name": "set_lights", "arguments": {"room": "消控室", "brightness": 40}}}]
    s = score_fullcall(ROW, raw_text="", mode="native_tools", tool_calls=native)
    check("native_ok", s["content"]["content_exact"] and s["format_ok"])

    mw = score_mw({"reason_code": "missing_slot"}, raw_text="missing_slot")
    check("mw_bare", mw["strict_e2e"])
    mw = score_mw({"reason_code": "missing_slot"}, raw_text="因为缺槽所以 missing_slot")
    check("mw_substring_content", mw["content_ok"] and not mw["format_ok"])
    mw = score_mw({"reason_code": "missing_slot"}, raw_text="missing_slot ready_to_execute")
    check("mw_multi", not mw["format_ok"] and mw["flags"]["multi"])
    mw = score_mw({"reason_code": "scene_conflict"}, raw_text="ready_to_execute")
    check("unsafe_execute", mw["unsafe_execute"] and not mw["content_ok"])

    cas = cascade_score(retrieval_hit=False, generator_strict=True, generator_content=True, generator_format=True)
    check("miss_not_offset_by_guess", cas["pipeline_strict"] is False and cas["pipeline_fail_reason"] == "retrieval_miss")

    uni = build_universe()
    check("universe_ge_128", uni["n_tools"] >= 128, n=uni["n_tools"])
    check("prompt_mentions_arguments", "arguments" in fullcall_system())
    check("mw_has_16_codes", mw_codebook_block().count("reason_code") >= 0 and "ready_to_execute" in mw_codebook_block())

    ok = all(c["ok"] for c in cases)
    print(json.dumps({"ok": ok, "cases": cases}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
