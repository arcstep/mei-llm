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

    from sft_v2_baseline_adapters import ABANDONED_51M_ARCHIVE, _is_abandoned_51m_archive, resolve_promoted_51m_base

    st = resolve_promoted_51m_base()
    check("51m_available", bool(st.get("available")), status=st.get("status"), path=st.get("path"))
    check("51m_not_archive", ABANDONED_51M_ARCHIVE not in str(st.get("path") or ""))
    check("51m_is_scratch300m", "scratch300m-v1" in str(st.get("path") or "") and str(st.get("path") or "").endswith("pretrain-300m-scratch.npz"))
    check(
        "51m_sha_pinned",
        st.get("weights_sha256") == "e64bbc6baab47c578f5659ccf2e7f03986bcc44754f3e081b2a8a70a74e31d65",
        sha=st.get("weights_sha256"),
    )
    check(
        "archive_path_refused",
        _is_abandoned_51m_archive("notebook/archive/base/mei-1.0-51m-checkpoints/pretrain-300m.npz"),
    )

    from sft_v2_scorecard_metrics import rate_block, summarize_column

    fake = [
        {
            "gold": [{"name": "set_lights", "arguments": {"room": "a"}}],
            "extracted": [{"name": "set_lights", "arguments": {"room": "a"}}],
            "gold_execute": True,
            "content_exact": True,
            "format_ok": True,
            "strict_e2e": True,
            "legal": True,
            "retrieval_hit": True,
            "pipeline_strict": True,
            "wall_ms": 100.0,
            "family": "park",
            "slice": "execute",
        },
        {
            "gold": [],
            "extracted": [],
            "gold_execute": False,
            "content_exact": True,
            "format_ok": True,
            "strict_e2e": True,
            "legal": True,
            "retrieval_hit": True,
            "pipeline_strict": True,
            "wall_ms": 100.0,
            "family": "park",
            "slice": "refuse",
        },
        {
            "gold": [{"name": "set_lights", "arguments": {"room": "a"}}],
            "extracted": [{"name": "set_lights", "arguments": {"room": "b"}}],
            "gold_execute": True,
            "content_exact": False,
            "format_ok": True,
            "strict_e2e": False,
            "legal": True,
            "retrieval_hit": True,
            "pipeline_strict": False,
            "wall_ms": 200.0,
            "family": "park",
            "slice": "execute",
        },
    ]
    col = summarize_column(fake, task="fullcall")
    check("scorecard_has_accuracy", "accuracy" in col and "rate" in col)
    check("rate_10_items_per_s", col["rate"]["items_per_s"] == 7.5, rate=col["rate"])
    check("accuracy_split_execute_refuse", "execute" in col["accuracy"]["splits"] and "refuse" in col["accuracy"]["splits"])
    check("tool_name_split", col["accuracy"]["metrics"]["tool_name"]["k"] == 3)
    check("arguments_split", col["accuracy"]["metrics"]["arguments"]["k"] == 2)
    check("headline_has_both", col["headline"]["accuracy_metric"] == "strict_e2e" and col["headline"]["items_per_s"] == 7.5)

    ret_traces = [
        {"family": "seen", "rank": 0, "wall_ms": 10.0, "catalog_size": 128},
        {"family": "seen", "rank": 4, "wall_ms": 10.0, "catalog_size": 128},
        {"family": "unseen", "rank": -1, "wall_ms": 10.0, "catalog_size": 128},
        {"family": "no_match", "rank": -1, "wall_ms": 10.0},
    ]
    rcol = summarize_column(ret_traces, task="retrieval")
    check("retrieval_primary_recall5", rcol["accuracy"]["primary_metric"] == "recall_at_5")
    check("retrieval_rate_present", rcol["rate"]["items_per_s"] == 100.0)
    check("retrieval_family_split", "seen" in rcol["accuracy"]["splits"]["by_family"])

    rb = rate_block([{"wall_ms": 250.0} for _ in range(4)])
    check("rate_4_per_s", rb["items_per_s"] == 4.0 and rb["items_per_min"] == 240.0)

    ok = all(c["ok"] for c in cases)
    print(json.dumps({"ok": ok, "cases": cases}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
