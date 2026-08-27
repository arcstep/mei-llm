#!/usr/bin/env python3
"""Oracle top-5 full-call / validator / runtime interface smoke. Not product quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repo_paths import ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from architecture import NeedleZh
from config import NeedleZhConfig
from provenance_validator_v2 import validate_generated_call
from runtime_v2 import RuntimeV2
from schema_render import load_toolset_json
from tokenizer import ZhTokenizerV1
from tool_call_protocol_v2 import dump_call, dump_empty
from tool_index import ToolIndex

import mlx.core as mx


def main() -> int:
    tools = load_toolset_json("needle-home-v0")["tools"]
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True, confidence_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    rt = RuntimeV2(model, tok, catalog=tools, index=ToolIndex(), confidence_threshold=0.0)
    cases = [
        {
            "query": "帮我查一下成都今天的天气",
            "gold": dump_call("get_weather", {"city": "成都"}),
            "expect_execute_gold": True,
        },
        {"query": "你好", "gold": dump_empty(), "expect_execute_gold": False},
        {
            "query": "查天气",
            "gold": dump_call("get_weather", {"city": "火星"}),
            "expect_execute_gold": False,
        },
    ]
    rows = []
    for case in cases:
        complete = rt.complete(case["query"], oracle_tools=tools)
        validated_gold = validate_generated_call(case["gold"], tools=tools, query=case["query"])
        rows.append(
            {
                "query": case["query"],
                "complete_keys": sorted(complete.keys()),
                "has_validated": "validated" in complete,
                "gold_ok": bool(validated_gold.get("ok")),
                "gold_execute": bool(validated_gold.get("function_calls")) and not validated_gold.get("refuse"),
                "gold_matches_expect": (
                    bool(validated_gold.get("function_calls")) and not validated_gold.get("refuse")
                )
                == bool(case["expect_execute_gold"]),
                "prompt_leaks": complete.get("prompt_leaks") or [],
            }
        )
    ok = all(r["has_validated"] and not r["prompt_leaks"] and r["gold_matches_expect"] for r in rows)
    out = {"ok": ok, "n": len(rows), "rows": rows, "interface": ["complete", "run", "oracle_tools"]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
