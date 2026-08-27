#!/usr/bin/env python3
"""Launch Bailian sidecar producers into independent dirs. Never writes qwen-v1."""

from __future__ import annotations

import argparse
import json
import sys

from colloquial_fleet import remaining_for_lane
from repo_paths import CORPORA_ROOT, ROOT, SCRIPTS_ROOT

PRODUCER = SCRIPTS_ROOT / "produce_colloquial_qwen.py"

# High-RPM, spoken-capable DashScope models. Skip OCR / code specialists / small 27b.
# Per-lane target is current unique + pooled remainder, not an independent 30M.
SIDECARS = [
    {
        "model": "deepseek-v4-flash-0731",
        "generator": "deepseek-v4-flash",
        "id_prefix": "csynth-dsflash-v1",
        "out_name": "zh-pretrain-colloquial-synth-dsflash-v1",
        "workers": 24,
        "qps": 16.0,
        "max_spend_cny": 200.0,
        "start_index": 1_000_000,
        "input_cny_per_million": 1.0,
        "output_cny_per_million": 2.0,
    },
    {
        "model": "kimi-k3",
        "generator": "kimi-k3",
        "id_prefix": "csynth-kimi-k3-v1",
        "out_name": "zh-pretrain-colloquial-synth-kimi-k3-v1",
        "workers": 16,
        "qps": 10.0,
        "max_spend_cny": 200.0,
        "start_index": 2_000_000,
        "input_cny_per_million": 10.0,
        "output_cny_per_million": 40.0,
    },
    {
        "model": "qwen3.7-plus",
        "generator": "qwen3.7-plus",
        "id_prefix": "csynth-qwen37plus-v1",
        "out_name": "zh-pretrain-colloquial-synth-qwen37plus-v1",
        "workers": 24,
        "qps": 16.0,
        "max_spend_cny": 200.0,
        "start_index": 3_000_000,
        "input_cny_per_million": 2.0,
        "output_cny_per_million": 8.0,
    },
    {
        "model": "glm-5.2",
        "generator": "glm-5.2",
        "id_prefix": "csynth-glm52-v1",
        "out_name": "zh-pretrain-colloquial-synth-glm52-v1",
        "workers": 10,
        "qps": 8.0,
        "max_spend_cny": 200.0,
        "start_index": 4_000_000,
        "input_cny_per_million": 8.0,
        "output_cny_per_million": 28.0,
    },
    {
        "model": "qwen3.6-plus",
        "generator": "qwen3.6-plus",
        "id_prefix": "csynth-qwen36plus-v1",
        "out_name": "zh-pretrain-colloquial-synth-qwen36plus-v1",
        "workers": 32,
        "qps": 20.0,
        "max_spend_cny": 200.0,
        "start_index": 5_000_000,
        "input_cny_per_million": 2.0,
        "output_cny_per_million": 12.0,
    },
]
def cmd_for(spec: dict) -> list[str]:
    out = CORPORA_ROOT / spec["out_name"]
    return [
        sys.executable,
        str(PRODUCER),
        "--out-dir",
        str(out),
        "--model",
        spec["model"],
        "--generator",
        spec["generator"],
        "--id-prefix",
        spec["id_prefix"],
        "--start-index",
        str(spec["start_index"]),
        "--target-unique-tokens",
        str(remaining_for_lane(spec["out_name"])),
        "--max-spend-cny",
        str(spec["max_spend_cny"]),
        "--workers",
        str(spec["workers"]),
        "--qps",
        str(spec["qps"]),
        "--input-cny-per-million",
        str(spec["input_cny_per_million"]),
        "--output-cny-per-million",
        str(spec["output_cny_per_million"]),
        "--release-kind",
        "sidecar-pilot",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma model ids to launch")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()
    want = {x.strip() for x in args.only.split(",") if x.strip()}
    specs = [s for s in SIDECARS if not want or s["model"] in want]
    if args.print_only:
        print(json.dumps({"sidecars": specs}, ensure_ascii=False, indent=2))
        return 0
    print("use produce_colloquial_qwen.py per sidecar; this helper only prints commands", file=sys.stderr)
    for spec in specs:
        print(" ".join(cmd_for(spec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
