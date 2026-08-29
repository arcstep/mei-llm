#!/usr/bin/env python3
"""Apple MLX Q4 complete() without candidate_text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from identity_51m import JOBS_DIR, Q4_PACKAGE_DIR, fail, write_json


def main() -> int:
    sdk = Path(__file__).resolve().parents[2] / "sdk" / "python"
    sys.path.insert(0, str(sdk))
    from mei_sdk import Engine

    engine = Engine.load(str(Q4_PACKAGE_DIR), backend="mlx-reference")
    caps = engine.capabilities()
    session = engine.create_session()
    tool = {
        "name": "light.set",
        "description": "开灯",
        "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}},
    }
    turn = session.complete(
        {
            "query": "厨房灯打开",
            "oracle_tools": [tool],
            "max_new": 16,
            "decode_mode": "constrained",
        }
    )
    report = {
        "ok": bool(caps.get("inference")) and "raw_text" in turn,
        "inference": bool(caps.get("inference")),
        "no_candidate_text": True,
        "backend": caps.get("backend"),
        "raw_text": turn.get("raw_text"),
        "refuse": turn.get("refuse"),
        "execution": turn.get("execution"),
        "confidence": turn.get("confidence"),
        "prompt_tokens": (turn.get("stats") or {}).get("prompt_tokens"),
        "output_tokens": (turn.get("stats") or {}).get("output_tokens"),
        "quantized_only": True,
    }
    blocked = write_json(JOBS_DIR / "apple-q4-smoke.json", report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
