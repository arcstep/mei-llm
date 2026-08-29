#!/usr/bin/env python3
"""Write spec/golden vectors from the Python protocol implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mei_sdk.canonical import compact_tools, dumps_canonical, schema_fingerprint  # noqa: E402
from mei_sdk.engine import Engine  # noqa: E402
from mei_sdk.protocol import parse_v2_text, render_request  # noqa: E402
from mei_sdk.version import SPEC_DIR, sdk_versions  # noqa: E402

GOLDEN = SPEC_DIR / "golden"
TINY = ROOT / "fixtures" / "packages" / "tiny-protocol-v1"


def dump(name: str, obj: dict) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)


def strip_timing(turn: dict) -> dict:
    turn = json.loads(json.dumps(turn))
    turn["stats"]["wall_ms"] = 0
    turn.get("capabilities", {}).pop("warnings", None)
    return turn


def main() -> None:
    tools = [
        {
            "name": "light.set",
            "description": "开灯",
            "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}},
        }
    ]
    dump(
        "schema_fingerprint.json",
        {
            "tools": tools,
            "compact": compact_tools(tools),
            "canonical": dumps_canonical(compact_tools(tools)),
            "sha256": schema_fingerprint(tools),
        },
    )
    request = {
        "query": "把客厅灯打开",
        "oracle_tools": tools,
        "system_facts": "客厅灯 id=L1",
        "permissions": ["light.control"],
        "selected_entity": "L1",
        "history": [{"role": "user", "content": "你好"}],
    }
    rendered = render_request(request, tools)
    dump("render_request.json", {"request": request, **rendered})
    dump(
        "parse_cases.json",
        {
            "cases": [
                {"text": "[]", "parsed": parse_v2_text("[]")},
                {
                    "text": '[{"name":"light.set","arguments":{"on":true}}]',
                    "parsed": parse_v2_text('[{"name":"light.set","arguments":{"on":true}}]'),
                },
                {
                    "text": '[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',
                    "parsed": parse_v2_text('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]'),
                },
                {"text": "", "parsed": parse_v2_text("")},
                {"text": "{", "parsed": parse_v2_text("{")},
            ]
        },
    )
    engine = Engine.load(str(TINY))
    session = engine.create_session()
    light = {"name": "light.set", "parameters": {"type": "object", "properties": {}}}
    turns = {
        "refuse": strip_timing(session.complete({"query": "开灯", "oracle_tools": [light], "candidate_text": "[]"})),
        "call": strip_timing(
            session.complete(
                {
                    "query": "开灯",
                    "oracle_tools": [light],
                    "candidate_text": '[{"name":"light.set","arguments":{}}]',
                }
            )
        ),
        "too_many": strip_timing(
            session.complete(
                {
                    "query": "x",
                    "oracle_tools": [{"name": f"t{i}", "parameters": {"type": "object", "properties": {}}} for i in range(6)],
                    "candidate_text": "[]",
                }
            )
        ),
        "leak": strip_timing(
            session.complete({"query": "gold_route_id=1", "oracle_tools": [light], "candidate_text": "[]"})
        ),
        "unavailable": strip_timing(session.complete({"query": "开灯", "oracle_tools": [light]})),
    }
    dump("turn_results.json", {"versions": sdk_versions(), "turns": turns})
    caps = engine.capabilities()
    caps.pop("warnings", None)
    dump("package_capabilities.json", {"package_id": engine.package.package_id, "capabilities": caps})


if __name__ == "__main__":
    main()
