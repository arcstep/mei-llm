#!/usr/bin/env python3
"""Validate needle-zh SFT seed against toolsets and phase-1 contract.

Not a synthesizer: reads an existing JSONL and fails on contract violations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from repo_paths import EVAL_SHARED_ROOT, ROOT, SEED_NEEDLE_ZH

REQUIRED_FIELDS = ("sample_id", "split", "lang", "family", "toolset_id", "query", "answers")
EMPTY_FAMILIES = {"weather_missing", "lights_missing", "offtopic"}
FAMILY_TOOL = {
    "weather": "get_weather",
    "lights": "set_lights",
    "invoice": "invoice",
}
ID_RE = re.compile(r"^TRAIN-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+$")
EVAL_RE = re.compile(r"\bEVAL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+\b")
ASK_RE = re.compile(r"请补充|请提供城市|请告诉我房间")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_toolset(toolset_id: str) -> dict:
    path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown toolset_id={toolset_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def tools_of(toolset: dict) -> dict[str, dict]:
    out = {}
    for tool in toolset.get("tools") or []:
        out[str(tool.get("name"))] = tool
    return out


def required_args(tool: dict) -> list[str]:
    params = tool.get("parameters") or {}
    return list(params.get("required") or [])


def validate_row(row: dict, seen_ids: set[str], seen_queries: set[str]) -> list[str]:
    errors: list[str] = []
    sid = str(row.get("sample_id") or "")
    loc = sid or "missing-id"
    blob = json.dumps(row, ensure_ascii=False)

    for key in REQUIRED_FIELDS:
        if key not in row:
            errors.append(f"{loc}: missing {key}")
    if not sid:
        return errors
    if not ID_RE.match(sid):
        errors.append(f"{sid}: sample_id must match TRAIN-*-N")
    if EVAL_RE.search(blob):
        errors.append(f"{sid}: embeds EVAL-* text")
    if sid in seen_ids:
        errors.append(f"{sid}: duplicate sample_id")
    seen_ids.add(sid)

    if row.get("split") not in {"train", "valid"}:
        errors.append(f"{sid}: split must be train|valid")
    if "act" not in row or row.get("act") is not None:
        errors.append(f"{sid}: act must be null")
    if ASK_RE.search(blob):
        errors.append(f"{sid}: natural-language fill prompt in gold")

    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"{sid}: empty query")
        return errors
    qn = " ".join(query.split())
    if qn in seen_queries:
        errors.append(f"{sid}: duplicate query")
    seen_queries.add(qn)

    answers = row.get("answers")
    if not isinstance(answers, list):
        errors.append(f"{sid}: answers must be a list")
        return errors

    family = str(row.get("family") or "")
    toolset_id = str(row.get("toolset_id") or "")
    try:
        toolset = load_toolset(toolset_id)
    except FileNotFoundError as exc:
        errors.append(f"{sid}: {exc}")
        return errors
    catalog = tools_of(toolset)

    if family in EMPTY_FAMILIES:
        if answers:
            errors.append(f"{sid}: family={family} must have answers=[]")
        return errors

    want_tool = FAMILY_TOOL.get(family)
    if not want_tool:
        errors.append(f"{sid}: unknown family={family}")
        return errors
    if len(answers) != 1:
        errors.append(f"{sid}: family={family} needs exactly one call")
        return errors

    call = answers[0]
    if not isinstance(call, dict):
        errors.append(f"{sid}: call must be object")
        return errors
    name = str(call.get("name") or "")
    args = call.get("arguments")
    if name != want_tool:
        errors.append(f"{sid}: expected tool {want_tool}, got {name}")
    if name not in catalog:
        errors.append(f"{sid}: tool {name} not in {toolset_id}")
        return errors
    if not isinstance(args, dict):
        errors.append(f"{sid}: arguments must be object")
        return errors

    tool = catalog[name]
    for req in required_args(tool):
        if req not in args or args[req] in (None, ""):
            errors.append(f"{sid}: missing required {req}")
    for key, val in args.items():
        if isinstance(val, str) and val.strip() == query.strip():
            errors.append(f"{sid}: slot {key} equals full query")
        if key in {"brightness", "total"} and not isinstance(val, (int, float)):
            errors.append(f"{sid}: {key} must be numeric, got {type(val).__name__}")
    if name == "set_lights":
        bright = args.get("brightness")
        if isinstance(bright, (int, float)) and not (0 <= float(bright) <= 100):
            errors.append(f"{sid}: brightness out of 0-100")
        room = args.get("room")
        if isinstance(room, str) and room and room not in query:
            errors.append(f"{sid}: room={room!r} not in query")
    if name == "get_weather":
        city = args.get("city")
        if isinstance(city, str) and city and city not in query:
            errors.append(f"{sid}: city={city!r} not in query")
    if name == "invoice":
        vendor = args.get("vendor")
        if isinstance(vendor, str) and vendor and vendor not in query:
            errors.append(f"{sid}: vendor={vendor!r} not in query")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path, default=SEED_NEEDLE_ZH)
    args = ap.parse_args()
    path = args.seed
    if not path.is_file():
        print(f"missing seed {path}", file=sys.stderr)
        return 1
    rows = load_jsonl(path)
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    n_empty = 0
    for row in rows:
        errors.extend(validate_row(row, seen_ids, seen_queries))
        if isinstance(row.get("answers"), list) and not row["answers"]:
            n_empty += 1
    if len(rows) < 80 or len(rows) > 120:
        errors.append(f"seed size {len(rows)} outside 80-120")
    if rows and n_empty / len(rows) < 0.10:
        errors.append(f"empty-answer ratio {n_empty}/{len(rows)} below ~1/8 floor")
    report = {
        "ok": not errors,
        "seed": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
        "n": len(rows),
        "empty_answers": n_empty,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
