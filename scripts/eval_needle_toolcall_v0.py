#!/usr/bin/env python3
"""Validate needle-toolcall bank schema and optionally score predictions.

Predictions JSONL fields: item_id, function_calls: [{name, arguments}]
Without --predictions, only schema-check the bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from repo_paths import BANK_NEEDLE_TOOLCALL, EVAL_SHARED_ROOT, ROOT

REFUSE_FAMILIES = {"missing", "scene_conflict", "illegal_pair", "offtopic"}
EXECUTE_FAMILIES = {"gesture", "home", "order", "sequence", "paraphrase"}

REQUIRED_BANK = ("item_id", "query", "gold", "pass", "toolset_id")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_toolset(toolset_id: str) -> dict:
    path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def norm_scalar(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return value


def values_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def norm_args(args: dict | None) -> dict:
    if not isinstance(args, dict):
        return {}
    return {str(k): norm_scalar(v) for k, v in args.items()}


def extract_arguments(call: dict, *, accept_parameters: bool = False) -> dict:
    args = call.get("arguments")
    if isinstance(args, dict):
        return args
    if accept_parameters and isinstance(call.get("parameters"), dict):
        return call["parameters"]
    if accept_parameters:
        rest = {k: v for k, v in call.items() if k not in {"name", "function", "description", "parameters", "type", "properties", "required"}}
        if rest:
            return rest
    return {}


def calls_equal(a, b) -> bool:
    left, right = norm_calls(a), norm_calls(b)
    if len(left) != len(right):
        return False
    for x, y in zip(left, right):
        if x.get("name") != y.get("name"):
            return False
        xa, ya = x.get("arguments") or {}, y.get("arguments") or {}
        if set(xa) != set(ya):
            return False
        for k in xa:
            if not values_equal(xa[k], ya[k]):
                return False
    return True


def norm_calls(calls, *, accept_parameters: bool = False) -> list[dict]:
    if not calls:
        return []
    out = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or c.get("function") or "")
        if isinstance(c.get("name"), dict):
            name = str(c["name"].get("name") or "")
        out.append({"name": name, "arguments": norm_args(extract_arguments(c, accept_parameters=accept_parameters))})
    return out


def tool_index(toolset: dict) -> dict[str, dict]:
    return {str(t.get("name")): t for t in (toolset.get("tools") or []) if t.get("name")}


def json_type_ok(value, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def call_schema_errors(iid: str, calls, toolset: dict) -> list[str]:
    errors: list[str] = []
    by_name = tool_index(toolset)
    for call in calls or []:
        if not isinstance(call, dict):
            errors.append(f"{iid}: gold call is not an object")
            continue
        name = str(call.get("name") or "")
        spec = by_name.get(name)
        if not spec:
            errors.append(f"{iid}: unknown tool {name!r}")
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties") or {}
        required = list(params.get("required") or [])
        args = call.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            errors.append(f"{iid}: {name} arguments must be an object")
            continue
        for key in required:
            if key not in args:
                errors.append(f"{iid}: {name} missing required {key}")
        for key, val in args.items():
            if key not in props:
                errors.append(f"{iid}: {name} unexpected argument {key}")
                continue
            schema = props[key] or {}
            declared = schema.get("type")
            if declared and not json_type_ok(val, str(declared)):
                errors.append(f"{iid}: {name}.{key} type want {declared}")
            enum = schema.get("enum")
            if enum is not None and val not in enum:
                errors.append(f"{iid}: {name}.{key}={val!r} not in enum")
    return errors


def schema_errors(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    toolsets: dict[str, dict] = {}
    for i, row in enumerate(rows, 1):
        iid = row.get("item_id")
        if not iid:
            errors.append(f"line {i}: missing item_id")
            continue
        if iid in seen:
            errors.append(f"{iid}: duplicate")
        seen.add(str(iid))
        if not str(iid).startswith("EVAL-"):
            errors.append(f"{iid}: item_id must start with EVAL-")
        for key in REQUIRED_BANK:
            if key not in row:
                errors.append(f"{iid}: missing {key}")
        gold = row.get("gold") or {}
        if "function_calls" not in gold:
            errors.append(f"{iid}: gold.function_calls missing")
        toolset_id = row.get("toolset_id")
        path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
        if not path.is_file():
            errors.append(f"{iid}: unknown toolset_id={toolset_id}")
            continue
        if row.get("pass") != "exact_match":
            errors.append(f"{iid}: only exact_match is supported in v0")
        key = str(toolset_id)
        if key not in toolsets:
            toolsets[key] = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(call_schema_errors(str(iid), gold.get("function_calls"), toolsets[key]))
    return errors


def _inc(slot: dict, passed: bool) -> None:
    slot["n"] += 1
    if passed:
        slot["n_pass"] += 1


def _finalize(table: dict[str, dict]) -> dict[str, dict]:
    for slot in table.values():
        slot["exact_match"] = round(slot["n_pass"] / slot["n"], 4) if slot["n"] else 0.0
    return table


def gold_baselines(rows: list[dict]) -> dict:
    """Always-refuse and strata from gold only (no model predictions)."""
    n = len(rows)
    n_empty = 0
    by_family: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_er: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_tool: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_call_count: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_style: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_case: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    for row in rows:
        gold = norm_calls((row.get("gold") or {}).get("function_calls"))
        empty = not gold
        n_empty += int(empty)
        passed_if_refuse = empty
        fam = str(row.get("family") or "?")
        _inc(by_family[fam], passed_if_refuse)
        er = "refuse" if fam in REFUSE_FAMILIES else "execute"
        _inc(by_er[er], passed_if_refuse)
        names = [c.get("name") or "?" for c in gold] or ["_empty"]
        for name in names:
            _inc(by_tool[str(name)], passed_if_refuse)
        _inc(by_call_count[str(len(gold))], passed_if_refuse)
        for tag in list(row.get("style_tags") or ["_none"]):
            _inc(by_style[str(tag)], passed_if_refuse)
        for tag in list(row.get("case_tags") or ["_none"]):
            _inc(by_case[str(tag)], passed_if_refuse)
    return {
        "n": n,
        "always_refuse_exact_match": round(n_empty / n, 4) if n else 0.0,
        "n_gold_empty": n_empty,
        "by_family": _finalize(dict(by_family)),
        "by_execute_refuse": _finalize(dict(by_er)),
        "by_tool": _finalize(dict(by_tool)),
        "by_call_count": _finalize(dict(by_call_count)),
        "by_style_tag": _finalize(dict(by_style)),
        "by_case_tag": _finalize(dict(by_case)),
        "note": "always_refuse is the constant-[] baseline; execute items score 0 under it.",
    }


def score(rows: list[dict], preds: list[dict]) -> dict:
    by_id = {r["item_id"]: r for r in rows}
    pred_by = {p.get("item_id"): p for p in preds if p.get("item_id")}
    details = []
    n_pass = 0
    by_family: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_er: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_tool: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_call_count: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_style: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_case: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    for iid, row in by_id.items():
        pred = pred_by.get(iid) or {}
        gold_calls = norm_calls((row.get("gold") or {}).get("function_calls"))
        got_calls = norm_calls(pred.get("function_calls"))
        passed = calls_equal(gold_calls, got_calls)
        if passed:
            n_pass += 1
        fam = str(row.get("family") or "?")
        _inc(by_family[fam], passed)
        er = "refuse" if fam in REFUSE_FAMILIES else "execute"
        if fam not in REFUSE_FAMILIES and fam not in EXECUTE_FAMILIES:
            er = "refuse" if not gold_calls else "execute"
        _inc(by_er[er], passed)
        names = [c.get("name") or "?" for c in gold_calls] or ["_empty"]
        for name in names:
            _inc(by_tool[str(name)], passed)
        _inc(by_call_count[str(len(gold_calls))], passed)
        for tag in list(row.get("style_tags") or ["_none"]):
            _inc(by_style[str(tag)], passed)
        for tag in list(row.get("case_tags") or ["_none"]):
            _inc(by_case[str(tag)], passed)
        details.append(
            {
                "item_id": iid,
                "lang": row.get("lang"),
                "family": fam,
                "pass": passed,
                "gold": gold_calls,
                "pred": got_calls,
            }
        )
    n = len(by_id)
    return {
        "n": n,
        "n_pass": n_pass,
        "n_pred": len(pred_by),
        "exact_match": round(n_pass / n, 4) if n else 0.0,
        "by_family": _finalize(dict(by_family)),
        "by_execute_refuse": _finalize(dict(by_er)),
        "by_tool": _finalize(dict(by_tool)),
        "by_call_count": _finalize(dict(by_call_count)),
        "by_style_tag": _finalize(dict(by_style)),
        "by_case_tag": _finalize(dict(by_case)),
        "always_refuse": gold_baselines(rows),
        "missing_pred": sorted(set(by_id) - set(pred_by)),
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_TOOLCALL)
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--split", default=None, help="Optional split filter (dev|eval)")
    args = ap.parse_args()
    bank = args.bank if args.bank.is_absolute() else (ROOT / args.bank)
    if not bank.is_file():
        print(f"missing bank: {bank}", file=sys.stderr)
        return 1
    rows = load_jsonl(bank)
    if args.split:
        rows = [r for r in rows if str(r.get("split") or "") == args.split]
    errors = schema_errors(rows)
    report: dict = {
        "ok": not errors,
        "bank": str(bank.resolve().relative_to(ROOT)),
        "n": len(rows),
        "schema_errors": errors,
        "always_refuse": gold_baselines(rows),
    }
    if args.predictions:
        preds = load_jsonl(args.predictions)
        scored = score(rows, preds)
        report["ok"] = report["ok"] and scored["n_pass"] == scored["n"] and not scored["missing_pred"]
        report["score"] = scored
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
