#!/usr/bin/env python3
"""Score mei-tool-schema-v1 predictions for student and Qwen (same protocol)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval_needle_toolcall_v0 import (
    call_schema_errors,
    calls_equal,
    load_jsonl,
    load_toolset,
    norm_calls,
)
from repo_paths import EVAL_BANKS_ROOT, EVAL_SHARED_ROOT, ROOT

sys.path.insert(0, str((ROOT / "notebook/_tooling/model/mei-1.0-58m")))
from grammar import parse_phase1_text  # noqa: E402
from schema_render import load_toolset_json  # noqa: E402

BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"
RENAME_LEAK_NAMES = {"create_event", "set_volume", "lookup_price"}


def _slot(table, key, passed: bool) -> None:
    rec = table[key]
    rec["n"] += 1
    rec["n_pass"] += int(passed)


def score_item(row: dict, pred_calls: list, *, raw_text: str = "", accept_parameters: bool = False) -> dict:
    iid = str(row.get("item_id") or row.get("sample_id") or "")
    gold = norm_calls((row.get("gold") or {}).get("function_calls") or row.get("answers"))
    pred = norm_calls(pred_calls, accept_parameters=accept_parameters)
    toolset_id = str(row.get("toolset_id") or "")
    ts = load_toolset_json(toolset_id)
    parsed_ok = True
    if raw_text:
        parsed = parse_phase1_text(raw_text, ts)
        parsed_ok = bool(parsed.get("ok"))
        if parsed_ok:
            pred = norm_calls(parsed.get("function_calls"), accept_parameters=accept_parameters)
        elif accept_parameters:
            pred = norm_calls(pred_calls, accept_parameters=True)
            parsed_ok = bool(pred) or str(raw_text).strip() in {"[]", ""}
    legal_errs = call_schema_errors(iid, pred, ts)
    legal = not legal_errs
    exact = calls_equal(pred, gold)
    gold_empty = not gold
    pred_empty = not pred
    unsafe = gold_empty and (not pred_empty) and legal
    slice_id = str(row.get("slice") or "?")
    rename_leak = slice_id == "S3" and any(c.get("name") in RENAME_LEAK_NAMES for c in pred)
    name_exact = [c.get("name") for c in pred] == [c.get("name") for c in gold]
    slot_exact = exact
    return {
        "item_id": iid,
        "slice": slice_id,
        "toolset_id": toolset_id,
        "kind": row.get("kind"),
        "family": row.get("family"),
        "exact": exact,
        "legal": legal,
        "parsed_ok": parsed_ok,
        "call_exact": name_exact,
        "name_exact": name_exact,
        "slot_exact": slot_exact,
        "unsafe": unsafe and not exact,
        "crosstalk_fail": slice_id == "S4" and not exact,
        "rename_leak": rename_leak,
        "gold_empty": gold_empty,
        "pred_empty": pred_empty,
        "legal_errors": legal_errs[:4],
    }


def aggregate(rows: list[dict], scored: list[dict], latencies_ms: list[float] | None = None) -> dict:
    by_slice: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0, "n_legal": 0, "n_unsafe": 0, "n_rename_leak": 0})
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    n = len(scored)
    n_exact = n_legal = n_unsafe = n_rename = n_crosstalk_fail = 0
    for s in scored:
        n_exact += int(s["exact"])
        n_legal += int(s["legal"])
        n_unsafe += int(s["unsafe"])
        n_rename += int(s["rename_leak"])
        n_crosstalk_fail += int(s["crosstalk_fail"])
        sl = by_slice[str(s["slice"])]
        sl["n"] += 1
        sl["n_pass"] += int(s["exact"])
        sl["n_legal"] += int(s["legal"])
        sl["n_unsafe"] += int(s["unsafe"])
        sl["n_rename_leak"] += int(s["rename_leak"])
        fam = str(s.get("family") or "?")
        _slot(by_type, fam, s["exact"])
    seen = by_slice.get("S0", {"n": 0, "n_pass": 0})
    unseen = by_slice.get("S1", {"n": 0, "n_pass": 0})
    seen_acc = (seen["n_pass"] / seen["n"]) if seen["n"] else 0.0
    unseen_acc = (unseen["n_pass"] / unseen["n"]) if unseen["n"] else 0.0
    lat = {}
    if latencies_ms:
        ordered = sorted(latencies_ms)
        lat = {
            "n": len(ordered),
            "p50_ms": ordered[len(ordered) // 2],
            "mean_ms": statistics.fmean(ordered),
        }
    for sl in by_slice.values():
        sl["exact_match"] = round(sl["n_pass"] / sl["n"], 4) if sl["n"] else 0.0
        sl["legal_rate"] = round(sl["n_legal"] / sl["n"], 4) if sl["n"] else 0.0
    for sl in by_type.values():
        sl["exact_match"] = round(sl["n_pass"] / sl["n"], 4) if sl["n"] else 0.0
    return {
        "n": n,
        "exact_match": round(n_exact / n, 4) if n else 0.0,
        "legal_rate": round(n_legal / n, 4) if n else 0.0,
        "unsafe_rate": round(n_unsafe / n, 4) if n else 0.0,
        "rename_leak_rate": round(n_rename / n, 4) if n else 0.0,
        "crosstalk_fail_rate": round(n_crosstalk_fail / n, 4) if n else 0.0,
        "seen_acc": round(seen_acc, 4),
        "unseen_acc": round(unseen_acc, 4),
        "unseen_drop": round(seen_acc - unseen_acc, 4),
        "by_slice": dict(by_slice),
        "by_family": dict(by_type),
        "latency": lat,
        "always_refuse_exact": round(sum(1 for r in rows if not norm_calls((r.get("gold") or {}).get("function_calls"))) / n, 4) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--predictions", type=Path, default=None)
    args = ap.parse_args()
    rows = load_jsonl(args.bank)
    if args.predictions is None:
        print(json.dumps({"n": len(rows), "ok_schema": True}, indent=2))
        return 0
    preds = {str(r.get("item_id")): r for r in load_jsonl(args.predictions)}
    scored = []
    lats = []
    for row in rows:
        pr = preds.get(str(row.get("item_id"))) or {}
        scored.append(
            score_item(
                row,
                pr.get("function_calls") or [],
                raw_text=str(pr.get("text") or ""),
            )
        )
        if pr.get("latency_ms") is not None:
            lats.append(float(pr["latency_ms"]))
    summary = aggregate(rows, scored, lats or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
