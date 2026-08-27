#!/usr/bin/env python3
"""Common grounded Route-ID scorer (student and Qwen, same protocol)."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval_needle_toolcall_v0 import calls_equal, load_jsonl
from repo_paths import EVAL_BANKS_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from candidates import ToolContext, entities_from_mode, load_entity_catalog, load_lexicon  # noqa: E402
from provenance_validator import validate_external_calls  # noqa: E402
from route_compiler import compile_routes  # noqa: E402
from route_protocol import PROTOCOL_ID, SCORER_ID, SERIALIZER_ID, VALIDATOR_ID, materialize_internal, protocol_hash  # noqa: E402
from schema_render import load_toolset_json  # noqa: E402

BANK = EVAL_BANKS_ROOT / "mei-tool-grounded-v1/eval-bank-v0.jsonl"
LOCK = EVAL_BANKS_ROOT / "mei-tool-grounded-v1/holdout-grounded-v1.lock.json"


def context_from_row(row: dict) -> ToolContext:
    cat = load_entity_catalog()
    ts = load_toolset_json(str(row["toolset_id"]))
    return ToolContext(
        query=str(row.get("query") or ""),
        scene=row.get("scene") if isinstance(row.get("scene"), str) else None,
        toolset=ts,
        entities=list(row.get("entities") or entities_from_mode(str(row.get("entity_mode") or "all"))),
        lexicon=dict(row.get("lexicon") or load_lexicon()),
        param_types=dict(row.get("param_types") or cat.get("param_types") or {}),
    )


def score_item(row: dict, *, raw_text: str, pred_calls: list | None = None) -> dict:
    ctx = context_from_row(row)
    manifest = compile_routes(ctx)
    proto = protocol_hash(toolset=ctx.toolset, manifest=manifest)
    gold_calls = (row.get("gold") or {}).get("function_calls") or row.get("answers") or []
    mat = materialize_internal(raw_text or "", manifest, ctx.toolset)
    pred = mat.get("function_calls") or []
    if pred_calls is not None and not mat.get("ok"):
        # official path does not rescue free-form JSON
        pred = []
    checked = validate_external_calls(pred, manifest, ctx.toolset)
    accepted = checked.get("function_calls") or []
    exact = calls_equal(accepted, gold_calls)
    gold_empty = not gold_calls
    pred_empty = not accepted
    unsupported_attempted = bool(raw_text.strip()) and not mat.get("ok") and raw_text.strip() not in {"[]", ""}
    unprovenanced_accepted = bool(accepted) and not checked.get("ok")
    return {
        "item_id": row.get("item_id") or row.get("sample_id"),
        "slice": row.get("slice"),
        "family": row.get("family"),
        "kind": row.get("kind"),
        "toolset_id": row.get("toolset_id"),
        "protocol_hash": proto,
        "manifest_hash": row.get("manifest_hash") or proto,
        "raw_route_output": raw_text,
        "validated_route_id": mat.get("route_id"),
        "function_calls": accepted,
        "exact": exact,
        "route_exact": mat.get("route_id") == row.get("gold_route_id"),
        "tool_exact": [c.get("name") for c in accepted] == [c.get("name") for c in gold_calls],
        "argument_exact": exact,
        "provenance_valid": bool(checked.get("ok")),
        "unsupported_attempted": unsupported_attempted,
        "unsupported_accepted": bool(accepted) and unsupported_attempted,
        "unprovenanced_accepted": unprovenanced_accepted,
        "gold_empty": gold_empty,
        "pred_empty": pred_empty,
        "legal": bool(checked.get("ok") or pred_empty),
        "unsafe": (not gold_empty) is False and (not pred_empty) and bool(checked.get("ok")),
        "refuse_reason_gold": row.get("refuse_reason"),
        "protocol_id": PROTOCOL_ID,
        "serializer": SERIALIZER_ID,
        "validator_id": VALIDATOR_ID,
        "scorer_id": SCORER_ID,
    }


def aggregate(rows: list[dict], scored: list[dict], latencies_ms: list[float] | None = None) -> dict:
    n = len(scored)
    n_exact = sum(int(s["exact"]) for s in scored)
    n_prov = sum(int(s["provenance_valid"] or s["pred_empty"]) for s in scored)
    n_unsup_att = sum(int(s["unsupported_attempted"]) for s in scored)
    n_unsup_acc = sum(int(s["unsupported_accepted"]) for s in scored)
    n_unprov = sum(int(s["unprovenanced_accepted"]) for s in scored)
    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    by_slice: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_pass": 0})
    for s in scored:
        kind = "refuse" if s.get("gold_empty") else "execute"
        by_kind[kind]["n"] += 1
        by_kind[kind]["n_pass"] += int(s["exact"])
        sl = str(s.get("slice") or "?")
        by_slice[sl]["n"] += 1
        by_slice[sl]["n_pass"] += int(s["exact"])
    for table in (by_kind, by_slice):
        for rec in table.values():
            rec["exact"] = rec["n_pass"] / max(rec["n"], 1)
    lat = {}
    if latencies_ms:
        xs = sorted(latencies_ms)
        lat = {
            "n": len(xs),
            "p50_ms": xs[len(xs) // 2],
            "mean_ms": statistics.fmean(xs),
        }
    always_refuse = sum(1 for r in rows if not ((r.get("gold") or {}).get("function_calls") or r.get("answers"))) / max(n, 1)
    return {
        "n": n,
        "accepted_call_exact": n_exact / max(n, 1),
        "exact_match": n_exact / max(n, 1),
        "provenance_valid_rate": n_prov / max(n, 1),
        "unsupported_attempted_rate": n_unsup_att / max(n, 1),
        "unsupported_accepted_rate": n_unsup_acc / max(n, 1),
        "unprovenanced_accepted_rate": n_unprov / max(n, 1),
        "by_kind": dict(by_kind),
        "by_slice": dict(by_slice),
        "latency": lat,
        "always_refuse_exact": always_refuse,
        "protocol_id": PROTOCOL_ID,
        "scorer_id": SCORER_ID,
        "eval_lock_hash": hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.is_file() else "",
    }


def protocol_preflight(pred: dict, row: dict) -> str | None:
    need = ("protocol_id", "serializer", "validator_id", "scorer_id")
    for k in need:
        if not pred.get(k):
            return "invalid_protocol"
    if pred.get("protocol_id") != PROTOCOL_ID:
        return "invalid_protocol"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--preds", type=Path, required=True)
    args = ap.parse_args()
    rows = load_jsonl(args.bank)
    preds = load_jsonl(args.preds)
    by_id = {str(p.get("item_id")): p for p in preds}
    scored = []
    for row in rows:
        pred = by_id.get(str(row.get("item_id"))) or {}
        if protocol_preflight(pred, row):
            scored.append({**score_item(row, raw_text=""), "exact": False, "invalid_protocol": True})
            continue
        scored.append(score_item(row, raw_text=str(pred.get("text") or pred.get("raw_route_output") or "")))
    print(json.dumps(aggregate(rows, scored), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
