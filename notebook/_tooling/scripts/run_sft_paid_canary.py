#!/usr/bin/env python3
"""Same-item paid canary. Cap 50 CNY. Park enters retrieval/full-call; MW stays generic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import FLEET_SFT_SYNTH, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from park_toolcall_lib import park_fullcall_cases, park_retrieval_cases  # noqa: E402
from sft_canonical_lib import dump_jsonl, ensure_train_valid_split, retrieval_case_bank  # noqa: E402
from sft_synth_lib import bakeoff_metrics, dump_json, lanes_of, load_budget_plan, load_fleet  # noqa: E402
from build_mei_mw_disposition_v2 import cases_from_seed  # noqa: E402
from run_sft_synth_bakeoff import _ledger_for, run_lane, share_decision  # noqa: E402


GATES = {
    "compiler_pass_min": 0.98,
    "gold_leak_max": 0,
    "max_429_rate": 0.01,
    "max_retry_error_rate": 0.02,
}


def canary_cases(task: str, n: int) -> list[dict]:
    if task == "retrieval":
        park = park_retrieval_cases(max(4, n // 3))
        generic = retrieval_case_bank()[: max(0, n - len(park))]
        rows = (park + generic)[:n]
    elif task == "fullcall":
        rows = park_fullcall_cases(quota={"L0": max(3, n // 4), "L1": 2, "L2": 2, "L3": 2})[:n]
        if len(rows) < n:
            from build_mei_toolcall_v2 import cases_from_intents

            rows = (rows + cases_from_intents(n, seed=20260827))[:n]
    else:
        rows = cases_from_seed(limit=n)
    ensure_train_valid_split(rows)
    for i, row in enumerate(rows):
        row["idx"] = i
    return rows[:n]


def lane_http_stats(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    n429 = sum(1 for r in rows if r.get("http_429") or r.get("reject_reason") == "429")
    nerr = sum(1 for r in rows if r.get("status") == "error")
    return {"rate_429": n429 / n, "retry_error_rate": nerr / n}


def evaluate_gates(report: dict, *, by_lane: dict[str, list[dict]]) -> dict:
    fails = []
    paid_seen = False
    for name, stats in (report.get("lanes") or {}).items():
        http = lane_http_stats(by_lane.get(name) or [])
        stats["rate_429"] = http["rate_429"]
        stats["retry_error_rate"] = http["retry_error_rate"]
        if name == "template":
            continue
        paid_seen = True
        n = max(1, int(stats.get("n") or 0))
        if float(stats.get("compiler_pass_rate") or 0) < GATES["compiler_pass_min"]:
            fails.append(f"{name}:compiler")
        if http["rate_429"] > GATES["max_429_rate"]:
            fails.append(f"{name}:429")
        if http["retry_error_rate"] > GATES["max_retry_error_rate"]:
            fails.append(f"{name}:retry_error")
        if int(stats.get("n_accepted") or 0) and float(stats.get("protected_slot_keep_rate") or 0) * n + 1e-9 < float(
            stats.get("n_accepted") or 0
        ):
            fails.append(f"{name}:slots")
    spend = float(report.get("spend_cny") or 0)
    if spend > float(report.get("max_spend_cny") or 50) + 1e-6:
        fails.append("budget")
    return {"ok": not fails, "fails": fails, "paid_lanes_ran": paid_seen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--fleet", type=Path, default=FLEET_SFT_SYNTH)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--lanes", default="template,deepseek-v4-flash-0731,qwen-plus-2025-12-01,qwen3.7-plus")
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "notebook/jobs/toolcall-sft/outbox/draft/canary")
    args = ap.parse_args()
    fleet = load_fleet(args.fleet)
    plan = load_budget_plan(fleet)
    cap = float((plan.get("phases") or {}).get("canary", {}).get("max_spend_cny") or fleet.get("canary_budget_cny") or 50)
    lanes = lanes_of(fleet)
    wanted = [p.strip() for p in args.lanes.split(",") if p.strip()]
    enabled = (
        json.loads(args.probe.read_text(encoding="utf-8")).get("enabled_lanes")
        if args.probe and args.probe.is_file()
        else None
    )
    if enabled is not None:
        wanted = [n for n in wanted if n == "template" or n in enabled]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = _ledger_for(fleet, "canary") if args.allow_spend else None
    remaining = cap
    tasks = {}
    paid_ran = False
    for task in ("retrieval", "fullcall", "mw"):
        cases = canary_cases(task, args.n)
        dump_jsonl(args.out_dir / f"{task}-cases.jsonl", cases)
        by_lane = {}
        for name in wanted:
            if name not in lanes:
                continue
            by_lane[name] = run_lane(
                task,
                cases,
                lanes[name],
                allow_spend=args.allow_spend,
                seed=20260827,
                max_spend_cny=remaining if lanes[name].provider != "offline" else None,
                ledger=ledger,
            )
            dump_jsonl(args.out_dir / f"{task}-{name}-rows.jsonl", by_lane[name])
            spent = sum(float(r.get("spend_cny") or 0) for r in by_lane[name])
            if lanes[name].provider != "offline":
                remaining = max(0.0, remaining - spent)
                paid_ran = True
        metrics = bakeoff_metrics(by_lane)
        metrics["task"] = task
        metrics["n_cases"] = len(cases)
        metrics["max_spend_cny"] = cap
        metrics["spend_cny"] = sum(float(s.get("spend_cny") or 0) for s in metrics["lanes"].values())
        metrics["share"] = share_decision(metrics, fleet)
        metrics["gates"] = evaluate_gates(metrics, by_lane=by_lane)
        metrics["ledger"] = ledger.snapshot() if ledger else None
        tasks[task] = metrics
        dump_json(args.out_dir / f"{task}-canary.json", metrics)
    total_spend = sum(float(t.get("spend_cny") or 0) for t in tasks.values())
    gates_ok = all(t.get("gates", {}).get("ok") for t in tasks.values())
    out = {
        "ok": gates_ok and total_spend <= cap + 1e-6,
        "spend_cny": total_spend,
        "max_spend_cny": cap,
        "tasks": tasks,
        "scale_2k_template": True,
        "scale_2k_paid": bool(paid_ran and gates_ok and total_spend <= cap + 1e-6),
        "note": "Fail any paid gate → do not scale 2k paid. Template 2k candidates may still be built. Do not majority-vote gold. Do not replace missing snapshots.",
    }
    dump_json(args.out_dir / "canary-report.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
