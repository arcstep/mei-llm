#!/usr/bin/env python3
"""Same-item multi-model canary. Default is template-only; paid lanes need --allow-spend."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from repo_paths import FLEET_SFT_SYNTH, ROOT, SFT_RECIPES

sys.path.insert(0, str(Path(__file__).resolve().parent))

from park_toolcall_lib import python_park_call_ok, rewrite_guard  # noqa: E402
from sft_canonical_lib import compile_fullcall_row, compile_mw_row, compile_retrieval_row, load_jsonl  # noqa: E402
from sft_ledger import BudgetHardStop, ProviderDisabled, SharedLedger  # noqa: E402
from sft_synth_lib import (  # noqa: E402
    PROMPT_VERSION,
    SpendNotAllowed,
    bakeoff_metrics,
    dump_json,
    lanes_of,
    load_budget_plan,
    load_fleet,
    require_spend_allowed,
    rewrite_query,
)


COMPILERS = {
    "retrieval": lambda case, query, rng, model: compile_retrieval_row(case, query, teacher_model=model),
    "fullcall": lambda case, query, rng, model: compile_fullcall_row(case, query, rng=rng, teacher_model=model),
    "mw": lambda case, query, rng, model: compile_mw_row(case, query, teacher_model=model),
}


def _ledger_for(fleet: dict, phase: str) -> SharedLedger | None:
    rel = fleet.get("shared_ledger")
    if not rel:
        return None
    plan = load_budget_plan(fleet)
    phases = (plan.get("phases") or {}).get(phase) or {}
    phase_raw = phases.get("max_spend_cny")
    return SharedLedger(
        ROOT / str(rel),
        global_limit=float(fleet.get("global_budget_cny") or plan.get("global_budget_cny") or 0),
        warn_frac=float(fleet.get("warn_frac") or 0.8),
        hard_frac=float(fleet.get("hard_stop_frac") or 1.0),
        phase=phase,
        phase_limit=None if phase_raw is None else float(phase_raw),
        lane_caps={k: float(v) for k, v in (plan.get("lane_caps_cny") or {}).items()},
    )


def run_lane(
    task: str,
    cases: list[dict],
    lane,
    *,
    allow_spend: bool,
    seed: int,
    max_spend_cny: float | None = None,
    ledger: SharedLedger | None = None,
) -> list[dict]:
    require_spend_allowed(lane, allow_spend=allow_spend)
    rng = random.Random(seed)
    compile_row = COMPILERS[task]
    out = []
    spent = 0.0
    for case in cases:
        case_id = str(case.get("case_id") or f"case-{len(out)}")
        if max_spend_cny is not None and spent >= max_spend_cny:
            out.append(
                {
                    "case_id": case_id,
                    "lane": lane.name,
                    "status": "skipped",
                    "reject_reason": "budget_stop",
                    "spend_cny": 0.0,
                    "latency_ms": 0.0,
                    "gold_kept": False,
                    "slots_kept": False,
                    "compiler_ok": False,
                }
            )
            continue
        reserved = 0.0
        if ledger and lane.provider != "offline":
            try:
                reserved = ledger.reserve(
                    task=task,
                    case_id=case_id,
                    lane=lane,
                    prompt_version=PROMPT_VERSION,
                )
            except (BudgetHardStop, ProviderDisabled) as exc:
                out.append(
                    {
                        "case_id": case_id,
                        "lane": lane.name,
                        "status": "error",
                        "reject_reason": str(exc)[:240],
                        "spend_cny": 0.0,
                        "latency_ms": 0.0,
                        "gold_kept": False,
                        "slots_kept": False,
                        "compiler_ok": False,
                    }
                )
                continue
        try:
            produced = rewrite_query(case, lane, allow_spend=allow_spend)
        except Exception as exc:  # noqa: BLE001 — canary must record provider failures
            err = str(exc)
            billed = 0.0
            if ledger and lane.provider != "offline":
                billed = ledger.settle(
                    task=task,
                    case_id=case_id,
                    lane=lane,
                    prompt_version=PROMPT_VERSION,
                    actual_cny=None,
                    reserved_cny=reserved,
                    usage={},
                )
            spent += billed
            out.append(
                {
                    "case_id": case_id,
                    "lane": lane.name,
                    "status": "error",
                    "query": None,
                    "spend_cny": billed,
                    "latency_ms": 0.0,
                    "gold_kept": False,
                    "slots_kept": False,
                    "compiler_ok": False,
                    "reject_reason": err[:240],
                    "http_429": "429" in err,
                }
            )
            continue
        billed = float(produced.get("spend_cny") or 0.0)
        if ledger and lane.provider != "offline":
            billed = ledger.settle(
                task=task,
                case_id=case_id,
                lane=lane,
                prompt_version=PROMPT_VERSION,
                actual_cny=billed,
                reserved_cny=reserved,
                usage=produced.get("usage") or {},
            )
        spent += billed
        query = produced.get("query")
        reason = produced.get("reject_reason")
        if query and not reason:
            reason = rewrite_guard(case, query)
        row = {
            "case_id": case_id,
            "lane": lane.name,
            "status": "rejected",
            "query": query,
            "spend_cny": billed,
            "latency_ms": produced.get("latency_ms") or 0.0,
            "gold_kept": False,
            "slots_kept": False,
            "compiler_ok": False,
            "reject_reason": reason,
        }
        if query and not reason:
            try:
                compiled = compile_row(case, query, rng, lane.model_snapshot)
                if task == "fullcall" and case.get("toolset_id") == "mei-park-room-v1":
                    call = (compiled.get("answers") or [None])
                    call = call[0] if call else None
                    if not python_park_call_ok(case, call):
                        raise ValueError("park_validator")
                row["status"] = "accepted"
                row["gold_kept"] = True
                row["slots_kept"] = True
                row["compiler_ok"] = True
                row["sample_id"] = compiled.get("sample_id")
            except ValueError as exc:
                row["reject_reason"] = str(exc)
        out.append(row)
    return out


def share_decision(report: dict, fleet: dict) -> dict:
    """Pareto: rank paid lanes by compiler_pass then ¥/1k unique. Never majority-vote gold."""
    candidate = dict(fleet.get("candidate_share") or {})
    lanes = report.get("lanes") or {}
    paid = {
        name: stats
        for name, stats in lanes.items()
        if name != "template" and stats.get("n_accepted_unique", 0) > 0
    }
    ranked = sorted(
        paid.items(),
        key=lambda kv: (
            -float(kv[1].get("compiler_pass_rate") or 0),
            float(kv[1].get("cny_per_1k_accepted_unique") or 0),
        ),
    )
    usable = [name for name, _ in ranked]
    policy = "pareto_from_canary" if paid else "candidate_until_paid_canary"
    return {
        "policy": policy,
        "usable_lanes": usable or [name for name, stats in lanes.items() if stats.get("n_accepted_unique", 0) > 0],
        "ranked": [{"lane": n, **s} for n, s in ranked],
        "candidate_share": candidate,
        "note": (
            "Production share stays candidate until compiler_pass and ¥/1k unique are compared across paid lanes."
            if not paid
            else "Ranked by compiler_pass_rate then ¥/1k accepted unique. Do not majority-vote gold."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["retrieval", "fullcall", "mw"])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--lanes", default="template", help="comma-separated lane names")
    ap.add_argument("--fleet", type=Path, default=FLEET_SFT_SYNTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use first N canonical cases; 0 = all")
    ap.add_argument("--max-spend-cny", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--phase", default="canary", choices=["canary", "2k", "10k", "smoke"])
    args = ap.parse_args()
    fleet = load_fleet(args.fleet)
    wanted = [part.strip() for part in args.lanes.split(",") if part.strip()]
    lanes = lanes_of(fleet)
    cases = load_jsonl(args.cases)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    ledger = _ledger_for(fleet, args.phase) if args.allow_spend else None
    by_lane: dict[str, list[dict]] = {}
    remaining = args.max_spend_cny
    for name in wanted:
        if name not in lanes:
            print(json.dumps({"ok": False, "error": f"unknown lane {name}"}), file=sys.stderr)
            return 2
        cap = remaining
        try:
            by_lane[name] = run_lane(
                args.task,
                cases,
                lanes[name],
                allow_spend=args.allow_spend,
                seed=args.seed,
                max_spend_cny=cap,
                ledger=ledger,
            )
        except SpendNotAllowed as exc:
            print(json.dumps({"ok": False, "error": str(exc), "lane": name}), file=sys.stderr)
            return 3
        spent = sum(float(r.get("spend_cny") or 0) for r in by_lane[name])
        if remaining is not None:
            remaining = max(0.0, remaining - spent)
    report = bakeoff_metrics(by_lane)
    report["ok"] = True
    report["task"] = args.task
    report["n_cases"] = len(cases)
    report["max_spend_cny"] = args.max_spend_cny
    report["spend_cny"] = sum(float(s.get("spend_cny") or 0) for s in report["lanes"].values())
    report["share"] = share_decision(report, fleet)
    report["ledger"] = ledger.snapshot() if ledger else None
    report["adaptive"] = {
        "start": {n: {"workers": lanes[n].start_workers, "qps": lanes[n].start_qps} for n in wanted},
        "cap": {n: {"workers": lanes[n].max_workers, "qps": lanes[n].max_qps} for n in wanted},
        "ramp": fleet.get("ramp"),
        "note": "Ramp only when 429≤1%, retry/error≤2%, and p95 is stable.",
    }
    out = args.out or (SFT_RECIPES / f"bakeoff-{args.task}-template.json")
    dump_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
