#!/usr/bin/env python3
"""Teacher rewrite worker: query-only. Gold is recompiled after every rewrite."""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo_paths import FLEET_SFT_SYNTH, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from park_toolcall_lib import python_park_call_ok, rewrite_guard  # noqa: E402
from sft_canonical_lib import (  # noqa: E402
    compile_fullcall_row,
    compile_mw_row,
    compile_retrieval_row,
    load_jsonl,
    query_banned,
)
from sft_ledger import BudgetHardStop, ProviderDisabled, SharedLedger, default_reserve_cny  # noqa: E402
from sft_synth_lib import (  # noqa: E402
    PROMPT_VERSION,
    SpendNotAllowed,
    TokenBucket,
    accepted_unique_count,
    bind_resume_contract,
    claim_item,
    connect_queue,
    dump_json,
    dump_jsonl,
    enqueue_cases,
    export_queue_rows,
    finish_item,
    fleet_price_fingerprint,
    lanes_of,
    load_budget_plan,
    load_fleet,
    maybe_ramp,
    queue_sums,
    reclaim_running,
    require_spend_allowed,
    rewrite_query,
    sha256_text,
)


def _compile(task: str, case: dict, query: str, rng: random.Random, teacher_model: str) -> dict:
    if task == "retrieval":
        return compile_retrieval_row(case, query, teacher_model=teacher_model)
    if task == "fullcall":
        return compile_fullcall_row(case, query, rng=rng, teacher_model=teacher_model)
    if task == "mw":
        return compile_mw_row(case, query, teacher_model=teacher_model)
    raise ValueError(f"unknown task {task}")


def _ledger(fleet: dict, phase: str) -> SharedLedger | None:
    rel = fleet.get("shared_ledger")
    if not rel:
        return None
    plan = load_budget_plan(fleet)
    phases = (plan.get("phases") or {}).get(phase) or {}
    phase_raw = phases.get("max_spend_cny")
    phase_limit = None if phase_raw is None else float(phase_raw)
    return SharedLedger(
        ROOT / str(rel),
        global_limit=float(fleet.get("global_budget_cny") or plan.get("global_budget_cny") or 0),
        warn_frac=float(fleet.get("warn_frac") or 0.8),
        hard_frac=float(fleet.get("hard_stop_frac") or 1.0),
        phase=phase,
        phase_limit=phase_limit,
        lane_caps={k: float(v) for k, v in (plan.get("lane_caps_cny") or {}).items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["retrieval", "fullcall", "mw"])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lane", default="template")
    ap.add_argument("--fleet", type=Path, default=FLEET_SFT_SYNTH)
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--phase", default="canary", choices=["canary", "2k", "10k", "smoke"])
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    fleet = load_fleet(args.fleet)
    lanes = lanes_of(fleet)
    if args.lane not in lanes:
        print(json.dumps({"ok": False, "error": f"unknown lane {args.lane}"}), file=sys.stderr)
        return 2
    lane = lanes[args.lane]
    try:
        require_spend_allowed(lane, allow_spend=args.allow_spend)
    except SpendNotAllowed as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 3

    cases = load_jsonl(args.cases)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    (work / "raw").mkdir(exist_ok=True)
    conn = connect_queue(work / "queue.sqlite")
    lock = threading.Lock()
    hashes = {str(c.get("toolset_hash") or "") for c in cases if c.get("toolset_hash")}
    toolset_hash = next(iter(hashes)) if len(hashes) == 1 else None
    bind_resume_contract(
        conn,
        fleet_id=str(fleet.get("id") or "sft-synth-fleet-v1"),
        lane=lane.name,
        model_snapshot=lane.model_snapshot,
        prompt_version=PROMPT_VERSION,
        price_fingerprint=fleet_price_fingerprint(fleet),
        serializer_version=str(fleet.get("serializer_version") or "mei-tool-call-serializer-v2"),
        toolset_hash=toolset_hash,
    )
    reclaim_running(conn, lock)
    enqueue_cases(conn, lock, cases, lane=lane.name)

    rng = random.Random(args.seed)
    seen: set[str] = set()
    for row in conn.execute("SELECT query FROM items WHERE status='accepted' AND query IS NOT NULL"):
        seen.add(sha256_text(str(row["query"]).strip()))
    ledger = _ledger(fleet, "smoke" if args.phase == "smoke" else args.phase)
    n_workers = args.workers or max(1, lane.start_workers if lane.provider != "offline" else 1)
    bucket = TokenBucket(lane.start_qps)
    ramp_state = {"workers": n_workers, "qps": lane.start_qps}
    stats = {"n429": 0, "nerr": 0, "n": 0, "retries": 0, "latencies": []}
    stats_lock = threading.Lock()
    ramp_cfg = fleet.get("ramp") or {}

    def process_one() -> None:
        while True:
            item = claim_item(conn, lock)
            if item is None:
                return
            case = item["case"]
            reserved = 0.0
            if ledger and lane.provider != "offline":
                try:
                    reserved = ledger.reserve(
                        task=args.task,
                        case_id=item["case_id"],
                        lane=lane,
                        prompt_version=PROMPT_VERSION,
                    )
                except (BudgetHardStop, ProviderDisabled) as exc:
                    finish_item(conn, lock, item["case_id"], status="error", reject_reason=str(exc)[:200], task=args.task)
                    if isinstance(exc, BudgetHardStop):
                        return
                    continue
            bucket.take()
            produced = None
            attempts = 0
            while produced is None:
                try:
                    produced = rewrite_query(case, lane, allow_spend=args.allow_spend)
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                    is_429 = "429" in err
                    if ledger and lane.provider != "offline":
                        ledger.void(
                            task=args.task,
                            case_id=item["case_id"],
                            lane=lane,
                            prompt_version=PROMPT_VERSION,
                        )
                    if is_429 and attempts < 5:
                        attempts += 1
                        time.sleep(min(16.0, 2.0**attempts))
                        bucket.take()
                        if ledger and lane.provider != "offline":
                            try:
                                reserved = ledger.reserve(
                                    task=args.task,
                                    case_id=item["case_id"],
                                    lane=lane,
                                    prompt_version=PROMPT_VERSION,
                                )
                            except (BudgetHardStop, ProviderDisabled) as stop:
                                finish_item(
                                    conn,
                                    lock,
                                    item["case_id"],
                                    status="error",
                                    reject_reason=str(stop)[:200],
                                    task=args.task,
                                )
                                if isinstance(stop, BudgetHardStop):
                                    return
                                produced = None
                                break
                        continue
                    finish_item(
                        conn,
                        lock,
                        item["case_id"],
                        status="error",
                        reject_reason=err[:200],
                        model_snapshot=lane.model_snapshot,
                        prompt_version=PROMPT_VERSION,
                        task=args.task,
                        http_status=429 if is_429 else None,
                    )
                    with stats_lock:
                        stats["nerr"] += 1
                        if is_429:
                            stats["n429"] += 1
                        stats["n"] += 1
                        n = stats["n"]
                        rate_429 = stats["n429"] / max(1, n)
                        retry_error = stats["nerr"] / max(1, n)
                        workers, qps = maybe_ramp(
                            lane,
                            {
                                "n": n,
                                "workers": ramp_state["workers"],
                                "qps": ramp_state["qps"],
                                "rate_429": rate_429,
                                "retry_error_rate": retry_error,
                                "p95_stable": True,
                            },
                            ramp_cfg,
                        )
                        ramp_state["workers"] = workers
                        ramp_state["qps"] = qps
                        bucket.set_qps(qps)
                    produced = None
                    break
            if produced is None:
                continue
            billed = float(produced.get("spend_cny") or 0.0)
            if ledger and lane.provider != "offline":
                billed = ledger.settle(
                    task=args.task,
                    case_id=item["case_id"],
                    lane=lane,
                    prompt_version=PROMPT_VERSION,
                    actual_cny=billed,
                    reserved_cny=reserved,
                    usage=produced.get("usage") or {},
                )
            query = produced.get("query")
            reason = produced.get("reject_reason") or (query_banned(query) if query else "empty")
            if query and not reason:
                reason = rewrite_guard(case, query)
            row = None
            gold_kept = False
            if query and not reason:
                try:
                    row = _compile(args.task, case, query, rng, lane.model_snapshot)
                    if args.task == "fullcall" and case.get("toolset_id") == "mei-park-room-v1":
                        call = (row.get("answers") or [None])
                        call = call[0] if call else None
                        if not python_park_call_ok(case, call):
                            raise ValueError("park_validator")
                    gold_kept = True
                except ValueError as exc:
                    reason = str(exc)
                    row = None
                    gold_kept = False
            key = sha256_text((query or "") + "\n" + str(case.get("scene") or "") + "\n" + str(case.get("system_facts_text") or ""))
            with stats_lock:
                stats["n"] += 1
                if key in seen and row:
                    reason = "near_dup"
                    row = None
                    gold_kept = False
                elif row:
                    seen.add(key)
            status = "accepted" if row else "rejected"
            if row:
                row["status"] = "accepted"
                row["gold_kept"] = gold_kept
                row["slots_kept"] = gold_kept
                row["compiler_ok"] = True
                row["spend_cny"] = billed
                row["request_sha256"] = produced["request_sha256"]
                row["response_sha256"] = produced["response_sha256"]
                row["latency_ms"] = produced["latency_ms"]
            finish_item(
                conn,
                lock,
                item["case_id"],
                status=status,
                query=query,
                reject_reason=None if row else reason,
                prompt_tokens=produced.get("prompt_tokens") or 0,
                completion_tokens=produced.get("completion_tokens") or 0,
                spend=billed,
                request_sha256=produced.get("request_sha256"),
                response_sha256=produced.get("response_sha256"),
                model_snapshot=lane.model_snapshot,
                prompt_version=PROMPT_VERSION,
                latency_ms=produced.get("latency_ms") or 0.0,
                result_json=json.dumps(row, ensure_ascii=False) if row else None,
                task=args.task,
            )
            with stats_lock:
                stats["latencies"].append(float(produced.get("latency_ms") or 0.0))
                n = stats["n"]
                lat = sorted(stats["latencies"])
                p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
                p50 = lat[len(lat) // 2] if lat else 0.0
                workers, qps = maybe_ramp(
                    lane,
                    {
                        "n": n,
                        "workers": ramp_state["workers"],
                        "qps": ramp_state["qps"],
                        "rate_429": stats["n429"] / max(1, n),
                        "retry_error_rate": stats["nerr"] / max(1, n),
                        "p95_stable": p95 <= max(p50 * 4, 1.0),
                    },
                    ramp_cfg,
                )
                ramp_state["workers"] = workers
                ramp_state["qps"] = qps
                bucket.set_qps(qps)

    if n_workers <= 1:
        process_one()
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(process_one) for _ in range(n_workers)]
            for fut in futs:
                fut.result()

    accepted, rejected = export_queue_rows(conn)
    dump_jsonl(work / "raw" / "accepted.jsonl", accepted)
    dump_jsonl(work / "raw" / "rejected.jsonl", rejected)
    sums = queue_sums(conn)
    report = {
        "ok": True,
        "task": args.task,
        "lane": lane.name,
        "model_snapshot": lane.model_snapshot,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_accepted_unique": accepted_unique_count(conn),
        "spend_cny": sums["spend"],
        "cny_per_1k_accepted_unique": (
            (float(sums["spend"]) / max(1, accepted_unique_count(conn))) * 1000.0
        ),
        "work_dir": str(work.relative_to(ROOT)) if work.is_relative_to(ROOT) else str(work),
        "ledger": ledger.snapshot() if ledger else None,
        "http_stats": {
            "n": stats["n"],
            "n429": stats["n429"],
            "nerr": stats["nerr"],
            "rate_429": stats["n429"] / max(1, stats["n"]),
            "retry_error_rate": stats["nerr"] / max(1, stats["n"]),
            "workers": ramp_state["workers"],
            "qps": ramp_state["qps"],
        },
    }
    dump_json(work / "cost-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
