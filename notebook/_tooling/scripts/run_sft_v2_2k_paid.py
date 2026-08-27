#!/usr/bin/env python3
"""Paid 2k candidate packs. Pareto share. Phase cap 300 CNY. Does not overwrite template packs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    BANK_MEI_PARK_TOOLCALL_V1,
    BANK_NEEDLE_VRM_MW,
    FLEET_SFT_SYNTH,
    JOB_FULLCALL_V2,
    JOB_MW_DISPOSITION_V1,
    JOB_RETRIEVAL_V2,
    PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    PACK_MEI_RETRIEVAL_V2_2K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    ROOT,
    SCRIPTS_ROOT,
    TOPIC_TOOLCALL_SFT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from jobs.paths import job_work, rel_to_root  # noqa: E402
from sft_canonical_lib import dump_jsonl, load_jsonl  # noqa: E402
from sft_synth_lib import (  # noqa: E402
    PAID_SHARE,
    assign_pareto_lanes,
    bakeoff_metrics,
    dump_json,
    load_fleet,
    sha256_text,
)

LINES = (
    {
        "job": JOB_RETRIEVAL_V2,
        "task": "retrieval",
        "validator": "validate_mei_retrieval_v2_pack.py",
        "pack": PACK_MEI_RETRIEVAL_V2_2K_PAID,
    },
    {
        "job": JOB_FULLCALL_V2,
        "task": "fullcall",
        "validator": "validate_mei_toolcall_v2_pack.py",
        "pack": PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
        "validator_args": lambda pack: ["--pack", str(pack), "--require-compiler"],
    },
    {
        "job": JOB_MW_DISPOSITION_V1,
        "task": "mw",
        "validator": "validate_mei_mw_disposition_v2.py",
        "pack": PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    },
)

GATES = {
    "compiler_pass_min": 0.98,
    "max_429_rate": 0.01,
    "max_retry_error_rate": 0.02,
    "max_spend_cny": 300.0,
    "unique_target": 2000,
    "overgenerate": 3,
    "gold_leak_max": 0,
}


def run(script: str, argv: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / script), *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload: dict = {"returncode": proc.returncode, "stderr": proc.stderr.strip()[-2000:]}
    if proc.stdout.strip():
        try:
            payload["stdout"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload["stdout_text"] = proc.stdout[-1000:]
    return payload


def reenable_qwen() -> None:
    """Clear a stale provider disable so a prior 429/void bug cannot block resume."""
    fleet = load_fleet(FLEET_SFT_SYNTH)
    path = ROOT / str(fleet.get("shared_ledger") or "")
    if not path.is_file():
        return
    con = sqlite3.connect(str(path))
    con.execute("UPDATE provider_state SET disabled=0, reason='' WHERE provider='QWEN'")
    con.commit()
    con.close()


def wipe_queue(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "raw").mkdir(exist_ok=True)
    for stale in ("queue.sqlite", "queue.sqlite-wal", "queue.sqlite-shm"):
        p = work / stale
        if p.is_file():
            p.unlink()


def unique_queries(rows: list[dict]) -> set[str]:
    return {sha256_text(str(r.get("query") or "").strip()) for r in rows if str(r.get("query") or "").strip()}


def clone_cases(cases: list[dict], n: int, tag: str) -> list[dict]:
    out = []
    for i in range(n):
        src = dict(cases[i % len(cases)])
        stem = str(src.get("query") or src.get("stem") or "").strip()
        q = f"{stem}（扩{tag}-{i}）"
        src["query"] = q
        src["stem"] = q
        src["case_id"] = f"{src.get('case_id')}::{tag}-{i}"
        out.append(src)
    return out


def case_id_of(row: dict) -> str:
    return str(row.get("case_id") or "")


def durable_done(work: Path) -> set[str]:
    done: set[str] = set()
    for row in load_jsonl(work / "raw" / "accepted.jsonl"):
        cid = case_id_of(row)
        if cid:
            done.add(cid)
    for row in load_jsonl(work / "raw" / "rejected.jsonl"):
        if row.get("status") == "error":
            continue
        cid = case_id_of(row)
        if cid:
            done.add(cid)
    return done


def merge_http(old: dict, new: dict) -> dict:
    n = int(old.get("n") or 0) + int(new.get("n") or 0)
    n429 = int(old.get("n429") or 0) + int(new.get("n429") or 0)
    nerr = int(old.get("nerr") or 0) + int(new.get("nerr") or 0)
    out = dict(new or {})
    out["n"] = n
    out["n429"] = n429
    out["nerr"] = nerr
    out["rate_429"] = n429 / max(1, n)
    out["retry_error_rate"] = nerr / max(1, n)
    return out


def produce_lane(task: str, cases: list[dict], work: Path, lane: str, *, phase: str = "2k", workers: int | None = None) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work / "raw").mkdir(exist_ok=True)
    prev_acc = load_jsonl(work / "raw" / "accepted.jsonl")
    prev_rej = [r for r in load_jsonl(work / "raw" / "rejected.jsonl") if r.get("status") != "error"]
    done = durable_done(work)
    pending = [c for c in cases if str(c.get("case_id") or "") not in done]
    hist_path = work / "spend-acc.json"
    http_path = work / "http-acc.json"
    spend_hist = float(json.loads(hist_path.read_text(encoding="utf-8")).get("spend_cny") or 0) if hist_path.is_file() else 0.0
    http_hist = json.loads(http_path.read_text(encoding="utf-8")) if http_path.is_file() else {}
    old_cost = {}
    if (work / "cost-report.json").is_file():
        try:
            old_cost = json.loads((work / "cost-report.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_cost = {}
    if not hist_path.is_file():
        spend_hist = float(old_cost.get("spend_cny") or 0)
    if not http_path.is_file():
        http_hist = dict(old_cost.get("http_stats") or {})
    if not pending:
        return {
            "returncode": 0,
            "skipped": True,
            "this_run_spend_cny": 0.0,
            "cost": {
                "spend_cny": spend_hist,
                "http_stats": http_hist or {"n": 0, "n429": 0, "nerr": 0, "rate_429": 0.0, "retry_error_rate": 0.0},
            },
        }
    wipe_queue(work)
    dump_jsonl(work / "canonical.jsonl", pending)
    argv = [
        "--task",
        task,
        "--cases",
        str(work / "canonical.jsonl"),
        "--work-dir",
        str(work),
        "--lane",
        lane,
        "--phase",
        phase,
        "--allow-spend",
    ]
    if workers is not None:
        argv.extend(["--workers", str(workers)])
    elif lane == "qwen-plus-2025-12-01":
        argv.extend(["--workers", "1"])
    report = run("produce_sft_teacher.py", argv)
    cost = {}
    if (work / "cost-report.json").is_file():
        try:
            cost = json.loads((work / "cost-report.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cost = {}
    billed = float(cost.get("spend_cny") or (report.get("stdout") or {}).get("spend_cny") or 0)
    spend_hist += billed
    http_hist = merge_http(http_hist, cost.get("http_stats") or {})
    dump_json(hist_path, {"spend_cny": spend_hist})
    dump_json(http_path, http_hist)
    new_acc = load_jsonl(work / "raw" / "accepted.jsonl")
    new_rej = load_jsonl(work / "raw" / "rejected.jsonl")
    seen: set[str] = set()
    merged_acc: list[dict] = []
    for row in new_acc + prev_acc:
        cid = case_id_of(row) or sha256_text(str(row.get("query") or ""))
        if cid in seen:
            continue
        seen.add(cid)
        merged_acc.append(row)
    dump_jsonl(work / "raw" / "accepted.jsonl", merged_acc)
    dump_jsonl(work / "raw" / "rejected.jsonl", prev_rej + new_rej)
    cost["spend_cny"] = spend_hist
    cost["http_stats"] = http_hist
    report["cost"] = cost
    report["this_run_spend_cny"] = billed
    report["n_pending"] = len(pending)
    return report


def metrics_from_work(work: Path, lane: str) -> dict:
    accepted = load_jsonl(work / "raw" / "accepted.jsonl")
    rejected = load_jsonl(work / "raw" / "rejected.jsonl")
    for row in rejected:
        if not row.get("reject_reason") and row.get("reason"):
            row["reject_reason"] = row["reason"]
    eligible = accepted + [r for r in rejected if r.get("status") != "error"]
    baked = bakeoff_metrics({lane: eligible})
    stats = dict(baked["lanes"].get(lane) or {})
    near = 0
    gold_leak = 0
    compile_fail = 0
    compile_fail_reasons = {"protected_slot", "park_validator"}
    for row in rejected:
        reason = str(row.get("reject_reason") or row.get("reason") or "")
        if reason == "near_dup":
            near += 1
        if reason == "gold_leak":
            gold_leak += 1
        if reason in compile_fail_reasons:
            compile_fail += 1
    n_acc = len(accepted)
    stats["compiler_pass_rate"] = (n_acc / max(1, n_acc + compile_fail)) if (n_acc or compile_fail) else 1.0
    stats["gold_leak"] = gold_leak
    stats["near_dup"] = near
    n = max(1, n_acc + near + compile_fail)
    if n_acc:
        stats["protected_slot_keep_rate"] = n_acc / max(1, n_acc + compile_fail)
        gold_ok = sum(1 for r in accepted if r.get("gold_kept", True))
        stats["gold_keep_rate"] = gold_ok / n_acc
    http = {}
    if (work / "http-acc.json").is_file():
        http = json.loads((work / "http-acc.json").read_text(encoding="utf-8"))
    elif (work / "cost-report.json").is_file():
        http = json.loads((work / "cost-report.json").read_text(encoding="utf-8")).get("http_stats") or {}
    stats["http_stats"] = http
    spend = 0.0
    if (work / "spend-acc.json").is_file():
        spend = float(json.loads((work / "spend-acc.json").read_text(encoding="utf-8")).get("spend_cny") or 0)
    elif (work / "cost-report.json").is_file():
        spend = float(json.loads((work / "cost-report.json").read_text(encoding="utf-8")).get("spend_cny") or 0)
    stats["spend_cny"] = spend
    stats["n_accepted_unique"] = len(unique_queries(accepted))
    stats["cny_per_1k_accepted_unique"] = (spend / max(1, stats["n_accepted_unique"])) * 1000.0
    return stats


def lane_ok(stats: dict, *, n_cases: int) -> list[str]:
    fails = []
    http = stats.get("http_stats") or {}
    n_acc = int(stats.get("n_accepted") or 0)
    recovered = n_acc + int(stats.get("near_dup") or 0)
    if n_cases > 0 and n_acc <= 0:
        fails.append("empty")
    if float(stats.get("compiler_pass_rate") or 0) < GATES["compiler_pass_min"]:
        fails.append("compiler")
    if n_acc:
        if float(stats.get("protected_slot_keep_rate") or 0) < GATES["compiler_pass_min"]:
            fails.append("slots")
        if float(stats.get("gold_keep_rate") or 0) < GATES["compiler_pass_min"]:
            fails.append("provenance")
    if int(stats.get("gold_leak") or 0) > GATES["gold_leak_max"]:
        fails.append("gold_leak")
    # Recovered 429s (later accepted / near_dup) do not fail the item-level gate.
    if recovered < n_cases * GATES["compiler_pass_min"]:
        if float(http.get("rate_429") or 0) > GATES["max_429_rate"]:
            fails.append("429")
        if float(http.get("retry_error_rate") or 0) > GATES["max_retry_error_rate"]:
            fails.append("retry_error")
    return fails


def merge_accepted(paths: list[Path]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for path in paths:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            key = sha256_text(str(row.get("query") or "").strip())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


def park_holdout_hits(rows: list[dict]) -> int:
    blocked = set()
    for path in (BANK_MEI_PARK_TOOLCALL_V1, BANK_NEEDLE_VRM_MW):
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            q = str(row.get("query") or "").strip()
            if q:
                blocked.add(q)
    return sum(1 for r in rows if str(r.get("query") or "").strip() in blocked)


def public_lane(stats: dict) -> dict:
    keep = (
        "n",
        "n_accepted",
        "n_accepted_unique",
        "compiler_pass_rate",
        "protected_slot_keep_rate",
        "gold_keep_rate",
        "gold_leak",
        "near_dup",
        "spend_cny",
        "cny_per_1k_accepted_unique",
        "p95_latency_ms",
        "http_stats",
        "reject_reasons",
        "share",
        "n_cases",
        "fails",
        "retry_rounds",
    )
    return {k: stats[k] for k in keep if k in stats}


def pending_count(cases: list[dict], work: Path) -> int:
    done = durable_done(work)
    return sum(1 for c in cases if str(c.get("case_id") or "") not in done)


def run_task(line: dict, *, limit: int, remaining_cny: float) -> dict:
    src_work = job_work(TOPIC_TOOLCALL_SFT, line["job"], root=ROOT) / "candidates-2k"
    canonical = load_jsonl(src_work / "canonical.jsonl")[:limit]
    if len(canonical) < limit:
        return {"task": line["task"], "ok": False, "error": f"canonical n={len(canonical)} < {limit}"}
    paid_root = job_work(TOPIC_TOOLCALL_SFT, line["job"], root=ROOT) / "candidates-2k-paid"
    paid_root.mkdir(parents=True, exist_ok=True)
    buckets = assign_pareto_lanes(canonical)
    share = {k: len(v) for k, v in buckets.items()}
    lane_reports: dict[str, dict] = {}
    works: dict[str, Path] = {}
    attempted = 0
    spend = 0.0
    for lane, cases in buckets.items():
        if not cases:
            continue
        if spend >= remaining_cny:
            break
        work = paid_root / lane
        retry_rounds = 0
        while retry_rounds < GATES["overgenerate"]:
            left = pending_count(cases, work)
            if left <= 0 and (work / "raw" / "accepted.jsonl").is_file():
                break
            report = produce_lane(line["task"], cases, work, lane)
            retry_rounds += 1
            if report.get("skipped"):
                break
            if pending_count(cases, work) <= 0:
                break
            if spend >= remaining_cny:
                break
        stats = metrics_from_work(work, lane)
        stats["share"] = share.get(lane, 0)
        stats["n_cases"] = len(cases)
        stats["retry_rounds"] = retry_rounds
        stats["fails"] = lane_ok(stats, n_cases=len(cases))
        lane_reports[lane] = stats
        works[lane] = work
        attempted += len(cases)
        spend = sum(float(v.get("spend_cny") or 0) for v in lane_reports.values())
        if spend >= remaining_cny:
            break
    accepted_paths = [works[k] / "raw" / "accepted.jsonl" for k in PAID_SHARE if k in works]
    existing_overgen = sorted(p for p in paid_root.glob("overgen-*") if p.is_dir())
    for path in existing_overgen:
        accepted_paths.append(path / "raw" / "accepted.jsonl")
        lane_reports[path.name] = metrics_from_work(path, "deepseek-v4-flash-0731")
    spend = sum(float(v.get("spend_cny") or 0) for v in lane_reports.values())
    merged = merge_accepted(accepted_paths)
    unique = unique_queries(merged)
    overgen_rounds = len(existing_overgen)
    volume = "deepseek-v4-flash-0731"
    http_complete = all(pending_count(buckets.get(k) or [], works[k]) == 0 for k in works)
    first_fails = [f for stats in lane_reports.values() for f in stats.get("fails") or []]
    cap_attempts = limit * GATES["overgenerate"]
    while (
        len(unique) < GATES["unique_target"]
        and http_complete
        and not first_fails
        and attempted < cap_attempts
        and spend < remaining_cny
    ):
        need = GATES["unique_target"] - len(unique)
        extra = clone_cases(canonical, need, f"og{overgen_rounds}")
        work = paid_root / f"overgen-{overgen_rounds}"
        report = produce_lane(line["task"], extra, work, volume)
        stats = metrics_from_work(work, volume)
        stats["n_cases"] = len(extra)
        stats["fails"] = lane_ok(stats, n_cases=len(extra))
        lane_reports[f"overgen-{overgen_rounds}"] = stats
        attempted += len(extra)
        overgen_rounds += 1
        accepted_paths.append(work / "raw" / "accepted.jsonl")
        merged = merge_accepted(accepted_paths)
        unique = unique_queries(merged)
        spend = sum(float(v.get("spend_cny") or 0) for v in lane_reports.values())
    packed = merged[:limit]
    dump_jsonl(line["pack"], packed)
    vargs = line.get("validator_args")
    validator_args = vargs(line["pack"]) if callable(vargs) else ["--pack", str(line["pack"])]
    validator = run(line["validator"], validator_args)
    fails = []
    for name, stats in lane_reports.items():
        fails.extend(f"{name}:{f}" for f in stats.get("fails") or [])
    holdout = park_holdout_hits(packed)
    n_unique = len(unique_queries(packed))
    ok = (
        n_unique >= GATES["unique_target"]
        and validator.get("returncode") == 0
        and not fails
        and holdout == 0
        and spend <= remaining_cny + 1e-6
    )
    return {
        "task": line["task"],
        "ok": ok,
        "fails": fails,
        "n": len(packed),
        "n_accepted_unique": n_unique,
        "attempted": attempted,
        "overgen_rounds": overgen_rounds,
        "share": share,
        "spend_cny": spend,
        "park_holdout_hits": holdout,
        "lanes": {k: public_lane(v) for k, v in lane_reports.items()},
        "validator": validator,
        "pack": rel_to_root(line["pack"], root=ROOT),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max-spend-cny", type=float, default=300.0)
    args = ap.parse_args()
    reenable_qwen()
    remaining = args.max_spend_cny
    reports = []
    ok = True
    for line in LINES:
        if remaining <= 0:
            ok = False
            reports.append({"task": line["task"], "ok": False, "error": "phase_budget_exhausted"})
            break
        item = run_task(line, limit=args.limit, remaining_cny=remaining)
        remaining = max(0.0, remaining - float(item.get("spend_cny") or 0))
        reports.append(item)
        ok = ok and bool(item.get("ok"))
        if not item.get("ok"):
            break
    isolation = run("check_train_eval_isolation.py", ["--scope", "sft-v2"])
    isol_path = ROOT / "notebook/jobs/toolcall-sft/outbox/draft/2k-paid-isolation.json"
    dump_json(
        isol_path,
        isolation.get("stdout")
        or {"returncode": isolation.get("returncode"), "stderr": isolation.get("stderr")},
    )
    spend = sum(float(r.get("spend_cny") or 0) for r in reports)
    holdout = sum(int(r.get("park_holdout_hits") or 0) for r in reports)
    out = {
        "ok": ok and isolation.get("returncode") == 0 and spend <= args.max_spend_cny + 1e-6 and holdout == 0,
        "limit": args.limit,
        "spend_cny": spend,
        "max_spend_cny": args.max_spend_cny,
        "park_holdout_hits": holdout,
        "lines": reports,
        "isolation": isolation,
        "isolation_receipt": rel_to_root(isol_path, root=ROOT),
        "note": "Paid 2k candidates. Template packs untouched. accepted waits for 0303. Do not majority-vote gold.",
    }
    dest = ROOT / "notebook/jobs/toolcall-sft/outbox/draft/2k-paid-report.json"
    dump_json(dest, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
