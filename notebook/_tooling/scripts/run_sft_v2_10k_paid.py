#!/usr/bin/env python3
"""Paid 10k candidate packs. Reuse accepted 2k, add ~8k unique per task. Phase cap 500 CNY."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    JOB_FULLCALL_V2,
    JOB_MW_DISPOSITION_V1,
    JOB_RETRIEVAL_V2,
    PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
    PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    PACK_MEI_RETRIEVAL_V2_10K_PAID,
    PACK_MEI_RETRIEVAL_V2_2K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    ROOT,
    SCRIPTS_ROOT,
    TOPIC_TOOLCALL_SFT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from jobs.paths import job_work, rel_to_root  # noqa: E402
from run_sft_v2_2k_paid import (  # noqa: E402
    clone_cases,
    lane_ok,
    merge_accepted,
    metrics_from_work,
    park_holdout_hits,
    pending_count,
    produce_lane,
    public_lane,
    reenable_qwen,
    unique_queries,
)
from sft_canonical_lib import dump_jsonl, load_jsonl  # noqa: E402
from sft_synth_lib import PAID_SHARE, assign_pareto_lanes, dump_json, sha256_text  # noqa: E402

LINES = (
    {
        "job": JOB_RETRIEVAL_V2,
        "task": "retrieval",
        "builder": "build_mei_retrieval_v2.py",
        "validator": "validate_mei_retrieval_v2_pack.py",
        "pack": PACK_MEI_RETRIEVAL_V2_10K_PAID,
        "seed_pack": PACK_MEI_RETRIEVAL_V2_2K_PAID,
    },
    {
        "job": JOB_FULLCALL_V2,
        "task": "fullcall",
        "builder": "build_mei_toolcall_v2.py",
        "validator": "validate_mei_toolcall_v2_pack.py",
        "pack": PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
        "seed_pack": PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
        "validator_args": lambda pack: ["--pack", str(pack), "--require-compiler"],
    },
    {
        "job": JOB_MW_DISPOSITION_V1,
        "task": "mw",
        "builder": "build_mei_mw_disposition_v2.py",
        "validator": "validate_mei_mw_disposition_v2.py",
        "pack": PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
        "seed_pack": PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    },
)

GATES = {
    "compiler_pass_min": 0.98,
    "max_429_rate": 0.01,
    "max_retry_error_rate": 0.02,
    "max_near_dup_rate": 0.05,
    "max_spend_cny": 500.0,
    "unique_target": 10000,
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


def namespace_10k(cases: list[dict]) -> list[dict]:
    out = []
    for src in cases:
        row = dict(src)
        cid = str(row.get("case_id") or "")
        if cid and not cid.startswith("10k::"):
            row["case_id"] = f"10k::{cid}"
        out.append(row)
    return out


def ensure_canonical(line: dict, *, limit: int, paid_root: Path) -> list[dict]:
    src = paid_root / "_canonical"
    canon_path = src / "canonical.jsonl"
    if canon_path.is_file():
        rows = load_jsonl(canon_path)
        if len(rows) >= limit:
            return rows[:limit]
    src.mkdir(parents=True, exist_ok=True)
    argv = [
        "--limit",
        str(limit),
        "--work-dir",
        str(src),
        "--out-dir",
        str(src),
        "--pack-name",
        "rows.jsonl",
    ]
    report = run(line["builder"], argv)
    if report.get("returncode") != 0:
        raise RuntimeError(f"builder {line['builder']} failed: {report.get('stderr') or report}")
    rows = load_jsonl(canon_path)
    if len(rows) < limit:
        raise RuntimeError(f"canonical n={len(rows)} < {limit} for {line['task']}")
    return rows[:limit]


def run_task(line: dict, *, limit: int, remaining_cny: float) -> dict:
    seed_pack: Path = line["seed_pack"]
    if not seed_pack.is_file():
        return {"task": line["task"], "ok": False, "error": f"missing 2k seed {rel_to_root(seed_pack, root=ROOT)}"}
    seed_rows = load_jsonl(seed_pack)
    seed_unique = unique_queries(seed_rows)
    if len(seed_unique) < 2000:
        return {"task": line["task"], "ok": False, "error": f"2k seed unique={len(seed_unique)} < 2000"}
    paid_root = job_work(TOPIC_TOOLCALL_SFT, line["job"], root=ROOT) / "candidates-10k-paid"
    paid_root.mkdir(parents=True, exist_ok=True)
    canonical = ensure_canonical(line, limit=limit, paid_root=paid_root)
    dump_jsonl(paid_root / "canonical.jsonl", canonical)
    need = max(0, limit - len(seed_unique))
    fresh = []
    for case in canonical:
        qh = sha256_text(str(case.get("query") or "").strip())
        if not qh or qh in seed_unique:
            continue
        fresh.append(case)
        if len(fresh) >= need:
            break
    fresh = namespace_10k(fresh)
    buckets = assign_pareto_lanes(fresh)
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
            report = produce_lane(
                line["task"],
                cases,
                work,
                lane,
                phase="10k",
                workers=4 if lane == "qwen-plus-2025-12-01" else None,
            )
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
        print(
            json.dumps(
                {
                    "progress": line["task"],
                    "lane": lane,
                    "n_accepted_unique": int(stats.get("n_accepted_unique") or 0),
                    "spend_cny": spend,
                    "fails": stats.get("fails") or [],
                    "skipped": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if spend >= remaining_cny:
            break
    accepted_paths = [works[k] / "raw" / "accepted.jsonl" for k in PAID_SHARE if k in works]
    existing_overgen = sorted(p for p in paid_root.glob("overgen-*") if p.is_dir())
    for path in existing_overgen:
        accepted_paths.append(path / "raw" / "accepted.jsonl")
        lane_reports[path.name] = metrics_from_work(path, "deepseek-v4-flash-0731")
    spend = sum(float(v.get("spend_cny") or 0) for v in lane_reports.values())
    merged_new = merge_accepted(accepted_paths)
    packed_seed_first = merge_accepted([seed_pack] + accepted_paths)
    unique = unique_queries(packed_seed_first)
    overgen_rounds = len(existing_overgen)
    volume = "deepseek-v4-flash-0731"
    http_complete = all(pending_count(buckets.get(k) or [], works[k]) == 0 for k in works)
    first_fails = [f for stats in lane_reports.values() for f in stats.get("fails") or []]
    cap_attempts = need * GATES["overgenerate"]
    while (
        len(unique) < GATES["unique_target"]
        and http_complete
        and not first_fails
        and attempted < max(cap_attempts, 1)
        and spend < remaining_cny
    ):
        gap = GATES["unique_target"] - len(unique)
        extra = namespace_10k(clone_cases(fresh or canonical, gap, f"og{overgen_rounds}"))
        work = paid_root / f"overgen-{overgen_rounds}"
        produce_lane(line["task"], extra, work, volume, phase="10k")
        stats = metrics_from_work(work, volume)
        stats["n_cases"] = len(extra)
        stats["fails"] = lane_ok(stats, n_cases=len(extra))
        lane_reports[f"overgen-{overgen_rounds}"] = stats
        attempted += len(extra)
        overgen_rounds += 1
        accepted_paths.append(work / "raw" / "accepted.jsonl")
        packed_seed_first = merge_accepted([seed_pack] + accepted_paths)
        unique = unique_queries(packed_seed_first)
        spend = sum(float(v.get("spend_cny") or 0) for v in lane_reports.values())
        if stats.get("fails"):
            first_fails.extend(stats["fails"])
            break
    packed = packed_seed_first[:limit]
    dump_jsonl(line["pack"], packed)
    vargs = line.get("validator_args")
    validator_args = vargs(line["pack"]) if callable(vargs) else ["--pack", str(line["pack"])]
    validator = run(line["validator"], validator_args)
    fails = []
    for name, stats in lane_reports.items():
        fails.extend(f"{name}:{f}" for f in stats.get("fails") or [])
    holdout = park_holdout_hits(packed)
    n_unique = len(unique_queries(packed))
    near_dup_rate = 0.0
    n_acc = sum(int(v.get("n_accepted") or 0) for v in lane_reports.values())
    n_near = sum(int(v.get("near_dup") or 0) for v in lane_reports.values())
    if n_acc + n_near:
        near_dup_rate = n_near / (n_acc + n_near)
    ok = (
        n_unique >= GATES["unique_target"]
        and validator.get("returncode") == 0
        and not fails
        and holdout == 0
        and spend <= remaining_cny + 1e-6
        and (n_unique >= GATES["unique_target"] or near_dup_rate <= GATES["max_near_dup_rate"] + 1e-12)
    )
    return {
        "task": line["task"],
        "ok": ok,
        "fails": fails,
        "n": len(packed),
        "n_accepted_unique": n_unique,
        "n_seed_unique": len(seed_unique),
        "n_fresh": len(fresh),
        "n_new_accepted": len(merged_new),
        "attempted": attempted,
        "overgen_rounds": overgen_rounds,
        "share": share,
        "spend_cny": spend,
        "near_dup_rate": near_dup_rate,
        "park_holdout_hits": holdout,
        "lanes": {k: public_lane(v) for k, v in lane_reports.items()},
        "validator": validator,
        "pack": rel_to_root(line["pack"], root=ROOT),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--max-spend-cny", type=float, default=500.0)
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
        print(
            json.dumps(
                {
                    "task_done": line["task"],
                    "ok": item.get("ok"),
                    "n_accepted_unique": item.get("n_accepted_unique"),
                    "spend_cny": item.get("spend_cny"),
                    "remaining_cny": remaining,
                    "fails": item.get("fails") or item.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        ok = ok and bool(item.get("ok"))
        if not item.get("ok"):
            break
    isolation = run("check_train_eval_isolation.py", ["--scope", "sft-v2"])
    isol_path = ROOT / "notebook/jobs/toolcall-sft/outbox/draft/10k-paid-isolation.json"
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
        "note": "Paid 10k candidates reuse 2k unique and add ~8k. Template/2k packs untouched. accepted waits for 0303. Do not majority-vote gold.",
    }
    dest = ROOT / "notebook/jobs/toolcall-sft/outbox/draft/10k-paid-report.json"
    dump_json(dest, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
