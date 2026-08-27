#!/usr/bin/env python3
"""Build 10k-evidence.json from template 2k vs paid 2k reports. Does not spend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    PACK_MEI_MW_DISPOSITION_V2_2K,
    PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    PACK_MEI_RETRIEVAL_V2_2K,
    PACK_MEI_RETRIEVAL_V2_2K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_canonical_lib import load_jsonl  # noqa: E402
from sft_synth_lib import dump_json, sha256_text  # noqa: E402

DRAFT = ROOT / "notebook/jobs/toolcall-sft/outbox/draft"

TEMPLATE_PACKS = {
    "retrieval": PACK_MEI_RETRIEVAL_V2_2K,
    "fullcall": PACK_MEI_TOOLCALL_V2_ORACLE_2K,
    "mw": PACK_MEI_MW_DISPOSITION_V2_2K,
}
PAID_PACKS = {
    "retrieval": PACK_MEI_RETRIEVAL_V2_2K_PAID,
    "fullcall": PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    "mw": PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
}


def pack_unique(path: Path) -> int:
    if not path.is_file():
        return 0
    return len({sha256_text(str(r.get("query") or "").strip()) for r in load_jsonl(path) if str(r.get("query") or "").strip()})


def unique_of(report: dict, task: str) -> int:
    for line in report.get("lines") or []:
        if line.get("task") == task:
            return int(line.get("n_accepted_unique") or 0)
    return 0


def compiler_floor(report: dict) -> float:
    rates = []
    for line in report.get("lines") or []:
        for stats in (line.get("lanes") or {}).values():
            if "compiler_pass_rate" in stats:
                rates.append(float(stats["compiler_pass_rate"] or 0))
        teacher = ((line.get("teacher") or {}).get("stdout") or {})
        if teacher.get("n_accepted") and teacher.get("n_rejected") is not None:
            n = int(teacher["n_accepted"]) + int(teacher["n_rejected"])
            if n:
                rates.append(int(teacher["n_accepted"]) / n)
    return min(rates) if rates else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=DRAFT / "2k-template-report.json")
    ap.add_argument("--paid", type=Path, default=DRAFT / "2k-paid-report.json")
    ap.add_argument("--canary", type=Path, default=DRAFT / "canary/canary-report.json")
    ap.add_argument("--out", type=Path, default=DRAFT / "10k-evidence.json")
    args = ap.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    paid = json.loads(args.paid.read_text(encoding="utf-8"))
    canary = json.loads(args.canary.read_text(encoding="utf-8")) if args.canary.is_file() else {}
    tasks = ("retrieval", "fullcall", "mw")
    t_unique = {t: pack_unique(TEMPLATE_PACKS[t]) for t in tasks}
    p_unique = {t: pack_unique(PAID_PACKS[t]) for t in tasks}
    n_unique = min(p_unique.values()) if p_unique else 0
    isolation_ok = bool((paid.get("isolation") or {}).get("returncode") == 0) and bool(
        ((paid.get("isolation") or {}).get("stdout") or {}).get("ok", True)
    )
    holdout = int(paid.get("park_holdout_hits") or 0)
    spend = float(paid.get("spend_cny") or 0)
    cap = float(paid.get("max_spend_cny") or 300)
    paid_compiler = compiler_floor(paid)
    unique_ok = all(t_unique[t] >= 2000 and p_unique[t] >= 2000 for t in tasks)
    quality_ok = bool(paid.get("ok")) and unique_ok and paid_compiler >= 0.98 and holdout == 0
    cost_ok = spend <= cap + 1e-6
    curve_ok = unique_ok and isolation_ok and cost_ok
    out = {
        "allow_spend": True,
        "global_budget_cny": 1000.0,
        "quality_ok": quality_ok,
        "isolation_ok": isolation_ok,
        "cost_ok": cost_ok,
        "park_blind_ok": holdout == 0,
        "learning_curve_ok": curve_ok,
        "n_accepted_unique": n_unique,
        "scale_2k_paid": bool(paid.get("ok")),
        "scale_2k_template": bool(template.get("ok")),
        "sources": {
            "canary": "notebook/jobs/toolcall-sft/outbox/draft/canary/canary-report.json",
            "template_2k": "notebook/jobs/toolcall-sft/outbox/draft/2k-template-report.json",
            "paid_2k": "notebook/jobs/toolcall-sft/outbox/draft/2k-paid-report.json",
            "paid_isolation": "notebook/jobs/toolcall-sft/outbox/draft/2k-paid-isolation.json",
            "probe": "notebook/jobs/toolcall-sft/outbox/draft/snapshot-probe.json",
        },
        "canary": {
            "ok": bool(canary.get("ok")),
            "spend_cny": canary.get("spend_cny"),
            "scale_2k_paid": canary.get("scale_2k_paid"),
        },
        "template_2k": {
            "retrieval_accepted_unique": t_unique["retrieval"],
            "fullcall_accepted_unique": t_unique["fullcall"],
            "mw_accepted_unique": t_unique["mw"],
            "spend_cny": float(template.get("spend_cny") or 0),
            "isolation_ok": bool((template.get("isolation") or {}).get("returncode") == 0),
            "note": "Unique counted from live packs, not stale report fields.",
        },
        "paid_2k": {
            "ok": bool(paid.get("ok")),
            "retrieval_accepted_unique": p_unique["retrieval"],
            "fullcall_accepted_unique": p_unique["fullcall"],
            "mw_accepted_unique": p_unique["mw"],
            "spend_cny": spend,
            "max_spend_cny": cap,
            "compiler_pass_floor": paid_compiler,
            "isolation_ok": isolation_ok,
            "park_holdout_hits": holdout,
            "fails": [line.get("fails") for line in paid.get("lines") or []],
        },
        "compare": {
            "unique_delta": {t: p_unique[t] - t_unique[t] for t in tasks},
            "paid_vs_template_unique_met": unique_ok,
            "note": "Compare quality/isolation/cost. Do not majority-vote gold. Do not auto-train.",
        },
        "note": (
            "Paid 2k curve justifies 10k phase cap 500 CNY. Do not auto-train."
            if quality_ok and isolation_ok and cost_ok
            else "Hold 10k. 20k/50k remain zero this envelope."
        ),
    }
    dump_json(args.out, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if quality_ok and isolation_ok and cost_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
