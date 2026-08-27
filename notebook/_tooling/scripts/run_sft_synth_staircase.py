#!/usr/bin/env python3
"""2k→10k→20k/50k staircase. Paid tiers refuse without --allow-spend and a positive budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import FLEET_SFT_SYNTH, ROOT, STAIRCASE_SFT_SYNTH

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_synth_lib import dump_json, load_fleet  # noqa: E402


def load_staircase(path: Path | None = None) -> dict:
    return json.loads((path or STAIRCASE_SFT_SYNTH).read_text(encoding="utf-8"))


def evaluate_tier(name: str, spec: dict, *, allow_spend: bool, evidence: dict) -> dict:
    need_spend = bool(spec.get("allow_spend"))
    target = int(spec.get("accepted_unique") or 0)
    gates = {
        "allow_spend_ok": (not need_spend) or allow_spend,
        "budget_positive": (not need_spend) or float(evidence.get("global_budget_cny") or 0) > 0,
        "phase_budget_ok": float(spec.get("max_spend_cny") or 0) > 0 or not need_spend,
        "quality_ok": bool(evidence.get("quality_ok", name in {"smoke", "canary"})),
        "isolation_ok": bool(evidence.get("isolation_ok", name in {"smoke", "canary"})),
        "cost_ok": bool(evidence.get("cost_ok", name in {"smoke", "canary"})),
        "learning_curve_ok": name not in {"20k", "50k"} or bool(evidence.get("learning_curve_ok")),
        "accepted_unique_met": (int(evidence.get("n_accepted_unique") or 0) >= target) if target else True,
    }
    if name in {"20k", "50k"} and float(spec.get("max_spend_cny") or 0) <= 0:
        gates["phase_budget_ok"] = False
    blocked = [k for k, ok in gates.items() if not ok]
    if name == "smoke":
        ok = gates["allow_spend_ok"] and gates["quality_ok"] and gates["isolation_ok"]
    elif name == "canary":
        ok = gates["allow_spend_ok"] and gates["budget_positive"] and gates["phase_budget_ok"]
    else:
        ok = not blocked
    return {
        "tier": name,
        "target_accepted_unique": target,
        "max_spend_cny": spec.get("max_spend_cny"),
        "ok": ok,
        "blocked": blocked,
        "gates": gates,
        "next": None if ok else "refuse_until_budget_and_gates",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=["smoke", "canary", "2k", "10k", "20k", "50k"])
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--evidence", type=Path, default=None, help="optional JSON evidence from a prior run")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stair = load_staircase()
    fleet = load_fleet(FLEET_SFT_SYNTH)
    evidence = {}
    if args.evidence and args.evidence.is_file():
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence.setdefault("global_budget_cny", fleet.get("global_budget_cny") or 0)
    spec = (stair.get("tiers") or {})[args.tier]
    result = evaluate_tier(args.tier, spec, allow_spend=args.allow_spend, evidence=evidence)
    result["overgenerate"] = stair.get("overgenerate")
    result["fleet_allow_spend_default"] = bool(fleet.get("allow_spend_default"))
    result["note"] = stair.get("note")
    out = args.out
    if out:
        dump_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.tier in {"2k", "10k", "20k", "50k", "canary"} and not args.allow_spend:
        return 4
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
