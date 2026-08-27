#!/usr/bin/env python3
"""2k → 10k decision. 20k/50k stay at zero budget this round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import ROOT, STAIRCASE_SFT_SYNTH

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sft_synth_staircase import evaluate_tier, load_staircase  # noqa: E402
from sft_synth_lib import dump_json, load_fleet  # noqa: E402
from repo_paths import FLEET_SFT_SYNTH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "notebook/jobs/toolcall-sft/outbox/draft/10k-decision.json")
    args = ap.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    stair = load_staircase(STAIRCASE_SFT_SYNTH)
    fleet = load_fleet(FLEET_SFT_SYNTH)
    two_k = evaluate_tier("2k", stair["tiers"]["2k"], allow_spend=bool(evidence.get("allow_spend")), evidence=evidence)
    ten = evaluate_tier("10k", stair["tiers"]["10k"], allow_spend=bool(evidence.get("allow_spend")), evidence=evidence)
    curve_ok = bool(evidence.get("quality_ok")) and bool(evidence.get("isolation_ok")) and bool(evidence.get("cost_ok"))
    park_ok = bool(evidence.get("park_blind_ok", False))
    decide_10k = bool(two_k["ok"] and curve_ok and park_ok and evidence.get("n_accepted_unique", 0) >= 2000)
    out = {
        "ok": True,
        "scale_10k": decide_10k,
        "scale_20k": False,
        "scale_50k": False,
        "budget_10k_cny": 500.0 if decide_10k else 0.0,
        "budget_20k_cny": 0.0,
        "budget_50k_cny": 0.0,
        "gates": {
            "2k": two_k,
            "10k": ten,
            "learning_curve": curve_ok,
            "park_blind": park_ok,
        },
        "reason": (
            "2k curve, isolation, cost, and park blind all passed; 10k may use remaining 500 CNY phase cap."
            if decide_10k
            else "Hold 10k. 20k/50k remain zero this envelope."
        ),
        "fleet_global_budget_cny": fleet.get("global_budget_cny"),
    }
    dump_json(args.out, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if decide_10k else 4


if __name__ == "__main__":
    raise SystemExit(main())
