#!/usr/bin/env python3
"""Freeze the conditional 2B unique-expansion ledger. Does not download FineWeb2."""

from __future__ import annotations

import json
from pathlib import Path

from repo_paths import ROOT, TRAINING_V1

CONTRACT = TRAINING_V1 / "recipes/cpt-2b-conditional.json"
BLOCK = ROOT / "notebook/corpus/lm-v3/BLOCKED-until-1b-gate.json"


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    block = json.loads(BLOCK.read_text(encoding="utf-8"))
    quotas = contract["incremental_quotas"]
    if sum(quotas.values()) != 1_000_000_000:
        raise RuntimeError("2B incremental quotas must sum to 1B")
    if sum(contract["cumulative_quotas"].values()) != 2_000_000_000:
        raise RuntimeError("2B cumulative quotas must sum to 2B")
    needed = contract["fresh_unique_needed"]
    if int(needed["total"]) != 849_912_161:
        raise RuntimeError("fresh unique total drifted")
    if int(needed["collection_target"]) < 900_000_000:
        raise RuntimeError("collection target must be >= 900M")
    if contract.get("append_into_lm_v2") is not False:
        raise RuntimeError("2B must not append into lm-v2")
    if "cwt2" not in json.dumps(contract["language_source"]["exclude"]).lower():
        raise RuntimeError("2B must exclude CWT2")
    if contract["status"] != "blocked_until_1b_benefit_gate":
        raise RuntimeError("2B recipe must stay blocked until the 1B gate")
    if int(block["fresh_unique_needed"]) != int(needed["total"]):
        raise RuntimeError("lm-v3 block ledger mismatch")
    print(json.dumps({"ok": True, "status": contract["status"], "fresh_unique_needed": needed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
