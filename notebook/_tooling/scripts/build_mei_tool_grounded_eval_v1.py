#!/usr/bin/env python3
"""Freeze mei-tool-grounded-v1 eval universe and lock before SFT."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from mei_tool_grounded_lib import build_universe, dump_jsonl, sha256_text
from repo_paths import EVAL_BANKS_ROOT, ROOT

BANK = EVAL_BANKS_ROOT / "mei-tool-grounded-v1"
QUOTA = {"S0": 24, "S1": 16, "S2": 16, "S3": 12, "S4": 4, "S5": 12, "S6": 4, "S7": 6}


def main() -> int:
    universe = build_universe(seed=7)
    by_slice: dict[str, list] = defaultdict(list)
    for row in universe:
        by_slice[str(row.get("slice") or "S0")].append(row)
    eval_rows: list[dict] = []
    used: set[tuple[str, str]] = set()
    for sl, n in QUOTA.items():
        picked = 0
        for row in by_slice.get(sl, []):
            key = (row["query"], row["toolset_id"])
            if key in used:
                continue
            rec = dict(row)
            rec["split"] = "eval"
            rec["item_id"] = f"EVAL-GRD-{sl}-{picked:04d}"
            rec["sample_id"] = rec["item_id"]
            eval_rows.append(rec)
            used.add(key)
            picked += 1
            if picked >= n:
                break
    eval_rows.sort(key=lambda r: r["item_id"])
    BANK.mkdir(parents=True, exist_ok=True)
    bank_path = BANK / "eval-bank-v0.jsonl"
    dump_jsonl(bank_path, eval_rows)
    lock = {
        "id": "mei-tool-grounded-v1",
        "n_eval": len(eval_rows),
        "eval_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "generator_version": "mei-tool-grounded-v1",
        "protocol": "mei-route-protocol-v1",
        "serializer": "mei-route-serializer-v1",
        "quota": QUOTA,
        "slices": {sl: sum(1 for r in eval_rows if r["slice"] == sl) for sl in QUOTA},
        "used_keys": sorted(f"{q}|{t}" for q, t in used),
        "universe_n": len(universe),
        "note": "Frozen before SFT. Do not retarget items to raise scores.",
    }
    lock_path = BANK / "holdout-grounded-v1.lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BANK / "universe-pool-v0.jsonl").write_text(
        "".join(json.dumps({"query": r["query"], "toolset_id": r["toolset_id"], "slice": r["slice"], "kind": r["kind"]}, ensure_ascii=False) + "\n" for r in universe),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n_eval": len(eval_rows),
                "lock": str(lock_path.relative_to(ROOT)),
                "eval_sha256": lock["eval_sha256"],
                "slices": lock["slices"],
            },
            indent=2,
        )
    )
    return 0 if eval_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
