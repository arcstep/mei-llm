#!/usr/bin/env python3
"""Freeze mei-51m-longitudinal-eval-v8-retrieval-depth.

Extends eval-v7 (which stays authoritative and unmodified for
base-language/agent/mw/narration/schema-holdout banks -- see
cycles/mei-1.0-51m/exp-000600m/corpus/eval.json) with the retrieval-depth
coverage v7 lacked: only 60 no-match rows and 6-7 rank>5 rows, not enough to
measure the <=5% / >=99% gates. This lock reuses the `dev` (threshold
calibration) and `test` (frozen, never tuned) partitions of the
sft-zh-rebuild-v1 release's family/group-isolated split -- the same
cf_group-hash isolation that keeps SFT train/valid separate from these rows
-- rather than a second independent generation pass. `test` here is
literally never read by anything in this cycle after being written; no
threshold in gates.json was tuned against it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebuild_zh_v1 import common as C

SFT_RELEASE_ID = "mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2"
SFT_RELEASE_DIR = C.RELEASE_ROOT / SFT_RELEASE_ID
EVAL_LOCK_ID = "mei-51m-longitudinal-eval-v8-retrieval-depth-v2"
EVAL_LOCK_DIR = C.EVAL_LOCK_ROOT / EVAL_LOCK_ID

BANKS = ("retrieval", "mw_disposition", "full_call", "agent", "confidence", "narration")


def main() -> int:
    if not SFT_RELEASE_DIR.is_dir():
        raise SystemExit(f"run build.py first -- missing {SFT_RELEASE_DIR}")

    manifest = json.loads((SFT_RELEASE_DIR / "manifests" / "artifact-manifest.json").read_text(encoding="utf-8"))
    sft_merkle = manifest["artifact_merkle_root"]

    (EVAL_LOCK_DIR / "banks").mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    row_counts: dict[str, dict[str, int]] = {}
    for family in BANKS:
        row_counts[family] = {}
        for split, out_name in (("dev", "dev"), ("test", "test")):
            src = SFT_RELEASE_DIR / "compiled" / family / f"{split}.jsonl"
            if not src.is_file():
                continue
            data = src.read_bytes()
            dst = EVAL_LOCK_DIR / "banks" / f"{family}.{out_name}.jsonl"
            dst.write_bytes(data)
            rel = str(dst.relative_to(EVAL_LOCK_DIR))
            artifacts[rel] = {"bytes": len(data), "rows": data.count(b"\n"), "sha256": C.sha256_bytes(data)}
            row_counts[family][out_name] = data.count(b"\n")

    merkle = C.merkle_root(artifacts)

    lock = {
        "schema": "mei-51m-longitudinal-eval-lock-v8",
        "lock_id": EVAL_LOCK_ID,
        "cycle_id": C.CYCLE_ID,
        "extends": "mei-51m-longitudinal-eval-v7",
        "extends_relationship": "additive_retrieval_depth_coverage_only; v7 base-language/agent/mw/narration/schema-holdout banks remain authoritative and are not duplicated here",
        "source_sft_release_id": SFT_RELEASE_ID,
        "source_sft_release_merkle_root": sft_merkle,
        "isolation_mechanism": "cf_group sha256 hash bucketing (same mechanism separating SFT train/valid); dev+test partitions were never included in the SFT release's train/valid compiled splits",
        "locked_test_used_for_threshold_tuning": False,
        "banks": row_counts,
        "gate_targets": {
            "retrieval_no_match_false_selection_rate_max": 0.05,
            "retrieval_eligible_gold_recall_drop_max_pp": 0.5,
            "retrieval_rank_gt5_gold_retention_min": 0.99,
            "retrieval_batches_2_3_4_must_be_covered": True,
            "mw_macro_f1_min": None,
            "confidence_min_class_rows": {"positive": 100, "negative": 100},
        },
        "artifact_merkle_root": merkle,
        "process_complete": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (EVAL_LOCK_DIR / "lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (EVAL_LOCK_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    (EVAL_LOCK_DIR / "manifests" / "artifact-manifest.json").write_text(
        json.dumps({"artifacts": artifacts, "artifact_merkle_root": merkle}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"lock_id": EVAL_LOCK_ID, "merkle_root": merkle, "row_counts": row_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
