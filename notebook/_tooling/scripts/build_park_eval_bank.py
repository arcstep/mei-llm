#!/usr/bin/env python3
"""Freeze mei-park EVAL bank, scenario/quick-phrase lock, and scorer spec."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repo_paths import EVAL_BANKS_ROOT, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from park_toolcall_lib import (  # noqa: E402
    PARK_BANK_DIR,
    holdout_lock_payload,
    park_eval_blind_rows,
    park_fingerprint,
)
from sft_canonical_lib import dump_jsonl  # noqa: E402
from sft_synth_lib import dump_json, sha256_obj  # noqa: E402


SCORER = {
    "id": "mei-park-toolcall-v1-scorer",
    "dynamic_scenarios": {
        "n": 12,
        "score_by": ["final_state", "validator_receipt", "tool_effect"],
        "forbid_fake_as_oracle": True,
    },
    "l0_l1_blind": {
        "match": "exact_normalized",
        "layers": ["command", "schedule", "autonomy", "refuse"],
        "tools": [
            "control_room_devices",
            "create_or_update_reservation",
            "cancel_reservation",
            "update_room_policy",
            "empty",
        ],
        "selected_entity": True,
        "absolute_relative_clamp": True,
        "state_tool_result_provenance": True,
    },
    "keep_0206": ["retrieval", "oracle-top5", "learned-top5", "grammar", "safety"],
}


def main() -> int:
    PARK_BANK_DIR.mkdir(parents=True, exist_ok=True)
    lock = holdout_lock_payload()
    rows = park_eval_blind_rows(160)
    dump_jsonl(PARK_BANK_DIR / "eval-bank-v1.jsonl", rows)
    dump_json(PARK_BANK_DIR / "holdout-v1.lock.json", lock)
    dump_json(PARK_BANK_DIR / "holdout-v1.recipe.json", {
        "bank_id": "mei-park-toolcall-v1",
        "n_blind": len(rows),
        "n_scenarios": 12,
        "n_quick_phrases": 5,
        "toolset_fingerprint": park_fingerprint(),
    })
    dump_json(PARK_BANK_DIR / "scorer-v1.json", SCORER)
    manifest = {
        "bank_id": "mei-park-toolcall-v1",
        "lock_sha256": sha256_obj(lock),
        "n_blind": len(rows),
        "toolset_fingerprint": park_fingerprint(),
        "promotable_to_train": False,
    }
    dump_json(PARK_BANK_DIR / "holdout-v1.manifest.json", manifest)
    (PARK_BANK_DIR / "NOT_PROMOTABLE.json").write_text(
        json.dumps({"ok": True, "reason": "eval_holdout", "train": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (PARK_BANK_DIR / "README.md").write_text(
        "# mei-park-toolcall-v1\n\n"
        "12 场景 + 5 快捷语 + L0/L1 exact-match blind。禁止进入 train。\n"
        "动态 12 场景按最终状态 + validator receipt + 工具效果评分，不用 fake 当 oracle。\n",
        encoding="utf-8",
    )
    report = {
        "ok": len(rows) >= 160 and bool(lock["toolset_fingerprint"]),
        "n_blind": len(rows),
        "dir": str(PARK_BANK_DIR.relative_to(ROOT)),
        "fingerprint": park_fingerprint(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
