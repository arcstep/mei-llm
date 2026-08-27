#!/usr/bin/env python3
"""Build mei-tool-schema-v1 eval bank (seven slices). Does not modify v1/v2/MW bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mei_tool_schema_lib import GENERATOR_VERSION, SERIALIZER_ID, dump_jsonl, make_eval_rows
from repo_paths import EVAL_BANKS_ROOT, ROOT

BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1"


def main() -> int:
    eval_rows, dev_rows = make_eval_rows()
    BANK.mkdir(parents=True, exist_ok=True)
    eval_path = BANK / "eval-bank-v0.jsonl"
    dev_path = BANK / "dev-bank-v0.jsonl"
    dump_jsonl(eval_path, eval_rows)
    dump_jsonl(dev_path, dev_rows)
    lock = {
        "id": "mei-tool-schema-v1",
        "serializer": SERIALIZER_ID,
        "generator_version": GENERATOR_VERSION,
        "n_eval": len(eval_rows),
        "n_dev": len(dev_rows),
        "slices": sorted({r["slice"] for r in eval_rows + dev_rows}),
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "does_not_modify": [
            "notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl",
            "notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl",
            "notebook/evaluation/banks/needle-vrm-mw-v0/eval-bank-v0.jsonl",
        ],
    }
    (BANK / "holdout-schema-v1.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    recipe = {
        "id": "mei-tool-schema-v1",
        "slices": {
            "S0": "seen",
            "S1": "unseen",
            "S2": "schema-mutation",
            "S3": "rename",
            "S4": "crosstalk",
            "S5": "type-stress",
            "S6": "refuse",
        },
        "isolation": "whole toolset / canonical family / template / counterfactual",
    }
    (BANK / "holdout-schema-v1.recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
