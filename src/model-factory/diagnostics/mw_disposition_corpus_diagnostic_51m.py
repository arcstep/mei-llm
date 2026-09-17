#!/usr/bin/env python3
"""Diagnostic-only: does the rebuild-v2 MW disposition corpus carry a
learnable signal for the 20-class disposition head, trained directly on the
RAW (float, unquantized, no retrieval/agent-alignment) 600M base LM?

This is explicitly NOT a reproduction of the formal
`mei-51m-adaptive-productization-v5` pipeline's `mw_disposition_v5` stage,
which trains on top of a CQ2-quantized, retrieval-R2-and-agent-aligned
checkpoint (stage order: Base -> QAT -> retrieval R0 -> full-call SFT ->
retrieval R1 -> alignment replay -> retrieval R2 -> agent alignment ->
MW disposition). Reproducing those prerequisites is the full formal run this
diagnostic exists to avoid committing to blindly. Oracle-mode candidate
batches are used throughout (`_oracle_ranking` needs no retrieval head --
only the `retrieved_tools` field already present in the corpus), since no
trained retrieval head exists yet on this raw base.

Writes only to a scoped `cycles/mei-1.1-51m/exp-00600m/runs/`
diagnostic run directory. Never touches CURRENT.json, any release, or a
formal cycle stage receipt. `model-factory/contracts/CODE_CATALOG.json`
should classify this file `diagnostic_only`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _ensure_python_paths() -> None:
    for path in (
        ROOT / "src/architecture/mei-1.2-51m",
        ROOT / "src/model-factory",
        ROOT / "src/platform/python-sdk",
        ROOT / "src/platform/_shared/runtime",
        ROOT / "src/corpus-factory/generators",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_python_paths()

import orchestration.productize_controller as lifecycle  # noqa: E402
import training.tool_use.adaptive_tool_context_51m as adaptive  # noqa: E402
import training.tool_use.sft_v3_training_51m as training  # noqa: E402
import evaluation.tool_use.sft_v3_eval_51m as evaluation  # noqa: E402
from rebuild_zh_v1 import common as C  # noqa: E402

BASE_WEIGHTS = (
    ROOT / "models/mei-1.2-51m/releases/exp-000600m/base"
    / "mei-1.0-51m-base-cpt600m-clean-source-v3-v1/mei-1.0-51m-base-cpt600m-clean-source-v3-v1.npz"
)
RELEASE_ID = "mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2"
RELEASE_DIR = C.RELEASE_ROOT / RELEASE_ID
RUNS_ROOT = ROOT / "cycles/mei-1.1-51m/exp-00600m/runs"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _remap_source_row(row: dict[str, Any]) -> dict[str, Any]:
    """rebuild_zh_v1 mw.py row -> the `source` shape iter_mw_visible_batches
    expects: raw (not effective) label under `reason_class_id`/`reason_code`."""

    return {
        "sample_id": row["case_id"],
        "split": row["split"],
        "family": row.get("candidate_tool") or "unknown",
        "reason_class_id": row["raw_reason_class_id"],
        "reason_code": row["raw_reason_code"],
        "candidate_tool": row.get("candidate_tool"),
        "retrieved_tools": row.get("retrieved_tools") or [],
        "query": row.get("query") or "",
        "context": row.get("context") or {},
        "evidence": row.get("evidence") or [],
        "permissions": row.get("permissions") or {},
        "state": row.get("state") or {},
        "history": row.get("history") or [],
        "tool_results": [],
    }


def load_split(split: str) -> list[dict[str, Any]]:
    path = RELEASE_DIR / "compiled" / "mw_disposition" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [_remap_source_row(row) for row in rows]


def build_visible_batches(runtime: Any, rows: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in ("compact", "standard"):
        for view in adaptive.iter_mw_visible_batches(
            runtime, rows, catalog, calibration={}, retrieval_mode="oracle", profile=profile,
        ):
            out.append(view)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-limit", type=int, default=400)
    args = parser.parse_args()

    run_id = f"mw-corpus-diagnostic-v2-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    _write_json(run_dir / "STATUS.json", {"status": "running", "run_id": run_id, "diagnostic_only": True})

    log(f"loading raw float 600M base runtime from {BASE_WEIGHTS.name} (quantized=False, no heads)")
    runtime = lifecycle._load_runtime(BASE_WEIGHTS, quantized=False)

    log("loading deploy tool catalog + rebuild-v2 mw_disposition compiled splits")
    deploy = C.load_deploy_tools()
    # portable wire schema only allows name/description/parameters(+permission
    # /state keys) on a tool object -- project away the registry's extra
    # provenance metadata (family/fingerprint/similar_to/...) before it is
    # validated by platform/_shared/runtime/schema_subset.py.
    catalog = [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in deploy.tools
    ]
    train_source = load_split("train")
    dev_source = load_split("dev")
    test_source = load_split("test")
    log(f"source rows: train={len(train_source)} dev={len(dev_source)} test={len(test_source)}")

    log("building oracle-mode mw-visible-batches (compact+standard) -- no model calls, no retrieval head needed")
    train_views = build_visible_batches(runtime, train_source, catalog)
    dev_views = build_visible_batches(runtime, dev_source, catalog)
    test_views = build_visible_batches(runtime, test_source, catalog)
    _write_jsonl(run_dir / "mw-visible.train.oracle.jsonl", train_views)
    _write_jsonl(run_dir / "mw-visible.dev.oracle.jsonl", dev_views)
    _write_jsonl(run_dir / "mw-visible.test.oracle.jsonl", test_views)
    train_eligible = [r for r in train_views if r.get("mw_eligible")]
    log(f"visible-batch rows: train={len(train_views)} (eligible={len(train_eligible)}) dev={len(dev_views)} test={len(test_views)}")
    classes_present = sorted({int(r["effective_reason_class_id"]) for r in train_eligible})
    log(f"effective classes present in train: {len(classes_present)}/20 -> {classes_present}")

    log(f"training MW disposition head for {args.steps} steps on raw float base (mechanism+corpus-signal diagnostic only)")
    train_report = training.train_mw_disposition_v3(
        runtime, train_eligible, {}, [], steps=args.steps, checkpoint_dir=run_dir / "checkpoints",
    )
    log(f"train done: last_loss={train_report['last_loss']:.4f} relabeled_class0_to_10={train_report['ready_relabelled_capability_insufficient']}")

    log("evaluating on dev/test")
    dev_report, dev_predictions = evaluation.evaluate_frozen_mw_batches(runtime, dev_views, limit=args.eval_limit)
    test_report, test_predictions = evaluation.evaluate_frozen_mw_batches(runtime, test_views, limit=args.eval_limit)
    _write_jsonl(run_dir / "mw-frozen.dev.predictions.jsonl", dev_predictions)
    _write_jsonl(run_dir / "mw-frozen.test.predictions.jsonl", test_predictions)
    log(f"dev: accuracy={dev_report.get('accuracy')} macro_f1={dev_report.get('macro_f1')}")
    log(f"test: accuracy={test_report.get('accuracy')} macro_f1={test_report.get('macro_f1')}")

    report = {
        "schema": "mei-51m-mw-corpus-diagnostic-v1",
        "diagnostic_only": True,
        "not_a_formal_release": True,
        "not_representative_of_final_cq2_aligned_checkpoint": True,
        "run_id": run_id,
        "base_weights_sha256": C.sha256_file(BASE_WEIGHTS),
        "source_release_id": RELEASE_ID,
        "steps": args.steps,
        "classes_present_in_train": classes_present,
        "train_report": train_report,
        "dev_report": dev_report,
        "test_report": test_report,
        "baseline_from_real_adaptive_v5_pipeline_for_reference_only": {
            "mw_dev_accuracy": 0.226,
            "mw_dev_macro_f1": 0.030093,
            "note": "measured on the CQ2-quantized, retrieval-R2/agent-aligned checkpoint at the end of the formal pipeline -- not directly comparable to this diagnostic's raw-float-base numbers, reference only",
        },
        "current_json_mutated": False,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(run_dir / "diagnostic-report.json", report)
    _write_json(run_dir / "STATUS.json", {"status": "complete", "run_id": run_id, "diagnostic_only": True})
    log("diagnostic complete")
    print(json.dumps({"run_dir": str(run_dir), "dev": dev_report, "test": test_report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
