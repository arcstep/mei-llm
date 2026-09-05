#!/usr/bin/env python3
"""Orchestrator for mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1.

Runs all six family generators, audits (dedup/leakage/budget/coverage),
writes cycle-bound immutable artifacts under
.local/artifacts/mei-1.0-51m/exp-000600m/corpus/sft-suite/<release-id>/, and
reports gate status. Never touches CURRENT.json, an existing release, or any
training run. Offline only -- no provider calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebuild_zh_v1 import common as C
from rebuild_zh_v1 import retrieval as RETR
from rebuild_zh_v1 import fullcall as FC
from rebuild_zh_v1 import agent as AG
from rebuild_zh_v1 import mw as MW
from rebuild_zh_v1 import narration as NARR
from rebuild_zh_v1 import confidence as CONF

RELEASE_ID = "mei-1.0-51m-exp-000600m-sft-zh-rebuild-v2"
RELEASE_DIR = C.RELEASE_ROOT / RELEASE_ID
CYCLE_ID = C.CYCLE_ID
BASE_BINDING = {
    "base_id": "mei-1.0-51m-base-cpt600m-clean-source-v3-v1",
    "base_weights_sha256": C.BASE_WEIGHTS_SHA256,
}
SPLIT_NAMES = C.SPLIT_NAMES

# v2 vs v1: MW scaled up ~3.6x (140->500/class) after auditing the old
# v4-300m-v4 mw-disposition corpus and finding 12/20 classes (including
# capability_insufficient itself, 0/375 rows) had candidate_tool=null --
# no tool-level ground truth at all, so an in-place relabel-repair of the
# old 13,763-row corpus was not viable and v1's 140/class was thin relative
# to the old corpus's 294-3263/class range. Every v2 MW row carries a
# concrete, grounded candidate_tool by construction (see mw_scenarios.py).
#
# v3 (2026-09-05, user-approved rebalance): v2's 500/class MW was
# over-compensation (59% of the release) and parameter filling lacked the
# hard normalization skills a tool agent needs. v3 quotas rebalance toward
# retrieval (throughput-critical gate) and full_call, and full_call gains
# six arg_norm_* scenarios (clock/ISO time, enum aliases, duration
# conversion, large/decimal Chinese numerals, entity resolution) whose gold
# is locally compiled deterministic normalization. MW keeps full 20-class
# coverage at 200/class.
RETRIEVAL_TARGETS = dict(
    rank_1_5=700, rank_6_10=700, rank_11_15=450, rank_16_20=450,
    no_match=1000, cross_batch_exhausted=500, stop_before_scan=700,
    hard_negative_discrimination=500,
)
FULLCALL_TARGETS = dict(per_scenario_count=400, per_refusal_count=180)
AGENT_TARGETS = dict(per_kind_count=333)
MW_TARGETS = dict(per_class_count=200, visibility_pair_count=400, neighbor_pair_count=500)
NARRATION_TARGETS = dict(max_rows=2000)
CONFIDENCE_TARGETS = dict(fullcall_n=600, agent_n=400, retrieval_n=500)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------
def visible_text_of(row: Mapping[str, Any]) -> str:
    parts = [str(row.get("query") or "")]
    for turn in row.get("history") or []:
        parts.append(str(turn.get("content") or ""))
    return " ".join(parts)


def audit_dedup(rows: Sequence[Mapping[str, Any]], *, designed_dup_key: str | None) -> dict[str, Any]:
    index = C.DedupIndex()
    exact_dup = 0
    near_dup = 0
    undesigned_exact_dup = 0
    for row in rows:
        text = visible_text_of(row)
        is_dup, kind = index.check_and_add(row["case_id"], text)
        if is_dup:
            if kind == "exact":
                exact_dup += 1
                if not (designed_dup_key and row.get(designed_dup_key)):
                    undesigned_exact_dup += 1
            else:
                near_dup += 1
    return {
        "total_rows": len(rows),
        "exact_duplicate_rows": exact_dup,
        "near_duplicate_rows": near_dup,
        "undesigned_exact_duplicate_rows": undesigned_exact_dup,
        "undesigned_exact_duplicate_rate": round(undesigned_exact_dup / len(rows), 4) if rows else 0.0,
    }


def audit_leakage(rows: Sequence[Mapping[str, Any]], *, forbidden_tokens: Sequence[str] = ()) -> dict[str, Any]:
    hits = 0
    examples: list[str] = []
    for row in rows:
        text = visible_text_of(row)
        found = C.scan_leakage(text, forbidden_tokens)
        if found:
            hits += 1
            if len(examples) < 5:
                examples.append(f"{row['case_id']}: {found}")
    return {"total_rows": len(rows), "rows_with_leakage_markers": hits, "examples": examples}


def audit_budget(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fits = sum(1 for r in rows if r.get("budget", {}).get("fits"))
    unrepresentable = sum(1 for r in rows if r.get("budget", {}).get("context_unrepresentable"))
    return {
        "total_rows": len(rows),
        "fits": fits,
        "fit_rate": round(fits / len(rows), 4) if rows else 0.0,
        "context_unrepresentable": unrepresentable,
    }


# ---------------------------------------------------------------------------
# Gates (corpus-side preconditions; model-quality gates are explicitly
# reported as pending_sft, never asserted from corpus statistics alone).
# ---------------------------------------------------------------------------
def compute_retrieval_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_scen = ("rank_6_10", "rank_11_15", "rank_16_20")
    rank_rows = [r for r in rows if r["scenario"] in rank_scen]
    gold_ok = sum(
        1 for r in rank_rows
        if r["gold_name"] in r["catalog_tool_names"]
        and r["catalog_tool_names"][r["gold_rank"] - 1] == r["gold_name"]
    )
    no_match_rows = [r for r in rows if r["scenario"].startswith("no_match")]
    no_match_correctly_labeled = sum(1 for r in no_match_rows if r["gold_name"] is None and r["retrieval_terminal_reason"] == "retrieval_no_match")
    batch_hits = {i: sum(1 for r in rows if r.get("found_at_batch_index") == i) for i in range(4)}
    stop_rows = [r for r in rows if r["retrieval_terminal_reason"].startswith("stop_")]
    return {
        "rank_gt5_rows": len(rank_rows),
        "rank_gt5_corpus_side_retention": round(gold_ok / len(rank_rows), 4) if rank_rows else None,
        "rank_gt5_retention_model_measurement": "pending_sft",
        "no_match_rows": len(no_match_rows),
        "no_match_correctly_labeled_rate": round(no_match_correctly_labeled / len(no_match_rows), 4) if no_match_rows else None,
        "no_match_false_selection_rate_model_measurement": "pending_sft",
        "batch_hit_counts": {f"batch_{i}": n for i, n in batch_hits.items()},
        "batch_2_3_4_covered": all(batch_hits[i] > 0 for i in (1, 2, 3)),
        "scan_stop_rows": len(stop_rows),
        "scan_stop_classes_covered": sorted({r["retrieval_terminal_reason"] for r in stop_rows}),
    }


def compute_mw_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class_split: dict[str, dict[str, int]] = {}
    for r in rows:
        code = r["reason_code"]
        split = r["split"]
        by_class_split.setdefault(code, {}).setdefault(split, 0)
        by_class_split[code][split] += 1
    all_codes = set(C.mw_reason_codes())
    missing = sorted(all_codes - set(by_class_split))
    thin = {
        code: splits for code, splits in by_class_split.items()
        if any(splits.get(s, 0) == 0 for s in SPLIT_NAMES)
    }
    return {
        "classes_present": len(by_class_split),
        "classes_expected": len(all_codes),
        "classes_missing_entirely": missing,
        "classes_missing_a_split": sorted(thin),
        "min_rows_per_class_per_split": min(
            (min(splits.get(s, 0) for s in SPLIT_NAMES) for splits in by_class_split.values()), default=0
        ),
    }


def compute_agent_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    non_empty_verified = sum(
        1 for r in rows if all(s["tool_result"].get("verified") and s["tool_result"].get("payload") for s in r["steps"])
    )
    return {
        "total_rows": len(rows),
        "non_empty_verified_result_rate": round(non_empty_verified / len(rows), 4) if rows else None,
        "by_step_count": {n: sum(1 for r in rows if r["step_count"] == n) for n in (1, 2, 3, 4)},
    }


def compute_fullcall_gates(rows: Sequence[Mapping[str, Any]], deploy: C.ToolRegistry) -> dict[str, Any]:
    execute_rows = [r for r in rows if r["kind"] == "execute"]
    schema_valid = 0
    for r in execute_rows:
        tool = deploy.by_name.get(r["gold_name"])
        if tool is None:
            continue
        errors = C.validate_instance(tool.get("parameters") or {}, r["gold_args"])
        if not errors:
            schema_valid += 1
    return {
        "execute_rows": len(execute_rows),
        "full_schema_valid_rate": round(schema_valid / len(execute_rows), 4) if execute_rows else None,
        "refusal_rows": len(rows) - len(execute_rows),
    }


def compute_confidence_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels_all_null = all(r["label"] is None for r in rows)
    return {
        "total_rows": len(rows),
        "all_labels_null_pre_harvest": labels_all_null,
        "positive_class_present": any(r["expected_kind"] == "execute" for r in rows),
        "negative_class_present": any(r["expected_kind"] != "execute" for r in rows),
    }


def compute_narration_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grounded = 0
    for r in rows:
        result = r.get("verified_terminal_result")
        if r["outcome_kind"] == "refused":
            grounded += 1  # refusal narration has no ToolResult by design
        elif result and result.get("verified"):
            grounded += 1
    return {
        "total_rows": len(rows),
        "grounded_in_verified_terminal_result_rate": round(grounded / len(rows), 4) if rows else None,
        "by_outcome_kind": {k: sum(1 for r in rows if r["outcome_kind"] == k) for k in {r["outcome_kind"] for r in rows}},
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_family(rows: list[dict[str, Any]], family_dir_name: str, *, base_dir: Path) -> dict[str, Any]:
    semantic_path = base_dir / "semantic" / f"{family_dir_name}.jsonl"
    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    with semantic_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(C.canonical_json(row) + "\n")

    compiled_dir = base_dir / "compiled" / family_dir_name
    compiled_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in SPLIT_NAMES:
        split_rows = [r for r in rows if r["split"] == split]
        path = compiled_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in split_rows:
                fh.write(C.canonical_json(row) + "\n")
        split_counts[split] = len(split_rows)
    return {"semantic_path": str(semantic_path.relative_to(base_dir)), "split_counts": split_counts, "total": len(rows)}


def build_manifest(base_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(base_dir.rglob("*.jsonl")):
        rel = str(path.relative_to(base_dir))
        data = path.read_bytes()
        artifacts[rel] = {
            "bytes": len(data),
            "rows": data.count(b"\n"),
            "sha256": C.sha256_bytes(data),
        }
    return {"artifacts": artifacts, "artifact_merkle_root": C.merkle_root(artifacts)}


def main() -> int:
    global RELEASE_ID, RELEASE_DIR, CYCLE_ID, BASE_BINDING
    ap = argparse.ArgumentParser(description="rebuild-zh-v1 six-family SFT release builder")
    ap.add_argument("--release-id", default=RELEASE_ID)
    ap.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    ap.add_argument("--cycle-id", default=CYCLE_ID)
    ap.add_argument("--base-id", default=BASE_BINDING["base_id"])
    ap.add_argument("--base-weights-sha256", default=BASE_BINDING["base_weights_sha256"])
    args = ap.parse_args()
    if args.release_dir.exists():
        raise SystemExit(f"write-once refusal: release dir already exists: {args.release_dir}")
    RELEASE_ID = args.release_id
    RELEASE_DIR = args.release_dir
    CYCLE_ID = args.cycle_id
    BASE_BINDING = {"base_id": args.base_id, "base_weights_sha256": args.base_weights_sha256}

    t0 = time.time()
    log("loading registries + tokenizer")
    deploy = C.load_deploy_tools()
    training = C.load_training_tools()
    tokenizer_fallback = C.tokenizer_is_fallback()

    log(f"generating retrieval ({sum(RETRIEVAL_TARGETS.values())} target rows)")
    retrieval_rows, retrieval_cov = RETR.generate(deploy, RETRIEVAL_TARGETS)

    log("generating full_call")
    fullcall_rows, fullcall_cov = FC.generate(deploy, **FULLCALL_TARGETS)

    log("generating agent")
    agent_rows, agent_cov = AG.generate(deploy, **AGENT_TARGETS)

    log("generating mw_disposition")
    mw_rows, mw_cov = MW.generate(deploy, **MW_TARGETS)

    log("generating narration (terminal only)")
    narration_rows, narration_cov = NARR.generate(deploy, fullcall_rows, agent_rows, **NARRATION_TARGETS)

    log("freezing confidence candidates (label=null)")
    confidence_rows, confidence_cov = CONF.generate(fullcall_rows, agent_rows, retrieval_rows, **CONFIDENCE_TARGETS)

    families = {
        "retrieval": retrieval_rows,
        "full_call": fullcall_rows,
        "agent": agent_rows,
        "mw_disposition": mw_rows,
        "narration": narration_rows,
        "confidence": confidence_rows,
    }

    log("auditing dedup/leakage/budget")
    audits: dict[str, Any] = {}
    for name, rows in families.items():
        designed_key = "minimal_pair_id" if name == "mw_disposition" else None
        audits[name] = {
            "dedup": audit_dedup(rows, designed_dup_key=designed_key),
            "leakage": audit_leakage(rows, forbidden_tokens=C.mw_reason_codes() if name != "mw_disposition" else ()),
            "budget": audit_budget(rows) if rows and "budget" in rows[0] else None,
        }

    log("computing gates")
    gates = {
        "retrieval": compute_retrieval_gates(retrieval_rows),
        "mw_disposition": compute_mw_gates(mw_rows),
        "agent": compute_agent_gates(agent_rows),
        "full_call": compute_fullcall_gates(fullcall_rows, deploy),
        "confidence": compute_confidence_gates(confidence_rows),
        "narration": compute_narration_gates(narration_rows),
        "budget_2048_joint_contract": {
            name: audits[name]["budget"] for name in families if audits[name]["budget"]
        },
        "leakage": {name: audits[name]["leakage"]["rows_with_leakage_markers"] for name in families},
    }

    log("writing family artifacts")
    write_reports: dict[str, Any] = {}
    for name, rows in families.items():
        write_reports[name] = write_family(rows, name, base_dir=RELEASE_DIR)

    manifest = build_manifest(RELEASE_DIR)

    coverage = {
        "retrieval": retrieval_cov, "full_call": fullcall_cov, "agent": agent_cov,
        "mw_disposition": mw_cov, "narration": narration_cov, "confidence": confidence_cov,
    }

    governance_dir = RELEASE_DIR / "governance"
    governance_dir.mkdir(parents=True, exist_ok=True)
    (governance_dir / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (governance_dir / "audits.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (governance_dir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    release_manifest = {
        "schema": "mei-51m-sft-zh-rebuild-release-v1",
        "release_id": RELEASE_ID,
        "cycle_id": CYCLE_ID,
        "product": C.PRODUCT,
        "params": C.PARAMS,
        "generator_id": C.GENERATOR_ID,
        "generator_version": C.GENERATOR_VERSION,
        "base_binding": BASE_BINDING,
        "tool_registry": {
            "deploy_tools": deploy.n_tools, "deploy_families": len(deploy.families),
            "training_tools": training.n_tools, "training_families": len(training.families),
            "deploy_sha256": deploy.sha256, "training_sha256": training.sha256,
        },
        "tokenizer_fallback_used": tokenizer_fallback,
        "families": {name: write_reports[name] for name in families},
        "artifact_merkle_root": manifest["artifact_merkle_root"],
        "process_complete": True,
        "training_started": False,
        "current_mutated": False,
        "provider_calls": 0,
        "paid_cny": 0,
        "human_review": {
            "method": "automated_multi_gate_audit_plus_agent_sampled_review",
            "human_review_performed_by_named_reviewer": False,
            "note": "Unlike mei-1.0-51m-sft-gap-pilot-v1 (100% human-reviewed by a named reviewer), this release relies on automated schema/dedup/leakage/budget/coverage gates plus a documented agent-sampled audit -- not a claim of full human review.",
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    (RELEASE_DIR / "release-manifest.json").write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = RELEASE_DIR / "manifests" / "artifact-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    log(f"done in {time.time() - t0:.1f}s; total rows = {sum(len(r) for r in families.values())}")
    print(json.dumps({"release_id": RELEASE_ID, "merkle_root": manifest["artifact_merkle_root"], "row_totals": {k: len(v) for k, v in families.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
