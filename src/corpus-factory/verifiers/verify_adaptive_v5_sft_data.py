#!/usr/bin/env python3
"""Fail-closed verifier for the adaptive-v5 SFT/data v2 design freeze.

It verifies only control-plane artifacts.  It does not create samples, start
CPT/SFT, inspect a live run, or mutate CURRENT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
TRACKS = {
    "retrieval_r3": (40, {"train": 24, "dev": 8, "valid": 8, "locked_test": 0}),
    "full_call_v2": (40, {"train": 24, "dev": 8, "valid": 8, "locked_test": 0}),
    "mw_v2": (100, {"train": 60, "dev": 20, "valid": 20, "locked_test": 0}),
    "agent_outcome_v2": (40, {"train": 24, "dev": 8, "valid": 8, "locked_test": 0}),
    "narration_terminal_v2": (40, {"train": 24, "dev": 8, "valid": 8, "locked_test": 0}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def verify(design_path: Path) -> dict[str, Any]:
    design = load(design_path)
    errors: list[str] = []
    if design.get("schema") != "mei-51m-adaptive-v5-sft-data-design-v2":
        errors.append("schema mismatch")
    if design.get("status") != "design_frozen_no_samples":
        errors.append("design is not a no-sample freeze")
    invariants = design.get("global_invariants") or {}
    for key, expected in {
        "provider_calls": 0, "paid_cny": 0, "generated_semantic_tasks": 0,
        "training_started": False, "current_mutated": False, "cpt_started": False,
        "mtp_in_scope": False, "eval_fixture_may_enter_training": False,
    }.items():
        if invariants.get(key) != expected:
            errors.append(f"invariant mismatch: {key}")
    base = design.get("base") or {}
    if base.get("role") != "frozen_experiment_base_only" or base.get("continuation_cpt_parent_eligible") is not False:
        errors.append("600M Base role is not experiment-only")
    if set(base.get("known_confounds") or []) != {"hybrid_recovery", "corpus_diversity_degraded"}:
        errors.append("known 600M confounds are incomplete")
    evidence = design.get("evidence") or {}
    source_merkle = hashlib.sha256(canonical({"base": base, "evidence": evidence})).hexdigest()
    if design.get("source_merkle_root") != source_merkle:
        errors.append("source Merkle root mismatch")
    for name, path_key, hash_key in (
        ("base", "release_path", "release_sha256"),
        ("paired", "paired_comparison_path", "paired_comparison_sha256"),
        ("eval", "locked_eval_v7_path", "locked_eval_v7_sha256"),
        ("sft", "sft_v4_manifest_path", "sft_v4_manifest_sha256"),
    ):
        source = base if name == "base" else evidence
        path = ROOT / str(source.get(path_key) or "")
        if not path.is_file() or sha256(path) != source.get(hash_key):
            errors.append(f"source hash mismatch: {name}")
    rows = design.get("tracks") or []
    indexed = {str(row.get("track_id")): row for row in rows if isinstance(row, Mapping)}
    if set(indexed) != set(TRACKS):
        errors.append("track set mismatch")
    for track_id, (worlds, splits) in TRACKS.items():
        row = indexed.get(track_id) or {}
        if row.get("fixture_worlds") != worlds or row.get("splits") != splits:
            errors.append(f"fixture contract mismatch: {track_id}")
        if (row.get("splits") or {}).get("locked_test") != 0:
            errors.append(f"locked test leaked into fixture: {track_id}")
    r3 = indexed.get("retrieval_r3") or {}
    if r3.get("catalog_sizes") != [10, 20, 50] or r3.get("required_case_counts") != {"no_match": 8, "multiple_plausible": 8, "gold_rank_6_20": 12}:
        errors.append("R3 coverage contract mismatch")
    mw = indexed.get("mw_v2") or {}
    if mw.get("classes") != 20 or mw.get("per_class") != 5 or mw.get("forbidden_label_confound") != "retrieval_failure":
        errors.append("MW class/isolation contract mismatch")
    narration = indexed.get("narration_terminal_v2") or {}
    if narration.get("input_contract") != "verified_terminal_result_view_only" or set(narration.get("forbidden") or []) != {"intermediate_tool_loop", "execution_authority"}:
        errors.append("narration boundary mismatch")
    recipe = design.get("ab_recipe") or {}
    if recipe.get("control") != "same_600m_base_plus_unchanged_sft_v4_recipe" or recipe.get("treatment") != "one_track_delta_only":
        errors.append("A/B isolation contract mismatch")
    required_match = {"base_weights", "cq2_master", "recipe", "seed", "serializer", "grammar", "scorer", "runtime_scope", "eval_lock"}
    if set(recipe.get("must_match") or []) != required_match or recipe.get("model_release_eligible") is not False:
        errors.append("A/B lineage contract mismatch")
    return {
        "schema": "mei-51m-adaptive-v5-sft-data-design-verification-v2",
        "status": "passed" if not errors else "blocked",
        "design_sha256": sha256(design_path),
        "errors": errors,
        "provider_calls": 0,
        "generated_semantic_tasks": 0,
        "training_started": False,
        "current_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.design)
    data = canonical(receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_bytes() != data:
        raise SystemExit(f"immutable receipt differs: {args.out}")
    args.out.write_bytes(data)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
