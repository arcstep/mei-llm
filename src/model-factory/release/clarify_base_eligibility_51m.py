#!/usr/bin/env python3
"""Write a hash-bound observer receipt separating Base eligibility axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from common.paths import CURRENT_PATH


EXPECTED_PARAMS = 51_463_797
EXPECTED_TENSORS = 400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _artifact(release_path: Path, release: dict[str, Any], role: str) -> Path:
    files = release.get("files") or {}
    value = files.get(role)
    if not value:
        value = release.get("weights" if role == "weights" else "train_state")
    if not value:
        raise RuntimeError(f"Base release does not declare {role}")
    return (release_path.parent / str(value)).resolve()


def _scan_checkpoint(weights: Path, state: Path) -> dict[str, Any]:
    with np.load(weights, allow_pickle=False) as archive:
        weight_names = list(archive.files)
        weight_shapes: dict[str, tuple[int, ...]] = {}
        scalar_count = 0
        weights_finite = True
        for name in weight_names:
            array = archive[name]
            weight_shapes[name] = tuple(array.shape)
            scalar_count += int(array.size)
            if array.dtype.kind in {"f", "c"} and not bool(
                np.isfinite(array).all()
            ):
                weights_finite = False
    with np.load(state, allow_pickle=False) as archive:
        state_names = list(archive.files)
        parameter_names = [name[2:] for name in state_names if name.startswith("p.")]
        optimizer_names = [name[2:] for name in state_names if name.startswith("o.")]
        shapes_match = True
        state_finite = True
        for name in state_names:
            array = archive[name]
            if name.startswith("p."):
                parameter_name = name[2:]
                if weight_shapes.get(parameter_name) != tuple(array.shape):
                    shapes_match = False
            if array.dtype.kind in {"f", "c"} and not bool(
                np.isfinite(array).all()
            ):
                state_finite = False
    passed = (
        len(weight_names) == EXPECTED_TENSORS
        and scalar_count == EXPECTED_PARAMS
        and parameter_names == weight_names
        and bool(optimizer_names)
        and shapes_match
        and weights_finite
        and state_finite
    )
    return {
        "status": "passed" if passed else "blocked",
        "parameter_tensor_count": len(weight_names),
        "parameter_count": scalar_count,
        "optimizer_tensor_count": len(optimizer_names),
        "parameter_names_and_order_exact": parameter_names == weight_names,
        "parameter_shapes_exact": shapes_match,
        "all_numeric_tensors_finite": weights_finite and state_finite,
    }


def build_receipt(release_path: Path, diversity_path: Path) -> dict[str, Any]:
    release_path = release_path.resolve()
    diversity_path = diversity_path.resolve()
    release = load_json(release_path)
    diversity = load_json(diversity_path)
    if diversity.get("schema") != "mei-synthetic-corpus-diversity-receipt-v1":
        raise RuntimeError("unsupported diversity audit schema")
    if diversity.get("terminal_status") not in {"passed", "degraded"}:
        raise RuntimeError("diversity audit is not terminal")
    weights = _artifact(release_path, release, "weights")
    state = _artifact(release_path, release, "train_state")
    expected_weights = str(release.get("weights_sha256") or "")
    expected_state = str(release.get("state_sha256") or "")
    if (
        not weights.is_file()
        or not state.is_file()
        or sha256_file(weights) != expected_weights
        or sha256_file(state) != expected_state
    ):
        raise RuntimeError("Base checkpoint artifacts are missing or hash-drifted")
    scan = _scan_checkpoint(weights, state)
    declared = release.get("numeric_integrity") or {}
    if declared and declared.get("status") != "passed":
        raise RuntimeError("Base release numeric_integrity is not passed")
    continuation = scan["status"] == "passed"
    degraded = bool(diversity.get("corpus_diversity_degraded"))
    productization = bool(
        release.get("productization_experiment_eligible", continuation)
    ) and continuation
    return {
        "schema": "mei-base-eligibility-clarification-v1",
        "base": {
            "model_id": release.get("model_id"),
            "tokens_seen_exposure": int(release.get("tokens_seen_exposure") or 0),
            "release": str(release_path),
            "release_sha256": sha256_file(release_path),
            "weights": str(weights),
            "weights_sha256": expected_weights,
            "train_state": str(state),
            "state_sha256": expected_state,
            "lineage_assurance": release.get("lineage_assurance", "legacy_frozen_base"),
        },
        "numeric_integrity": scan,
        "corpus_diversity": {
            "receipt": str(diversity_path),
            "receipt_sha256": sha256_file(diversity_path),
            "status": diversity.get("terminal_status"),
            "corpus_diversity_degraded": degraded,
        },
        "eligibility": {
            "continuation_checkpoint_eligible": continuation,
            "productization_experiment_eligible": productization,
            "automatic_parent_promotion_eligible": continuation and not degraded,
            "corpus_reuse_eligible": not degraded,
        },
        "semantics": {
            "continuation_checkpoint_eligible": (
                "weights, optimizer state, tensor identity and numeric integrity permit CPT continuation"
            ),
            "automatic_parent_promotion_eligible": (
                "candidate may be promoted without first resolving recorded corpus-quality blockers"
            ),
            "corpus_reuse_eligible": (
                "the audited synthetic slices may be rolled forward unchanged"
            ),
        },
        "current_sha256_observed": sha256_file(CURRENT_PATH),
        "current_mutated": False,
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return
        raise FileExistsError(f"write-once clarification already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--diversity-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.base_release, args.diversity_audit)
    write_once(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
