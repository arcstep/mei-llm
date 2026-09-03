#!/usr/bin/env python3
"""Register CPT bases without implicitly mutating CURRENT.json.

Registration, freeze proposal, and CURRENT finalization deliberately require
different authority.  Only ``finalize_current`` may update CURRENT, and it
requires both an explicit confirmation and a compare-and-swap hash.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from common._repo import (
    ARTIFACT_ROOT,
    CURRENT_PATH,
    ROOT,
    TOKENIZER_ZH_V1,
    architecture_contracts,
)
from orchestration.lifecycle_51m import (
    EXPECTED_PARAMS,
    atomic_json,
    current_hash,
    identity_errors,
    live_stage_state,
    load_json,
    output_manifest_valid,
    recipe,
    relative,
    run_dir,
    sha256_file,
    sha256_json,
    stage_fingerprint,
    utc_now,
)


# Tests and migration tooling may override this boundary.  Production writes
# derive the cycle-local Base directory from cumulative exposure.
BASE_ROOT: Path | None = None
CANDIDATE_RE = re.compile(r"^mei-1\.0-51m-base-cpt[a-z0-9._-]+-v1$")


def _scan_numeric_npz(path: Path, *, label: str) -> dict:
    import numpy as np

    names: list[str] = []
    shapes: dict[str, list[int]] = {}
    parameter_count = 0
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            array = archive[name]
            if array.dtype.kind not in {"b", "u", "i", "f", "c"}:
                raise RuntimeError(f"{label} contains a non-numeric tensor: {name}")
            if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
                raise RuntimeError(f"{label} contains a non-finite tensor: {name}")
            names.append(name)
            shapes[name] = list(array.shape)
            parameter_count += int(array.size)
    if not names:
        raise RuntimeError(f"{label} has no tensors")
    return {
        "tensor_names": names,
        "tensor_shapes": shapes,
        "scalar_count": parameter_count,
    }


def _numeric_integrity(
    weights: Path,
    train_state: Path,
    train_state_meta: Path,
    summary: dict,
    config: dict,
    *,
    expected_parameter_tensors: int = 400,
    expected_params: int = EXPECTED_PARAMS,
) -> dict:
    weight_scan = _scan_numeric_npz(weights, label="CPT weights")
    state_scan = _scan_numeric_npz(train_state, label="CPT train state")
    weight_names = weight_scan["tensor_names"]
    state_parameter_names = [
        name[2:] for name in state_scan["tensor_names"] if name.startswith("p.")
    ]
    optimizer_names = [
        name[2:] for name in state_scan["tensor_names"] if name.startswith("o.")
    ]
    if (
        len(weight_names) != expected_parameter_tensors
        or weight_names != state_parameter_names
        or not optimizer_names
        or int(weight_scan["scalar_count"]) != expected_params
    ):
        raise RuntimeError("CPT weights/train-state tensor identity is incomplete")
    for name in weight_names:
        if weight_scan["tensor_shapes"][name] != state_scan["tensor_shapes"].get(
            "p." + name
        ):
            raise RuntimeError(f"CPT train-state parameter shape differs: {name}")
    metrics = {
        "valid_loss": summary.get("valid_loss"),
        "valid_loss_hq": summary.get("valid_loss_hq"),
        "valid_loss_structure": summary.get("valid_loss_structure"),
        "valid_loss_colloquial": summary.get("valid_loss_colloquial"),
        "final_probe_mean_nll": summary.get(
            "final_probe_mean_nll", summary.get("probe_mean_nll")
        ),
    }
    if any(
        value is None or not math.isfinite(float(value)) for value in metrics.values()
    ):
        raise RuntimeError("CPT terminal validation metrics are missing or non-finite")
    meta = load_json(train_state_meta)
    expected_contract = str(
        config.get("weight_contract_sha256")
        or architecture_contracts()["weight_contract_sha256"]
    )
    if (
        int(meta.get("tokens_seen") or 0) != int(summary.get("tokens_seen") or 0)
        or int(meta.get("params") or 0) != expected_params
        or meta.get("architecture_id") != config.get("architecture_id")
        or meta.get("weight_contract_sha256") != expected_contract
        or meta.get("tokenizer_sha256") != summary.get("tokenizer_sha256")
        or not meta.get("sampler_state")
    ):
        raise RuntimeError("CPT terminal train-state metadata is inconsistent")
    return {
        "schema": "mei-cpt-numeric-integrity-v1",
        "status": "passed",
        "parameter_tensor_count": len(weight_names),
        "parameter_count": int(weight_scan["scalar_count"]),
        "optimizer_tensor_count": len(optimizer_names),
        "all_numeric_tensors_finite": True,
        "parameter_names_and_order_exact": True,
        "parameter_shapes_exact": True,
        "terminal_metrics": metrics,
        "train_state_tokens_seen": int(meta["tokens_seen"]),
    }


def _candidate_id(config: dict, requested: str | None) -> str:
    value = requested or f"mei-1.0-51m-base-cpt{config['rung']}-v1"
    if not CANDIDATE_RE.fullmatch(value):
        raise ValueError("candidate id must be mei-1.0-51m-base-cpt<exposure>-v1")
    return value


def _base_root(config: dict) -> Path:
    if BASE_ROOT is not None:
        return BASE_ROOT
    target = int(config.get("target_exposure_tokens") or config["target_exposure"])
    millions = target // 1_000_000
    if millions <= 0:
        raise ValueError("target exposure must identify a positive cycle")
    return ARTIFACT_ROOT / f"exp-{millions:06d}m/models/base"


def _verified_inputs(run_id: str) -> tuple[Path, dict, dict]:
    directory = run_dir(run_id)
    config = load_json(directory / "run.json")
    if not config:
        raise RuntimeError(f"unknown lifecycle run {run_id}")
    gate = load_json(directory / "stages/cpt_gate/receipt.json")
    gate_row = recipe()["stages"]["cpt_gate"]
    expected_fingerprint = stage_fingerprint(directory, config, "cpt_gate", gate_row)
    gate_manifest = gate.get("output_manifest") or {}
    live = live_stage_state(directory)
    if (
        gate.get("status") != "passed"
        or gate.get("fingerprint") != expected_fingerprint
        or not gate_manifest
        or gate.get("output_manifest_sha256") != sha256_json(gate_manifest)
        or not output_manifest_valid(gate_manifest)
        or gate.get("current_sha256") != current_hash()
        or live.get("lock_held") is True
        or live.get("pid_alive") is True
    ):
        raise RuntimeError("register requires a current, hash-valid terminal cpt_gate")
    summary = load_json(directory / "checkpoints/cpt/summary.json")
    errors = identity_errors(summary, config)
    if errors:
        raise RuntimeError(f"CPT identity gate failed: {errors}")
    target = int(
        config.get("target_exposure_tokens") or config.get("target_exposure") or 0
    )
    tokens = int(summary.get("tokens_seen") or 0)
    if not (target <= tokens <= target + 8192):
        raise RuntimeError(f"CPT exposure mismatch: summary={tokens} target={target}")
    return directory, config, summary


def _formal_parent(config: dict, summary: dict) -> dict:
    schedule_path = ROOT / str(config["corpus"]["schedule"])
    schedule = load_json(schedule_path)
    checkpoint_rel = str(schedule.get("parent_checkpoint") or "")
    if not checkpoint_rel:
        raise RuntimeError(
            "continued CPT registration requires a formal parent checkpoint"
        )
    checkpoint = ROOT / checkpoint_rel
    release_path = checkpoint.parent / "RELEASE.json"
    release = load_json(release_path)
    if not checkpoint.is_file() or not release:
        raise RuntimeError("formal parent checkpoint or RELEASE.json is missing")
    expected_exposure = int(config["corpus"]["parent_exposure"])
    if int(release.get("tokens_seen_exposure") or 0) != expected_exposure:
        raise RuntimeError("formal parent exposure differs from the CPT schedule")
    if release.get("state_sha256") != sha256_file(checkpoint):
        raise RuntimeError("formal parent checkpoint SHA differs from RELEASE.json")
    tokenizer_sha = str(summary.get("tokenizer_sha256") or "")
    if not tokenizer_sha or tokenizer_sha != release.get("tokenizer_sha256"):
        raise RuntimeError("formal parent and child tokenizer identities differ")
    if not TOKENIZER_ZH_V1.is_file() or sha256_file(TOKENIZER_ZH_V1) != tokenizer_sha:
        raise RuntimeError("CPT tokenizer SHA does not match the canonical tokenizer")
    weights_name = str(release.get("weights") or "")
    weights = checkpoint.parent / weights_name
    if not weights_name or not weights.is_file():
        raise RuntimeError("formal parent weights are missing")
    if release.get("weights_sha256") != sha256_file(weights):
        raise RuntimeError("formal parent weights SHA differs from RELEASE.json")
    return {
        "model_id": release["model_id"],
        "release": relative(release_path),
        "release_sha256": sha256_file(release_path),
        "weights": relative(weights),
        "weights_sha256": release["weights_sha256"],
        "train_state": checkpoint_rel,
        "state_sha256": release["state_sha256"],
        "tokens_seen_exposure": expected_exposure,
    }


def _recovery_provenance(directory: Path, config: dict) -> tuple[str, dict]:
    receipt_path = directory / "jobs/recovery-source.json"
    receipt = load_json(receipt_path)
    configured = config.get("resume_checkpoint") or {}
    if not receipt and not configured:
        return "direct_parent", {}
    recovery = receipt or configured
    checkpoint_rel = str(recovery.get("checkpoint") or "")
    checkpoint = ROOT / checkpoint_rel
    expected_sha = str(recovery.get("checkpoint_sha256") or "")
    if (
        not checkpoint.is_file()
        or not expected_sha
        or sha256_file(checkpoint) != expected_sha
    ):
        raise RuntimeError("recovery checkpoint provenance is missing or drifted")
    metadata_rel = str(recovery.get("metadata") or "")
    metadata_sha = str(recovery.get("metadata_sha256") or "")
    if bool(metadata_rel) != bool(metadata_sha):
        raise RuntimeError("recovery metadata provenance is incomplete")
    if metadata_rel:
        metadata_path = ROOT / metadata_rel
        if not metadata_path.is_file() or sha256_file(metadata_path) != metadata_sha:
            raise RuntimeError("recovery metadata provenance is missing or drifted")
    source_receipt_rel = str(recovery.get("source_receipt") or "")
    source_receipt_sha = str(recovery.get("source_receipt_sha256") or "")
    if bool(source_receipt_rel) != bool(source_receipt_sha):
        raise RuntimeError("recovery source receipt provenance is incomplete")
    source_status = str(recovery.get("source_receipt_status") or "unknown")
    if source_receipt_rel:
        source_receipt_path = ROOT / source_receipt_rel
        if (
            not source_receipt_path.is_file()
            or sha256_file(source_receipt_path) != source_receipt_sha
        ):
            raise RuntimeError("recovery source receipt is missing or drifted")
        source_receipt = load_json(source_receipt_path)
        observed_status = str(
            source_receipt.get("terminal_status")
            or source_receipt.get("status")
            or "unknown"
        )
        if observed_status != source_status:
            raise RuntimeError("recovery source receipt status is inconsistent")
    elif source_status != "unknown":
        raise RuntimeError("recovery source receipt status has no bound receipt")
    assurance = (
        "verified_continuation" if source_status == "passed" else "hybrid_recovery"
    )
    return assurance, {
        "source_run_id": recovery.get("source_run_id"),
        "checkpoint": checkpoint_rel,
        "checkpoint_sha256": expected_sha,
        "metadata": metadata_rel or None,
        "metadata_sha256": metadata_sha or None,
        "tokens_seen": int(recovery.get("tokens_seen") or 0),
        "source_receipt": source_receipt_rel or None,
        "source_receipt_sha256": source_receipt_sha or None,
        "source_receipt_status": source_status,
        "source_receipt_artifacts_match": recovery.get(
            "source_receipt_artifacts_match"
        ),
        "receipt": relative(receipt_path) if receipt else None,
        "receipt_sha256": sha256_file(receipt_path) if receipt else None,
    }


def _corpus_quality(directory: Path) -> dict:
    receipt_path = directory / "jobs/synthetic-diversity-receipt.json"
    receipt = load_json(receipt_path)
    if not receipt:
        return {
            "status": "not_audited",
            "corpus_diversity_degraded": True,
            "receipt": None,
            "receipt_sha256": None,
            "corpus_reuse_eligible": False,
            "automatic_parent_promotion_eligible": False,
            "future_parent_eligible": False,
        }
    if receipt.get("schema") != "mei-synthetic-corpus-diversity-receipt-v1":
        raise RuntimeError("synthetic corpus diversity receipt schema is invalid")
    terminal_status = str(receipt.get("terminal_status") or "")
    if terminal_status not in {"passed", "degraded"}:
        raise RuntimeError("synthetic corpus diversity audit is not terminal")
    roles = receipt.get("roles") or {}
    if not isinstance(roles, dict) or not roles:
        raise RuntimeError("synthetic corpus diversity audit has no role evidence")
    role_degraded = False
    for role, report in roles.items():
        if not isinstance(report, dict) or report.get("status") not in {
            "passed",
            "degraded",
        }:
            raise RuntimeError(f"synthetic corpus diversity role is invalid: {role}")
        role_degraded = role_degraded or report.get("status") != "passed"
        source = report.get("input") or {}
        source_path = Path(str(source.get("path") or ""))
        source_sha = str(source.get("sha256") or "")
        if (
            not source_path.is_file()
            or not source_sha
            or sha256_file(source_path) != source_sha
        ):
            raise RuntimeError(
                f"synthetic corpus diversity input is missing or drifted: {role}"
            )
    degraded = bool(receipt.get("corpus_diversity_degraded"))
    if degraded != role_degraded or (terminal_status == "degraded") != degraded:
        raise RuntimeError("synthetic corpus diversity terminal status is inconsistent")
    return {
        "status": terminal_status,
        "corpus_diversity_degraded": degraded,
        "receipt": relative(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "corpus_reuse_eligible": not degraded,
        "automatic_parent_promotion_eligible": not degraded,
        "future_parent_eligible": not degraded,
    }


def register_base_candidate(run_id: str, candidate_id: str | None = None) -> dict:
    directory, config, summary = _verified_inputs(run_id)
    model_id = _candidate_id(config, candidate_id)
    base_root = _base_root(config)
    destination = base_root / model_id
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing base candidate: {destination}"
        )
    source_dir = directory / "checkpoints/cpt"
    source_files = {
        "weights": source_dir / "pretrain-cpt.npz",
        "train_state": source_dir / "pretrain-cpt-state.npz",
        "train_state_meta": source_dir / "pretrain-cpt-state.meta.json",
        "summary": source_dir / "summary.json",
    }
    missing = [name for name, path in source_files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing CPT candidate artifacts: {missing}")
    numeric_integrity = _numeric_integrity(
        source_files["weights"],
        source_files["train_state"],
        source_files["train_state_meta"],
        summary,
        config,
    )
    contracts = architecture_contracts()
    schedule = load_json(ROOT / config["corpus"]["schedule"])
    parent_checkpoint = str(schedule.get("parent_checkpoint") or "")
    formal_parent = _formal_parent(config, summary)
    lineage_assurance, recovery = _recovery_provenance(directory, config)
    corpus_quality = _corpus_quality(directory)
    source_manifest_path = directory / "source-manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_manifest_sha = str(config.get("source_manifest_sha256") or "")
    if (
        not source_manifest
        or source_manifest.get("manifest_sha256") != source_manifest_sha
    ):
        raise RuntimeError("launch source manifest identity is missing or drifted")
    release_blockers = []
    if corpus_quality["corpus_diversity_degraded"]:
        release_blockers.append("corpus_diversity_degraded")
    base_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{model_id}.register-", dir=base_root
    ) as raw:
        staging = Path(raw)
        copied: dict[str, str] = {}
        target_names = {
            "weights": f"{model_id}.npz",
            "train_state": f"{model_id}-state.npz",
            "train_state_meta": f"{model_id}-state.meta.json",
            "summary": "summary.json",
        }
        for role, source in source_files.items():
            target = staging / target_names[role]
            shutil.copy2(source, target)
            copied[role] = target.name
        release = {
            "schema_version": 2,
            "model_id": model_id,
            "kind": "base-cpt-candidate",
            "status": "registered_candidate",
            "product": "mei-1.0-51m",
            "architecture_id": config["architecture_id"],
            "weight_contract_sha256": contracts["weight_contract_sha256"],
            "runtime_profile_sha256": contracts["runtime_profile_sha256"],
            "training_aux_sha256": contracts["training_aux_sha256"],
            "architecture_source_sha256": summary.get("architecture_sha256"),
            "params": EXPECTED_PARAMS,
            "rung": config["rung"],
            "tokens_seen_exposure": int(summary["tokens_seen"]),
            "target_exposure_tokens": int(
                config.get("target_exposure_tokens") or config["target_exposure"]
            ),
            "incremental_exposure_tokens": int(
                config["corpus"]["incremental_exposure"]
            ),
            "parent_exposure_tokens": int(config["corpus"]["parent_exposure"]),
            "parent_checkpoint": parent_checkpoint,
            "formal_parent": formal_parent,
            "recovery_checkpoint": recovery,
            "lineage_assurance": lineage_assurance,
            "pure_single_source_exposure_comparison": False,
            "source_run": relative(directory),
            "source_capture_mode": config.get("source_capture_mode") or "reconstructed",
            "source_manifest": relative(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_file_sha256": sha256_file(source_manifest_path),
            "corpus_snapshot_sha256": config["corpus"]["snapshot_sha256"],
            "corpus_artifact_merkle_sha256": config["corpus"].get(
                "artifact_merkle_sha256"
            ),
            "corpus_quality": corpus_quality,
            "numeric_integrity": numeric_integrity,
            "tokenizer_sha256": summary.get("tokenizer_sha256"),
            "manifest_sha256": summary.get("manifest_sha256"),
            "schedule_sha256": summary.get("schedule_sha256"),
            "corpus_sha256": summary.get("corpus_sha256"),
            "valid_loss": summary.get("valid_loss"),
            "valid_loss_hq": summary.get("valid_loss_hq"),
            "valid_loss_structure": summary.get("valid_loss_structure"),
            "valid_loss_colloquial": summary.get("valid_loss_colloquial"),
            "final_probe_mean_nll": summary.get(
                "final_probe_mean_nll", summary.get("probe_mean_nll")
            ),
            "last_train_loss": summary.get("last_loss"),
            "files": copied,
            "weights_sha256": sha256_file(staging / copied["weights"]),
            "state_sha256": sha256_file(staging / copied["train_state"]),
            "state_meta_sha256": sha256_file(staging / copied["train_state_meta"]),
            "summary_sha256": sha256_file(staging / copied["summary"]),
            "registered_at": utc_now(),
            "current_promoted": False,
            "process_complete": True,
            "productization_experiment_eligible": True,
            "continuation_checkpoint_eligible": True,
            "automatic_parent_promotion_eligible": not release_blockers,
            "corpus_reuse_eligible": not corpus_quality[
                "corpus_diversity_degraded"
            ],
            "future_parent_eligible": not release_blockers,
            "release_eligible": not release_blockers,
            "promotion_prohibited": bool(release_blockers),
            "release_blockers": release_blockers,
            "public_distribution_clearance_asserted": False,
            "not_a_claim": "LM gates are not tool-calling ability.",
        }
        atomic_json(staging / "RELEASE.json", release)
        staging.rename(destination)
    return {
        "ok": True,
        "candidate": relative(destination),
        "release": relative(destination / "RELEASE.json"),
        "current_unchanged": True,
    }


def _candidate_dir(candidate: Path) -> Path:
    destination = candidate if candidate.is_absolute() else ROOT / candidate
    resolved = destination.resolve()
    if BASE_ROOT is not None:
        allowed = resolved.parent == BASE_ROOT.resolve()
    else:
        try:
            relative_parent = resolved.parent.relative_to(ARTIFACT_ROOT.resolve())
        except ValueError:
            allowed = False
        else:
            parts = relative_parent.parts
            allowed = (
                len(parts) == 3
                and re.fullmatch(r"exp-[0-9]{6}m", parts[0]) is not None
                and parts[1:] == ("models", "base")
            )
    if not allowed:
        raise ValueError("candidate must be a direct cycle models/base child")
    return resolved


def propose_freeze(candidate: Path) -> dict:
    destination = _candidate_dir(candidate)
    release = load_json(destination / "RELEASE.json")
    if release.get("status") != "registered_candidate":
        raise RuntimeError("freeze proposal requires a registered candidate")
    hash_keys = {
        "weights": "weights_sha256",
        "train_state": "state_sha256",
        "train_state_meta": "state_meta_sha256",
        "summary": "summary_sha256",
    }
    for role, name in (release.get("files") or {}).items():
        path = destination / str(name)
        if not path.is_file() or sha256_file(path) != release.get(hash_keys[role]):
            raise RuntimeError(f"candidate {role} hash mismatch")
    release_eligible = release.get("release_eligible") is True
    blockers = list(release.get("release_blockers") or [])
    proposal = {
        "schema_version": 1,
        "kind": "base-freeze-proposal",
        "candidate": relative(destination),
        "process_complete": release.get("process_complete") is True,
        "release_eligible": release_eligible,
        "eligible": release_eligible,
        "productization_experiment_eligible": (
            release.get("productization_experiment_eligible") is True
        ),
        "continuation_checkpoint_eligible": (
            release.get("continuation_checkpoint_eligible") is True
        ),
        "automatic_parent_promotion_eligible": (
            release.get("automatic_parent_promotion_eligible") is True
        ),
        "corpus_reuse_eligible": release.get("corpus_reuse_eligible") is True,
        "future_parent_eligible": release.get("future_parent_eligible") is True,
        "unmet_gates": blockers,
        "proposed_at": utc_now(),
        "current_sha256_observed": current_hash(),
        "current_mutated": False,
        "requires_explicit_finalize": True,
        "finalize_confirmation": "freeze mei-1.0-51m base",
    }
    atomic_json(destination / "freeze-proposal.json", proposal)
    return proposal


def finalize_current(
    candidate: Path, *, expected_current_sha256: str, confirmation: str
) -> dict:
    if confirmation != "freeze mei-1.0-51m base":
        raise PermissionError("explicit user freeze confirmation is required")
    if current_hash() != expected_current_sha256:
        raise RuntimeError(
            "CURRENT compare-and-swap failed; file changed after proposal"
        )
    destination = _candidate_dir(candidate)
    proposal = load_json(destination / "freeze-proposal.json")
    if proposal.get("eligible") is not True or proposal.get("candidate") != relative(
        destination
    ):
        raise RuntimeError("eligible freeze proposal is missing or mismatched")
    current = load_json(CURRENT_PATH)
    if current.get("product") != "mei-1.0-51m":
        raise RuntimeError("CURRENT product identity mismatch")
    release = load_json(destination / "RELEASE.json")
    current["base"] = relative(destination)
    current["stage"] = f"cpt-{release['rung']}-frozen"
    atomic_json(CURRENT_PATH, current)
    return {"ok": True, "base": current["base"], "current_sha256": current_hash()}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    register = sub.add_parser("register-base-candidate")
    register.add_argument("--run-id", required=True)
    register.add_argument("--candidate-id")
    propose = sub.add_parser("propose-freeze")
    propose.add_argument("--candidate", type=Path, required=True)
    finalize = sub.add_parser("finalize-current")
    finalize.add_argument("--candidate", type=Path, required=True)
    finalize.add_argument("--expected-current-sha256", required=True)
    finalize.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.action == "register-base-candidate":
        payload = register_base_candidate(args.run_id, args.candidate_id)
    elif args.action == "propose-freeze":
        payload = propose_freeze(args.candidate)
    else:
        payload = finalize_current(
            args.candidate,
            expected_current_sha256=args.expected_current_sha256,
            confirmation=args.confirmation,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
