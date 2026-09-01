#!/usr/bin/env python3
"""Register CPT bases without implicitly mutating CURRENT.json.

Registration, freeze proposal, and CURRENT finalization deliberately require
different authority.  Only ``finalize_current`` may update CURRENT, and it
requires both an explicit confirmation and a compare-and-swap hash.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from _repo import CURRENT_PATH, ROOT, architecture_contracts
from lifecycle_51m import (
    EXPECTED_PARAMS,
    atomic_json,
    current_hash,
    identity_errors,
    load_json,
    relative,
    run_dir,
    sha256_file,
    utc_now,
)


BASE_ROOT = ROOT / "base"
CANDIDATE_RE = re.compile(r"^mei-1\.0-51m-base-cpt[a-z0-9._-]+-v1$")


def _candidate_id(config: dict, requested: str | None) -> str:
    value = requested or f"mei-1.0-51m-base-cpt{config['rung']}-v1"
    if not CANDIDATE_RE.fullmatch(value):
        raise ValueError("candidate id must be mei-1.0-51m-base-cpt<exposure>-v1")
    return value


def _verified_inputs(run_id: str) -> tuple[Path, dict, dict]:
    directory = run_dir(run_id)
    config = load_json(directory / "run.json")
    if not config:
        raise RuntimeError(f"unknown lifecycle run {run_id}")
    gate = load_json(directory / "stages/cpt_gate/receipt.json")
    if gate.get("status") != "passed":
        raise RuntimeError("register requires a passed cpt_gate receipt")
    summary = load_json(directory / "checkpoints/cpt/summary.json")
    errors = identity_errors(summary, config)
    if errors:
        raise RuntimeError(f"CPT identity gate failed: {errors}")
    target = int(config.get("target_exposure_tokens") or config.get("target_exposure") or 0)
    tokens = int(summary.get("tokens_seen") or 0)
    if not (target <= tokens <= target + 8192):
        raise RuntimeError(f"CPT exposure mismatch: summary={tokens} target={target}")
    return directory, config, summary


def register_base_candidate(run_id: str, candidate_id: str | None = None) -> dict:
    directory, config, summary = _verified_inputs(run_id)
    model_id = _candidate_id(config, candidate_id)
    destination = BASE_ROOT / model_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing base candidate: {destination}")
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
    contracts = architecture_contracts()
    schedule = load_json(ROOT / config["corpus"]["schedule"])
    parent_checkpoint = str(schedule.get("parent_checkpoint") or "")
    BASE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{model_id}.register-", dir=BASE_ROOT) as raw:
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
            "incremental_exposure_tokens": int(config["corpus"]["incremental_exposure"]),
            "parent_exposure_tokens": int(config["corpus"]["parent_exposure"]),
            "parent_checkpoint": parent_checkpoint,
            "source_run": relative(directory),
            "source_capture_mode": config.get("source_capture_mode") or "reconstructed",
            "corpus_snapshot_sha256": config["corpus"]["snapshot_sha256"],
            "corpus_artifact_merkle_sha256": config["corpus"].get("artifact_merkle_sha256"),
            "files": copied,
            "weights_sha256": sha256_file(staging / copied["weights"]),
            "state_sha256": sha256_file(staging / copied["train_state"]),
            "state_meta_sha256": sha256_file(staging / copied["train_state_meta"]),
            "summary_sha256": sha256_file(staging / copied["summary"]),
            "registered_at": utc_now(),
            "current_promoted": False,
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
    if resolved.parent != BASE_ROOT.resolve():
        raise ValueError("candidate must be a direct base/ sibling")
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
    proposal = {
        "schema_version": 1,
        "kind": "base-freeze-proposal",
        "candidate": relative(destination),
        "eligible": True,
        "proposed_at": utc_now(),
        "current_sha256_observed": current_hash(),
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
        raise RuntimeError("CURRENT compare-and-swap failed; file changed after proposal")
    destination = _candidate_dir(candidate)
    proposal = load_json(destination / "freeze-proposal.json")
    if proposal.get("eligible") is not True or proposal.get("candidate") != relative(destination):
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
