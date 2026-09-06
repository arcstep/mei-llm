#!/usr/bin/env python3
"""Offline, resumable lifecycle ledger for the single 51M model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common._repo import (
    ARCHITECTURE_DIR,
    ARCHITECTURE_ID,
    ARTIFACT_ROOT,
    CURRENT_PATH,
    RECIPES_DIR,
    ROOT,
    TOKENIZER_ZH_V1,
    TRAIN_RUNS,
    architecture_contracts,
    cycle_artifacts,
    frozen_tokenizer_path,
    architecture_sha256,
    legacy_weight_contract_sha256,
)
from common.run_lock import lock_is_held, pid_alive_from_meta, read_lock_meta


HERE = Path(__file__).resolve().parent
RECIPE_PATH = RECIPES_DIR / "cpt-training-v1.json"
RUN_ROOT = TRAIN_RUNS / "mei-1.0-51m"


def run_roots() -> list[Path]:
    """全部可能的 run 根：旧链 + 新链（ARTIFACT_ROOT 下带 -v 后缀的 cycle）。"""
    roots = [RUN_ROOT]
    cycles_root = ARTIFACT_ROOT / "cycles"
    if cycles_root.is_dir():
        for cycle_dir in sorted(cycles_root.iterdir()):
            if cycle_dir.is_dir() and re.fullmatch(r"exp-\d{6}m", cycle_dir.name):
                candidate = cycle_dir / "runs/mei-1.0-51m"
                if candidate.is_dir():
                    roots.append(candidate)
    return roots


def run_root_for(cycle_id: str | None) -> Path:
    """cycle_id 给定且非旧链 → 该 cycle 的 runs/mei-1.0-51m；否则旧 RUN_ROOT。"""
    if cycle_id and re.fullmatch(r"exp-\d{6}m-v\d+", cycle_id):
        return cycle_artifacts(cycle_id) / "runs/mei-1.0-51m"
    return RUN_ROOT
EXPECTED_PARAMS = 51_463_797
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
HASH_INPUTS = ("RELEASE.json", "manifest.json", "hashes.json", "mix.json")
SOURCE_SUFFIXES = {".py", ".json", ".rs", ".js", ".mjs", ".toml", ".yaml", ".yml"}
SOURCE_EXCLUDES = {"__pycache__", "node_modules", "target", "packages", "runs", ".git"}
HEARTBEAT_FRESH_SECONDS = 180.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _npy_member_shape(archive: zipfile.ZipFile, name: str) -> list[int]:
    with archive.open(name) as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"invalid NPY member: {name}")
        major, _minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported NPY version {major}: {name}")
        header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
        return [int(dim) for dim in header["shape"]]


def train_state_contract_report(state_path: Path) -> dict:
    """Validate a paused train state without loading its tensor payloads."""
    state_path = Path(state_path).resolve()
    meta_path = state_path.with_suffix(".meta.json")
    if not state_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"resume checkpoint or metadata missing: {state_path}")
    meta = load_json(meta_path)
    contracts = architecture_contracts()
    expected_rows = contracts["weight_contract"]["tensor_order"]
    expected_names = [str(row["name"]) for row in expected_rows]
    expected_shapes = {str(row["name"]): list(row["shape"]) for row in expected_rows}
    with zipfile.ZipFile(state_path) as archive:
        members = archive.namelist()
        parameter_members = [
            name for name in members if name.startswith("p.") and name.endswith(".npy")
        ]
        actual_names = [name[2:-4] for name in parameter_members]
        actual_shapes = {
            name[2:-4]: _npy_member_shape(archive, name) for name in parameter_members
        }
        shape_mismatches = {
            name: {"expected": expected_shapes[name], "actual": actual_shapes.get(name)}
            for name in expected_names
            if actual_shapes.get(name) != expected_shapes[name]
        }
        optimizer_missing = [
            f"o.{name}.{slot}.npy"
            for name in expected_names
            for slot in ("m", "v")
            if f"o.{name}.{slot}.npy" not in members
        ]
        optimizer_scalars = all(
            name in members for name in ("o.step.npy", "o.learning_rate.npy")
        )
    legacy_hash = str(meta.get("architecture_sha256") or "")
    declared_weight = str(meta.get("weight_contract_sha256") or "")
    resolved_weight = declared_weight or str(legacy_weight_contract_sha256(legacy_hash) or "")
    source_run = state_path.parent.parent.parent
    receipt_path = source_run / "stages/cpt_gate/receipt.json"
    receipt = load_json(receipt_path)
    manifest = receipt.get("output_manifest") or {}
    state_row = manifest.get(relative(state_path)) or {}
    meta_row = manifest.get(relative(meta_path)) or {}
    receipt_state_ok = (
        state_row.get("sha256") == sha256_file(state_path)
        and int(state_row.get("bytes") or -1) == state_path.stat().st_size
    )
    receipt_meta_ok = (
        meta_row.get("sha256") == sha256_file(meta_path)
        and int(meta_row.get("bytes") or -1) == meta_path.stat().st_size
    )
    return {
        "schema_version": 1,
        "kind": "cpt-recovery-source",
        "source_run_id": source_run.name,
        "checkpoint": relative(state_path),
        "checkpoint_sha256": sha256_file(state_path),
        "checkpoint_bytes": state_path.stat().st_size,
        "metadata": relative(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "metadata_bytes": meta_path.stat().st_size,
        "source_receipt": relative(receipt_path),
        "source_receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
        "source_receipt_status": receipt.get("status"),
        "source_receipt_artifacts_match": receipt_state_ok and receipt_meta_ok,
        "architecture_id": meta.get("architecture_id"),
        "legacy_architecture_sha256": legacy_hash,
        "resolved_weight_contract_sha256": resolved_weight,
        "params": meta.get("params"),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "schedule_sha256": meta.get("schedule_sha256"),
        "parent_tokens_seen": meta.get("parent_tokens_seen"),
        "tokens_seen": meta.get("tokens_seen"),
        "target_tokens": meta.get("target_tokens"),
        "step": meta.get("step"),
        "window_index": meta.get("window_index"),
        "parameter_tensor_count": len(actual_names),
        "parameter_names_and_order_exact": actual_names == expected_names,
        "parameter_shapes_exact": not shape_mismatches,
        "shape_mismatches": shape_mismatches,
        "optimizer_slots_complete": not optimizer_missing and optimizer_scalars,
        "optimizer_missing": optimizer_missing,
        "sampler_state_present": bool(meta.get("sampler_state")),
    }


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 3-80 lowercase letters, digits, dot, underscore or dash")
    for root in run_roots():
        candidate = root / run_id
        if candidate.is_dir():
            return candidate
    return RUN_ROOT / run_id


def locate_run(run_id: str) -> Path:
    """status 用：在全部 run 根中定位 run 目录；找不到返回旧根路径。"""
    for root in run_roots():
        candidate = root / run_id
        if (candidate / "run.json").is_file():
            return candidate
    return run_dir(run_id)


def current_hash() -> str:
    return sha256_file(CURRENT_PATH)


def code_revision() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_manifest() -> dict:
    """Hash runnable source, excluding generated packages and training outputs."""
    roots = (
        ARCHITECTURE_DIR,
        HERE,
        ROOT / "platform/python-sdk",
        ROOT / "platform/browser-sdk",
        ROOT / "platform/_shared",
    )
    files: dict[str, dict[str, object]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if any(part in SOURCE_EXCLUDES for part in rel_parts):
                continue
            files[relative(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": 1,
        "capture_mode": "launch",
        "files": files,
        "manifest_sha256": sha256_json(files),
    }


def environment_contract() -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "offline": {
            "MEI_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "CARGO_NET_OFFLINE": "true",
        },
    }


def recipe() -> dict:
    data = load_json(RECIPE_PATH)
    if data.get("architecture_id") != ARCHITECTURE_ID:
        raise RuntimeError("lifecycle recipe architecture identity mismatch")
    if int(data.get("params") or 0) != EXPECTED_PARAMS:
        raise RuntimeError("lifecycle recipe parameter identity mismatch")
    actual = architecture_contracts()["weight_contract_sha256"]
    if data.get("weight_contract_sha256") != actual:
        raise RuntimeError(
            f"weight contract drift: recipe={data.get('weight_contract_sha256')} actual={actual}"
        )
    if data.get("current_policy") != "read_only_until_explicit_user_freeze":
        raise RuntimeError("recipe must keep CURRENT read-only")
    return data


def schedule_path(corpus_dir: Path, target: int) -> Path:
    candidates = sorted(corpus_dir.glob("schedule-cpt*.json"))
    exact = corpus_dir / f"schedule-cpt-{rung_name(target)}.json"
    found = [exact] if exact.is_file() else candidates
    if not found:
        scratch = corpus_dir / "schedule-scratch.json"
        if scratch.is_file():
            return scratch
        raise RuntimeError(f"missing immutable CPT schedule under {corpus_dir}")
    matches = []
    for path in dict.fromkeys(found):
        row = load_json(path)
        if int(row.get("cumulative_exposure_tokens") or 0) == int(target):
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"multiple CPT schedules declare cumulative exposure {target}")
    raise RuntimeError(f"no CPT schedule declares cumulative exposure {target}")


def rung_name(target: int) -> str:
    if int(target) <= 0:
        raise ValueError("target exposure must be positive")
    names = recipe().get("rung_names") or {}
    if str(target) in names:
        return str(names[str(target)])
    if target % 1_000_000_000 == 0:
        return f"{target // 1_000_000_000}b"
    if target % 1_000_000 == 0:
        return f"{target // 1_000_000}m"
    return f"t{target}"


def _resolved_corpus_artifacts(corpus_dir: Path, manifest: dict) -> list[Path]:
    mix = load_json(corpus_dir / "mix.json")
    if not (mix.get("sources") or {}) and manifest.get("sliced_from"):
        source_root = ROOT / str(manifest["sliced_from"])
        mix = load_json(source_root / "mix.json")
    paths: list[Path] = []
    for row in (mix.get("sources") or {}).values():
        for key in ("train_shards", "valid_shards"):
            for raw in (row or {}).get(key) or []:
                path = Path(str(raw))
                paths.append(path if path.is_absolute() else ROOT / path)
    contamination = manifest.get("contamination") or {}
    for raw in contamination.get("receipts") or []:
        path = Path(str(raw))
        paths.append(path if path.is_absolute() else ROOT / path)
    return sorted(set(path.resolve() for path in paths))


def corpus_snapshot(corpus_dir: Path, target: int) -> dict:
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise RuntimeError(f"missing local corpus directory: {corpus_dir}")
    schedule = schedule_path(corpus_dir, target)
    release = load_json(corpus_dir / "RELEASE.json")
    manifest = load_json(corpus_dir / "manifest.json")
    sched = load_json(schedule)
    if release.get("training_mode") != "cpt":
        raise RuntimeError("corpus release must declare training_mode=cpt")
    if sched.get("kind") not in {"cpt", "scratch"}:
        raise RuntimeError("schedule kind must be cpt or scratch")
    if bool(sched.get("allow_repeat")):
        raise RuntimeError("CPT schedule cannot allow repeated exposure")
    parent = int(sched.get("parent_tokens_seen") or 0)
    incremental = int(sched.get("exposure_tokens") or 0)
    if sched.get("kind") == "scratch":
        cumulative = incremental
        if parent != 0:
            raise RuntimeError("scratch schedule cannot declare parent exposure")
    else:
        cumulative = int(sched.get("cumulative_exposure_tokens") or 0)
    if cumulative != target or parent + incremental != target:
        raise RuntimeError("parent + incremental exposure must equal the requested cumulative target")
    sources = sched.get("sources") or {}
    mix_document = load_json(corpus_dir / "mix.json")
    mix_sources = mix_document.get("sources")
    if mix_sources and set(sources) != set(mix_sources):
        # 旧布局 mix 不声明 sources（仅 schedule 四角色）；v2 布局必须两者一致。
        raise RuntimeError("CPT schedule roles must match the mix sources exactly")
    if sum(int((row or {}).get("token_quota") or 0) for row in sources.values()) != incremental:
        raise RuntimeError("CPT source quotas must sum to incremental exposure")
    if release.get("public_distribution_clearance_asserted") is not True and not (
        release.get("license") or release.get("licenses")
    ):
        raise RuntimeError("corpus release lacks licence/clearance registration")
    contamination = (
        release.get("contamination")
        or release.get("contamination_report")
        or manifest.get("contamination")
        or manifest.get("contamination_report")
    )
    if not contamination:
        raise RuntimeError("corpus release lacks contamination evidence")
    unique_tokens = int(manifest.get("n_unique_train_tokens") or 0)
    train_tokens = int(manifest.get("n_train_tokens") or 0)
    if not release.get("dedup") and not (0 < unique_tokens <= train_tokens):
        raise RuntimeError("corpus release lacks dedup/unique-token evidence")
    files = [corpus_dir / name for name in HASH_INPUTS]
    files.append(schedule)
    missing = [relative(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing immutable corpus evidence: {missing}")
    hashes = {relative(path): sha256_file(path) for path in files}
    artifacts = _resolved_corpus_artifacts(corpus_dir, manifest)
    missing_artifacts = [relative(path) for path in artifacts if not path.is_file()]
    if missing_artifacts:
        raise RuntimeError(f"missing resolved corpus artifacts: {missing_artifacts[:8]}")
    artifact_hashes = {relative(path): sha256_file(path) for path in artifacts}
    return {
        "corpus_dir": relative(corpus_dir),
        "schedule": relative(schedule),
        "target_exposure": target,
        "parent_exposure": parent,
        "incremental_exposure": incremental,
        "evidence_sha256": hashes,
        "legacy_evidence_snapshot_sha256": sha256_json(hashes),
        "artifact_sha256": artifact_hashes,
        "artifact_merkle_sha256": sha256_json(artifact_hashes),
        "snapshot_sha256": sha256_json(
            {"evidence_sha256": hashes, "artifact_sha256": artifact_hashes}
        ),
    }


def init_run(
    run_id: str,
    corpus_dir: Path,
    target: int,
    *,
    resume_checkpoint: Path | None = None,
    cycle_id: str | None = None,
) -> dict:
    data = recipe()
    root = run_root_for(cycle_id)
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    directory = root / run_id
    snapshot = corpus_snapshot(corpus_dir, target)
    contracts = architecture_contracts()
    sources = source_manifest()
    frozen = frozen_tokenizer_path()
    tokenizer_sha = sha256_file(frozen) if frozen.is_file() else "missing"
    recovery = None
    if resume_checkpoint is not None:
        recovery_path = Path(resume_checkpoint)
        if not recovery_path.is_absolute():
            recovery_path = ROOT / recovery_path
        recovery = train_state_contract_report(recovery_path)
        expected_schedule_sha = snapshot["evidence_sha256"][snapshot["schedule"]]
        recovery_errors = []
        if recovery.get("source_receipt_artifacts_match") is not True:
            recovery_errors.append("source receipt does not pin checkpoint and metadata hashes")
        if recovery.get("architecture_id") != ARCHITECTURE_ID:
            recovery_errors.append("architecture_id mismatch")
        if int(recovery.get("params") or 0) != EXPECTED_PARAMS:
            recovery_errors.append("parameter count mismatch")
        if recovery.get("resolved_weight_contract_sha256") != contracts["weight_contract_sha256"]:
            recovery_errors.append("weight contract is not recognized")
        if recovery.get("tokenizer_sha256") != tokenizer_sha:
            recovery_errors.append("tokenizer mismatch")
        if recovery.get("manifest_sha256") != snapshot["evidence_sha256"].get(
            relative(corpus_dir / "manifest.json")
        ):
            recovery_errors.append("corpus manifest mismatch")
        if recovery.get("schedule_sha256") != expected_schedule_sha:
            recovery_errors.append("schedule mismatch")
        resume_tokens = int(recovery.get("tokens_seen") or 0)
        if not int(snapshot["parent_exposure"]) < resume_tokens < int(target):
            recovery_errors.append("resume exposure is outside parent/target interval")
        if int(recovery.get("parent_tokens_seen") or 0) != int(snapshot["parent_exposure"]):
            recovery_errors.append("parent exposure mismatch")
        if int(recovery.get("target_tokens") or 0) != int(target):
            recovery_errors.append("target exposure mismatch")
        for key in (
            "parameter_names_and_order_exact",
            "parameter_shapes_exact",
            "optimizer_slots_complete",
            "sampler_state_present",
        ):
            if recovery.get(key) is not True:
                recovery_errors.append(f"{key} failed")
        if recovery_errors:
            raise RuntimeError("resume checkpoint rejected: " + "; ".join(recovery_errors))
    config = {
        "schema_version": 2,
        "run_id": run_id,
        "cycle_id": cycle_id,
        "tokenizer_id": frozen.name.replace(".model", ""),
        "status": "planned",
        "created_at": utc_now(),
        "product": "mei-1.0-51m",
        "architecture_id": ARCHITECTURE_ID,
        "weight_contract_sha256": contracts["weight_contract_sha256"],
        "runtime_profile_sha256": contracts["runtime_profile_sha256"],
        "training_aux_sha256": contracts["training_aux_sha256"],
        # Retained as source provenance for old tooling, never for compatibility.
        "architecture_sha256": architecture_sha256(),
        "params": EXPECTED_PARAMS,
        "rung": rung_name(target),
        "target_exposure": target,
        "target_exposure_tokens": target,
        "corpus": snapshot,
        "recipe": relative(RECIPE_PATH),
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "code_revision": code_revision(),
        "source_capture_mode": "launch",
        "source_manifest": relative(directory / "source-manifest.json"),
        "source_manifest_sha256": sources["manifest_sha256"],
        "tokenizer_sha256": tokenizer_sha,
        "environment_contract": environment_contract(),
        "current_sha256_at_init": current_hash(),
        "current_mutation_allowed": False,
        "offline": True,
    }
    if recovery is not None:
        config["resume_checkpoint"] = {
            key: recovery[key]
            for key in (
                "source_run_id",
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_bytes",
                "metadata",
                "metadata_sha256",
                "metadata_bytes",
                "source_receipt",
                "source_receipt_sha256",
                "tokens_seen",
                "step",
                "window_index",
                "resolved_weight_contract_sha256",
            )
        }
    path = directory / "run.json"
    if path.is_file():
        old = load_json(path)
        stable = (
            "product",
            "architecture_id",
            "weight_contract_sha256",
            "params",
            "rung",
            "target_exposure",
            "source_manifest_sha256",
            "tokenizer_sha256",
            "resume_checkpoint",
        )
        if any(old.get(key) != config.get(key) for key in stable) or old.get("corpus") != snapshot:
            raise RuntimeError("run_id already exists with different immutable inputs")
        return old
    for child in ("stages", "jobs", "checkpoints", "packages", "shards", "logs"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    seed_source = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/jobs/mei-1.0-51m"
    seed_names = (
        "q4-baseline-bit-map.json",
        "q2q4-product-candidate-bit-map.json",
        "sft-v2-51m-thresholds-preregister.json",
        "qat-q4-logit-threshold-preregister.json",
    )
    seed_hashes = {}
    for name in seed_names:
        source = seed_source / name
        if not source.is_file():
            raise RuntimeError(f"missing pre-registered lifecycle input: {source}")
        target_path = directory / "jobs" / name
        shutil.copy2(source, target_path)
        seed_hashes[name] = sha256_file(target_path)
    if recovery is not None:
        recovery_path = directory / "jobs/recovery-source.json"
        atomic_json(recovery_path, recovery)
        seed_hashes[recovery_path.name] = sha256_file(recovery_path)
    config["seed_input_sha256"] = seed_hashes
    atomic_json(directory / "source-manifest.json", sources)
    atomic_json(path, config)
    atomic_json(directory / "recipe.snapshot.json", data)
    return config


def load_run(run_id: str) -> tuple[Path, dict]:
    directory = locate_run(run_id)
    config = load_json(directory / "run.json")
    if not config:
        raise RuntimeError(f"unknown run_id: {run_id}")
    return directory, config


def stage_receipt(directory: Path, stage: str) -> Path:
    return directory / "stages" / stage / "receipt.json"


def receipt_digest(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def stage_fingerprint(directory: Path, config: dict, stage: str, row: dict) -> str:
    dependencies = {
        dep: receipt_digest(stage_receipt(directory, dep)) for dep in (row.get("requires") or [])
    }
    return sha256_json(
        {
            "stage": stage,
            "recipe_sha256": config["recipe_sha256"],
            "corpus_snapshot_sha256": config["corpus"]["snapshot_sha256"],
            "corpus_artifact_merkle_sha256": config["corpus"].get(
                "artifact_merkle_sha256", "legacy-unavailable"
            ),
            "target_exposure": config["target_exposure"],
            "weight_contract_sha256": config.get("weight_contract_sha256")
            or architecture_contracts()["weight_contract_sha256"],
            "runtime_profile_sha256": config.get("runtime_profile_sha256", "legacy-unavailable"),
            "training_aux_sha256": config.get("training_aux_sha256", "legacy-unavailable"),
            "source_capture_mode": config.get("source_capture_mode", "reconstructed"),
            "source_manifest_sha256": config.get("source_manifest_sha256", "legacy-unavailable"),
            "tokenizer_sha256": config.get("tokenizer_sha256", "legacy-unavailable"),
            "seed_input_sha256": config.get("seed_input_sha256") or {},
            "resume_checkpoint": config.get("resume_checkpoint"),
            "environment_contract": config.get("environment_contract") or environment_contract(),
            "command": row.get("command"),
            "dependencies": dependencies,
        }
    )


def source_capture_status(config: dict) -> dict:
    mode = str(config.get("source_capture_mode") or "reconstructed")
    expected = str(config.get("source_manifest_sha256") or "")
    if mode != "launch" or not expected:
        return {
            "mode": "reconstructed",
            "enforced": False,
            "unchanged": None,
            "expected_sha256": expected or None,
        }
    actual = source_manifest()["manifest_sha256"]
    return {
        "mode": "launch",
        "enforced": True,
        "unchanged": actual == expected,
        "expected_sha256": expected,
        "actual_sha256": actual,
    }


def immutable_input_errors(directory: Path, config: dict) -> list[str]:
    errors: list[str] = []
    capture = source_capture_status(config)
    if capture.get("enforced") and capture.get("unchanged") is not True:
        errors.append("source_manifest_sha256 changed")
    expected_tokenizer = str(config.get("tokenizer_sha256") or "")
    if expected_tokenizer:
        frozen = frozen_tokenizer_path()
        actual = sha256_file(frozen) if frozen.is_file() else "missing"
        if actual != expected_tokenizer:
            errors.append("tokenizer_sha256 changed")
    for name, expected in (config.get("seed_input_sha256") or {}).items():
        path = directory / "jobs" / str(name)
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            errors.append(f"seed input changed: {name}")
    recovery = config.get("resume_checkpoint") or {}
    for path_key, hash_key, bytes_key in (
        ("checkpoint", "checkpoint_sha256", "checkpoint_bytes"),
        ("metadata", "metadata_sha256", "metadata_bytes"),
        ("source_receipt", "source_receipt_sha256", None),
    ):
        if not recovery.get(path_key):
            continue
        path = Path(str(recovery[path_key]))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256_file(path) != recovery.get(hash_key):
            errors.append(f"resume input changed: {path_key}")
            continue
        if bytes_key and path.stat().st_size != int(recovery.get(bytes_key) or -1):
            errors.append(f"resume input size changed: {path_key}")
    return errors


def output_manifest(directory: Path, stage: str) -> dict[str, dict]:
    roots = [directory / "jobs"]
    if stage in {"cpt", "cpt_gate"}:
        roots.append(directory / "checkpoints/cpt")
    elif stage == "qat_q4":
        roots.append(directory / "checkpoints/qat-q4")
    elif stage == "cq2":
        roots.append(directory / "checkpoints/qat-cq2")
    elif stage == "pack_q4":
        roots.append(directory / "packages/qat-q4")
    elif stage == "quant_aware_sft":
        roots.extend((directory / "checkpoints/sft", directory / "packages/sft"))
    elif stage == "lock_v2":
        roots.append(directory / "shards/lock-v2")
    manifest = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "cargo-target" not in path.parts:
                manifest[relative(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return manifest


def output_manifest_valid(manifest: dict[str, dict]) -> bool:
    for name, expected in manifest.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if (
            not path.is_file()
            or path.stat().st_size != int(expected.get("bytes") or -1)
            or sha256_file(path) != expected.get("sha256")
        ):
            return False
    return True


def write_receipt(directory: Path, stage: str, payload: dict) -> dict:
    path = stage_receipt(directory, stage)
    payload = {"stage": stage, **payload}
    atomic_json(path, payload)
    return payload


def adoptable_completed_cpt(directory: Path, config: dict) -> tuple[bool, dict]:
    """Recognize a detached CPT worker that finished after the ledger blocked."""
    summary_path = directory / "checkpoints/cpt/summary.json"
    summary = load_json(summary_path)
    target = int(config.get("target_exposure_tokens") or config.get("target_exposure") or 0)
    tokens = int(summary.get("tokens_seen") or 0)
    artifacts = (
        directory / "checkpoints/cpt/pretrain-cpt.npz",
        directory / "checkpoints/cpt/pretrain-cpt-state.npz",
        directory / "checkpoints/cpt/pretrain-cpt-state.meta.json",
    )
    complete = (
        target > 0
        and target <= tokens <= target + 8192
        and all(path.is_file() for path in artifacts)
        and not bool(summary.get("paused"))
        and not bool(summary.get("exhausted"))
    )
    return complete, {
        "tokens_seen": tokens,
        "target_exposure_tokens": target,
        "artifacts": [relative(path) for path in artifacts],
        "summary": relative(summary_path),
        "source_capture_mode": config.get("source_capture_mode") or "reconstructed",
    }


def render_context(directory: Path, config: dict) -> dict[str, str]:
    cpt_dir = directory / "checkpoints/cpt"
    cpt_state = cpt_dir / "pretrain-cpt-state.npz"
    schedule = load_json(ROOT / config["corpus"]["schedule"])
    declared_parent = Path(str(schedule.get("parent_checkpoint") or ""))
    if not declared_parent.is_absolute():
        declared_parent = ROOT / declared_parent
    recovery = config.get("resume_checkpoint") or {}
    recovery_state = Path(str(recovery.get("checkpoint") or ""))
    if recovery_state and not recovery_state.is_absolute():
        recovery_state = ROOT / recovery_state
    if recovery.get("checkpoint"):
        parent_state = recovery_state
    else:
        parent_state = cpt_state if cpt_state.is_file() else declared_parent
    cq2_map = load_json(directory / "jobs/q2q4-product-bit-map.json")
    cq2_package = directory / "packages/cq2"
    use_cq2 = bool(cq2_map.get("product_final")) and (cq2_package / "weights.q4").is_file()
    selected_package = cq2_package if use_cq2 else directory / "packages/qat-q4"
    selected_master = (
        directory / "checkpoints/qat-cq2/best.npz"
        if use_cq2
        else directory / "checkpoints/qat-q4/best.npz"
    )
    selected_bit_map = (
        directory / "jobs/q2q4-product-bit-map.json"
        if use_cq2
        else directory / "jobs/q4-baseline-bit-map.json"
    )
    cpt_weights = cpt_dir / "pretrain-cpt.npz"
    cpt_model_id = f"mei-1.0-51m-base-cpt{config['rung']}-v1"
    qat_package_id = f"{cpt_model_id}-qat-q4"
    return {
        "python": sys.executable,
        "model_factory": str(ROOT / "model-factory"),
        "corpus_dir": str(ROOT / config["corpus"]["corpus_dir"]),
        "target_exposure": str(config["target_exposure"]),
        "rung": config["rung"],
        "parent_state": str(parent_state),
        "jobs": str(directory / "jobs"),
        "checkpoints": str(directory / "checkpoints"),
        "packages": str(directory / "packages"),
        "shards": str(directory / "shards"),
        "cpt_weights": str(cpt_weights),
        "cpt_weights_sha256": sha256_file(cpt_weights) if cpt_weights.is_file() else "missing",
        "cpt_model_id": cpt_model_id,
        "qat_package_id": qat_package_id,
        "qat_master": str(directory / "checkpoints/qat-q4/best.npz"),
        "qat_model_id": f"mei-1.0-51m-base-cpt{config['rung']}-qat-q4-v1",
        "selected_package": str(selected_package),
        "selected_master": str(selected_master),
        "selected_bit_map": str(selected_bit_map),
        "sft_package": str(directory / "packages/sft"),
        "sft_master": str(directory / "checkpoints/sft/sft-master.npz"),
        "sft_model_id": f"mei-1.0-51m-cpt{config['rung']}-sft-qat-v1",
    }


def render_command(command: list[str], context: dict[str, str]) -> list[str]:
    return [part.format(**context) for part in command]


def identity_errors(summary: dict, config: dict) -> list[str]:
    errors = []
    params = int(summary.get("params") or 0)
    if params != EXPECTED_PARAMS:
        errors.append(f"params drift: got={params} expected={EXPECTED_PARAMS}")
    if summary.get("architecture_id") != ARCHITECTURE_ID:
        errors.append(
            f"architecture_id drift: got={summary.get('architecture_id')!r} expected={ARCHITECTURE_ID!r}"
        )
    expected_weight = str(
        config.get("weight_contract_sha256")
        or architecture_contracts()["weight_contract_sha256"]
    )
    actual_weight = str(summary.get("weight_contract_sha256") or "")
    legacy_sha = str(summary.get("architecture_sha256") or "")
    if actual_weight:
        if actual_weight != expected_weight:
            errors.append("weight_contract_sha256 drift")
    elif legacy_weight_contract_sha256(legacy_sha) != expected_weight:
        errors.append("unrecognized legacy architecture_sha256")
    return errors


def internal_gate(stage: str, directory: Path, config: dict) -> tuple[bool, dict]:
    if stage == "corpus_freeze":
        fresh = corpus_snapshot(ROOT / config["corpus"]["corpus_dir"], int(config["target_exposure"]))
        expected = str(config["corpus"]["snapshot_sha256"])
        actual = (
            fresh["snapshot_sha256"]
            if config["corpus"].get("artifact_merkle_sha256")
            else fresh["legacy_evidence_snapshot_sha256"]
        )
        ok = actual == expected
        return ok, {"fresh_snapshot": fresh, "immutable": ok}
    if stage == "cpt_gate":
        schedule_file = ROOT / config["corpus"]["schedule"]
        schedule = load_json(schedule_file)
        summary_path = directory / "checkpoints/cpt/summary.json"
        if schedule.get("kind") == "scratch" and not summary_path.is_file():
            candidates = sorted((directory / "checkpoints/cpt").glob("pretrain-*/summary.json"))
            if candidates:
                summary_path = candidates[-1]
        summary = load_json(summary_path)
        readiness = load_json(directory / "jobs/cpt-readiness.json")
        params = int(summary.get("params") or 0)
        tokens = int(summary.get("tokens_seen") or 0)
        arch = summary.get("architecture_id")
        arch_hash = summary.get("architecture_sha256")
        weight_hash = summary.get("weight_contract_sha256") or legacy_weight_contract_sha256(
            str(arch_hash or "")
        )
        probe = summary.get("final_probe_mean_nll", summary.get("probe_mean_nll"))
        no_repeat = not bool(summary.get("allow_repeat"))
        tolerance = 2048
        expected_incremental = int(schedule.get("exposure_tokens") or 0)
        segment_tokens = int(summary.get("segment_tokens") or 0)
        stage_drawn = summary.get("stage_tokens_drawn") or {}
        if schedule.get("kind") == "scratch":
            # scratch run: no parent rung, roles come from the mix, not the
            # legacy wiki/hq/structure/colloquial set.
            roles = [
                role
                for role, spec in (schedule.get("sources") or {}).items()
                if int((spec or {}).get("token_quota") or 0) > 0
            ]
            role_losses = {role: summary.get(f"valid_loss_{role}") for role in roles}
            quality = summary.get("valid_loss") is not None and probe is not None
            # source_tokens_drawn 是跨 curriculum 阶段的累计值；stage_tokens_drawn
            # 仅含最后一个阶段（s3），不能对总配额做账。
            cumulative_drawn = summary.get("source_tokens_drawn") or stage_drawn
            # scratch batches are 4096 tokens (s1); allow one batch of slack.
            quota_tolerance = tolerance * 4
            quota_ok = all(
                abs(
                    int(cumulative_drawn.get(role) or 0)
                    - int(((schedule.get("sources") or {}).get(role) or {}).get("token_quota") or 0)
                )
                <= quota_tolerance
                for role in roles
            )
            benefit_ok = True  # no parent rung to compare against
            parent = {}
        else:
            role_losses = {
                role: summary.get("valid_loss" if role == "wiki" else f"valid_loss_{role}")
                for role in ("wiki", "hq", "structure", "colloquial")
            }
            quality = all(value is not None for value in role_losses.values()) and probe is not None
            quota_ok = all(
                abs(
                    int(stage_drawn.get(role) or 0)
                    - int(((schedule.get("sources") or {}).get(role) or {}).get("token_quota") or 0)
                )
                <= tolerance
                for role in ("wiki", "hq", "structure", "colloquial")
            )
            parent_path = Path(str(schedule.get("parent_checkpoint") or ""))
            if not parent_path.is_absolute():
                parent_path = ROOT / parent_path
            parent = load_json(parent_path.parent / "RELEASE.json")
            benefit_pairs = []
            for role, current_loss in role_losses.items():
                parent_loss = parent.get("valid_loss" if role == "wiki" else f"valid_loss_{role}")
                if current_loss is not None and parent_loss is not None:
                    benefit_pairs.append(float(current_loss) <= float(parent_loss))
            benefit_ok = bool(benefit_pairs) and any(benefit_pairs)
        if schedule.get("kind") == "scratch":
            # scratch trainer names checkpoints after the run subdir:
            # checkpoints/cpt/pretrain-<run>/pretrain-<run>.npz (+ -state.npz).
            scratch_dirs = [p for p in (directory / "checkpoints/cpt").glob("pretrain-*") if p.is_dir()]
            scratch_ckpts: list[Path] = []
            for sub in scratch_dirs:
                name = sub.name.removeprefix("pretrain-")
                scratch_ckpts.append(sub / f"pretrain-{name}.npz")
                scratch_ckpts.append(sub / f"pretrain-{name}-state.npz")
            checkpoint_ok = bool(scratch_ckpts) and all(p.is_file() for p in scratch_ckpts)
            # summary.json has no segment_tokens field; for scratch the segment
            # is the whole run (parent_tokens_seen == 0).
            segment_tokens = tokens
        else:
            checkpoint_ok = all(
                path.is_file()
                for path in (
                    directory / "checkpoints/cpt/pretrain-cpt.npz",
                    directory / "checkpoints/cpt/pretrain-cpt-state.npz",
                )
            )
        schedule_ok = (
            summary.get("schedule_sha256") == sha256_file(schedule_file)
            and int(summary.get("parent_tokens_seen") or 0) == int(schedule.get("parent_tokens_seen") or 0)
        )
        exposure_ok = (
            int(config["target_exposure"]) <= tokens <= int(config["target_exposure"]) + tolerance * 4
            and expected_incremental <= segment_tokens <= expected_incremental + tolerance * 4
        )
        identity = identity_errors(summary, config)
        ok = (
            readiness.get("ready") is True
            and exposure_ok
            and not identity
            and quality
            and no_repeat
            and quota_ok
            and benefit_ok
            and checkpoint_ok
            and schedule_ok
            and not bool(summary.get("paused"))
            and not bool(summary.get("exhausted"))
        )
        return ok, {
            "tokens_seen": tokens,
            "segment_tokens": segment_tokens,
            "params": params,
            "architecture_id": arch,
            "architecture_sha256": arch_hash,
            "weight_contract_sha256": weight_hash,
            "identity_errors": identity,
            "readiness_passed": readiness.get("ready") is True,
            "quality_evidence_present": quality,
            "role_valid_loss": role_losses,
            "probe_mean_nll": probe,
            "benefit_vs_parent_ok": benefit_ok,
            "allow_repeat": not no_repeat,
            "exposure_ok": exposure_ok,
            "quota_ok": quota_ok,
            "checkpoint_ok": checkpoint_ok,
            "schedule_ok": schedule_ok,
            "summary": relative(directory / "checkpoints/cpt/summary.json"),
        }
    raise RuntimeError(f"unknown internal stage: {stage}")


def offline_env(directory: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "MEI_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": str(directory / "checkpoints/cargo-target"),
        }
    )
    existing_pythonpath = env.get("PYTHONPATH")
    required_pythonpath = [
        str(ROOT / "model-factory"),
        str(ROOT / "platform/python-sdk"),
    ]
    if existing_pythonpath:
        required_pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(required_pythonpath)
    return env


def run_logged_process(
    command: list[str], *, directory: Path, log_path: Path, env: dict[str, str]
) -> int:
    """Stream one stage through Popen; isolated for deterministic tests."""
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        proc.wait()
    return int(proc.returncode or 0)


def live_stage_state(directory: Path) -> dict:
    """Reconcile the ledger with the trainer lock and heartbeat, without mutation."""
    cpt_dir = directory / "checkpoints/cpt"
    lock = lock_is_held(cpt_dir)
    lock_meta = lock or read_lock_meta(cpt_dir)
    heartbeat = load_json(cpt_dir / "heartbeat.json")
    pid = int((lock_meta or heartbeat).get("pid") or 0)
    alive = pid_alive_from_meta({"pid": pid}) if pid else False
    heartbeat_unix = float(heartbeat.get("unix") or 0.0)
    heartbeat_age = max(0.0, time.time() - heartbeat_unix) if heartbeat_unix else None
    fresh = heartbeat_age is not None and heartbeat_age <= HEARTBEAT_FRESH_SECONDS
    if bool(lock) and fresh:
        state = "live"
    elif bool(lock) and alive:
        state = "stale"
    elif lock_meta and not alive:
        state = "orphaned"
    else:
        state = "idle"
    return {
        "state": state,
        "lock_held": bool(lock),
        "pid": pid or None,
        "pid_alive": alive,
        "heartbeat_age_seconds": heartbeat_age,
        "tokens_seen": heartbeat.get("tokens_seen"),
        "loss": heartbeat.get("loss"),
        "tok_s": heartbeat.get("tok_s"),
    }


SCRATCH_STAGE_COMMAND = [
    "{python}",
    "{model_factory}/training/cpt/run_scratch_curriculum_51m.py",
    "--corpus-dir",
    "{corpus_dir}",
    "--out-dir",
    "{checkpoints}/cpt",
    "--no-compile",
]


def _scratch_stage_command(directory: Path) -> list[str]:
    """run 内已有完整 checkpoint 时加 --resume-existing（原地续跑）。"""
    command = list(SCRATCH_STAGE_COMMAND)
    state = (
        directory
        / "checkpoints/cpt/pretrain-mei-1.0-51m-base-scratch300m-v1"
        / "pretrain-mei-1.0-51m-base-scratch300m-v1-state.npz"
    )
    if state.is_file():
        command.append("--resume-existing")
    return command


def run_stage(
    directory: Path,
    config: dict,
    stage: str,
    row: dict,
    *,
    dry_run: bool,
) -> tuple[bool, dict]:
    fingerprint = stage_fingerprint(directory, config, stage, row)
    prior = load_json(stage_receipt(directory, stage))
    if (
        prior.get("status") in {"passed", "degraded"}
        and prior.get("fingerprint") == fingerprint
        and output_manifest_valid(prior.get("output_manifest") or {})
    ):
        return True, {"stage": stage, "status": "reused", "receipt": relative(stage_receipt(directory, stage))}
    if not dry_run:
        dependency_errors = []
        for dependency in row.get("requires") or []:
            dependency_status = load_json(stage_receipt(directory, dependency)).get("status")
            if dependency_status not in {"passed", "degraded"}:
                dependency_errors.append(f"{dependency}={dependency_status or 'missing'}")
        if dependency_errors:
            receipt = write_receipt(
                directory,
                stage,
                {
                    "status": "blocked",
                    "fingerprint": fingerprint,
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                    "failure_reason": "unsatisfied stage dependencies: "
                    + ", ".join(dependency_errors),
                    "output_manifest": {},
                    "output_manifest_sha256": sha256_json({}),
                    "current_sha256": current_hash(),
                },
            )
            return False, receipt
    if stage == "cpt" and not dry_run:
        adoptable, detail = adoptable_completed_cpt(directory, config)
        if adoptable:
            outputs = output_manifest(directory, stage)
            receipt = write_receipt(
                directory,
                stage,
                {
                    "status": "passed",
                    "completion_mode": "adopted_detached_worker",
                    "fingerprint": fingerprint,
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                    "detail": detail,
                    "output_manifest": outputs,
                    "output_manifest_sha256": sha256_json(outputs),
                    "current_sha256": current_hash(),
                },
            )
            return True, receipt
    if dry_run:
        planned = {"stage": stage, "status": "planned", "kind": row.get("kind")}
        if row.get("kind") != "internal":
            if stage == "cpt" and load_json(ROOT / config["corpus"]["schedule"]).get("kind") == "scratch":
                planned["command"] = render_command(
                    _scratch_stage_command(directory), render_context(directory, config)
                )
            else:
                planned["command"] = render_command(row["command"], render_context(directory, config))
        return True, planned
    started = utc_now()
    if row.get("kind") == "internal":
        ok, detail = internal_gate(stage, directory, config)
        outputs = output_manifest(directory, stage)
        receipt = write_receipt(
            directory,
            stage,
            {
                "status": "passed" if ok else "blocked",
                "fingerprint": fingerprint,
                "started_at": started,
                "finished_at": utc_now(),
                "detail": detail,
                "output_manifest": outputs,
                "output_manifest_sha256": sha256_json(outputs),
                "current_sha256": current_hash(),
            },
        )
        return ok, receipt
    if stage == "cpt" and load_json(ROOT / config["corpus"]["schedule"]).get("kind") == "scratch":
        command = render_command(
            _scratch_stage_command(directory), render_context(directory, config)
        )
    else:
        command = render_command(row["command"], render_context(directory, config))
    if dry_run:
        return True, {"stage": stage, "status": "planned", "command": command}
    before_current = current_hash()
    before_outputs = output_manifest(directory, stage)
    log_path = directory / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = offline_env(directory)
    env.setdefault("PYTHONUNBUFFERED", "1")
    returncode = run_logged_process(command, directory=directory, log_path=log_path, env=env)
    after_current = current_hash()
    if after_current != before_current:
        receipt = write_receipt(
            directory,
            stage,
            {
                "status": "blocked",
                "fingerprint": fingerprint,
                "started_at": started,
                "finished_at": utc_now(),
                "command": command,
                "exit_code": returncode,
                "log": relative(log_path),
                "failure_reason": "CURRENT.json changed during stage",
                "current_sha256_before": before_current,
                "current_sha256_after": after_current,
                "output_manifest": {},
                "output_manifest_sha256": sha256_json({}),
            },
        )
        return False, receipt
    ok = returncode == 0
    degrade_evidence = {}
    if row.get("kind") == "command_or_degrade" and row.get("degrade_receipt"):
        degrade_evidence = load_json(directory / "jobs" / str(row["degrade_receipt"]))
    degraded = (
        row.get("kind") == "command_or_degrade"
        and not ok
        and bool(degrade_evidence.get(str(row.get("degrade_key") or "")))
    )
    status = "passed" if ok else ("degraded" if degraded else "blocked")
    after_outputs = output_manifest(directory, stage)
    outputs = {name: value for name, value in after_outputs.items() if before_outputs.get(name) != value}
    receipt = write_receipt(
        directory,
        stage,
        {
            "status": status,
            "fingerprint": fingerprint,
            "started_at": started,
            "finished_at": utc_now(),
            "command": command,
            "exit_code": returncode,
            "log": relative(log_path),
            "failure_branch": row.get("failure_branch") if degraded else None,
            "degrade_evidence": (
                relative(directory / "jobs" / str(row["degrade_receipt"])) if degraded else None
            ),
            "output_manifest": outputs,
            "output_manifest_sha256": sha256_json(outputs),
            "current_sha256_before": before_current,
            "current_sha256_after": after_current,
        },
    )
    return ok or degraded, receipt


def execute(run_id: str, *, dry_run: bool, until: str | None, track: str = "cpt") -> int:
    directory, config = load_run(run_id)
    data = recipe()
    if current_hash() != config["current_sha256_at_init"]:
        raise RuntimeError("CURRENT.json changed since run initialization")
    live = live_stage_state(directory)
    if not dry_run and live["state"] in {"live", "stale"}:
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run_id,
                    "already_running": True,
                    "live": live,
                    "note": "no second lifecycle stage was started",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    input_errors = immutable_input_errors(directory, config)
    if input_errors:
        raise RuntimeError(
            "immutable run inputs changed; create a new run or explicit reconstructed adoption: "
            + "; ".join(input_errors)
        )
    if not dry_run:
        config["status"] = "running"
        config["blocked_stage"] = None
        config["updated_at"] = utc_now()
        atomic_json(directory / "run.json", config)
    if track == "full":
        selected_stages = list(data["stage_order"])
    else:
        selected_stages = list((data.get("tracks") or {}).get(track) or [])
        if not selected_stages:
            raise ValueError(f"unknown or empty lifecycle track: {track}")
    if until is not None and until not in selected_stages:
        raise ValueError(f"stage {until} is outside lifecycle track {track}")
    results = []
    for stage in selected_stages:
        ok, result = run_stage(directory, config, stage, data["stages"][stage], dry_run=dry_run)
        results.append(result)
        if not ok:
            config["status"] = "blocked"
            config["blocked_stage"] = stage
            config["updated_at"] = utc_now()
            atomic_json(directory / "run.json", config)
            print(json.dumps({"ok": False, "run_id": run_id, "results": results}, ensure_ascii=False, indent=2))
            return 2
        if stage == until:
            break
    completed_stage = until or selected_stages[-1]
    if not dry_run:
        if completed_stage == data["stage_order"][-1]:
            config["status"] = "freeze_proposed"
            config["freeze_requires_explicit_user_command"] = True
        else:
            config["status"] = "passed"
            config["completed_stage"] = completed_stage
        config["updated_at"] = utc_now()
        atomic_json(directory / "run.json", config)
    print(json.dumps({"ok": True, "run_id": run_id, "track": track, "dry_run": dry_run, "results": results}, ensure_ascii=False, indent=2))
    return 0


def status(run_id: str) -> dict:
    directory, config = load_run(run_id)
    rows = []
    for stage in recipe()["stage_order"]:
        receipt = load_json(stage_receipt(directory, stage))
        rows.append({"stage": stage, "status": receipt.get("status") or "pending"})
    live = live_stage_state(directory)
    ledger_status = str(config.get("status") or "planned")
    if live["state"] in {"live", "stale"}:
        effective_status = "running"
    elif ledger_status == "initialized":
        effective_status = "planned"
    else:
        effective_status = ledger_status
    return {
        "run_id": run_id,
        "status": effective_status,
        "ledger_status": ledger_status,
        "live": live,
        "target_exposure": config.get("target_exposure"),
        "parent_exposure": (config.get("corpus") or {}).get("parent_exposure"),
        "incremental_exposure": (config.get("corpus") or {}).get("incremental_exposure"),
        "source_capture": source_capture_status(config),
        "current_unchanged": current_hash() == config.get("current_sha256_at_init"),
        "stages": rows,
    }


def baseline_report() -> dict:
    sources = source_manifest()
    live_runs = []
    for root in run_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not (path / "run.json").is_file():
                continue
            live = live_stage_state(path)
            if live["state"] in {"live", "stale"}:
                live_runs.append({"run_id": path.name, **live})
    return {
        "schema_version": 1,
        "kind": "mei-1.0-51m-read-only-baseline",
        "root": str(ROOT),
        "code_revision": code_revision(),
        "source_manifest_sha256": sources["manifest_sha256"],
        "source_file_count": len(sources["files"]),
        "contracts": {
            key: value
            for key, value in architecture_contracts().items()
            if key.endswith("_sha256")
        },
        "current_sha256": current_hash(),
        "live_runs": live_runs,
        "mutated": False,
    }


def verify_static() -> dict:
    data = recipe()
    current = load_json(CURRENT_PATH)
    forbidden = []
    for base in (ROOT, ROOT.parent / "docs"):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if "archive" in path.parts:
                continue
            if "artifacts/mei-1.0-51m/legacy/exp-000300m/runs/mei-1.0-51m" in str(path):
                continue
            if path.suffix.lower() not in {".py", ".json", ".md", ".rs", ".js", ".mjs", ".toml", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            # Long base64 runs (embedding blobs) can contain arbitrary byte
            # pairs that accidentally match the identity pattern below; they
            # are not identity references and must not trip the scan.
            text = re.sub(r"[A-Za-z0-9+/=]{64,}", "", text)
            if re.search(r"(?i)mei[-_.]?1\.0[-_.]?5[8]m|mei[-_.]?5[8]m|5[8]m", text):
                forbidden.append(relative(path))
    legacy_dir = RUN_ROOT / "legacy-51m-qat-20260829"
    legacy = load_json(legacy_dir / "run.json")
    legacy_hash_errors = []
    for row in [*(legacy.get("accepted_evidence") or []), *(legacy.get("blocked_evidence") or [])]:
        row_path = str(row.get("path") or "")
        # 旧时代证据路径基准不统一：依次尝试仓库根、_legacy 根、300M models 根、
        # 300M 根（剥 sdk/ 前缀）。hash 校验不变，只修复基准解析。
        candidates = [
            ROOT / row_path,
            ROOT / "artifacts/mei-1.0-51m/legacy/_legacy" / row_path,
            ARTIFACT_ROOT / "exp-000300m/models" / row_path,
        ]
        if row_path.startswith("base/"):
            tail = row_path[len("base/") :]
            candidates.extend(
                (
                    ARTIFACT_ROOT / "exp-000300m/models/qat" / tail,
                    ARTIFACT_ROOT / "exp-000300m/models/product" / tail,
                )
            )
        if row_path.startswith("sdk/"):
            candidates.append(ARTIFACT_ROOT / "exp-000300m" / row_path[len("sdk/"):])
        resolved = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        actual = sha256_file(resolved) if resolved else "missing"
        if actual != row.get("sha256"):
            legacy_hash_errors.append(
                {
                    "path": row_path,
                    "expected": row.get("sha256"),
                    "actual": actual,
                }
            )
    legacy_ok = (
        legacy.get("status") == "imported_read_only"
        and legacy.get("immutable") is True
        and legacy.get("resumable") is False
        and legacy.get("promotion_allowed") is False
        and (legacy_dir / "READ_ONLY").is_file()
        and not legacy_hash_errors
    )
    return {
        "ok": (
            not forbidden
            and legacy_ok
            and current.get("product") == "mei-1.0-51m"
            and current.get("sft") is None
            and current.get("runtime") is None
        ),
        "recipe": data["id"],
        "architecture_source_sha256": architecture_sha256(),
        "contracts": {
            key: value
            for key, value in architecture_contracts().items()
            if key.endswith("_sha256")
        },
        "params": EXPECTED_PARAMS,
        "current_sft": current.get("sft"),
        "current_runtime": current.get("runtime"),
        "forbidden_identity_refs": forbidden,
        "legacy_import": {
            "ok": legacy_ok,
            "run_id": legacy.get("run_id"),
            "hash_errors": legacy_hash_errors,
        },
        "offline_required": data.get("offline_required"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-id", required=True)
    init.add_argument("--cycle-id", help="新链 cycle（exp-XXXXXXm-vN）；缺省 = 旧链 RUN_ROOT")
    init.add_argument("--corpus-dir", type=Path, required=True)
    init.add_argument("--target-exposure", type=int, required=True)
    init.add_argument("--resume-checkpoint", type=Path)
    for name in ("run", "resume", "plan", "status"):
        child = sub.add_parser(name)
        child.add_argument("--run-id", required=True)
        if name in {"run", "resume", "plan"}:
            child.add_argument("--until", choices=recipe()["stage_order"])
            child.add_argument("--track", choices=("cpt",), default="cpt")
    sub.add_parser("verify")
    sub.add_parser("baseline")
    args = parser.parse_args()
    if args.action == "init":
        print(
            json.dumps(
                init_run(
                    args.run_id,
                    args.corpus_dir,
                    args.target_exposure,
                    resume_checkpoint=args.resume_checkpoint,
                    cycle_id=args.cycle_id,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "status":
        print(json.dumps(status(args.run_id), ensure_ascii=False, indent=2))
        return 0
    if args.action == "verify":
        report = verify_static()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2
    if args.action == "baseline":
        print(json.dumps(baseline_report(), ensure_ascii=False, indent=2))
        return 0
    return execute(args.run_id, dry_run=args.action == "plan", until=args.until, track=args.track)


if __name__ == "__main__":
    raise SystemExit(main())
