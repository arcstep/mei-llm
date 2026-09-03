"""Locate the mei-llm root and expose the single supported 51M identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "CURRENT.json").is_file():
            return candidate
    raise RuntimeError("cannot locate mei-llm root (need CURRENT.json)")


ROOT = find_root()
CURRENT_PATH = ROOT / "CURRENT.json"
TOKENIZER_DIR = ROOT / "models/mei-1.0-51m/tokenizer"
TOKENIZER_ZH_V1 = TOKENIZER_DIR / "zh-24k-v1.model"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "tokenizer-v1-manifest.json"

_ARCHITECTURE_ID_RE = re.compile(r"^mei-1\.0-51m-arch-v\d+$")
ARCHITECTURE_ID = os.environ.get("MEI_ARCHITECTURE_ID", "mei-1.0-51m-arch-v1")
if "/" in ARCHITECTURE_ID or "\\" in ARCHITECTURE_ID or not _ARCHITECTURE_ID_RE.fullmatch(ARCHITECTURE_ID):
    raise RuntimeError(f"unsupported architecture identity: {ARCHITECTURE_ID}")
ARCHITECTURE_DIR = ROOT / "models/mei-1.0-51m/architecture"
if not (ARCHITECTURE_DIR / "spec/model.json").is_file():
    raise RuntimeError(f"missing architecture spec: {ARCHITECTURE_DIR / 'spec/model.json'}")
ARCHITECTURE_SPEC = ARCHITECTURE_DIR / "spec"
ARCHITECTURE_CONTRACT = ARCHITECTURE_DIR / "architecture_contract.py"

MODEL_FACTORY_DIR = ROOT / "model-factory"
RECIPES_DIR = MODEL_FACTORY_DIR / "recipes"
# Compatibility name used by older pipeline code. It now points at the visible
# model-factory package root, not at a hidden flat scripts directory.
TRAINING_DIR = MODEL_FACTORY_DIR
ARTIFACT_ROOT = ROOT / ".local/artifacts/mei-1.0-51m"
TRAIN_RUNS = ARTIFACT_ROOT / "exp-000300m/runs"
RUNTIME_SHARED = ROOT / "platform/_shared/runtime"
CORPUS_LM_V1 = ARTIFACT_ROOT / "exp-000300m/corpus/cpt-delta/lm-v1"
CORPUS_LM_V2 = ROOT / ".local/artifacts/_legacy/corpus/planned-exp-001000m-lm-v2"
CORPUS_ZH_PRETRAIN = CORPUS_LM_V1 / "language/zh-pretrain-v0"
BASE_SCRATCH300M = ARTIFACT_ROOT / "exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1"
BASE_CPT1B = ARTIFACT_ROOT / "exp-001000m/models/base/mei-1.0-51m-base-cpt1b-v1"
CPT_1B_RUN = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
EVAL_SHARED_ROOT = ROOT / ".local/artifacts/_legacy/notebook/evaluation/shared"


def cycle_artifacts(cycle_id: str) -> Path:
    if not re.fullmatch(r"exp-[0-9]{6}m", cycle_id):
        raise ValueError(f"invalid cycle id: {cycle_id}")
    return ARTIFACT_ROOT / cycle_id


def cycle_runs(cycle_id: str) -> Path:
    return cycle_artifacts(cycle_id) / "runs"


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a current path or an immutable pre-migration receipt path."""

    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        try:
            candidate = candidate.relative_to(ROOT)
        except ValueError:
            return candidate
    direct = ROOT / candidate
    if direct.exists():
        return direct

    raw = candidate.as_posix()
    old_run_prefix = "training/runs/mei-1.0-51m/"
    if raw.startswith(old_run_prefix):
        suffix = raw[len(old_run_prefix) :]
        candidates = [
            *ARTIFACT_ROOT.glob(f"exp-*/runs/{suffix}"),
            ARTIFACT_ROOT / "comparisons/runs" / suffix,
        ]
        matches = sorted(path for path in candidates if path.exists())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous migrated run path: {raw}")

    migration_path = ROOT / ".internal/registry/migrations/2026-09-four-domain.json"
    if migration_path.is_file():
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        exact = migration.get("exact") or {}
        if raw in exact:
            return ROOT / str(exact[raw])
        for old, new in sorted(exact.items(), key=lambda item: -len(item[0])):
            prefix = old.rstrip("/") + "/"
            if raw.startswith(prefix):
                return ROOT / str(new) / raw[len(prefix) :]
        prefixes = migration.get("prefix") or {}
        for old, new in sorted(prefixes.items(), key=lambda item: -len(item[0])):
            if raw.startswith(old):
                return ROOT / str(new) / raw[len(old) :]
    return direct


def architecture_sha256() -> str:
    """Hash source bytes for provenance only; never use as weight compatibility."""
    digest = hashlib.sha256()
    for rel in (
        "architecture.py",
        "config.py",
        "architecture_contract.py",
        "spec/model.json",
        "spec/legacy-architecture-aliases.json",
    ):
        path = ARCHITECTURE_DIR / rel
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def architecture_contracts() -> dict:
    """Return the three normalized architecture contracts without importing MLX."""
    text = str(ARCHITECTURE_DIR)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from architecture_contract import contract_bundle

        return contract_bundle()
    finally:
        if added:
            sys.path.remove(text)


def weight_contract_sha256() -> str:
    return str(architecture_contracts()["weight_contract_sha256"])


def runtime_profile_sha256() -> str:
    return str(architecture_contracts()["runtime_profile_sha256"])


def training_aux_sha256() -> str:
    return str(architecture_contracts()["training_aux_sha256"])


def legacy_weight_contract_sha256(legacy_architecture_sha256: str) -> str | None:
    text = str(ARCHITECTURE_DIR)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from architecture_contract import resolve_legacy_weight_contract

        return resolve_legacy_weight_contract(str(legacy_architecture_sha256 or ""))
    finally:
        if added:
            sys.path.remove(text)


def load_current() -> dict:
    if not CURRENT_PATH.is_file():
        return {}
    return json.loads(CURRENT_PATH.read_text(encoding="utf-8"))


def ensure_formal_on_path() -> None:
    for path in reversed((ROOT / "src", ARCHITECTURE_DIR, MODEL_FACTORY_DIR, RUNTIME_SHARED)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
