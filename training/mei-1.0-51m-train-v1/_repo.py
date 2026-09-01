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
TOKENIZER_DIR = ROOT / "tokenizer/zh-24k-v1"
TOKENIZER_ZH_V1 = TOKENIZER_DIR / "zh-24k-v1.model"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "tokenizer-v1-manifest.json"

_ARCHITECTURE_ID_RE = re.compile(r"^mei-1\.0-51m-arch-v\d+$")
ARCHITECTURE_ID = os.environ.get("MEI_ARCHITECTURE_ID", "mei-1.0-51m-arch-v1")
if "/" in ARCHITECTURE_ID or "\\" in ARCHITECTURE_ID or not _ARCHITECTURE_ID_RE.fullmatch(ARCHITECTURE_ID):
    raise RuntimeError(f"unsupported architecture identity: {ARCHITECTURE_ID}")
ARCHITECTURE_DIR = ROOT / "architecture" / ARCHITECTURE_ID
if not (ARCHITECTURE_DIR / "spec/model.json").is_file():
    raise RuntimeError(f"missing architecture spec: {ARCHITECTURE_DIR / 'spec/model.json'}")
ARCHITECTURE_SPEC = ARCHITECTURE_DIR / "spec"
ARCHITECTURE_CONTRACT = ARCHITECTURE_DIR / "architecture_contract.py"

TRAINING_DIR = ROOT / "training/mei-1.0-51m-train-v1"
TRAIN_RUNS = ROOT / "training/runs"
RUNTIME_SHARED = ROOT / "runtime/_shared"
CORPUS_LM_V1 = ROOT / "corpus/lm-v1"
CORPUS_LM_V2 = ROOT / "corpus/lm-v2"
CORPUS_ZH_PRETRAIN = CORPUS_LM_V1 / "language/zh-pretrain-v0"
BASE_SCRATCH300M = ROOT / "base/mei-1.0-51m-base-scratch300m-v1"
BASE_CPT1B = ROOT / "base/mei-1.0-51m-base-cpt1b-v1"
CPT_1B_RUN = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
EVAL_SHARED_ROOT = ROOT / "notebook/evaluation/shared"


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
    for path in reversed((ARCHITECTURE_DIR, TRAINING_DIR, RUNTIME_SHARED)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
