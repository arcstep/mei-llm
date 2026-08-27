"""Locate the mei-llm repo root from any formal release tree.

Formal code must not import notebook/_tooling/scripts/repo_paths.
Root is the directory that contains CURRENT.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "CURRENT.json").is_file():
            return p
    raise RuntimeError("cannot locate mei-llm root (need CURRENT.json)")


ROOT = find_root()
CURRENT_PATH = ROOT / "CURRENT.json"

TOKENIZER_DIR = ROOT / "tokenizer" / "zh-24k-v1"
TOKENIZER_ZH_V1 = TOKENIZER_DIR / "zh-24k-v1.model"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "tokenizer-v1-manifest.json"

ARCHITECTURE_DIR = ROOT / "architecture" / "mei-1.0-58m-arch-v1"
ARCHITECTURE_SPEC = ARCHITECTURE_DIR / "spec"
TRAINING_DIR = ROOT / "training" / "mei-1.0-58m-train-v1"
TRAIN_RUNS = ROOT / "training" / "runs"
RUNTIME_SHARED = ROOT / "runtime" / "_shared"
RUNTIME_ROUTE_V1 = ROOT / "runtime" / "mei-1.0-58m-route-v1"
RUNTIME_NEEDLE2_V2 = ROOT / "runtime" / "mei-1.0-58m-needle2-v2"
CORPUS_LM_V1 = ROOT / "corpus" / "lm-v1"
CORPUS_ZH_PRETRAIN = CORPUS_LM_V1 / "language" / "zh-pretrain-v0"

EVAL_SHARED_ROOT = ROOT / "notebook" / "evaluation" / "shared"


def load_current() -> dict:
    if not CURRENT_PATH.is_file():
        return {}
    return json.loads(CURRENT_PATH.read_text(encoding="utf-8"))


def formal_pythonpath() -> list[Path]:
    return [
        ARCHITECTURE_DIR,
        TRAINING_DIR,
        RUNTIME_SHARED,
        RUNTIME_ROUTE_V1,
        RUNTIME_NEEDLE2_V2,
    ]


def ensure_formal_on_path() -> None:
    for path in reversed(formal_pythonpath()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
