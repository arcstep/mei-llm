"""Repository paths shared by architecture-neutral runtime helpers."""

from __future__ import annotations

from pathlib import Path


def find_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "CURRENT.json").is_file():
            return candidate
    raise RuntimeError("cannot locate mei-llm root")


ROOT = find_root()
EVAL_SHARED_ROOT = ROOT / "notebook/evaluation/shared"
