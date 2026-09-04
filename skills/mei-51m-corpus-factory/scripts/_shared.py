#!/usr/bin/env python3
"""Shared repository discovery for corpus-factory skill entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_PATHS = (
    Path("corpus-factory/FACTORY.json"),
    Path("corpus-factory/generators/rebuild_zh_v1/build.py"),
    Path("CURRENT.json"),
)


def _ancestors(path: Path) -> list[Path]:
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    return [start, *start.parents]


def find_repo_root() -> Path:
    """Find the mei-llm code root from source or projected skill locations."""
    candidates: list[Path] = []
    configured = os.environ.get("MEI_LLM_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())

    for base in (*_ancestors(Path(__file__)), *_ancestors(Path.cwd())):
        candidates.extend((base, base / "mei-llm" / "mei-llm", base / "mei-llm"))

    visited: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        if all((resolved / relative).is_file() for relative in REQUIRED_PATHS):
            return resolved

    expected = ", ".join(str(path) for path in REQUIRED_PATHS)
    raise SystemExit(
        "无法定位 mei-llm 代码仓。请在 mei-projects 工作区运行，"
        f"或设置 MEI_LLM_ROOT。所需文件：{expected}"
    )


def preferred_python(root: Path) -> Path:
    venv_python = root / ".venv" / "bin" / "python"
    return venv_python if venv_python.is_file() else Path(sys.executable)
