from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DOMAIN = "qat"


def root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "src/mei_llm/cli.py").is_file():
            return candidate
        sibling = candidate / "mei-llm"
        if (sibling / "src/mei_llm/cli.py").is_file():
            return sibling
    raise RuntimeError("cannot locate mei-llm repository")


def run(action: str, argv: list[str]) -> int:
    repository = root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "mei_llm", DOMAIN, action, *argv],
        cwd=repository,
        env=env,
    ).returncode
