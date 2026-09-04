from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def root() -> Path:
    configured = os.environ.get("MEI_LLM_ROOT")
    candidates = [Path(configured)] if configured else []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start.parent, *start.parents):
            candidates.extend((parent, parent / "mei-llm/mei-llm", parent / "mei-llm"))
    for candidate in candidates:
        if (candidate / "CURRENT.json").is_file() and (candidate / "src/mei_llm").is_dir():
            return candidate.resolve()
    raise SystemExit("cannot locate mei-llm; set MEI_LLM_ROOT")


def run(action: str) -> int:
    repo = root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "mei_llm", "cycle", action, *sys.argv[1:]],
        cwd=repo,
        env=env,
    ).returncode
