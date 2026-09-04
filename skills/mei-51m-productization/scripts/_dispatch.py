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
        if (candidate / "src/mei_llm").is_dir() and (candidate / "CURRENT.json").is_file():
            return candidate.resolve()
    raise SystemExit("cannot locate mei-llm; set MEI_LLM_ROOT")


def _confirmed(arguments: list[str]) -> list[str]:
    if "--help" in arguments or "-h" in arguments:
        return arguments
    flag = "--confirm-training"
    if flag not in arguments:
        raise SystemExit("training refused: pass --confirm-training explicitly")
    os.environ["MEI_TRAINING_CONFIRMED"] = "1"
    return [value for value in arguments if value != flag]


def run(action: str, *, changes_weights: bool = False) -> int:
    repo = root()
    arguments = list(sys.argv[1:])
    if changes_weights:
        arguments = _confirmed(arguments)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "mei_llm", "productization", action, *arguments],
        cwd=repo,
        env=env,
    ).returncode
