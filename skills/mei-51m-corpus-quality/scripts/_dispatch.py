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
        if (candidate / "corpus-factory/quality/audit.py").is_file():
            return candidate.resolve()
    raise SystemExit("cannot locate mei-llm; set MEI_LLM_ROOT")


def run(action: str) -> int:
    repo = root()
    if action == "diagnose-signal":
        arguments = [
            sys.executable,
            str(
                repo
                / "model-factory/diagnostics/mw_disposition_corpus_diagnostic_51m.py"
            ),
            *sys.argv[1:],
        ]
    else:
        arguments = [
            sys.executable,
            "-m",
            "mei_llm",
            "corpus",
            "evaluate",
            action,
            *sys.argv[1:],
        ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(arguments, cwd=repo, env=env).returncode
