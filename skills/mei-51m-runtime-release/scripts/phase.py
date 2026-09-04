#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "src/mei_llm/cli.py").is_file():
            return candidate
        sibling = candidate / "mei-llm"
        if (sibling / "src/mei_llm/cli.py").is_file():
            return sibling
    raise RuntimeError("cannot locate mei-llm repository")


if len(sys.argv) < 2 or sys.argv[1] not in {
    "template", "doctor", "plan", "run", "resume", "status"
}:
    raise SystemExit("usage: phase.py {template|doctor|plan|run|resume|status} [arguments...]")
repository = root()
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(
    [str(repository / "src"), environment.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
raise SystemExit(
    subprocess.run(
        [sys.executable, "-m", "mei_llm", "runtime-release", *sys.argv[1:]],
        cwd=repository,
        env=environment,
    ).returncode
)
