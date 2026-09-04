#!/usr/bin/env python3
"""Compatibility router; delegates to one specialized Skill script."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROUTES = {
    "cycle": ("mei-51m-cycle-orchestrator", "action_script"),
    "sourcing": ("mei-51m-corpus-sourcing", "action_script"),
    "synthesis": ("mei-51m-corpus-factory", "action_script"),
    "quality": ("mei-51m-corpus-quality", "action_script"),
    "cpt": ("mei-51m-cpt-training", "action_script"),
    "qat": ("mei-51m-qat-training", "action_script"),
    "sft": ("mei-51m-sft-alignment", "phase_script"),
    "evaluation": ("mei-51m-model-evaluation", "phase_script"),
    "runtime": ("mei-51m-runtime-release", "phase_script"),
    "productization": ("mei-51m-productization", "action_script"),
}


def repo_root() -> Path:
    configured = os.environ.get("MEI_LLM_ROOT")
    candidates = [Path(configured)] if configured else []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start.parent, *start.parents):
            candidates.extend((parent, parent / "mei-llm/mei-llm", parent / "mei-llm"))
    for candidate in candidates:
        if (candidate / "skills/mei-51m-cycle-orchestrator").is_dir():
            return candidate.resolve()
    raise SystemExit("cannot locate mei-llm; set MEI_LLM_ROOT")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(ROUTES), required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    skill, style = ROUTES[args.task]
    script = repo_root() / "skills" / skill / "scripts"
    command_arguments = arguments
    if style == "phase_script":
        script = script / "phase.py"
        command_arguments = [args.action, *arguments]
    else:
        script = script / f"{args.action.replace('-', '_')}.py"
    if not script.is_file():
        raise SystemExit(f"unsupported routed action: {args.task}/{args.action}")
    return subprocess.run([sys.executable, str(script), *command_arguments]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
