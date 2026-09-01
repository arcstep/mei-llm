#!/usr/bin/env python3
"""Run the frozen scratch 300M curriculum, or the 5M mechanism drill."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ARCHITECTURE_ID, ROOT, TRAIN_RUNS, ensure_formal_on_path

ensure_formal_on_path()

from pretrain_gates import refuse_non_scratch_source

SCRATCH_PROFILES = {
    "mei-1.0-51m-arch-v1": {
        "recipe": "pretrain-51m-rungs.json",
        "300m_run": "pretrain-mei-1.0-51m-base-scratch300m-v1",
        "pilot_run": "pretrain-mei-1.0-51m-pilot-5m-v1",
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def apply_layouts(stages: list[dict], recipe: dict) -> list[dict]:
    layouts = recipe.get("layouts") or {}
    out = []
    for stage in stages:
        row = dict(stage)
        layout = layouts.get(str(row.get("seq_len"))) or {}
        if layout.get("batch_size"):
            row["batch_size"] = int(layout["batch_size"])
        if layout.get("grad_accum"):
            row["grad_accum"] = int(layout["grad_accum"])
        out.append(row)
    return out


def first_incomplete_stage(stages: list[dict], tokens_seen: int) -> int:
    for i, stage in enumerate(stages):
        if int(tokens_seen) < int(stage["stop_at_tokens"]):
            return i
    return len(stages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-5m", action="store_true", help="2.5M@512 + 1.5M@1024 + 1M@2048 mechanism drill")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume pretrain-300m-scratch (or pilot) from the current state instead of restarting S1",
    )
    args = ap.parse_args()

    profile = SCRATCH_PROFILES.get(ARCHITECTURE_ID)
    if profile is None:
        print(f"no scratch curriculum profile for {ARCHITECTURE_ID}", file=sys.stderr)
        return 2
    recipe = load_json(_HERE / "recipes" / profile["recipe"])
    blocked = refuse_non_scratch_source(ROOT / "corpus/lm-v1", "pilot-5m" if args.pilot_5m else "300m")
    if blocked:
        print(blocked, file=sys.stderr)
        return 4

    if args.pilot_5m:
        stages = list(recipe.get("pilot_curriculum") or [])
        rung = "pilot-5m"
        run_name = profile["pilot_run"]
    else:
        stages = apply_layouts(list(recipe.get("curriculum") or []), recipe)
        rung = "300m"
        run_name = profile["300m_run"]
    if len(stages) != 3:
        print("recipe curriculum must have three stages", file=sys.stderr)
        return 2

    trainer = _HERE / "train_pretrain.py"
    out_dir = TRAIN_RUNS / run_name
    last_state = out_dir / f"{run_name}-state.npz"
    compile_train = (not args.no_compile) and (args.compile or bool(recipe.get("compile_train")))
    start_index = 0
    if args.resume_existing:
        meta = load_json(last_state.with_suffix(".meta.json"))
        if not last_state.is_file() or not meta:
            print(f"missing resume state {last_state}", file=sys.stderr)
            return 3
        ckpt_arch = meta.get("architecture_id")
        if ckpt_arch:
            if str(ckpt_arch) != ARCHITECTURE_ID:
                print(
                    f"architecture_id mismatch: ckpt={ckpt_arch} expected={ARCHITECTURE_ID}",
                    file=sys.stderr,
                )
                return 4
        elif ARCHITECTURE_ID != "mei-1.0-51m-arch-v1":
            print(
                f"architecture_id missing in {last_state.with_suffix('.meta.json')}; "
                f"expected={ARCHITECTURE_ID}",
                file=sys.stderr,
            )
            return 4
        start_index = first_incomplete_stage(stages, int(meta.get("tokens_seen") or 0))
        if start_index >= len(stages):
            print(json.dumps({"ok": True, "already_complete": True, "tokens_seen": meta.get("tokens_seen")}))
            return 0
    runnable: list[tuple[dict, list[str]]] = []
    for i, stage in enumerate(stages):
        if i < start_index:
            continue
        cmd = [
            args.python,
            str(trainer),
            "--rung",
            rung,
            "--corpus-dir",
            "corpus/lm-v1",
            "--schedule-kind",
            "scratch",
            "--curriculum-stage",
            str(stage["id"]),
            "--seq-len",
            str(stage["seq_len"]),
            "--batch-size",
            str(stage["batch_size"]),
            "--grad-accum",
            str(stage["grad_accum"]),
            "--stop-at-tokens",
            str(stage["stop_at_tokens"]),
            "--out-dir",
            str(out_dir.relative_to(ROOT)),
        ]
        if compile_train:
            cmd.append("--compile")
        else:
            cmd.append("--no-compile")
        if i > 0 or args.resume_existing:
            cmd.extend(["--resume", str(last_state), "--resume-mode", "curriculum"])
        runnable.append((stage, cmd))

    print(
        json.dumps(
            {
                "run": run_name,
                "architecture_id": ARCHITECTURE_ID,
                "resume_existing": bool(args.resume_existing),
                "compile_train": compile_train,
                "start_stage": stages[start_index]["id"] if start_index < len(stages) else None,
                "stages": stages,
                "commands": [" ".join(cmd) for _, cmd in runnable],
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    for stage, cmd in runnable:
        print("+", " ".join(cmd), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MEI_ARCHITECTURE_ID"] = ARCHITECTURE_ID
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            return int(proc.returncode)
        if not last_state.is_file():
            print(f"missing {last_state} after stage {stage['id']}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
