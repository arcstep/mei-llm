#!/usr/bin/env python3
"""Detached locked CPT service: start/status/pause/resume/tail.

The trainer is launched with start_new_session=True so Cursor/chat exit
does not stop the run. Pause writes STOP after the current optimizer update.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import CORPUS_LM_V2, ROOT, TRAIN_RUNS, ensure_formal_on_path

ensure_formal_on_path()
from cpt_gates import refuse_cpt_parent, refuse_cpt_source
from run_lock import lock_is_held, other_held_runs, pid_alive_from_meta, read_lock_meta

PARENT_STATE = ROOT / "base/mei-1.0-51m-base-scratch300m-v1/mei-1.0-51m-base-scratch300m-v1-state.npz"
DEFAULT_RUN = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
TRAINER = _HERE / "train_pretrain.py"


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dir_for(rung: str, out_dir: Path | None) -> Path:
    if out_dir:
        path = Path(out_dir)
        return path if path.is_absolute() else ROOT / path
    if rung == "1b":
        return DEFAULT_RUN
    return TRAIN_RUNS / f"pretrain-{rung}"


def state_path(run_dir: Path) -> Path:
    name = run_dir.name
    return run_dir / f"{name}-state.npz"


def live_training_pids() -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", "train_pretrain.py|train_sft.py"], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pid = int(line)
            if pid != os.getpid():
                pids.append(pid)
    return pids


def memory_ok() -> tuple[bool, str]:
    try:
        out = subprocess.check_output(["memory_pressure"], text=True, timeout=5)
    except Exception as exc:  # noqa: BLE001
        return True, f"memory_pressure unavailable: {exc}"
    low = "warning" in out.lower() or "critical" in out.lower() or "urgent" in out.lower()
    return (not low), out.splitlines()[-1] if out.strip() else "ok"


def resolve_resume(run_dir: Path) -> Path:
    last = state_path(run_dir)
    if last.is_file():
        return last
    return PARENT_STATE


def spawn_trainer(
    *,
    python: str,
    rung: str,
    run_dir: Path,
    resume: Path,
    corpus_dir: Path,
    target_tokens: int | None,
    extra: list[str],
) -> subprocess.Popen:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "daemon.log"
    cmd = [
        python,
        str(TRAINER),
        "--rung",
        rung,
        "--schedule-kind",
        "cpt",
        "--corpus-dir",
        str(corpus_dir),
        "--resume",
        str(resume),
        "--resume-mode",
        "auto",
        "--out-dir",
        str(run_dir),
    ]
    if target_tokens is not None:
        cmd.extend(("--target-tokens", str(int(target_tokens))))
    cmd.extend(extra)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("ab") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(ROOT),
            close_fds=True,
            env=env,
        )
    dump(
        run_dir / "daemon.json",
        {
            "pid": proc.pid,
            "pgid": os.getpgid(proc.pid),
            "host": socket.gethostname(),
            "start_unix": time.time(),
            "cmd": cmd,
            "resume": str(resume),
            "rung": rung,
            "corpus_dir": str(corpus_dir),
            "target_tokens": target_tokens,
        },
    )
    return proc


def cmd_start(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.rung, args.out_dir)
    if run_dir.resolve() == (TRAIN_RUNS / "pretrain-300m-scratch").resolve():
        print("refusing to overwrite the frozen 300M scratch run", file=sys.stderr)
        return 2
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    blocked = refuse_cpt_source(corpus_dir, args.rung)
    if blocked:
        print(blocked, file=sys.stderr)
        return 4
    parent_err = refuse_cpt_parent()
    if parent_err:
        print(parent_err, file=sys.stderr)
        return 4
    held = lock_is_held(run_dir)
    if held and pid_alive_from_meta(held):
        print(json.dumps({"error": "run already locked", "lock": held}, ensure_ascii=False), file=sys.stderr)
        return 2
    others = other_held_runs(except_dir=run_dir)
    if others:
        print(json.dumps({"error": "other formal GPU run lock held", "held": others}, ensure_ascii=False), file=sys.stderr)
        return 4
    live = live_training_pids()
    if live and not args.allow_live:
        print(json.dumps({"error": "other train_pretrain/train_sft pid live", "pids": live}, ensure_ascii=False), file=sys.stderr)
        return 4
    ok_mem, mem_note = memory_ok()
    if not ok_mem and not args.allow_pressure:
        print(json.dumps({"error": "memory pressure not healthy", "note": mem_note}, ensure_ascii=False), file=sys.stderr)
        return 4
    stop = run_dir / "STOP"
    if stop.is_file():
        stop.unlink()
    resume = Path(args.resume) if args.resume else resolve_resume(run_dir)
    extra = []
    if args.compile:
        extra.append("--compile")
    if args.no_compile:
        extra.append("--no-compile")
    proc = spawn_trainer(
        python=args.python,
        rung=args.rung,
        run_dir=run_dir,
        resume=resume,
        corpus_dir=corpus_dir,
        target_tokens=args.target_tokens,
        extra=extra,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "action": "start",
                "pid": proc.pid,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "resume": str(resume),
                "memory": mem_note,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.rung, args.out_dir)
    lock = lock_is_held(run_dir)
    heart = load(run_dir / "heartbeat.json")
    summary = load(run_dir / "summary.json")
    daemon = load(run_dir / "daemon.json")
    pid = int((lock or daemon or heart).get("pid") or 0)
    alive = pid_alive_from_meta({"pid": pid}) if pid else False
    payload = {
        "run_dir": str(run_dir),
        "lock_held": bool(lock),
        "pid": pid,
        "pid_alive": alive,
        "host": (lock or daemon or {}).get("host"),
        "heartbeat": heart,
        "tokens_seen": heart.get("tokens_seen") or summary.get("tokens_seen"),
        "loss": heart.get("loss") or summary.get("last_loss"),
        "tok_s": heart.get("tok_s") or summary.get("segment_tok_s"),
        "paused": bool(summary.get("paused")) or (run_dir / "STOP").is_file(),
        "latest_checkpoint": str(state_path(run_dir)) if state_path(run_dir).is_file() else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if alive or not args.require_alive else 1


def cmd_pause(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.rung, args.out_dir)
    lock = lock_is_held(run_dir)
    if not lock:
        print(json.dumps({"ok": True, "action": "pause", "already_stopped": True}))
        return 0
    (run_dir / "STOP").write_text("pause\n", encoding="utf-8")
    pid = int(lock.get("pid") or 0)
    deadline = time.time() + float(args.wait)
    while time.time() < deadline:
        if not pid_alive_from_meta({"pid": pid}):
            break
        time.sleep(1)
    alive = pid_alive_from_meta({"pid": pid})
    print(json.dumps({"ok": not alive, "action": "pause", "pid": pid, "still_running": alive}, indent=2))
    return 0 if not alive else 1


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.rung, args.out_dir)
    stop = run_dir / "STOP"
    if stop.is_file():
        stop.unlink()
    last = state_path(run_dir)
    if not last.is_file():
        print("no CPT state to resume", file=sys.stderr)
        return 2
    args.resume = last
    return cmd_start(args)


def cmd_tail(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.rung, args.out_dir)
    path = run_dir / ("progress.log" if not args.daemon else "daemon.log")
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-int(args.n) :]:
        print(line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "status", "pause", "resume", "tail"])
    ap.add_argument("--rung", default="1b")
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_LM_V2)
    ap.add_argument("--target-tokens", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--allow-live", action="store_true")
    ap.add_argument("--allow-pressure", action="store_true")
    ap.add_argument("--require-alive", action="store_true")
    ap.add_argument("--wait", type=int, default=600, help="pause wait seconds")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--daemon", action="store_true", help="tail daemon.log instead of progress.log")
    args = ap.parse_args()
    if args.action == "start":
        return cmd_start(args)
    if args.action == "status":
        return cmd_status(args)
    if args.action == "pause":
        return cmd_pause(args)
    if args.action == "resume":
        return cmd_resume(args)
    if args.action == "tail":
        return cmd_tail(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
