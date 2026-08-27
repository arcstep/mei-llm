#!/usr/bin/env python3
"""SIGINT sidecar producers when plus+sidecar unique sums to 30M. Never touches qwen-v1."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timedelta, timezone

from colloquial_fleet import FORMAL_DIR, POOLED_UNIQUE_TARGET, SIDECAR_DIRS, pooled_unique
from repo_paths import EXPERIMENTS_RUNS, ROOT

BJ = timezone(timedelta(hours=8))
MARKER = EXPERIMENTS_RUNS / "colloquial-fleet" / "pooled-stop.json"


def sidecar_pids() -> list[dict]:
    import subprocess

    out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if "produce_colloquial_qwen.py" not in line:
            continue
        if "/bin/zsh" in line or "builtin eval" in line:
            continue
        if FORMAL_DIR in line or "zh-pretrain-colloquial-synth-qwen-v1" in line:
            continue
        if not any(name in line for name in SIDECAR_DIRS):
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        rows.append({"pid": pid, "cmd": cmd.strip()})
    return rows


def stop_sidecars(rows: list[dict]) -> list[dict]:
    stopped = []
    for row in rows:
        pid = int(row["pid"])
        try:
            os.kill(pid, signal.SIGINT)
            stopped.append({**row, "signal": "SIGINT"})
        except ProcessLookupError:
            stopped.append({**row, "signal": "gone"})
    return stopped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    while True:
        snap = pooled_unique()
        rows = sidecar_pids()
        payload = {
            "ts_bj": datetime.now(tz=BJ).strftime("%Y-%m-%d %H:%M:%S"),
            "cwd": str(ROOT),
            "pooled_unique": snap["total"],
            "pooled_target": POOLED_UNIQUE_TARGET,
            "remain": snap["remain"],
            "per_dir": snap["per_dir"],
            "sidecar_pids": rows,
            "stopped": False,
        }
        if snap["total"] >= POOLED_UNIQUE_TARGET:
            payload["stopped"] = True
            payload["actions"] = stop_sidecars(rows)
            MARKER.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return 0
        print(
            json.dumps(
                {
                    "waiting": True,
                    "pooled_unique": snap["total"],
                    "remain": snap["remain"],
                    "n_sidecar_procs": len(rows),
                }
            ),
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(max(args.interval, 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
