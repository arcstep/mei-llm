#!/usr/bin/env python3
"""Kill lock-v2 if free RAM drops below a floor. Coordinator PID is argv1."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


def unused_g() -> float:
    out = subprocess.check_output(["vm_stat"], text=True)
    match = re.search(r"page size of (\d+)", out)
    page = int(match.group(1)) if match else 16384
    stats = {}
    for line in out.splitlines()[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        stats[key.strip()] = int(re.sub(r"\D", "", value) or 0)
    unused = (
        stats.get("Pages free", 0) + stats.get("Pages speculative", 0) + stats.get("Pages purgeable", 0)
    ) * page
    return unused / (1024**3)


def workers() -> list[tuple[int, int]]:
    ps = subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True)
    rows = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, cmd = int(parts[0]), int(parts[1]), parts[2]
        if "eval_lock_v2_51m.py" in cmd:
            rows.append((pid, rss // 1024))
    return rows


def main() -> int:
    coord = int(sys.argv[1])
    watch = Path(sys.argv[2])
    min_unused_g = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
    max_rss_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    started = time.time()
    watch.write_text("elapsed_s\tphys_unused_g\tworker_rss_mb\tworker_pids\n", encoding="utf-8")
    while True:
        try:
            os.kill(coord, 0)
        except OSError:
            print("coordinator gone", flush=True)
            return 0
        elapsed = int(time.time() - started)
        unused = unused_g()
        rows = workers()
        rss_sum = sum(row[1] for row in rows)
        pids = ",".join(str(row[0]) for row in rows) or str(coord)
        with watch.open("a", encoding="utf-8") as handle:
            handle.write(f"{elapsed}\t{unused:.1f}\t{rss_sum}\t{pids}\n")
        print(f"t={elapsed}s unused={unused:.1f}G workers_rss={rss_sum}MB pids={pids}", flush=True)
        if unused < min_unused_g or any(row[1] > max_rss_mb for row in rows):
            print("KILL memory cap", flush=True)
            subprocess.call(["pkill", "-TERM", "-f", "eval_lock_v2_51m.py"])
            time.sleep(2)
            subprocess.call(["pkill", "-KILL", "-f", "eval_lock_v2_51m.py"])
            return 2
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
