#!/usr/bin/env python3
"""Run-lock and detached-service tests. No GPU training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import ROOT, ensure_formal_on_path

ensure_formal_on_path()
from common.run_lock import acquire_run_lock, lock_is_held, other_held_runs, write_heartbeat
import common.run_lock
import training.cpt.cpt_service as service


def test_dual_lock_refuses() -> None:
    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw) / "run-a"
        fd = acquire_run_lock(run_dir, {"recipe": "test"})
        held = lock_is_held(run_dir)
        assert held and int(held["pid"]) == os.getpid()
        raised = False
        try:
            acquire_run_lock(run_dir, {"recipe": "test2"})
        except RuntimeError as exc:
            raised = True
            assert "run lock held" in str(exc)
        assert raised
        os.close(fd)


def test_stale_lock_recovers_after_release() -> None:
    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw) / "run-b"
        fd = acquire_run_lock(run_dir, {"recipe": "test"})
        os.close(fd)
        assert lock_is_held(run_dir) is None
        fd2 = acquire_run_lock(run_dir, {"recipe": "recovered"})
        os.close(fd2)


def test_pause_marker_and_heartbeat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw)
        write_heartbeat(run_dir / "heartbeat.json", {"tokens_seen": 12, "loss": 1.2})
        heart = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
        assert heart["tokens_seen"] == 12
        assert heart["pid"] == os.getpid()
        (run_dir / "STOP").write_text("pause\n", encoding="utf-8")
        assert (run_dir / "STOP").is_file()


def test_detached_session_survives_parent() -> None:
    with tempfile.TemporaryDirectory() as raw:
        marker = Path(raw) / "alive.txt"
        log = Path(raw) / "out.txt"
        script = (
            "import os,time,pathlib,sys;"
            f"p=pathlib.Path({str(marker)!r});"
            "p.write_text(str(os.getpid()));"
            "time.sleep(2)"
        )
        with log.open("wb") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        time.sleep(0.4)
        assert proc.poll() is None
        assert marker.is_file()
        os.kill(proc.pid, 15)
        proc.wait(timeout=5)


def test_other_held_runs_lists_foreign_lock() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        a = root / "mei-1.0-51m/run-a/checkpoints/cpt"
        fd = acquire_run_lock(a, {"recipe": "x"})
        assert lock_is_held(a)
        with patch.object(run_lock, "TRAIN_RUNS", root):
            held = other_held_runs()
        assert len(held) == 1
        assert Path(held[0]["run_dir"]) == a
        os.close(fd)


def test_custom_exposure_is_forwarded_to_detached_trainer() -> None:
    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw) / "run-900m"
        with patch.object(service.subprocess, "Popen", return_value=SimpleNamespace(pid=42)) as popen, patch.object(
            service.os, "getpgid", return_value=42
        ):
            service.spawn_trainer(
                python=sys.executable,
                rung="900m",
                run_dir=run_dir,
                resume=Path("/tmp/parent-state.npz"),
                corpus_dir=Path("/tmp/corpus-900m"),
                target_tokens=900_000_000,
                extra=[],
            )
        command = popen.call_args.args[0]
        assert command[command.index("--rung") + 1] == "900m"
        assert command[command.index("--target-tokens") + 1] == "900000000"
        assert command[command.index("--corpus-dir") + 1] == "/tmp/corpus-900m"
        assert other_held_runs() is not None


def main() -> int:
    test_dual_lock_refuses()
    test_stale_lock_recovers_after_release()
    test_pause_marker_and_heartbeat()
    test_detached_session_survives_parent()
    test_other_held_runs_lists_foreign_lock()
    test_custom_exposure_is_forwarded_to_detached_trainer()
    print({"ok": True, "tests": 6})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
