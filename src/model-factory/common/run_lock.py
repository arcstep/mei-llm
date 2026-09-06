"""OS file locks for formal training runs."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from pathlib import Path

from common.paths import TRAIN_RUNS


def lock_file(run_dir: Path) -> Path:
    return Path(run_dir) / "run.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A sandbox may forbid signalling a foreign process while the OS lock
        # and fresh heartbeat still prove that it exists.
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(run_dir: Path, meta: dict) -> int:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = lock_file(run_dir)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        existing = {}
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
        raise RuntimeError(f"run lock held: {path} meta={existing}") from exc
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "start_unix": time.time(),
        **meta,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.write(fd, encoded)
    os.fsync(fd)
    return fd


def lock_is_held(run_dir: Path) -> dict | None:
    path = lock_file(run_dir)
    if not path.is_file():
        return None
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"held": True, "path": str(path)}
        finally:
            os.close(fd)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    return None


def other_held_runs(except_dir: Path | None = None) -> list[dict]:
    held: list[dict] = []
    except_res = Path(except_dir).resolve() if except_dir else None
    if not TRAIN_RUNS.is_dir():
        return held
    for lock in TRAIN_RUNS.rglob("run.lock"):
        run_dir = lock.parent
        if except_res and run_dir.resolve() == except_res:
            continue
        info = lock_is_held(run_dir)
        if info:
            row = dict(info)
            row["run_dir"] = str(run_dir)
            held.append(row)
    return held


def write_heartbeat(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    body = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "unix": time.time(),
        **payload,
    }
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_lock_meta(run_dir: Path) -> dict:
    path = lock_file(run_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def pid_alive_from_meta(meta: dict) -> bool:
    return _pid_alive(int(meta.get("pid") or 0))
