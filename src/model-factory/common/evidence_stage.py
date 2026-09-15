"""Immutable evidence-stage adapter over shared source capture and run locks."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from common import source_capture
from common.run_lock import acquire_run_lock, write_heartbeat

ROOT = Path(__file__).resolve().parents[3]
_ACTIVE = None


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def freeze(config, out):
    path = Path(out).resolve()
    if _ACTIVE is None or _ACTIVE != (path, source_capture.json_digest(config)):
        raise RuntimeError("worker must run inside the control-plane governed stage")
    return path


@contextmanager
def stage(config, *, track="diagnostic", pipeline_id=None):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("nested workers must use the control plane")
    out = Path(config["out"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if track not in {"diagnostic", "research_candidate"}:
        raise ValueError("this adapter cannot create canonical training/release lineage")
    fd = acquire_run_lock(out, {"action": config["action"], "track": track})
    stopped = threading.Event()
    started = time.time()
    previous = {}
    heartbeat = None
    terminal = {"ok": False, "process_complete": False, "release_eligible": False, "state": "failed"}
    try:
        write_json(out / "config.json", config)
        if pipeline_id:
            registry=ROOT/'src/model-factory/contracts/PIPELINES.json'
            selected=next(p for p in json.loads(registry.read_text())['pipelines'] if p['pipeline_id']==pipeline_id)
            if selected['status']!='current':raise ValueError('pipeline is not current')
            write_json(out/'PIPELINE.lock.json',{'pipeline':selected,'registry_sha256':sha256(registry),
                'config_sha256':source_capture.json_digest(config),'automatic_promotion':False})
        sources = source_capture.manifest(ROOT)
        export = None
        if not (ROOT / ".git").exists():
            patch = ROOT / "TRANSFER-dirty.patch"
            export = {"git_revision": (ROOT / "TRANSFER-REVISION.txt").read_text().strip(),
                      "dirty_patch": str(patch), "dirty_patch_sha256": sha256(patch)}
        binding = source_capture.capture(ROOT, out / "source", sources, [config["action"]], exported_checkout=export)
        errors = source_capture.verify(sources, binding)
        if errors:
            raise ValueError(str(errors))
        write_json(out / "source-manifest.json", sources)
        write_json(out / "run-binding.json", {"schema": "mei-research-stage-v1", "track": track,
                   "pipeline_id": pipeline_id,
                   "parent": None, "release_eligible": False, "source": binding,
                   "config_sha256": source_capture.json_digest(config), "host": socket.gethostname(),
                   "python": sys.version, "started_unix": started})
        def beat():
            while not stopped.is_set():
                write_heartbeat(out / "stage-heartbeat.json", {"action": config["action"], "state": "running"})
                stopped.wait(10)
        heartbeat = threading.Thread(target=beat, daemon=True)
        heartbeat.start()
        def interrupted(sig, _frame):
            # Save only at an optimizer boundary, never with a partly consumed batch.
            (out / "STOP_REQUESTED").touch(exist_ok=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        _ACTIVE = (out, source_capture.json_digest(config))
        yield out
        if source_capture.manifest(ROOT)["manifest_sha256"] != sources["manifest_sha256"]:
            raise ValueError("source drift during stage")
        if source_capture.verify(sources, binding):
            raise ValueError("source archive changed during stage")
        report = json.loads((out / "receipt.json").read_text())
        if not report.get("ok", False):
            raise ValueError("stage quality receipt failed")
        terminal = {"ok": True, "process_complete": True, "release_eligible": False,
                    "state": "completed", "receipt_sha256": sha256(out / "receipt.json")}
    except BaseException as exc:
        terminal = {"ok": False, "process_complete": False, "release_eligible": False,
                    "state": "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                    "error_type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        _ACTIVE = None
        stopped.set()
        if heartbeat:
            heartbeat.join(timeout=2)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        terminal.update(ended_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(out / "terminal.json", terminal)
        write_heartbeat(out / "stage-heartbeat.json", {"action": config["action"], "state": terminal["state"]})
        os.close(fd)
