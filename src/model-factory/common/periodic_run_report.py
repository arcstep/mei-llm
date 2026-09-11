from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class PeriodicRunReport:
    def __init__(self, directory: Path, interval: float = 7200):
        self.directory = directory
        self.interval = interval
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def snapshot(self, reason: str) -> dict:
        evidence = {}
        for name in ("heartbeat.json", "training-receipt.json", "receipt.json", "failure.json"):
            path = self.directory / name
            if path.is_file():
                try:
                    evidence[name] = json.loads(path.read_text())
                except (OSError, ValueError):
                    evidence[name] = {"status": "unreadable_retry_next_report"}
        heartbeat = evidence.get("heartbeat.json", {})
        return {"schema": "mei-periodic-run-report-v1", "reason": reason,
                "reported_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self.started,
                "heartbeat_age_seconds": time.time() - heartbeat["unix"] if heartbeat.get("unix") else None,
                "delivery": "local_report_and_process_stdout_no_chat_wakeup", "evidence": evidence}

    def emit(self, reason: str) -> None:
        report = self.snapshot(reason)
        with (self.directory / "progress-reports.jsonl").open("a") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        print("CPT_PROGRESS_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.emit("scheduled_2h")

    def start(self) -> None:
        self.emit("started")
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.emit("process_finished")
