"""Job directory helpers. Canonical roots stay in repo_paths."""

from __future__ import annotations

from pathlib import Path

from repo_paths import ROOT, job_topic_dir, resolve_rel


def topic_dir(topic: str, *, root: Path | None = None) -> Path:
    return job_topic_dir(topic, root=root)


def job_dir(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return topic_dir(topic, root=root) / "jobs" / job_id


def job_work(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return job_dir(topic, job_id, root=root) / "work"


def job_scratch(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return job_dir(topic, job_id, root=root) / "_"


def job_card(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return job_dir(topic, job_id, root=root) / "job.json"


def job_outbox(topic: str, job_id: str, state: str, *, root: Path | None = None) -> Path:
    if state not in {"draft", "accepted", "archive"}:
        raise ValueError(f"outbox state must be draft|accepted|archive, got {state}")
    return topic_dir(topic, root=root) / "outbox" / state / job_id


def receipt_path(topic: str, job_id: str, state: str, *, root: Path | None = None) -> Path:
    return job_outbox(topic, job_id, state, root=root) / "receipt.json"


def inbox_dir(topic: str, *, root: Path | None = None) -> Path:
    return topic_dir(topic, root=root) / "inbox"


def rel_to_root(path: Path, *, root: Path | None = None) -> str:
    return resolve_rel(path, root=root).relative_to((root or ROOT).resolve()).as_posix()
