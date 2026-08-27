"""Lightweight required-key checks. jsonschema is not a repo dependency."""

from __future__ import annotations

JOB_REQUIRED = ("job_id", "topic", "kind", "status", "work_dir")
RECEIPT_REQUIRED = ("job_id", "topic", "status", "publish_policy")
CORPUS_REQUIRED = ("id", "role", "state", "path")

JOB_STATUSES = {"running", "draft", "accepted", "failed", "archived"}
RECEIPT_STATUSES = {"draft", "accepted", "failed", "archived"}
CORPUS_STATES = {
    "accepted",
    "draft",
    "fail_closed",
    "job_complete",
    "engineering_contrast",
    "archive",
    "invalid",
    "evaluation_only",
    "empty",
    "probe",
    "smoke",
    "engineering_smoke",
    "candidate",
}


def _require(payload: dict, keys: tuple[str, ...], label: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"{label} missing {missing}")


def validate_job(payload: dict) -> dict:
    _require(payload, JOB_REQUIRED, "job")
    if payload["status"] not in JOB_STATUSES:
        raise ValueError(f"invalid job status {payload['status']}")
    return payload


def validate_receipt(payload: dict) -> dict:
    _require(payload, RECEIPT_REQUIRED, "receipt")
    if payload["status"] not in RECEIPT_STATUSES:
        raise ValueError(f"invalid receipt status {payload['status']}")
    return payload


def validate_corpus_entry(payload: dict) -> dict:
    _require(payload, CORPUS_REQUIRED, "corpus-entry")
    if payload["state"] not in CORPUS_STATES:
        raise ValueError(f"invalid corpus state {payload['state']}")
    return payload
