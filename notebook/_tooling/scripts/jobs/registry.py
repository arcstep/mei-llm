"""Load and write jobs/index.json and corpora/index.json."""

from __future__ import annotations

import json
from pathlib import Path

from repo_paths import ROOT, JOBS_INDEX_OVERLAY, is_mei_llm_root, jobs_home, jobs_rel_prefix

from jobs.schema import validate_corpus_entry, validate_job, validate_receipt


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jobs_index_path(root: Path | None = None) -> Path:
    return jobs_home(root) / "index.json"


def corpora_index_path(root: Path | None = None) -> Path:
    return (root or ROOT) / "corpora" / "index.json"


def empty_jobs_index() -> dict:
    return {"version": 1, "topics": {}, "jobs": [], "sft_packs": []}


def empty_corpora_index() -> dict:
    return {"version": 1, "note": "Process state lives in jobs/. This file is the publish catalog.", "corpora": []}


def load_jobs(root: Path | None = None) -> dict:
    path = jobs_index_path(root)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if is_mei_llm_root(root or ROOT) and JOBS_INDEX_OVERLAY.is_file():
        return json.loads(JOBS_INDEX_OVERLAY.read_text(encoding="utf-8"))
    return empty_jobs_index()


def load_corpora(root: Path | None = None) -> dict:
    path = corpora_index_path(root)
    if not path.is_file():
        return empty_corpora_index()
    return json.loads(path.read_text(encoding="utf-8"))


def save_jobs(index: dict, *, root: Path | None = None) -> Path:
    path = jobs_index_path(root)
    dump_json(path, index)
    return path


def save_corpora(index: dict, *, root: Path | None = None) -> Path:
    path = corpora_index_path(root)
    dump_json(path, index)
    return path


def upsert_job(card: dict, *, root: Path | None = None) -> dict:
    validate_job(card)
    index = load_jobs(root)
    jobs = [row for row in (index.get("jobs") or []) if row.get("job_id") != card["job_id"]]
    jobs.append(card)
    index["jobs"] = jobs
    topic = card["topic"]
    topics = index.setdefault("topics", {})
    prefix = jobs_rel_prefix(root)
    topic_paths = {
        "inbox": f"{prefix}/{topic}/inbox",
        "jobs_glob": f"{prefix}/{topic}/jobs/*",
        "outbox": f"{prefix}/{topic}/outbox",
    }
    if topic in {"colloquial-cpt", "colloquial"}:
        topics.setdefault(topic, topic_paths)
    else:
        topics[topic] = topic_paths
    save_jobs(index, root=root)
    return index


def upsert_receipt_pointer(receipt: dict, *, root: Path | None = None) -> dict:
    validate_receipt(receipt)
    outbox = {"draft": "draft", "accepted": "accepted", "archived": "archive", "failed": "archive"}[
        receipt["status"]
    ]
    index = load_jobs(root)
    prefix = jobs_rel_prefix(root)
    for row in index.get("jobs") or []:
        if row.get("job_id") == receipt["job_id"]:
            row["status"] = receipt["status"]
            row["receipt"] = (
                f"{prefix}/{receipt['topic']}/outbox/{outbox}/{receipt['job_id']}/receipt.json"
            )
            if receipt.get("publish_corpus_id"):
                row["publish_corpus_id"] = receipt["publish_corpus_id"]
            if receipt.get("publish_path"):
                row["publish_path"] = receipt["publish_path"]
            break
    save_jobs(index, root=root)
    return index


def upsert_corpus(entry: dict, *, root: Path | None = None) -> dict:
    validate_corpus_entry(entry)
    index = load_corpora(root)
    rows = [row for row in (index.get("corpora") or []) if row.get("id") != entry["id"]]
    rows.append(entry)
    rows.sort(key=lambda r: str(r.get("id") or ""))
    index["corpora"] = rows
    save_corpora(index, root=root)
    return index
