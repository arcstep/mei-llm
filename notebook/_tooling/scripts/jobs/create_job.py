#!/usr/bin/env python3
"""Create a jobs/<topic>/jobs/<id> card and workdir. Does not write corpora/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobs.paths import inbox_dir, job_card, job_dir, job_scratch, job_work, rel_to_root, topic_dir  # noqa: E402
from jobs.registry import upsert_job  # noqa: E402
from jobs.schema import validate_job  # noqa: E402
from repo_paths import ROOT  # noqa: E402


def ensure_topic(topic: str, *, root: Path) -> None:
    base = topic_dir(topic, root=root)
    inbox_dir(topic, root=root).mkdir(parents=True, exist_ok=True)
    for name in ("draft", "accepted", "archive"):
        (base / "outbox" / name).mkdir(parents=True, exist_ok=True)
    (base / "jobs").mkdir(parents=True, exist_ok=True)
    readme = base / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# {topic}\n\nJob topic. Inbox is input; jobs/ is executable work; outbox is receipts.\n",
            encoding="utf-8",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--job", required=True, help="YYMMDD-NN-slug")
    ap.add_argument("--kind", default="produce")
    ap.add_argument("--task", default=None)
    ap.add_argument("--intent", default="")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    ensure_topic(args.topic, root=root)
    work = job_work(args.topic, args.job, root=root)
    work.mkdir(parents=True, exist_ok=True)
    job_scratch(args.topic, args.job, root=root).mkdir(parents=True, exist_ok=True)
    for name in ("raw", "state", "reviews", "tokens"):
        (work / name).mkdir(exist_ok=True)
    card = {
        "job_id": args.job,
        "topic": args.topic,
        "kind": args.kind,
        "status": "running",
        "task_id": args.task,
        "work_dir": rel_to_root(work, root=root),
        "receipt": None,
        "publish_corpus_id": None,
        "publish_path": None,
        "parents": [],
        "entry": rel_to_root(job_card(args.topic, args.job, root=root), root=root),
        "intent": args.intent,
    }
    validate_job(card)
    dump_path = job_card(args.topic, args.job, root=root)
    dump_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = job_dir(args.topic, args.job, root=root) / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# {args.job}\n\n- topic: `{args.topic}`\n- work: `{card['work_dir']}`\n- intent: {args.intent or '(none)'}\n",
            encoding="utf-8",
        )
    upsert_job(card, root=root)
    print(json.dumps({"ok": True, "job": card}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
