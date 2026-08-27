#!/usr/bin/env python3
"""Close a job by writing outbox/<state>/<job-id>/receipt.json. Never overwrites corpora/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobs.paths import job_card, job_work, receipt_path, rel_to_root  # noqa: E402
from jobs.registry import upsert_job, upsert_receipt_pointer  # noqa: E402
from jobs.schema import validate_receipt  # noqa: E402
from repo_paths import ROOT  # noqa: E402

OUTBOX_MAP = {"draft": "draft", "accepted": "accepted", "archived": "archive", "failed": "archive"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--status", required=True, choices=sorted(OUTBOX_MAP))
    ap.add_argument("--publish-id", default=None)
    ap.add_argument("--publish-path", default=None)
    ap.add_argument("--policy", default="promote_only", choices=["promote_only", "in_place_register", "none"])
    ap.add_argument("--unique", type=int, default=None)
    ap.add_argument("--spend-cny", type=float, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--gates", default="{}", help="JSON object of gate booleans")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    outbox_state = OUTBOX_MAP[args.status]
    work = job_work(args.topic, args.job, root=root)
    receipt = {
        "job_id": args.job,
        "topic": args.topic,
        "status": args.status if args.status != "archived" else "archived",
        "publish_policy": args.policy,
        "work_dir": rel_to_root(work, root=root) if work.exists() else None,
        "publish_corpus_id": args.publish_id,
        "publish_path": args.publish_path,
        "gates": json.loads(args.gates),
        "unique_train_tokens": args.unique,
        "spend_cny": args.spend_cny,
        "note": args.note,
    }
    validate_receipt(receipt)
    path = receipt_path(args.topic, args.job, outbox_state, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card_path = job_card(args.topic, args.job, root=root)
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["status"] = "draft" if args.status == "draft" else args.status
        if args.status == "failed":
            card["status"] = "failed"
        card["receipt"] = rel_to_root(path, root=root)
        card["publish_corpus_id"] = args.publish_id
        card["publish_path"] = args.publish_path
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        upsert_job(card, root=root)
    upsert_receipt_pointer(receipt, root=root)
    print(json.dumps({"ok": True, "receipt": rel_to_root(path, root=root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
