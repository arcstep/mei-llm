#!/usr/bin/env python3
"""Register a task-local SFT pack as accepted or invalid. Does not copy eval banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobs.registry import load_jobs, save_jobs  # noqa: E402
from repo_paths import ROOT, resolve_rel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="repo-relative jsonl path")
    ap.add_argument("--task", required=True)
    ap.add_argument("--state", default="draft", choices=["accepted", "draft", "invalid", "archive", "smoke", "engineering_smoke", "candidate"])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    pack = resolve_rel(args.pack, root=root)
    rel = pack.relative_to(root).as_posix()
    if "notebook/evaluation/banks" in rel or "/evaluation/banks/" in rel:
        print("refuse to register eval bank as SFT pack", file=sys.stderr)
        return 5
    blob = pack.read_text(encoding="utf-8") if pack.is_file() else ""
    if "<routes>" in blob or "gold_route_id" in blob:
        print("refuse to register Route-ID / <routes> pack as v2 SFT", file=sys.stderr)
        return 6
    if args.state == "accepted" and ("smoke" in pack.name or "isolation-smoke" in pack.name or "engineering_smoke" in pack.name):
        print("refuse to promote smoke pack to accepted", file=sys.stderr)
        return 7
    index = load_jobs(root)
    packs = [row for row in (index.get("sft_packs") or []) if row.get("path") != rel]
    entry = {
        "id": pack.stem,
        "path": rel,
        "task_id": args.task,
        "state": args.state,
    }
    packs.append(entry)
    packs.sort(key=lambda r: str(r.get("path") or ""))
    index["sft_packs"] = packs
    save_jobs(index, root=root)
    print(json.dumps({"ok": True, "pack": entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
