#!/usr/bin/env python3
"""Promote a job workdir into corpus/<id> as a frozen serving pack.

Never overwrites an existing accepted corpus. Never copies GB blobs: token bins
are hardlinked. Process files (raw/reviews/idx) stay in the notebook workdir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobs.paths import job_work, rel_to_root  # noqa: E402
from jobs.registry import upsert_corpus  # noqa: E402
from repo_paths import ROOT, resolve_rel  # noqa: E402

SERVING_META = (
    "README.md",
    "RELEASE.json",
    "manifest.json",
    "hashes.json",
    "SOURCES.md",
    "source-license.json",
    "mix.json",
    "schedule.json",
)


def rewrite_rel(text: str, old_prefix: str, new_prefix: str) -> str:
    return text.replace(old_prefix.rstrip("/") + "/", new_prefix.rstrip("/") + "/")


def rewrite_json_file(path: Path, old_prefix: str, new_prefix: str) -> None:
    raw = path.read_text(encoding="utf-8")
    updated = rewrite_rel(raw, old_prefix, new_prefix)
    if updated != raw:
        path.write_text(updated, encoding="utf-8")


def hardlink_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError as exc:
        raise RuntimeError(f"refuse to copy {src}; hardlink failed: {exc}") from exc


def promote_serving(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in SERVING_META:
        item = src / name
        if item.is_file():
            shutil.copy2(item, dest / name)
    token_src = src / "tokens"
    if token_src.is_dir():
        for bin_path in sorted(token_src.glob("*.bin")):
            hardlink_file(bin_path, dest / "tokens" / bin_path.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--publish-id", required=True)
    ap.add_argument("--in-place", action="store_true", help="Register existing corpus/<id>; do not copy")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--role", default="cpt-colloquial")
    ap.add_argument("--state", default="draft")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    dest = root / "corpus" / args.publish_id
    if args.in_place:
        work = dest
        old_prefix = f"corpus/{args.publish_id}"
    else:
        if args.work_dir is not None:
            work = resolve_rel(args.work_dir, root=root)
        elif args.topic and args.job:
            work = job_work(args.topic, args.job, root=root)
        else:
            print("--work-dir or --topic/--job required unless --in-place", file=sys.stderr)
            return 2
        old_prefix = rel_to_root(work, root=root)
    new_prefix = f"corpus/{args.publish_id}"
    if dest.exists() and not args.in_place:
        print(f"refuse to overwrite existing corpus {dest}", file=sys.stderr)
        return 5
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "from": old_prefix, "to": new_prefix}, indent=2))
        return 0
    if not args.in_place:
        promote_serving(work, dest)
        for name in SERVING_META:
            meta = dest / name
            if meta.is_file() and meta.suffix == ".json":
                rewrite_json_file(meta, old_prefix, new_prefix)
    release = dest / "RELEASE.json"
    unique = None
    generators = []
    formal = False
    if release.is_file():
        rel = json.loads(release.read_text(encoding="utf-8"))
        unique = rel.get("n_unique_train_tokens") or rel.get("unique_train_tokens")
        if rel.get("generator"):
            generators = [str(rel["generator"])]
        formal = bool(rel.get("formal_cpt_eligible"))
    entry = {
        "id": args.publish_id,
        "role": args.role,
        "state": args.state,
        "path": new_prefix,
        "job_id": args.job,
        "release": f"{new_prefix}/RELEASE.json" if release.is_file() else None,
        "generators": generators,
        "unique_train_tokens": unique,
        "formal_eligible": formal,
        "subscribers": [],
    }
    upsert_corpus(entry, root=root)
    print(json.dumps({"ok": True, "corpus": entry}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
