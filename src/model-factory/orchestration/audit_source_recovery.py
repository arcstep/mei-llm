#!/usr/bin/env python3
"""Audit whether a historical run's exact source bytes remain recoverable.

Run artifacts keep SHA-256 manifests, but a hash alone is not a source bundle.
This tool searches the current tree, HEAD, the index, and every local Git blob.
It reports missing bytes honestly and never rewrites the historical run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from common.paths import ROOT


SKIP_DIRECTORIES = {
    ".git",
    ".local",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def current_inventory(root: Path) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in SKIP_DIRECTORIES)
        base = Path(directory)
        for name in sorted(files):
            candidate = base / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                digest = sha256_file(candidate)
            except OSError:
                continue
            by_hash[digest].append(candidate.relative_to(root).as_posix())
    return dict(by_hash)


def recovery_archive_inventory(root: Path) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    recovery_root = root / ".local/recovery"
    if not recovery_root.is_dir():
        return {}
    for archive_path in sorted(recovery_root.glob("*.tar.gz")):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                digest = sha256_bytes(handle.read())
                by_hash[digest].append(
                    f"{archive_path.relative_to(root).as_posix()}!/{member.name}"
                )
    return dict(by_hash)


def _git_bytes(root: Path, spec: str) -> bytes | None:
    result = subprocess.run(
        ["git", "cat-file", "blob", spec],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def named_git_inventory(root: Path, manifest_paths: Iterable[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    head: dict[str, list[str]] = defaultdict(list)
    index: dict[str, list[str]] = defaultdict(list)
    for relative in sorted(set(manifest_paths)):
        head_bytes = _git_bytes(root, f"HEAD:{relative}")
        if head_bytes is not None:
            head[sha256_bytes(head_bytes)].append(relative)
        index_bytes = _git_bytes(root, f":{relative}")
        if index_bytes is not None:
            index[sha256_bytes(index_bytes)].append(relative)
    return dict(head), dict(index)


def all_git_blob_inventory(root: Path) -> dict[str, list[str]]:
    listing = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    object_ids = [
        line.split()[0]
        for line in listing.stdout.splitlines()
        if line.endswith(" blob")
    ]
    by_hash: dict[str, list[str]] = defaultdict(list)
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for object_id in object_ids:
            process.stdin.write((object_id + "\n").encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii").strip().split()
            if len(header) != 3 or header[1] != "blob":
                raise RuntimeError(f"unexpected git cat-file header: {header!r}")
            size = int(header[2])
            payload = process.stdout.read(size)
            if process.stdout.read(1) != b"\n":
                raise RuntimeError("git cat-file batch framing error")
            by_hash[sha256_bytes(payload)].append(object_id)
    finally:
        process.stdin.close()
        process.wait()
    if process.returncode:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"git cat-file failed: {stderr}")
    return dict(by_hash)


def classify_expected(
    manifest: Mapping[str, str],
    *,
    current: Mapping[str, list[str]],
    archives: Mapping[str, list[str]],
    head: Mapping[str, list[str]],
    index: Mapping[str, list[str]],
    git_objects: Mapping[str, list[str]],
) -> dict:
    rows = []
    methods: Counter[str] = Counter()
    for historical_path, expected_sha256 in sorted(manifest.items()):
        matches: list[str]
        if expected_sha256 in current:
            method, matches = "current_tree", list(current[expected_sha256])
        elif expected_sha256 in archives:
            method, matches = "recovery_archive", list(archives[expected_sha256])
        elif expected_sha256 in head:
            method, matches = "git_head", list(head[expected_sha256])
        elif expected_sha256 in index:
            method, matches = "git_index", list(index[expected_sha256])
        elif expected_sha256 in git_objects:
            method, matches = "git_object", list(git_objects[expected_sha256])
        else:
            method, matches = "unavailable", []
        methods[method] += 1
        rows.append(
            {
                "historical_path": historical_path,
                "sha256": expected_sha256,
                "recovery_method": method,
                "matches": matches[:8],
            }
        )
    unavailable = [row for row in rows if row["recovery_method"] == "unavailable"]
    return {
        "manifest_entries": len(rows),
        "exact_recoverable": len(rows) - len(unavailable),
        "exact_unavailable": len(unavailable),
        "recovery_method_counts": dict(sorted(methods.items())),
        "unavailable": unavailable,
        "entries": rows,
    }


def load_plan_manifest(plan_path: Path) -> dict[str, str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = ((plan.get("immutable") or {}).get("source_manifest") or plan.get("source_manifest"))
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"plan has no source manifest: {plan_path}")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in manifest.items()):
        raise RuntimeError("source manifest must map paths to SHA-256 strings")
    return dict(manifest)


def audit(plan_path: Path, *, root: Path = ROOT) -> dict:
    manifest = load_plan_manifest(plan_path)
    current = current_inventory(root)
    archives = recovery_archive_inventory(root)
    head, index = named_git_inventory(root, manifest)
    git_objects = all_git_blob_inventory(root)
    result = classify_expected(
        manifest,
        current=current,
        archives=archives,
        head=head,
        index=index,
        git_objects=git_objects,
    )
    return {
        "schema": "mei-model-factory-source-recovery-audit-v1",
        "plan": plan_path.resolve().relative_to(root.resolve()).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "source_manifest_sha256": canonical_sha256(manifest),
        **result,
        "claim": (
            "recoverable means exact bytes exist locally; unavailable means the run "
            "retains only the recorded hash and is not exactly source-reproducible"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = audit(args.plan)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.out)
    if not args.quiet:
        print(payload, end="")
    return 0 if report["exact_unavailable"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
