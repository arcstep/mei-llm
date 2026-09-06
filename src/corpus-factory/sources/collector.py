#!/usr/bin/env python3
"""Generic registry-driven downloader for natural-corpus sourcing v2.

Two-phase authorization: resolve the batch offline and print an authorization
token; no network call is made unless the token matches byte-for-byte. Every
batch produces a download-manifest receipt; hash drift against pinned
expectations fails closed without leaving partial artifacts on disk.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "mei-51m-download-manifest-v1"


class CollectorError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_token(source_id: str, files: list[dict[str, Any]]) -> str:
    listing = canonical([row["filename"] for row in files])
    return f"{source_id}:{hashlib.sha256(listing).hexdigest()[:16]}"


def list_hf_files(entry: dict[str, Any], *, cache_dir: Path | None) -> list[str]:
    """List candidate files for an hf_api source; cached when possible."""
    repository = entry.get("repository")
    if not repository:
        raise CollectorError(
            f"source {entry['source_id']!r}: hf_api requires a pinned repository"
        )
    cache_path = cache_dir / f"{entry['source_id']}.files.json" if cache_dir else None
    if cache_path is not None and cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))["files"]
        except (json.JSONDecodeError, KeyError) as error:
            raise CollectorError(f"corrupt collector cache: {cache_path}") from error
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise CollectorError(
            "listing hf_api sources requires huggingface_hub"
        ) from error
    files = list(
        HfApi().list_repo_files(
            repo_id=repository, repo_type=entry["manifest_source"].get("repo_type", "dataset")
        )
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"files": files}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return files


def resolve_batch(
    entry: dict[str, Any],
    *,
    subset: str | None = None,
    filenames: list[str] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve the download batch for a registry entry without downloading."""
    source_id = entry["source_id"]
    manifest_source = entry["manifest_source"]
    kind = manifest_source["type"]
    files: list[dict[str, Any]] = []
    if kind == "static_manifest":
        manifest_path = Path(__file__).with_name(
            "manifests"
        ) / manifest_source["manifest"]
        if not manifest_path.is_file():
            raise CollectorError(f"missing static manifest: {manifest_path}")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value.get("source_id") != source_id:
            raise CollectorError(
                f"manifest source_id {value.get('source_id')!r} != {source_id!r}"
            )
        files = value["files"]
    elif kind == "direct_urls":
        urls = entry.get("urls") or []
        if filenames:
            by_name = {row["filename"]: row for row in urls}
            files = [by_name[name] for name in filenames if name in by_name]
        else:
            files = urls
        if not files:
            raise CollectorError(f"source {source_id!r}: direct_urls manifest is empty")
    elif kind == "hf_api":
        if filenames:
            files = [{"filename": name} for name in filenames]
        else:
            patterns = manifest_source.get("patterns") or []
            listing = list_hf_files(entry, cache_dir=cache_dir)
            matched = [
                name
                for name in listing
                if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
            ]
            if subset:
                matched = [
                    name
                    for name in matched
                    if fnmatch.fnmatch(name, subset)
                ]
            files = [{"filename": name} for name in sorted(set(matched))]
    else:
        raise CollectorError(f"unsupported manifest_source type: {kind}")
    if not files:
        raise CollectorError(f"source {source_id!r}: batch resolved to zero files")
    return {
        "source_id": source_id,
        "files": files,
        "batch_token": batch_token(source_id, files),
    }


def _hardlink_or_copy(cache_dir: Path, digest: str, target: Path, source: Path) -> None:
    """Populate the content-addressed cache and link/copy into place."""
    cached = cache_dir / digest[:2] / digest
    if cached.is_file() and sha256_file(cached) == digest:
        source.unlink(missing_ok=True)
    else:
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            cached.unlink()
        os.rename(source, cached)
    try:
        os.link(cached, target)
    except OSError:
        shutil.copy2(cached, target)


def _fetch_file(
    entry: dict[str, Any],
    row: dict[str, Any],
    destination: Path,
) -> None:
    manifest_source = entry["manifest_source"]
    if manifest_source["type"] == "hf_api":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise CollectorError("hf_api download requires huggingface_hub") from error
        downloaded = Path(
            hf_hub_download(
                repo_id=entry["repository"],
                repo_type=manifest_source.get("repo_type", "dataset"),
                filename=row["filename"],
                local_dir=str(destination.parent),
            )
        )
        if downloaded.resolve() != destination.resolve():
            shutil.move(str(downloaded), destination)
        return
    url = row.get("url")
    if not url:
        raise CollectorError(f"no url for file: {row['filename']}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_batch(
    batch: dict[str, Any],
    out: Path,
    *,
    entry: dict[str, Any],
    authorization: str,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Download one resolved batch; refuses unless the token matches exactly."""
    if authorization != batch["batch_token"]:
        raise CollectorError(
            "download refused; pass --authorize-download "
            f"{batch['batch_token']!r} for this exact batch"
        )
    if out.exists():
        raise FileExistsError(f"refusing to overwrite download output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent.resolve())
    )
    try:
        files = []
        for row in batch["files"]:
            filename = row["filename"]
            safe_name = filename.replace("/", "__")
            destination = temporary / safe_name
            _fetch_file(entry, row, destination)
            digest = sha256_file(destination)
            expected = row.get("sha256")
            if expected and digest != expected:
                raise CollectorError(
                    f"hash drift for {filename}: expected {expected}, got {digest}"
                )
            if cache_dir is not None:
                _hardlink_or_copy(cache_dir, digest, destination, destination)
            files.append(
                {
                    "filename": filename,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                    "hash_pinned": bool(expected),
                }
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source_id": batch["source_id"],
            "batch_token": batch["batch_token"],
            "files": files,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "distribution_clearance_asserted": False,
        }
        (temporary / "download-manifest.json").write_bytes(canonical(manifest))
        temporary.replace(out)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
