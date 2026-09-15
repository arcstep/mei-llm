from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


SOURCE_ROOTS = (
    "src/architecture/mei-1.2-51m", "src/model-factory", "src/corpus-factory",
    "src/mei_llm", "src/platform",
)
SUFFIXES = {".py", ".json", ".metal", ".rs", ".js", ".mjs", ".ts", ".tsx",
            ".toml", ".yaml", ".yml", ".lock", ".h", ".c", ".cpp", ".sh"}
EXCLUDES = {"__pycache__", "node_modules", "target", "packages", "runs", ".git", ".venv"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def manifest(root: Path) -> dict:
    paths = {path for folder in SOURCE_ROOTS for path in (root / folder).rglob("*")
             if path.is_file() and path.suffix.lower() in SUFFIXES
             and not EXCLUDES.intersection(path.relative_to(root).parts)}
    paths.update(root / name for name in ("pyproject.toml", "uv.lock", "requirements.txt", "AGENTS.md")
                 if (root / name).is_file())
    files = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source escapes repository: {path}")
        files[path.relative_to(root).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    if not files:
        raise ValueError("empty source closure")
    return {"schema_version": 2, "capture_mode": "launch", "files": files,
            "manifest_sha256": json_digest(files)}


def capture(root: Path, directory: Path, sources: dict, stages: list[str], *, exported_checkout: dict | None = None) -> dict:
    if sources.get("schema_version") != 2 or json_digest(sources["files"]) != sources["manifest_sha256"]:
        raise ValueError("invalid source manifest")
    directory.mkdir(parents=True, exist_ok=False)
    archive_path = directory / "source.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        for name, expected in sources["files"].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"source escapes repository: {name}")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected["sha256"] or len(payload) != expected["bytes"]:
                raise ValueError(f"source changed during capture: {name}")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(payload))
    patch_path = directory / "dirty.patch"
    if exported_checkout is None:
        patch_path.write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root))
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    else:
        revision = exported_checkout["git_revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("invalid exported revision")
        origin_patch = Path(exported_checkout["dirty_patch"])
        if digest(origin_patch) != exported_checkout["dirty_patch_sha256"]:
            raise ValueError("exported patch hash changed")
        patch_path.write_bytes(origin_patch.read_bytes())
    closure = {"schema": "mei-source-capture-v2", "git_revision": revision,
               "manifest_sha256": sources["manifest_sha256"],
               "source_archive": str(archive_path.resolve()), "source_archive_sha256": digest(archive_path),
               "dirty_patch": str(patch_path.resolve()), "dirty_patch_sha256": digest(patch_path),
               "phase_closures": {stage: {"scope": "full_active_source_superset",
                                          "manifest_sha256": sources["manifest_sha256"]} for stage in stages}}
    (directory / "capture.json").write_text(json.dumps(closure, indent=2) + "\n")
    return closure


def verify(sources: dict, binding: dict) -> list[str]:
    errors = []
    expected = sources.get("files", {})
    manifest_hash = json_digest(expected)
    if not expected or sources.get("manifest_sha256") != manifest_hash or binding.get("manifest_sha256") != manifest_hash:
        errors.append("source manifest binding mismatch")
    for key in ("source_archive", "dirty_patch"):
        path = Path(binding.get(key, ""))
        if not path.is_file() or digest(path) != binding.get(f"{key}_sha256"):
            errors.append(f"{key} missing or changed")
    if errors:
        return errors
    try:
        with tarfile.open(binding["source_archive"], "r:gz") as archive:
            names = set()
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in names:
                    errors.append(f"unexpected source archive member: {member.name}")
                    continue
                names.add(member.name)
                with archive.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected[member.name]["sha256"] or member.size != expected[member.name]["bytes"]:
                    errors.append(f"source archive bytes mismatch: {member.name}")
            if names != set(expected):
                errors.append("source archive closure incomplete")
    except (OSError, tarfile.TarError, EOFError) as error:
        errors.append(f"source archive unreadable: {error}")
    return errors
