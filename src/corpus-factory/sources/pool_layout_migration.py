"""Same-filesystem pool grouping with reversible moves and preserved evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from profiling import ROOT


def snapshot(directory):
    paths = [directory]
    if directory.is_dir() and not directory.is_symlink():
        for base, dirs, files in os.walk(directory, followlinks=False):
            paths.extend(Path(base) / name for name in dirs + files)
    rows = []
    for path in sorted(paths):
        stat = path.lstat()
        row = {"relative": str(path.relative_to(directory)), "device": stat.st_dev,
               "inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
               "mode": stat.st_mode}
        if path.is_symlink():
            row["link"] = os.readlink(path)
        elif path.is_file() and path.name in {
            "RELEASE.json", "REPORT.json", "manifest.json", "config.json",
            "TOKENIZER.json", "SOURCE-BINDING.json", "FILES.json",
        } and stat.st_size < 16 * 1024**2:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    return rows


def apply_moves(root, moves, out):
    root, out = Path(root), Path(out)
    before = {}
    for old, new in moves.items():
        src, dst = root / old, root / new
        if src.parent != root / "corpus/pools":
            raise ValueError("only immediate pool entries may move")
        if not dst.resolve().is_relative_to((root / "corpus/pools").resolve()):
            raise ValueError("destination escapes pools")
        if not src.exists() or src.is_symlink() or dst.exists():
            raise ValueError(f"missing source or occupied destination: {old} -> {new}")
        if src.stat().st_dev != out.stat().st_dev:
            raise ValueError("only same-filesystem rename is allowed")
        before[old] = snapshot(src)
    (out / "before.json").write_text(json.dumps(before, ensure_ascii=False))
    completed = []
    try:
        for old, new in moves.items():
            src, dst = root / old, root / new
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            completed.append((old, new))
            if snapshot(dst) != before[old]:
                raise ValueError(f"asset identity changed during rename: {old}")
    except BaseException:
        for old, new in reversed(completed):
            (root / new).rename(root / old)
        raise
    return {"status": "moved_and_verified", "moves": len(completed),
            "filesystem_entries": sum(len(x) for x in before.values()),
            "metadata_hashes": sum("sha256" in r for x in before.values() for r in x),
            "verification": "same device/inode/size/mtime/mode and selected metadata hashes",
            "payload_rehashed": False, "legacy_symlinks_created": 0,
            "mappings": dict(completed)}


def run(config, out):
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    migration = ROOT / config["migration_manifest"]
    before_current = hashlib.sha256((ROOT / "CURRENT.json").read_bytes()).hexdigest()
    report = apply_moves(ROOT, json.loads(migration.read_text())["exact"], out)
    report["migration_sha256"] = hashlib.sha256(migration.read_bytes()).hexdigest()
    report["current_unchanged"] = before_current == hashlib.sha256((ROOT / "CURRENT.json").read_bytes()).hexdigest()
    if not report["current_unchanged"]:
        raise ValueError("CURRENT changed during migration")
    (out / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return {k: v for k, v in report.items() if k != "mappings"}
