#!/usr/bin/env python3
"""Store mixed 30M spoken pack under corpus/ without admitting CPT.

Same-disk rename + relative symlink. Does not copy shards.
Does not set colloquial_promoted, roles_complete, or clear CURRENT.blocked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1"
DST = ROOT / "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1"
OLD_REL = "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1"
NEW_REL = "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1"
TEXT_SUFFIXES = {".json", ".md", ".txt"}
HASH_KEYS = (
    "RELEASE.json",
    "unique-ledger.json",
    "manifest.json",
    "recipe-lock.json",
    "source-license.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_symlink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(target, link.parent), link)


def rewrite_tree(root: Path) -> int:
    n = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.is_symlink() or path.name == "hashes.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text.replace(OLD_REL, NEW_REL)
        if new != text:
            path.write_text(new, encoding="utf-8")
            n += 1
    return n


def refresh_pack_hashes(pack: Path) -> None:
    hashes_path = pack / "hashes.json"
    stored = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.is_file() else {}
    for name in HASH_KEYS:
        path = pack / name
        if path.is_file():
            stored[name] = file_sha256(path)
    hashes_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    current = json.loads((ROOT / "CURRENT.json").read_text(encoding="utf-8"))
    if "colloquial-contract-not-promoted" not in current.get("blocked", []):
        raise SystemExit("refusing: CURRENT.blocked lost colloquial-contract-not-promoted")
    mix_path = ROOT / "corpus/lm-v1/mix.json"
    mix = json.loads(mix_path.read_text(encoding="utf-8"))
    if mix.get("roles_complete") or mix.get("colloquial_promoted"):
        raise SystemExit("refusing: mix already admits colloquial")
    if mix.get("n_colloquial_train_tokens"):
        raise SystemExit("refusing: mix n_colloquial_train_tokens is not 0")

    if DST.exists():
        raise SystemExit(f"already relocated: {DST.relative_to(ROOT)}")
    if not SRC.is_dir() or SRC.is_symlink():
        raise SystemExit(f"expected real directory at {SRC.relative_to(ROOT)}")
    if SRC.stat().st_dev != DST.parent.stat().st_dev:
        raise SystemExit("refusing: source and dest are not on the same device")

    probe = SRC / "tokens/colloquial-train-0000.bin"
    inode_before = probe.stat().st_ino if probe.is_file() else None

    print(f"mv {SRC.relative_to(ROOT)} -> {DST.relative_to(ROOT)}", flush=True)
    os.rename(SRC, DST)
    rel_symlink(SRC, DST)
    print(f"  link {SRC.relative_to(ROOT)} -> {os.readlink(SRC)}", flush=True)

    probe_after = DST / "tokens/colloquial-train-0000.bin"
    if inode_before is not None and probe_after.stat().st_ino != inode_before:
        raise SystemExit("inode changed; shards were copied instead of renamed")

    n = rewrite_tree(DST)
    print(f"rewrote {n} pack text files", flush=True)
    refresh_pack_hashes(DST)

    release = json.loads((DST / "RELEASE.json").read_text(encoding="utf-8"))
    if release.get("formal_cpt_eligible") or release.get("register_v4"):
        raise SystemExit("refusing: relocated RELEASE claims CPT eligibility")
    n_unique = int(release["n_unique_train_tokens"])

    mix["colloquial_stored"] = {
        "id": "zh-pretrain-colloquial-synth-pooled-v1",
        "path": NEW_REL,
        "n_unique_train_tokens": n_unique,
        "admitted": False,
        "reason": "mixed_generator_not_qwen_plus",
    }
    mix["colloquial_promoted"] = False
    mix["roles_complete"] = False
    mix["n_colloquial_train_tokens"] = 0
    mix_path.write_text(json.dumps(mix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mix_hashes = ROOT / "corpus/lm-v1/hashes.json"
    hashes = json.loads(mix_hashes.read_text(encoding="utf-8"))
    hashes["mix.json"] = file_sha256(mix_path)
    mix_hashes.write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (ROOT / "corpus/lm-v1/colloquial/README.md").write_text(
        "# colloquial\n\n"
        "30M mixed-generator spoken pack is stored at "
        "`zh-pretrain-colloquial-synth-pooled-v1/`.\n\n"
        "Not admitted to the CPT mix (`colloquial_promoted: false`). "
        "Frozen contract still requires `qwen-plus-2025-12-01`. "
        "`CURRENT.blocked` keeps `colloquial-contract-not-promoted`.\n\n"
        "Notebook draft path is a relative symlink back here.\n",
        encoding="utf-8",
    )
    (DST / "README.md").write_text(
        "# zh-pretrain-colloquial-synth-pooled-v1\n\n"
        "Stored mixed 30M spoken-role pack. Canonical path is this directory.\n\n"
        "- Generator is `mixed-bailian-fleet`; frozen contract is qwen-plus only.\n"
        "- `formal_cpt_eligible` stays false. Do not write `approved-colloquial.json` from this pack.\n"
        "- Notebook `outbox/draft/colloquial-v1` is a symlink here.\n",
        encoding="utf-8",
    )

    job_path = ROOT / "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["publish_path"] = NEW_REL
    job["status"] = "draft"
    job["formal_cpt_eligible"] = False
    job["note"] = (
        "Pooled unique 30M mixed-generator pack stored under corpus/. "
        "Not admitted to formal CPT."
    )
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    contracts_path = ROOT / "notebook/corpus/lm-v1/colloquial/inbox/contracts.json"
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    contracts["note"] = (
        "Mixed-generator packs may be stored under corpus/lm-v1/colloquial/ "
        "without mix admission. CPT stays fail-closed until the frozen "
        "qwen-plus contract is met."
    )
    contracts_path.write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "stored": NEW_REL,
                "admitted": False,
                "n_unique_train_tokens": n_unique,
                "inode_unchanged": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
