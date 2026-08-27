#!/usr/bin/env python3
"""Promote accepted language/structure into corpus/lm-v1 with links, no GB copies.

- Move accepted trees to corpus/lm-v1/{language,structure}/
- Relative symlinks from notebook outbox back to corpus/
- Hardlink HQ shards from archived v2 (same inode)
- Publish v4 atlas JSON beside those dirs as corpus/lm-v1/mix.json
- Do not promote colloquial
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB_LM = ROOT / "notebook/corpus/lm-v1"
PUB = ROOT / "corpus/lm-v1"

MOVES = [
    (
        NB_LM / "language/outbox/accepted/zh-pretrain-v0",
        PUB / "language/zh-pretrain-v0",
    ),
    (
        NB_LM / "language/outbox/accepted/zh-pretrain-v1",
        PUB / "language/zh-pretrain-v1",
    ),
    (
        NB_LM / "structure/outbox/accepted/zh-pretrain-v3",
        PUB / "structure/zh-pretrain-v3",
    ),
]

PATH_REPLACEMENTS = [
    (
        "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v0",
        "corpus/lm-v1/language/zh-pretrain-v0",
    ),
    (
        "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v1",
        "corpus/lm-v1/language/zh-pretrain-v1",
    ),
    (
        "notebook/corpus/lm-v1/structure/outbox/accepted/zh-pretrain-v3",
        "corpus/lm-v1/structure/zh-pretrain-v3",
    ),
    (
        "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4",
        "corpus/lm-v1",
    ),
    (
        "notebook/archive/corpus/zh-pretrain-v2/tokens/hq-",
        "corpus/lm-v1/language/hq/tokens/hq-",
    ),
]

TEXT_SUFFIXES = {".json", ".md", ".txt"}


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
        if path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for old, dst in PATH_REPLACEMENTS:
            new = new.replace(old, dst)
        if new != text:
            path.write_text(new, encoding="utf-8")
            n += 1
    return n


def main() -> int:
    if (PUB / "language/zh-pretrain-v0").exists() and not (PUB / "language/zh-pretrain-v0").is_symlink():
        raise SystemExit("already promoted: corpus/lm-v1/language/zh-pretrain-v0 exists")

    (PUB / "language/hq/tokens").mkdir(parents=True, exist_ok=True)
    (PUB / "structure").mkdir(parents=True, exist_ok=True)
    (PUB / "colloquial").mkdir(parents=True, exist_ok=True)

    for src, dst in MOVES:
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists():
            raise FileExistsError(dst)
        print(f"mv {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}", flush=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        rel_symlink(src, dst)
        print(f"  link {src.relative_to(ROOT)} -> {os.readlink(src)}", flush=True)

    hq_src = ROOT / "notebook/archive/corpus/zh-pretrain-v2/tokens"
    hq_dst = PUB / "language/hq/tokens"
    n_link = 0
    for src in sorted(hq_src.glob("hq-*")):
        dst = hq_dst / src.name
        if dst.exists():
            continue
        os.link(src, dst)
        n_link += 1
    print(f"hardlinked {n_link} HQ shard files", flush=True)

    v4 = NB_LM / "assemble/work/zh-pretrain-v4"
    if v4.is_dir() and not v4.is_symlink():
        for item in v4.iterdir():
            dest = PUB / item.name
            if dest.exists():
                if item.is_file() and dest.is_file():
                    # keep already-written published README if we add one later
                    if item.name == "README.md":
                        dest.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
                        continue
                raise FileExistsError(dest)
            os.rename(item, dest)
        shutil.rmtree(v4)
        rel_symlink(v4, PUB)
        print(f"link {v4.relative_to(ROOT)} -> {os.readlink(v4)}", flush=True)

    n = rewrite_tree(PUB)
    print(f"rewrote {n} published text files", flush=True)

    # mix.json must not list colloquial train shards as active
    mix_path = PUB / "mix.json"
    mix = json.loads(mix_path.read_text(encoding="utf-8"))
    mix["id"] = "lm-v1"
    mix["published_path"] = "corpus/lm-v1"
    mix["copies_shards"] = False
    mix["colloquial_promoted"] = False
    mix_path.write_text(json.dumps(mix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (PUB / "colloquial" / "README.md").write_text(
        "# colloquial\n\nNot published. Spoken pack remains "
        "`notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1/` "
        "until the frozen contract is promoted.\n",
        encoding="utf-8",
    )
    (PUB / "language" / "hq" / "README.md").write_text(
        "# language/hq\n\nFineWeb2-HQ token shards. Files are **hardlinks** into "
        "`notebook/archive/corpus/zh-pretrain-v2/tokens/hq-*` (same inode; CWT2 remainder stays archived).\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
