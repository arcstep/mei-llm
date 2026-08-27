#!/usr/bin/env python3
"""Make corpus/lm-v1 a serving surface; move process assets to notebook.

Same-disk rename only. Does not copy shards, start CPT, or flip fail-closed flags.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / "corpus/lm-v1"
NB = ROOT / "notebook/corpus/lm-v1"
MIG = Path(__file__).resolve().parent
FREEZE = MIG / "freeze.json"

V0 = PUB / "language/zh-pretrain-v0"
V1 = PUB / "language/zh-pretrain-v1"
HQ = PUB / "language/hq"
V3 = PUB / "structure/zh-pretrain-v3"
COL = PUB / "colloquial/zh-pretrain-colloquial-synth-pooled-v1"

WORK_V0 = NB / "language/work/zh-pretrain-v0"
WORK_V1 = NB / "language/work/zh-pretrain-v1"
WORK_HQ = NB / "language/work/hq"
WORK_V3 = NB / "structure/work/zh-pretrain-v3"
DRAFT = NB / "colloquial/outbox/draft/colloquial-v1"
ASSEMBLE = NB / "assemble/work/zh-pretrain-v4"
ACCEPTED_V0 = NB / "language/outbox/accepted/zh-pretrain-v0"
ACCEPTED_V1 = NB / "language/outbox/accepted/zh-pretrain-v1"
ACCEPTED_V3 = NB / "structure/outbox/accepted/zh-pretrain-v3"

TOKENIZER = "zh-24k-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_stat(path: Path) -> dict:
    st = path.stat()
    return {"path": rel(path), "ino": st.st_ino, "size": st.st_size, "nlink": st.st_nlink}


def relocate(src: Path, dst: Path) -> str | None:
    if not src.exists() or src.is_symlink():
        return None
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
    print(f"mv {rel(src)} -> {rel(dst)}", flush=True)
    return rel(dst)


def move_idx(pack_tokens: Path, work_tokens: Path) -> int:
    n = 0
    if not pack_tokens.is_dir():
        return n
    for path in sorted(pack_tokens.glob("*.idx.jsonl")):
        relocate(path, work_tokens / path.name)
        n += 1
    return n


def unlink_if_symlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        print(f"unlink {rel(path)}", flush=True)


def rel_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(target, link.parent), link)
    print(f"link {rel(link)} -> {os.readlink(link)}", flush=True)


def list_bins(pack: Path) -> list[str]:
    token_dir = pack / "tokens"
    if not token_dir.is_dir():
        return []
    return [rel(p) for p in sorted(token_dir.glob("*.bin"))]


def serving_hashes(pack: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("README.md", "RELEASE.json", "manifest.json", "SOURCES.md", "source-license.json"):
        path = pack / name
        if path.is_file():
            out[name] = sha256_file(path)
    if extra:
        out.update(extra)
    return out


def snapshot() -> dict:
    mix = json.loads((PUB / "mix.json").read_text(encoding="utf-8"))
    current = json.loads((ROOT / "CURRENT.json").read_text(encoding="utf-8"))
    release = json.loads((PUB / "RELEASE.json").read_text(encoding="utf-8"))
    schedule = json.loads((PUB / "schedule.json").read_text(encoding="utf-8"))
    bins = []
    for shard in list(mix.get("train_shards") or []) + list(mix.get("valid_shards") or []):
        path = ROOT / shard
        if path.is_file():
            bins.append(file_stat(path))
    hq = HQ / "tokens/hq-train-0000.bin"
    arch = ROOT / "notebook/archive/corpus/zh-pretrain-v2/tokens/hq-train-0000.bin"
    return {
        "current_blocked": current.get("blocked"),
        "roles_complete": mix.get("roles_complete"),
        "colloquial_promoted": mix.get("colloquial_promoted"),
        "n_colloquial_train_tokens": mix.get("n_colloquial_train_tokens"),
        "release_roles_complete": release.get("roles_complete"),
        "formal_cpt": release.get("formal_cpt"),
        "schedule_sources": list((schedule.get("sources") or {}).keys()),
        "n_train_shards": len(mix.get("train_shards") or []),
        "active_bins": bins,
        "hq_hardlink": {
            "pub": file_stat(hq) if hq.is_file() else None,
            "archive": file_stat(arch) if arch.is_file() else None,
            "same_ino": hq.is_file() and arch.is_file() and hq.stat().st_ino == arch.stat().st_ino,
        },
    }


def assert_invariants(snap: dict) -> None:
    if "colloquial-contract-not-promoted" not in (snap.get("current_blocked") or []):
        raise SystemExit("refusing: CURRENT.blocked lost colloquial-contract-not-promoted")
    if snap.get("roles_complete") or snap.get("colloquial_promoted"):
        raise SystemExit("refusing: mix already admits colloquial")
    if snap.get("n_colloquial_train_tokens"):
        raise SystemExit("refusing: mix n_colloquial_train_tokens is not 0")
    if snap.get("formal_cpt") != "fail_closed":
        raise SystemExit("refusing: RELEASE.formal_cpt is not fail_closed")


def slim_v0_manifest(original: dict) -> dict:
    shards = []
    for row in original.get("token_shards") or []:
        shards.append(
            {
                "path": row.get("path"),
                "split": row.get("split"),
                "n_tokens": row.get("n_tokens"),
                "sha256": row.get("sha256"),
            }
        )
    return {
        "id": "zh-pretrain-v0",
        "status": "active",
        "tokenizer": TOKENIZER,
        "tokenizer_sha256": original.get("tokenizer_sha256"),
        "n_train_tokens": original.get("n_train_tokens"),
        "n_valid_tokens": original.get("n_valid_tokens"),
        "n_unique_train_tokens": original.get("n_train_tokens"),
        "unk_token_rate": original.get("unk_token_rate"),
        "token_shards": shards,
    }


def write_pack_readme(path: Path, title: str, status: str, body: str) -> None:
    path.write_text(
        f"# {title}\n\n"
        f"Status: `{status}`.\n\n"
        f"{body.rstrip()}\n",
        encoding="utf-8",
    )


def main() -> int:
    snap = snapshot()
    assert_invariants(snap)
    dump_json(FREEZE, snap)
    print(f"froze {len(snap['active_bins'])} active bins", flush=True)

    # --- relocate process trees ---
    relocate(V0 / "raw", WORK_V0 / "raw")
    relocate(V0 / "shards", WORK_V0 / "shards")
    relocate(V0 / "extract-manifest.json", WORK_V0 / "extract-manifest.json")
    if (V0 / "manifest.json").is_file():
        original = json.loads((V0 / "manifest.json").read_text(encoding="utf-8"))
        dump_json(WORK_V0 / "manifest.full.json", original)
        dump_json(V0 / "manifest.json", slim_v0_manifest(original))
    move_idx(V0 / "tokens", WORK_V0 / "tokens")

    relocate(V1 / "raw", WORK_V1 / "raw")
    relocate(V1 / "ingest-cursor.json", WORK_V1 / "ingest-cursor.json")
    relocate(V1 / "ingest-stats.json", WORK_V1 / "ingest-stats.json")
    relocate(V1 / "mix.json", WORK_V1 / "mix.json")
    relocate(V1 / "hashes", WORK_V1 / "hashes")
    move_idx(V1 / "tokens", WORK_V1 / "tokens")

    relocate(V3 / "raw", WORK_V3 / "raw")
    relocate(V3 / "reviews", WORK_V3 / "reviews")
    relocate(V3 / "unique-ledger.json", WORK_V3 / "unique-ledger.json")
    move_idx(V3 / "tokens", WORK_V3 / "tokens")
    move_idx(HQ / "tokens", WORK_HQ / "tokens")

    unlink_if_symlink(DRAFT)
    DRAFT.mkdir(parents=True, exist_ok=True)
    for name in (
        "raw",
        "reviews",
        "recipe-lock.json",
        "parents.json",
        "unique-ledger.json",
        "source-role.json",
        "hashes.json",
    ):
        src = COL / name
        if src.exists() and not src.is_symlink():
            relocate(src, DRAFT / name)
    move_idx(COL / "tokens", DRAFT / "tokens")

    unlink_if_symlink(ASSEMBLE)
    ASSEMBLE.mkdir(parents=True, exist_ok=True)
    dump_json(
        ASSEMBLE / "receipt.json",
        {
            "kind": "assemble-pointer",
            "published_path": "corpus/lm-v1",
            "writable": False,
            "note": "Process ledgers live here. Trainer consumes corpus/lm-v1, not this directory.",
        },
    )
    for name in (
        "approved-colloquial.json",
        "unique-ledger.json",
        "source-role.json",
        "source-license.json",
    ):
        src = PUB / name
        if src.is_file():
            relocate(src, ASSEMBLE / name)

    # accepted pointers stay, but rebuild if missing
    for link, target in (
        (ACCEPTED_V0, V0),
        (ACCEPTED_V1, V1),
        (ACCEPTED_V3, V3),
    ):
        if not link.exists():
            rel_symlink(link, target)

    # --- serving contracts ---
    dump_json(
        V0 / "RELEASE.json",
        {
            "id": "zh-pretrain-v0",
            "status": "active",
            "role": "wiki",
            "tokenizer": TOKENIZER,
            "n_train_tokens": 649904474,
            "n_valid_tokens": 34610229,
            "n_unique_train_tokens": 649904474,
        },
    )
    write_pack_readme(
        V0 / "README.md",
        "zh-pretrain-v0",
        "active",
        "Chinese Wikipedia unique train shards for `corpus/lm-v1` wiki role.\n\n"
        "Trainer reads `tokens/*.bin` via root `mix.json`. Build/extract state lives in "
        "`notebook/corpus/lm-v1/language/work/zh-pretrain-v0/`.",
    )
    (V0 / "SOURCES.md").write_text(
        "# zh-pretrain-v0 sources\n\n"
        "| Source | Path | License |\n"
        "|--------|------|----------|\n"
        "| Chinese Wikipedia dump | `notebook/corpus/tokenizer-v1/work/zh-vocab-v0/dumps/zhwiki-latest-pages-articles.xml.bz2` | CC BY-SA 3.0/4.0 + GFDL |\n"
        "| Serving shards | `tokens/{train,valid}-*.bin` | same |\n"
        "| Extract workbench | `notebook/corpus/lm-v1/language/work/zh-pretrain-v0/` | same |\n",
        encoding="utf-8",
    )

    v1_man = json.loads((V1 / "manifest.json").read_text(encoding="utf-8")) if (V1 / "manifest.json").is_file() else {}
    dump_json(WORK_V1 / "manifest.full.json", v1_man) if v1_man else None
    dump_json(
        V1 / "manifest.json",
        {
            "id": "zh-pretrain-v1",
            "status": "inactive",
            "tokenizer": TOKENIZER,
            "tokenizer_sha256": v1_man.get("tokenizer_sha256"),
            "n_train_tokens": v1_man.get("n_train_tokens"),
            "n_supplement_train_tokens": v1_man.get("n_supplement_train_tokens"),
            "n_valid_tokens": v1_man.get("n_valid_tokens"),
            "token_shards": list_bins(V1),
        },
    )
    dump_json(
        V1 / "RELEASE.json",
        {
            "id": "zh-pretrain-v1",
            "status": "inactive",
            "role": "language-1b-sidecar",
            "tokenizer": TOKENIZER,
            "n_train_tokens": v1_man.get("n_train_tokens"),
            "note": "Frozen 1B sidecar. Not an active source in corpus/lm-v1 mix.json.",
        },
    )
    write_pack_readme(
        V1 / "README.md",
        "zh-pretrain-v1",
        "inactive",
        "Frozen 1B language sidecar. Not listed in the current `corpus/lm-v1` schedule.\n\n"
        "Ingest cursors and FineWeb parquet live in "
        "`notebook/corpus/lm-v1/language/work/zh-pretrain-v1/`.",
    )
    (V1 / "SOURCES.md").write_text(
        "# zh-pretrain-v1 sources\n\n"
        "| Source | Serving / work path | License |\n"
        "|--------|---------------------|----------|\n"
        "| Wiki v0 | `corpus/lm-v1/language/zh-pretrain-v0/tokens/*.bin` | CC BY-SA 3.0/4.0 + GFDL |\n"
        "| FineWeb2-HQ parquet | `notebook/corpus/lm-v1/language/work/zh-pretrain-v1/raw/fineweb2-hq/` | ODC-By-1.0 + Common Crawl ToU |\n"
        "| SchemaStore / OAI | `notebook/corpus/lm-v1/language/work/zh-pretrain-v1/raw/` | Apache-2.0 |\n"
        "| This pack tokens | `tokens/*.bin` | mixed as above |\n",
        encoding="utf-8",
    )

    hq_bins = list_bins(HQ)
    dump_json(
        HQ / "RELEASE.json",
        {
            "id": "zh-pretrain-hq",
            "status": "active",
            "role": "hq",
            "tokenizer": TOKENIZER,
            "n_train_tokens": 383617452,
            "n_valid_tokens": 3778466,
            "hardlinked_from": "notebook/archive/corpus/zh-pretrain-v2/tokens/hq-*",
        },
    )
    dump_json(
        HQ / "manifest.json",
        {
            "id": "zh-pretrain-hq",
            "status": "active",
            "tokenizer": TOKENIZER,
            "n_train_tokens": 383617452,
            "n_valid_tokens": 3778466,
            "token_shards": hq_bins,
        },
    )
    (HQ / "SOURCES.md").write_text(
        "# language/hq sources\n\n"
        "FineWeb2-HQ `cmn_Hani` token shards. License: ODC-By-1.0 + Common Crawl ToU.\n"
        "Files are hardlinks of `notebook/archive/corpus/zh-pretrain-v2/tokens/hq-*`.\n",
        encoding="utf-8",
    )
    write_pack_readme(
        HQ / "README.md",
        "language/hq",
        "active",
        "FineWeb2-HQ serving shards for the current mix `hq` role.",
    )

    v3_rel = json.loads((V3 / "RELEASE.json").read_text(encoding="utf-8")) if (V3 / "RELEASE.json").is_file() else {}
    v3_rel["status"] = "active"
    v3_rel["tokenizer"] = TOKENIZER
    dump_json(V3 / "RELEASE.json", v3_rel)
    dump_json(
        V3 / "manifest.json",
        {
            "id": "zh-pretrain-v3",
            "status": "active",
            "tokenizer": TOKENIZER,
            "n_train_tokens": 4861158,
            "n_unique_train_tokens": 4861158,
            "n_valid_tokens": 105850,
            "unk_rate": 0.0,
            "token_shards": list_bins(V3),
        },
    )
    (V3 / "SOURCES.md").write_text(
        "# zh-pretrain-v3 sources\n\n"
        "Synthetic JSON Schema / OpenAPI and registered tool descriptions. "
        "Raw jsonl and audits live in `notebook/corpus/lm-v1/structure/work/zh-pretrain-v3/`.\n",
        encoding="utf-8",
    )
    write_pack_readme(
        V3 / "README.md",
        "zh-pretrain-v3",
        "active",
        "Clean structure shards for the current mix `structure` role.",
    )

    col_rel = json.loads((COL / "RELEASE.json").read_text(encoding="utf-8"))
    dump_json(
        COL / "RELEASE.json",
        {
            "id": "zh-pretrain-colloquial-synth-pooled-v1",
            "status": "not_admitted",
            "role": "colloquial",
            "tokenizer": TOKENIZER,
            "n_unique_train_tokens": col_rel.get("n_unique_train_tokens"),
            "n_valid_tokens": col_rel.get("n_valid_tokens"),
            "generator": col_rel.get("generator"),
            "model_snapshot": col_rel.get("model_snapshot"),
            "formal_cpt_eligible": False,
            "admitted": False,
            "train_shards": list_bins(COL),
        },
    )
    dump_json(
        COL / "manifest.json",
        {
            "id": "zh-pretrain-colloquial-synth-pooled-v1",
            "status": "not_admitted",
            "tokenizer": TOKENIZER,
            "n_unique_train_tokens": col_rel.get("n_unique_train_tokens"),
            "n_valid_tokens": col_rel.get("n_valid_tokens"),
            "token_shards": list_bins(COL),
        },
    )
    write_pack_readme(
        COL / "README.md",
        "zh-pretrain-colloquial-synth-pooled-v1",
        "not_admitted",
        "Frozen 30M mixed-generator spoken pack. Stored for consumption, not in the current mix.\n\n"
        "Process reviews and raw jsonl: `notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1/`.",
    )
    (PUB / "colloquial/README.md").write_text(
        "# colloquial\n\n"
        "Frozen spoken packs. `zh-pretrain-colloquial-synth-pooled-v1` is stored and "
        "`not_admitted` until the qwen-plus contract is met.\n",
        encoding="utf-8",
    )
    dump_json(
        DRAFT / "published.json",
        {
            "published_path": "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1",
            "status": "not_admitted",
        },
    )

    mix = json.loads((PUB / "mix.json").read_text(encoding="utf-8"))
    mix["roles_complete"] = False
    mix["colloquial_promoted"] = False
    mix["n_colloquial_train_tokens"] = 0
    mix["published_path"] = "corpus/lm-v1"
    mix["colloquial_stored"] = {
        "id": "zh-pretrain-colloquial-synth-pooled-v1",
        "path": "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1",
        "n_unique_train_tokens": col_rel.get("n_unique_train_tokens"),
        "admitted": False,
        "status": "not_admitted",
        "reason": "mixed_generator_not_qwen_plus",
    }
    dump_json(PUB / "mix.json", mix)

    root_release = json.loads((PUB / "RELEASE.json").read_text(encoding="utf-8"))
    root_release["roles_complete"] = False
    root_release["formal_cpt"] = "fail_closed"
    dump_json(PUB / "RELEASE.json", root_release)

    for pack in (V0, V1, HQ, V3, COL):
        dump_json(pack / "hashes.json", serving_hashes(pack))

    root_hashes = {
        "mix.json": sha256_file(PUB / "mix.json"),
        "schedule.json": sha256_file(PUB / "schedule.json"),
        "manifest.json": sha256_file(PUB / "manifest.json"),
        "RELEASE.json": sha256_file(PUB / "RELEASE.json"),
    }
    dump_json(PUB / "hashes.json", root_hashes)

    (PUB / "README.md").write_text(
        "# corpus/lm-v1\n\n"
        "Serving atlas for `mei-1.0-58m`. `CURRENT.corpus` points here.\n\n"
        "```text\n"
        "mix.json / schedule.json / RELEASE.json / manifest.json\n"
        "language/zh-pretrain-v0/   # active wiki\n"
        "language/hq/               # active FineWeb2-HQ\n"
        "language/zh-pretrain-v1/   # inactive 1B sidecar\n"
        "structure/zh-pretrain-v3/  # active structure\n"
        "colloquial/...pooled-v1/   # stored, not_admitted\n"
        "```\n\n"
        "Trainer consumes this tree via `mix.json` + `tokens/*.bin`. "
        "Build, extract, reviews, and indexes live under `notebook/corpus/lm-v1/`.\n"
        "Spoken CPT stays fail-closed until a frozen qwen-plus pack is admitted.\n",
        encoding="utf-8",
    )

    after = snapshot()
    assert_invariants(after)
    before_map = {row["path"]: row for row in snap["active_bins"]}
    after_map = {row["path"]: row for row in after["active_bins"]}
    if set(before_map) != set(after_map):
        raise SystemExit("active shard path set changed")
    for path, row in before_map.items():
        got = after_map[path]
        if got["ino"] != row["ino"] or got["size"] != row["size"]:
            raise SystemExit(f"inode/size changed: {path}")
    if not after["hq_hardlink"]["same_ino"]:
        raise SystemExit("HQ hardlink broken")
    dump_json(MIG / "verify.json", {"ok": True, "n_active_bins": len(after["active_bins"])})
    print(json.dumps({"ok": True, "n_active_bins": len(after["active_bins"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
