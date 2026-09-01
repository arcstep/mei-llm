#!/usr/bin/env python3
"""Move Needle architecture/training/runtime out of notebook into formal roots.

Same-disk rename + relative symlinks. Do not copy large files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_MODEL = ROOT / "notebook/_tooling/model/mei-1.0-51m"
SRC_SPEC = ROOT / "notebook/base/pretrain-v1/spec"
SRC_SCRIPTS = ROOT / "notebook/_tooling/scripts"
SRC_RECIPES = ROOT / "notebook/sft/mei-1.0-51m/recipes"

ARCH = ROOT / "architecture/mei-1.0-51m-arch-v1"
TRAIN = ROOT / "training/mei-1.0-51m-train-v1"
RUNS = ROOT / "training/runs"
SHARED = ROOT / "runtime/_shared"
ROUTE = ROOT / "runtime/mei-1.0-51m-route-v1"
V2 = ROOT / "runtime/mei-1.0-51m-needle2-v2"

ARCH_PY = [
    "architecture.py",
    "config.py",
    "hidden_cells.py",
    "contrastive_head.py",
    "confidence_v2.py",
    "parity.py",
    "tokenizer.py",
    "NOTICE",
]
TRAIN_PY = ["data.py", "train_common.py", "checkpoint.py"]
SHARED_PY = ["schema_render.py", "normalizers.py", "decode.py"]
ROUTE_PY = [
    "route_protocol.py",
    "route_compiler.py",
    "candidates.py",
    "provenance_validator.py",
    "schema_mask.py",
    "grammar.py",
]
V2_PY = [
    "runtime_v2.py",
    "prompt_v2.py",
    "byte_grammar.py",
    "kv_manager.py",
    "provenance_validator_v2.py",
    "tool_index.py",
    "tool_call_protocol_v2.py",
]
ARCH_SPEC = ["model.json", "model-target-v2.json"]
ROUTE_SPEC = [
    "route-protocol-v1.json",
    "schema-subset-v1.json",
    "source-grounding-v1.json",
    "runtime.md",
    "special-tokens.json",
    "gates.json",
    "execute-threshold.json",
]
V2_SPEC = [
    "tool-call-protocol-v2.target.json",
    "schema-subset-v2.target.json",
    "source-grounding-v2.target.json",
    "runtime-target-v2.md",
    "special-tokens-target-v2.json",
    "gates-v2.target.json",
]
TRAIN_SCRIPTS = {
    "train_needle_zh_pretrain.py": "train_pretrain.py",
    "train_mei_51m_sft.py": "train_sft.py",
    "mei_cpt_gates.py": "cpt_gates.py",
}


def rel_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise RuntimeError(f"refusing to replace directory {link}")
    link.symlink_to(os.path.relpath(target, start=link.parent))
    print(f"  link {link.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


def move_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if src.resolve() == dest.resolve():
            return
        raise RuntimeError(f"destination exists {dest}")
    if src.is_symlink():
        dest.symlink_to(os.path.relpath(src.resolve(), start=dest.parent))
        src.unlink()
    else:
        os.rename(src, dest)
    print(f"mv {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
    rel_link(src, dest)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_digest(base: Path, patterns: tuple[str, ...] = ("*.py", "*.json", "*.md")) -> str:
    h = hashlib.sha256()
    files: list[Path] = []
    for pat in patterns:
        files.extend(p for p in base.rglob(pat) if p.is_file() and not p.is_symlink())
    for path in sorted(set(files), key=lambda p: p.as_posix()):
        rel = path.relative_to(base).as_posix()
        h.update(rel.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    for d in (ARCH / "spec", TRAIN / "recipes", RUNS, SHARED, ROUTE / "spec", V2 / "spec"):
        d.mkdir(parents=True, exist_ok=True)

    for name in ARCH_PY:
        move_file(SRC_MODEL / name, ARCH / name)
    for name in TRAIN_PY:
        move_file(SRC_MODEL / name, TRAIN / name)
    for name in SHARED_PY:
        move_file(SRC_MODEL / name, SHARED / name)
    for name in ROUTE_PY:
        move_file(SRC_MODEL / name, ROUTE / name)
    for name in V2_PY:
        move_file(SRC_MODEL / name, V2 / name)

    for name in ARCH_SPEC:
        move_file(SRC_SPEC / name, ARCH / "spec" / name)
    for name in ROUTE_SPEC:
        move_file(SRC_SPEC / name, ROUTE / "spec" / name)
    for name in V2_SPEC:
        move_file(SRC_SPEC / name, V2 / "spec" / name)

    for old, new in TRAIN_SCRIPTS.items():
        move_file(SRC_SCRIPTS / old, TRAIN / new)

    mix = SRC_RECIPES / "sft-mixture-v1.json"
    if mix.is_file() and not mix.is_symlink():
        move_file(mix, TRAIN / "recipes" / "sft-mixture-v1.json")

    (RUNS / ".gitkeep").write_text("", encoding="utf-8")

    write_json(
        ARCH / "RELEASE.json",
        {
            "id": "mei-1.0-51m-arch-v1",
            "kind": "architecture",
            "product": "mei-1.0-51m",
            "status": "implemented_in_code",
            "tokenizer": "tokenizer/zh-24k-v1",
            "spec": "spec/model.json",
            "target_spec": "spec/model-target-v2.json",
            "params": 58541901,
            "digest_sha256": tree_digest(ARCH),
        },
    )
    write_json(
        TRAIN / "RELEASE.json",
        {
            "id": "mei-1.0-51m-train-v1",
            "kind": "training",
            "product": "mei-1.0-51m",
            "status": "implemented_in_code",
            "architecture": "architecture/mei-1.0-51m-arch-v1",
            "corpus": "corpus/lm-v1",
            "runs": "training/runs",
            "digest_sha256": tree_digest(TRAIN),
        },
    )
    write_json(
        ROUTE / "RELEASE.json",
        {
            "id": "mei-1.0-51m-route-v1",
            "kind": "runtime",
            "product": "mei-1.0-51m",
            "status": "legacy_frozen",
            "protocol": "mei-route-protocol-v1",
            "architecture": "architecture/mei-1.0-51m-arch-v1",
            "not_a_needle2_release": True,
            "digest_sha256": tree_digest(ROUTE),
        },
    )
    write_json(
        V2 / "RELEASE.json",
        {
            "id": "mei-1.0-51m-needle2-v2",
            "kind": "runtime",
            "product": "mei-1.0-51m",
            "status": "implemented_in_code",
            "not_a_toolcall_model_release": True,
            "architecture": "architecture/mei-1.0-51m-arch-v1",
            "digest_sha256": tree_digest(V2) + tree_digest(SHARED),
        },
    )
    print("digest architecture", (ARCH / "RELEASE.json").read_text()[:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
