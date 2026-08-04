#!/usr/bin/env python3
"""Build derived training chunks from a pinned Evidence knowledge release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


ARTIFACT_ID_RE = re.compile(r'<a id="(meilang\.[a-z0-9.-]+)"></a>')
SECTION_ANCHOR_RE = re.compile(r'<a id="(sec-[0-9]{2})"></a>\s*\n## +(.+)$', re.MULTILINE)
HEADING_RE = re.compile(r"^## +(.+)$", re.MULTILINE)


def load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def chunks(text: str, *, root_anchor: str) -> list[tuple[str, str, str]]:
    """Prefer explicit sec-NN anchors; never invent title-slug anchors."""
    section_matches = list(SECTION_ANCHOR_RE.finditer(text))
    if section_matches:
        result: list[tuple[str, str, str]] = []
        preface = text[: section_matches[0].start()].strip()
        if preface:
            result.append((root_anchor, "chapter", preface))
        for index, match in enumerate(section_matches):
            end = (
                section_matches[index + 1].start()
                if index + 1 < len(section_matches)
                else len(text)
            )
            anchor = match.group(1)
            title = match.group(2).strip()
            body = text[match.start() : end].strip()
            result.append((anchor, title, body))
        return result

    heading_matches = list(HEADING_RE.finditer(text))
    if not heading_matches:
        return [(root_anchor, "chapter", text.strip())]
    result = []
    preface = text[: heading_matches[0].start()].strip()
    if preface:
        result.append((root_anchor, "chapter", preface))
    for index, match in enumerate(heading_matches):
        end = (
            heading_matches[index + 1].start()
            if index + 1 < len(heading_matches)
            else len(text)
        )
        title = match.group(1).strip()
        body = text[match.start() : end].strip()
        result.append((f"h{index + 1}", title, body))
    return result


def iter_knowledge_files(evidence_root: Path) -> list[Path]:
    knowledge = evidence_root / "knowledge"
    files: list[Path] = []
    for path in sorted(knowledge.rglob("*.md")):
        if path == knowledge / "README.md":
            continue
        if path.name == "README.md" and "examples" not in path.parts and "cases" not in path.parts:
            continue
        if "expected" in path.parts:
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-release", default="mei-lang-evidence@HEAD")
    parser.add_argument("--task-catalog-release", default="meilang-task-catalog-v2")
    args = parser.parse_args()

    evidence_root = args.evidence_root.resolve()
    release = args.evidence_release
    if args.manifest and args.manifest.exists():
        manifest = load_manifest(args.manifest.resolve())
        release = manifest.get("evidence_release", release)

    rows = []
    for path in iter_knowledge_files(evidence_root):
        rel = str(path.relative_to(evidence_root))
        text = path.read_text(encoding="utf-8")
        kid_match = ARTIFACT_ID_RE.search(text)
        if not kid_match:
            continue
        root_anchor = kid_match.group(1)
        for anchor, title, body in chunks(text, root_anchor=root_anchor):
            sample_anchor = root_anchor if anchor == root_anchor else f"{root_anchor}#{anchor}"
            rows.append(
                {
                    "sample_id": f"{release}:{sample_anchor}",
                    "task_catalog_release": args.task_catalog_release,
                    "evidence_release": release,
                    "source_refs": [f"{rel}#{anchor}"],
                    "transform_recipe": "knowledge_section_chunk.v3",
                    "source_path": rel,
                    "knowledge_id": root_anchor,
                    "anchor": anchor,
                    "title": title,
                    "text": body,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} chunks from {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
