#!/usr/bin/env python3
"""Export eval-bank v0 Markdown → pending JSONL (draft only).

人审冻结前输出 review_status=pending；禁止把输出并入 train。
真源仍是分册 MD；改题先改 MD 再重跑本脚本。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT_DOCS = ROOT.parent / "docs/draft/mei-llm"
FACE_FILES = {
    "face.dev": DRAFT_DOCS / "2026-07-31-eval-bank-v0-dev-assist.md",
    "face.edge": DRAFT_DOCS / "2026-07-31-eval-bank-v0-qa-edge.md",
}

ITEM_HEAD = re.compile(
    r"^### (EVAL-(?:DEV|EDGE)-\d+) · 【([正负])】(.+)\s*$",
    re.MULTILINE,
)
META_ROW = re.compile(r"^\| ([^|]+?) \| (.+?) \|\s*$", re.MULTILINE)
CUT_MARK = re.compile(r"^## 已砍", re.MULTILINE)


def _polarity(zh: str) -> str:
    return "pol.pos" if zh == "正" else "pol.neg"


def _split_path(path: str) -> dict[str, str]:
    parts = [p.strip().strip("`") for p in path.split("/")]
    keys = ["face", "chapter", "topic", "move", "polarity"]
    out = {k: "" for k in keys}
    for i, p in enumerate(parts[:5]):
        out[keys[i]] = p
    return out


def _block_field(block: str, name: str) -> str | None:
    for m in META_ROW.finditer(block):
        if m.group(1).strip() == name:
            return m.group(2).strip().strip("`")
    return None


def _section(block: str, title: str) -> str:
    # **题干（给模型）** … until next **判定 or **审核
    pat = re.compile(
        rf"\*\*{re.escape(title)}\*\*\s*\n(?P<body>.*?)(?=\n\*\*|\Z)",
        re.DOTALL,
    )
    m = pat.search(block)
    if not m:
        return ""
    body = m.group("body").strip()
    # strip markdown quote fences
    lines = []
    for line in body.splitlines():
        if line.startswith("> "):
            lines.append(line[2:])
        elif line.startswith(">"):
            lines.append(line[1:].lstrip())
        elif line.strip() == ">":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def _bullets(block: str, title: str) -> list[str]:
    body = _section(block, title)
    items: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("- "):
            items.append(s[2:].strip())
    return items


def parse_file(path: Path, face_fallback: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    cut = CUT_MARK.search(text)
    if cut:
        text = text[: cut.start()]
    records: list[dict] = []
    matches = list(ITEM_HEAD.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        item_id = m.group(1)
        pol_zh = m.group(2)
        path_s = _block_field(block, "分类路径") or ""
        fields = _split_path(path_s) if path_s else {}
        if not fields.get("face"):
            fields["face"] = face_fallback
        if not fields.get("polarity"):
            fields["polarity"] = _polarity(pol_zh)
        ctx = _block_field(block, "背景包") or ""
        slots_raw = _block_field(block, "项目槽位") or ""
        project_slots: dict[str, str] | None
        if not slots_raw or slots_raw == "无":
            project_slots = None
        else:
            # keep names only: {{app_name}} …
            names = re.findall(r"\{\{([^}]+)\}\}", slots_raw)
            project_slots = {n: None for n in names} if names else {"_note": slots_raw}
        prompt = _section(block, "题干（给模型）")
        # also accept plain 题干
        if not prompt:
            prompt = _section(block, "题干")
        rec = {
            "item_id": item_id,
            "split": "eval",
            "face": fields.get("face") or face_fallback,
            "chapter": fields.get("chapter") or _block_field(block, "L2 Chapter"),
            "topic": fields.get("topic") or _block_field(block, "L3 Topic"),
            "move": fields.get("move") or _block_field(block, "L4 Move"),
            "polarity": fields.get("polarity") or _polarity(pol_zh),
            "signal": None,
            "context_ids": [ctx] if ctx else [],
            "project_slots": project_slots,
            "prompt": prompt,
            "pass_criteria": _bullets(block, "判定：通过"),
            "forbid": _bullets(block, "判定：禁止"),
            "review_status": "pending",
            "source_md": path.name,
        }
        records.append(rec)
    return records


def main() -> int:
    from repo_paths import BANK_MEI_EXPERT

    out_path = BANK_MEI_EXPERT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_recs: list[dict] = []
    for face, path in FACE_FILES.items():
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        all_recs.extend(parse_file(path, face))
    all_recs.sort(key=lambda r: r["item_id"])
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # quick sanity
    missing_prompt = [r["item_id"] for r in all_recs if not r["prompt"]]
    missing_pass = [r["item_id"] for r in all_recs if not r["pass_criteria"]]
    print(f"wrote {len(all_recs)} → {out_path.name}")
    print(f"dev={sum(1 for r in all_recs if r['face']=='face.dev')} "
          f"edge={sum(1 for r in all_recs if r['face']=='face.edge')} "
          f"neg={sum(1 for r in all_recs if r['polarity']=='pol.neg')}")
    if missing_prompt:
        print("WARN missing prompt:", ", ".join(missing_prompt), file=sys.stderr)
    if missing_pass:
        print("WARN missing pass_criteria:", ", ".join(missing_pass), file=sys.stderr)
    return 1 if missing_prompt or missing_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
