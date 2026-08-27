#!/usr/bin/env python3
"""Render base/lora predictions + eval criteria into a human rubric Markdown sheet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from repo_paths import BANK_MEI_EXPERT, ROOT

DRAFT_DOCS = ROOT.parent / "docs/draft/mei-llm"
BANK = BANK_MEI_EXPERT
DEFAULT_TAG = "20260731T072836Z"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def bank_by_id(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in load_jsonl(path)}


def preds_by_id(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in load_jsonl(path)}


def bullets(xs: list | None) -> str:
    if not xs:
        return "_（无）_"
    return "\n".join(f"- {x}" for x in xs)


def md_escape_fence(text: str) -> str:
    return (text or "").replace("```", "``\u200b`")


def section(item_id: str, bank: dict, base: dict | None, lora: dict | None) -> str:
    bmeta = bank.get(item_id) or {}
    prompt = bmeta.get("prompt") or (base or lora or {}).get("user_prompt") or ""
    pol = bmeta.get("polarity") or (base or lora or {}).get("polarity") or ""
    topic = bmeta.get("topic") or (base or lora or {}).get("topic") or ""
    face = bmeta.get("face") or (base or lora or {}).get("face") or ""
    base_ans = (base or {}).get("answer") or (base or {}).get("error") or "_（无 base 答卷）_"
    lora_ans = (lora or {}).get("answer") or (lora or {}).get("error") or "_（无 lora 答卷）_"

    return f"""### {item_id}

| 项 | 值 |
|----|-----|
| face | `{face}` |
| topic | `{topic}` |
| polarity | `{pol}` |
| **你的判定 base** | □ P　□ F　□ P?　□ F? |
| **你的判定 lora** | □ P　□ F　□ P?　□ F? |
| **谁更好** | □ base　□ lora　□ 平　□ 都烂 |

**题干**

> {prompt.replace(chr(10), chr(10) + '> ')}

**判定：通过**

{bullets(bmeta.get('pass_criteria'))}

**判定：禁止**

{bullets(bmeta.get('forbid'))}

**Base 答卷**

```text
{md_escape_fence(base_ans)}
```

**LoRA 答卷**

```text
{md_escape_fence(lora_ans)}
```

**批注**

> （你填写）

---
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG, help="runs/<tag>-… prefix")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = ROOT / "notebook/archive/runs"
    base_edge = runs / f"{args.tag}-base-faceedge-n8" / "predictions.jsonl"
    lora_edge = runs / f"{args.tag}-lora-faceedge-n8" / "predictions.jsonl"
    base_dev = runs / f"{args.tag}-base-facedev-n8" / "predictions.jsonl"
    lora_dev = runs / f"{args.tag}-lora-facedev-n8" / "predictions.jsonl"
    for p in (base_edge, lora_edge, base_dev, lora_dev):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    bank = bank_by_id(args.bank)
    be, le = preds_by_id(base_edge), preds_by_id(lora_edge)
    bd, ld = preds_by_id(base_dev), preds_by_id(lora_dev)

    # preserve edge then dev order from files
    edge_ids = [r["item_id"] for r in load_jsonl(base_edge)]
    dev_ids = [r["item_id"] for r in load_jsonl(base_dev)]

    out = args.out or (DRAFT_DOCS / f"2026-07-31-rubric-sheet-{args.tag}.md")
    parts = [
        f"# 人工 Rubric 答卷表 · `{args.tag}`",
        "",
        "> **桶**：`draft:mei-llm`  ",
        "> **用法**：对每题勾 base/lora 的 P/F；勿看 heuristic。  ",
        f"> **来源**：`mei-llm/notebook/archive/runs/{args.tag}-*-n8/predictions.jsonl` + pending 题库  ",
        "",
        "## 进度",
        "",
        f"- edge：{len(edge_ids)} 题  ",
        f"- dev：{len(dev_ids)} 题  ",
        f"- 合计：{len(edge_ids) + len(dev_ids)}  ",
        "",
        "---",
        "",
        "## face.edge",
        "",
    ]
    for iid in edge_ids:
        parts.append(section(iid, bank, be.get(iid), le.get(iid)))
    parts.append("## face.dev\n")
    for iid in dev_ids:
        parts.append(section(iid, bank, bd.get(iid), ld.get(iid)))

    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(edge_ids) + len(dev_ids)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
