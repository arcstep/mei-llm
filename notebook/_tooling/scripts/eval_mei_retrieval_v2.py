#!/usr/bin/env python3
"""Retrieval smoke: Recall@1/5, MRR, random/lexical floors. Not a publish eval bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from repo_paths import ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx

from architecture import NeedleZh
from config import NeedleZhConfig
from prompt_v2 import render_tools_block
from schema_render import load_toolset_json
from tokenizer import ZhTokenizerV1
from tool_index import ToolIndex

FIXTURES = TASKS_ROOT / TASK_NEEDLE_ZH / "eval/fixtures/retrieval-smoke-v2.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def lexical_rank(query: str, tools: list[dict]) -> list[str]:
    q = query
    scored = []
    for t in tools:
        blob = str(t.get("name") or "") + " " + str(t.get("description") or "")
        score = sum(1 for ch in q if ch in blob)
        scored.append((score, str(t.get("name"))))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored]


def metrics(ranks: list[int]) -> dict:
    n = max(1, len(ranks))
    r1 = sum(1 for r in ranks if r == 0) / n
    r5 = sum(1 for r in ranks if 0 <= r < 5) / n
    mrr = sum((1.0 / (r + 1) if r >= 0 else 0.0) for r in ranks) / n
    return {"recall_at_1": r1, "recall_at_5": r5, "mrr": mrr, "n": n}


def embed_text(model, tok, text: str) -> mx.array:
    ids = tok.encode(text, add_bos=True)[:48]
    ids = ids + [0] * max(0, 8 - len(ids))
    arr = mx.array([ids], dtype=mx.int32)
    out = model(arr, return_contrastive=True, return_cells=True)
    return out["contrastive"][0]


def stratified_ranks(rows: list[dict], ranks: list[int]) -> dict:
    by_family: dict[str, list[int]] = defaultdict(list)
    by_split: dict[str, list[int]] = defaultdict(list)
    for row, rank in zip(rows, ranks):
        by_family[str(row.get("family") or "na")].append(rank)
        by_split[str(row.get("split") or "na")].append(rank)
    return {
        "by_family": {k: metrics(v) for k, v in by_family.items()},
        "by_split": {k: metrics(v) for k, v in by_split.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--lexical-only", action="store_true")
    args = ap.parse_args()
    rows = load_jsonl(args.pack or FIXTURES)
    home = load_toolset_json("needle-home-v0")["tools"]
    extra = [
        {
            "name": "create_invoice",
            "description": "Create an invoice document.",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        },
        {
            "name": "get_weather_status",
            "description": "Get weather API health status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]
    extra.extend(
        {
            "name": f"aux_tool_{i}",
            "description": f"Auxiliary catalog filler {i} for random-floor measurement.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        for i in range(8)
    )
    catalog = home + extra
    if args.pack and rows and rows[0].get("catalog_tools"):
        merged = {}
        for row in rows:
            for tool in row.get("catalog_tools") or []:
                merged[str(tool.get("name"))] = tool
        if merged:
            catalog = list(merged.values())
    if args.lexical_only:
        lex_ranks = []
        for row in rows:
            gold = row.get("gold_tool")
            lex = lexical_rank(str(row.get("query") or ""), catalog)
            lex_ranks.append(lex.index(gold) if gold in lex else -1)
        out = {
            "ok": True,
            "smoke": bool(args.smoke),
            "lexical_only": True,
            "lexical_floor": metrics(lex_ranks),
            "random_floor": {
                "recall_at_5": min(1.0, 5 / max(1, len(catalog))),
                "recall_at_1": 1 / max(1, len(catalog)),
                "uniform_recall_at_5": min(1.0, 5 / max(1, len(catalog))),
            },
            "n_catalog": len(catalog),
            "n_queries": len(rows),
            **stratified_ranks(rows, lex_ranks),
        }
        out["ok"] = out["lexical_floor"]["n"] == len(rows) and out["random_floor"]["uniform_recall_at_5"] > 0
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out["ok"] else 1
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    idx = ToolIndex()
    idx.invalidate_if_stale(model_hash="tiny", head_hash="c", tokenizer_hash=tok.model_sha256)
    for tool in catalog:
        text = render_tools_block([tool])
        idx.upsert(tool, embed_text(model, tok, text))
    learned_ranks = []
    lex_ranks = []
    for i, row in enumerate(rows):
        gold = str(row["gold_tool"])
        q_emb = embed_text(model, tok, str(row["query"]))
        picked = idx.topk(q_emb, k=len(catalog))
        order = [r.tool_id for r in picked]
        learned_ranks.append(order.index(gold) if gold in order else -1)
        lex = lexical_rank(str(row["query"]), catalog)
        lex_ranks.append(lex.index(gold) if gold in lex else -1)
    out = {
        "ok": True,
        "smoke": bool(args.smoke),
        "learned": metrics(learned_ranks),
        "lexical_floor": metrics(lex_ranks),
        "random_floor": {
            "recall_at_5": min(1.0, 5 / max(1, len(catalog))),
            "recall_at_1": 1 / max(1, len(catalog)),
            "uniform_recall_at_5": min(1.0, 5 / max(1, len(catalog))),
        },
        "n_catalog": len(catalog),
        "n_queries": len(rows),
        "note": "Untrained tiny ContrastiveHead is not a quality claim. Gate: API works and random floor is defined.",
        **stratified_ranks(rows, lex_ranks),
    }
    out["ok"] = out["learned"]["n"] == len(rows) and out["random_floor"]["uniform_recall_at_5"] > 0
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
