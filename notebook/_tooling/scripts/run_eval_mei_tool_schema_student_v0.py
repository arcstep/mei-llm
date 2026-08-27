#!/usr/bin/env python3
"""Student runner for mei-tool-schema-v1 (schema-conditioned constrained decode)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval_mei_tool_schema_v1 import aggregate, score_item
from eval_needle_toolcall_v0 import load_jsonl
from repo_paths import EVAL_BANKS_ROOT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx

from architecture import NeedleZh
from checkpoint import load_params
from config import NeedleZhConfig
from decode import greedy_constrained
from schema_render import load_toolset_json, render_request
from tokenizer import ZhTokenizerV1

BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    rows = load_jsonl(args.bank)
    if args.limit:
        rows = rows[: args.limit]
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig().tiny() if args.tiny else NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, args.ckpt, strict=True)
    preds = []
    scored = []
    lats = []
    for row in rows:
        ts = load_toolset_json(str(row["toolset_id"]))
        user = render_request(row, ts)
        prompt = tok.encode_chat(user)["prompt_ids"]
        t0 = time.perf_counter()
        out = greedy_constrained(model, tok, prompt, ts, max_new=args.max_new)
        ms = (time.perf_counter() - t0) * 1000
        calls = (out.get("parsed") or {}).get("function_calls") or []
        pred = {
            "item_id": row.get("item_id"),
            "function_calls": calls,
            "text": out.get("text"),
            "latency_ms": round(ms, 3),
            "ok": bool((out.get("parsed") or {}).get("ok")),
        }
        preds.append(pred)
        scored.append(score_item(row, calls, raw_text=str(out.get("text") or "")))
        lats.append(ms)
    summary = aggregate(rows, scored, lats)
    summary["ckpt"] = str(args.ckpt)
    summary["n_pred"] = len(preds)
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / "mei-1.0-58m-schema-student")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
