#!/usr/bin/env python3
"""Student runner for mei-tool-grounded-v1 (route-ID decode + common scorer)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from eval_mei_tool_grounded_v1 import BANK, LOCK, aggregate, context_from_row, score_item
from eval_needle_toolcall_v0 import load_jsonl
from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx

from architecture import NeedleZh
from checkpoint import load_params
from config import NeedleZhConfig
from decode import greedy_route_id
from route_compiler import compile_routes
from route_protocol import PROTOCOL_ID, SCORER_ID, SERIALIZER_ID, VALIDATOR_ID, manifest_hash, protocol_hash
from schema_render import TASK_CONTRACT, render_route_request
from tokenizer import ZhTokenizerV1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out-dir", type=Path, default=None)
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
    load_params(model, args.ckpt, strict=not args.tiny)
    preds = []
    scored = []
    lats = []
    for row in rows:
        ctx = context_from_row(row)
        t_c = time.perf_counter()
        manifest = compile_routes(ctx)
        build_ms = (time.perf_counter() - t_c) * 1000
        user = render_route_request(
            row,
            ctx.toolset,
            manifest_dict=manifest.as_dict(),
            manifest_hash=manifest_hash(manifest),
            entities=ctx.entities,
        )
        prompt = tok.encode_chat(user)["prompt_ids"]
        t0 = time.perf_counter()
        out = greedy_route_id(model, tok, prompt, manifest=manifest, toolset=ctx.toolset)
        ms = (time.perf_counter() - t_c) * 1000
        text = str(out.get("text") or "")
        item = score_item(row, raw_text=text)
        item["latency_ms"] = round(ms, 3)
        timings = dict(out.get("timings") or {})
        timings["candidate_build_ms"] = build_ms
        pred = {
            "item_id": row.get("item_id"),
            "text": text,
            "raw_route_output": text,
            "function_calls": item["function_calls"],
            "route_id": out.get("route_id"),
            "latency_ms": round(ms, 3),
            "timings": timings,
            "protocol_id": PROTOCOL_ID,
            "serializer": SERIALIZER_ID,
            "validator_id": VALIDATOR_ID,
            "scorer_id": SCORER_ID,
            "protocol_hash": protocol_hash(toolset=ctx.toolset, manifest=manifest),
            "manifest_hash": manifest_hash(manifest),
            "eval_lock_hash": hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.is_file() else "",
            "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "decode_mode": "route_id_trie_kv",
            "task_contract": TASK_CONTRACT,
        }
        preds.append(pred)
        scored.append(item)
        lats.append(ms)
    summary = aggregate(rows, scored, lats)
    summary["ckpt"] = str(args.ckpt)
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / "mei-1.0-58m-grounded-student")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
