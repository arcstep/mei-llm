#!/usr/bin/env python3
"""Qwen baseline for mei-tool-schema-v1 using the same <tools> renderer as the student."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval_mei_tool_schema_v1 import aggregate, score_item
from eval_needle_toolcall_v0 import load_jsonl
from repo_paths import EVAL_BANKS_ROOT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH
from run_eval_needle_qwen_v0 import load_mlx, mlx_chat, ollama_chat, parse_function_calls

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from schema_render import load_toolset_json, render_request  # noqa: E402

BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"
SYSTEM = """你是按当次请求 schema 路由的闭集工具调用器。
只允许调用 <tools> 里出现的工具，或输出空列表。
缺必填、离题、schema 中无匹配工具：必须输出 []。
禁止提问、禁止解释、禁止自然语言。
只输出 JSON 数组。"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "ollama"], default="mlx")
    ap.add_argument("--model", default=None, help="mlx: HF id; ollama: tag. Defaults: Qwen/Qwen3.5-0.8B / qwen3.5:9b-mlx")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    if args.backend == "ollama":
        model = args.model or "qwen3.5:9b-mlx"
    else:
        model = args.model or "Qwen/Qwen3.5-0.8B"
        load_mlx(model)
    rows = load_jsonl(args.bank)
    if args.limit:
        rows = rows[: args.limit]
    preds = []
    scored = []
    lats = []
    for row in rows:
        ts = load_toolset_json(str(row["toolset_id"]))
        user = render_request(row, ts)
        t0 = time.perf_counter()
        if args.backend == "ollama":
            text, lat = ollama_chat(args.host, model, SYSTEM, user, args.timeout, 0.0)
            ms = float(lat.get("wall_ms") or ((time.perf_counter() - t0) * 1000))
        else:
            text = mlx_chat(SYSTEM, user, args.max_tokens, 0.0)
            ms = (time.perf_counter() - t0) * 1000
        calls, status = parse_function_calls(text)
        if status != "ok":
            calls = []
        pred = {
            "item_id": row.get("item_id"),
            "function_calls": calls,
            "text": text,
            "latency_ms": round(ms, 3),
            "parse_status": status,
        }
        preds.append(pred)
        scored.append(score_item(row, calls, raw_text=text))
        lats.append(ms)
        print(f"[{len(preds)}/{len(rows)}] {row.get('item_id')} {status} {ms:.0f}ms", flush=True)
    summary = aggregate(rows, scored, lats)
    summary["model"] = model
    summary["backend"] = args.backend
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / "mei-1.0-58m-schema-qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
