#!/usr/bin/env python3
"""Qwen3.5 9B zero-shot on the same Route-ID payload as the student. No extra system contract."""

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
from run_eval_needle_qwen_v0 import load_mlx, mlx_chat

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from route_compiler import compile_routes  # noqa: E402
from route_protocol import PROTOCOL_ID, SCORER_ID, SERIALIZER_ID, VALIDATOR_ID, manifest_hash, protocol_hash  # noqa: E402
from schema_render import TASK_CONTRACT, render_route_request  # noqa: E402


def chat_user_only(backend: str, host: str, model: str, user: str, timeout: int, max_tokens: int) -> tuple[str, float]:
    if backend == "ollama":
        import urllib.request

        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": user}],
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ms = (time.perf_counter() - t0) * 1000
        msg = data.get("message") or {}
        text = (msg.get("content") or msg.get("thinking") or "").strip()
        return text, ms
    t0 = time.perf_counter()
    text = mlx_chat("", user, max_tokens, 0.0)
    return text, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "ollama"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    if args.backend == "ollama":
        model = args.model or "qwen3.5:9b-mlx"
    else:
        model = args.model or "Qwen/Qwen3.5-9B"
        load_mlx(model)
    rows = load_jsonl(args.bank)
    if args.limit:
        rows = rows[: args.limit]
    preds = []
    scored = []
    lats = []
    for row in rows:
        ctx = context_from_row(row)
        manifest = compile_routes(ctx)
        user = render_route_request(
            row,
            ctx.toolset,
            manifest_dict=manifest.as_dict(),
            manifest_hash=manifest_hash(manifest),
            entities=ctx.entities,
        )
        text, ms = chat_user_only(args.backend, args.host, model, user, args.timeout, args.max_tokens)
        item = score_item(row, raw_text=text)
        pred = {
            "item_id": row.get("item_id"),
            "text": text,
            "raw_route_output": text,
            "function_calls": item["function_calls"],
            "route_id": item.get("validated_route_id"),
            "latency_ms": round(ms, 3),
            "protocol_id": PROTOCOL_ID,
            "serializer": SERIALIZER_ID,
            "validator_id": VALIDATOR_ID,
            "scorer_id": SCORER_ID,
            "protocol_hash": protocol_hash(toolset=ctx.toolset, manifest=manifest),
            "manifest_hash": manifest_hash(manifest),
            "eval_lock_hash": hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.is_file() else "",
            "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "decode_mode": "qwen_route_id_text",
            "task_contract": TASK_CONTRACT,
            "system": "",
        }
        preds.append(pred)
        scored.append(item)
        lats.append(ms)
        print(f"[{len(preds)}/{len(rows)}] {row.get('item_id')} exact={item['exact']} {ms:.0f}ms", flush=True)
    summary = aggregate(rows, scored, lats)
    summary["model"] = model
    summary["backend"] = args.backend
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / "mei-1.0-58m-grounded-qwen9b")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
