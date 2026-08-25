#!/usr/bin/env python3
"""Single-GPU throughput sweep for Needle-zh pretrain on a frozen 1M-token slice."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN, EXPERIMENTS_RUNS, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx  # noqa: E402

from architecture import NeedleZh, count_params  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from data import PackedTokenSource, list_token_shards, pack_windows  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from train_common import peak_bytes, train_lm_steps  # noqa: E402

SEQ = 256
EFFECTIVE = 8
LAYOUTS = ((2, 4), (4, 2), (8, 1))


def run_one(windows, *, batch_size: int, grad_accum: int, compile_train: bool, precision: str) -> dict:
    cfg = NeedleZhConfig.from_spec()
    mx.random.seed(0)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    t0 = time.time()
    result = train_lm_steps(
        model,
        windows,
        steps=12,
        lr=3e-4,
        seed=0,
        batch_size=batch_size,
        grad_accum=grad_accum,
        compile_train=compile_train,
        precision=precision,
        allow_repeat=True,
    )
    elapsed = max(1e-6, time.time() - t0)
    tokens = int(result["tokens_seen"])
    last = result["last_loss"]
    finite = last is not None and last == last and abs(float(last)) != float("inf")
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "compile": compile_train,
        "precision": precision,
        "steps": result["steps"],
        "tokens_seen": tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(tokens / elapsed, 1),
        "update_ms": round(1000.0 * elapsed / max(result["steps"], 1), 2),
        "last_loss": None if last is None else float(last),
        "finite": finite,
        "peak_bytes": peak_bytes(),
        "params": count_params(model),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4096)
    args = ap.parse_args()
    tok = ZhTokenizerV1()
    bins = list_token_shards(CORPUS_ZH_PRETRAIN, "train")
    if bins:
        src = PackedTokenSource(bins, SEQ, tok.pad_id)
        windows = src[: min(args.windows, len(src))]
    else:
        print("missing token shards; using synthetic ids", file=sys.stderr)
        ids = ([2, 11, 12, 13, 14, 1] * (SEQ // 2))[: SEQ * args.windows + 1]
        windows = pack_windows([ids], SEQ, pad_id=0)[: args.windows]
    rows = []
    for bs, accum in LAYOUTS:
        row = run_one(windows, batch_size=bs, grad_accum=accum, compile_train=False, precision="fp32")
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = max(rows, key=lambda r: r["tok_s"] if r["finite"] else -1)
    try:
        compiled = run_one(
            windows,
            batch_size=best["batch_size"],
            grad_accum=best["grad_accum"],
            compile_train=True,
            precision="fp32",
        )
    except Exception as exc:
        compiled = {
            "batch_size": best["batch_size"],
            "grad_accum": best["grad_accum"],
            "compile": True,
            "precision": "fp32",
            "finite": False,
            "tok_s": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    rows.append(compiled)
    print(json.dumps(compiled), flush=True)
    try:
        amp = run_one(
            windows,
            batch_size=best["batch_size"],
            grad_accum=best["grad_accum"],
            compile_train=False,
            precision="fp16",
        )
    except Exception as exc:
        amp = {
            "batch_size": best["batch_size"],
            "grad_accum": best["grad_accum"],
            "compile": False,
            "precision": "fp16",
            "finite": False,
            "tok_s": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    rows.append(amp)
    print(json.dumps(amp), flush=True)
    finite_rows = [r for r in rows if r.get("finite")]
    fastest = max(finite_rows, key=lambda r: r["tok_s"])
    official_rows = [
        r for r in finite_rows if r.get("precision") == "fp32" and not r.get("compile")
    ]
    chosen = max(official_rows, key=lambda r: r["tok_s"]) if official_rows else fastest
    out = {
        "seq_len": SEQ,
        "effective_batch": EFFECTIVE,
        "n_windows": len(windows),
        "rows": rows,
        "chosen": chosen,
        "fastest": fastest,
        "amp_gate_ok": False,
        "compile_ok": bool(compiled.get("finite")),
        "baseline_tok_s": 3475,
        "speedup_vs_baseline": round(chosen["tok_s"] / 3475.0, 3) if chosen["tok_s"] else None,
    }
    dest = EXPERIMENTS_RUNS / "needle-zh-pretrain-throughput"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if chosen["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
