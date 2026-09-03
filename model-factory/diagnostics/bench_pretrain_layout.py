#!/usr/bin/env python3
"""Short tok/s sweep for compile × batch at 512/1024/2048. Does not write checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common._repo import ensure_formal_on_path

ensure_formal_on_path()

import mlx.core as mx
import mlx.optimizers as optim

from architecture import NeedleZh
from config import NeedleZhConfig
from common.train_common import peak_bytes, train_lm_steps


def dummy_windows(n: int, seq: int) -> list[dict]:
    x = list(range(seq))
    y = x[1:] + [0]
    mask = [1.0] * (seq - 1) + [0.0]
    return [{"x": x, "y": y, "mask": mask} for _ in range(n)]


def run_case(seq: int, batch: int, compile_train: bool, warmup: int, steps: int) -> dict:
    mx.reset_peak_memory() if hasattr(mx, "reset_peak_memory") else None
    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=3e-4)
    windows = dummy_windows(max(batch * (warmup + steps + 2), batch * 4), seq)
    # warmup / compile
    train_lm_steps(
        model,
        windows,
        steps=warmup,
        lr=3e-4,
        seed=0,
        optimizer=opt,
        reseed=True,
        batch_size=batch,
        grad_accum=1,
        compile_train=compile_train,
        precision="fp32",
        allow_repeat=True,
        horizon_tokens=300_000_000,
    )
    t0 = time.time()
    result = train_lm_steps(
        model,
        windows,
        steps=steps,
        lr=3e-4,
        seed=1,
        optimizer=opt,
        reseed=False,
        batch_size=batch,
        grad_accum=1,
        compile_train=compile_train,
        precision="fp32",
        allow_repeat=True,
        horizon_tokens=300_000_000,
    )
    elapsed = max(time.time() - t0, 1e-6)
    tokens = int(result.get("tokens_seen") or 0)
    row = {
        "seq_len": seq,
        "batch_size": batch,
        "grad_accum": 1,
        "compile": compile_train,
        "steps": steps,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": tokens / elapsed,
        "last_loss": result.get("last_loss"),
        "peak_bytes": peak_bytes(),
        "ok": True,
    }
    del model, opt, windows
    gc.collect()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()
    grid = [
        (512, 8, False),
        (512, 16, False),
        (512, 32, False),
        (512, 64, False),
        (512, 8, True),
        (512, 16, True),
        (512, 32, True),
        (512, 64, True),
        (1024, 2, False),
        (1024, 4, False),
        (1024, 8, False),
        (1024, 16, False),
        (1024, 2, True),
        (1024, 4, True),
        (1024, 8, True),
        (1024, 16, True),
        (2048, 1, False),
        (2048, 2, False),
        (2048, 4, False),
        (2048, 8, False),
        (2048, 1, True),
        (2048, 2, True),
        (2048, 4, True),
        (2048, 8, True),
    ]
    rows = []
    for seq, batch, compile_train in grid:
        label = f"seq={seq} bs={batch} compile={compile_train}"
        print("+", label, flush=True)
        try:
            row = run_case(seq, batch, compile_train, args.warmup, args.steps)
        except Exception as exc:
            row = {
                "seq_len": seq,
                "batch_size": batch,
                "compile": compile_train,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=4),
            }
            print("  FAIL", row["error"], flush=True)
        else:
            print(
                f"  tok_s={row['tok_s']:.1f} loss={row['last_loss']} peak_giB={(row['peak_bytes'] or 0)/1024**3:.2f}",
                flush=True,
            )
        rows.append(row)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        gc.collect()
    winners: dict[int, dict] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        seq = int(row["seq_len"])
        prev = winners.get(seq)
        if prev is None or float(row["tok_s"]) > float(prev["tok_s"]) * 1.01:
            winners[seq] = row
    report = {"rows": rows, "winners": winners}
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
