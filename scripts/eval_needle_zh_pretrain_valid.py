#!/usr/bin/env python3
"""Streaming valid LM eval over mmap shards. Does not materialize all windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx  # noqa: E402

from architecture import NeedleZh  # noqa: E402
from checkpoint import load_params  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from data import PackedTokenSource, list_token_shards  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from train_common import eval_lm_loss, peak_bytes, stratified_window_indices  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mode", choices=["sentinel", "stratified", "full"], default="stratified")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--stratified-windows", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tok = ZhTokenizerV1()
    bins = list_token_shards(CORPUS_ZH_PRETRAIN, "valid")
    if not bins:
        print(f"missing valid shards under {CORPUS_ZH_PRETRAIN / 'tokens'}", file=sys.stderr)
        return 2
    valid = PackedTokenSource(bins, args.seq_len, tok.pad_id)
    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, args.ckpt, strict=True)
    indices = None
    max_windows: int | None = 128
    weighted = False
    if args.mode == "sentinel":
        max_windows = 128
    elif args.mode == "stratified":
        indices = stratified_window_indices(len(valid), args.stratified_windows)
        max_windows = None
        weighted = True
    else:
        max_windows = None
        weighted = True
    t0 = time.time()
    loss = eval_lm_loss(
        model,
        valid,
        max_windows=max_windows,
        batch_size=args.batch_size,
        indices=indices,
        token_weighted=weighted,
    )
    report = {
        "mode": args.mode,
        "ckpt": str(Path(args.ckpt).resolve().relative_to(ROOT))
        if Path(args.ckpt).resolve().is_relative_to(ROOT)
        else str(args.ckpt),
        "n_source_tokens": int(valid.n_tokens),
        "n_predictable_tokens": int(valid.n_predictable_tokens),
        "n_windows": len(valid),
        "eval_windows": len(indices) if indices is not None else min(len(valid), max_windows or len(valid)),
        "token_weighted": weighted,
        "valid_loss": loss,
        "elapsed_s": time.time() - t0,
        "peak_bytes": peak_bytes(),
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if loss == loss and abs(float(loss)) != float("inf") else 1


if __name__ == "__main__":
    raise SystemExit(main())
