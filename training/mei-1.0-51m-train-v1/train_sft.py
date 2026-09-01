#!/usr/bin/env python3
"""Canonical mei-1.0-51m Route-ID SFT with valid/best/early-stop. 2k and 10k start from the same 300M base."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ROOT, TRAIN_RUNS, ensure_formal_on_path

ensure_formal_on_path()

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from architecture import NeedleZh, count_params
from checkpoint import load_params, load_train_state, save_params, save_train_state
from config import NeedleZhConfig
from data import encode_sft_row
from schema_render import PRODUCT_SFT_SEQ_LEN, ROUTE_SERIALIZER_ID, SERIALIZER_ID
from tokenizer import ZhTokenizerV1
from train_common import eval_lm_loss, train_lm_steps


def load_pack(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def encode_rows(tok, rows, cap: int, inject: bool) -> tuple[list[dict], dict[str, int], int]:
    raw: list[tuple[dict, dict]] = []
    reject: dict[str, int] = {}
    for r in rows:
        e = encode_sft_row(tok, r, cap, inject_schema=inject)
        if e.get("n_unmasked", 0) > 0:
            raw.append((r, e))
        else:
            reason = str(e.get("reject_reason") or "empty")
            reject[reason] = reject.get(reason, 0) + 1
    seq = cap
    if raw and cap >= 64:
        seq = min(cap, max(64, max(int(e.get("n_total") or 0) for _, e in raw) + 2))
    encoded = []
    if seq != cap:
        for r, _ in raw:
            e = encode_sft_row(tok, r, seq, inject_schema=inject)
            if e.get("n_unmasked", 0) > 0:
                encoded.append(e)
            else:
                reject["repack"] = reject.get("repack", 0) + 1
    else:
        encoded = [e for _, e in raw]
    return encoded, reject, seq


def teacher_acc(model, windows: list[dict], n: int = 8) -> float:
    if not windows:
        return 0.0
    hit = 0.0
    tot = 0.0
    for w in windows[:n]:
        x = mx.array([w["x"]], dtype=mx.int32)
        y = np.array(w["y"])
        mask = np.array(w["mask"])
        logits = model(x)["logits"]
        mx.eval(logits)
        pred = np.array(logits[0].argmax(axis=-1))
        hit += float(((pred == y) * mask).sum())
        tot += float(mask.sum())
    return hit / max(tot, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", type=Path, default=None, help="weights-only init; omit for random")
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--valid", type=Path, default=None)
    ap.add_argument("--recipe", type=Path, default=None)
    ap.add_argument("--seq-len", type=int, default=PRODUCT_SFT_SEQ_LEN)
    ap.add_argument("--steps", type=int, default=None, help="legacy fixed steps; default is epoch loop")
    ap.add_argument("--max-epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = once per epoch")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overfit-gate", action="store_true")
    ap.add_argument("--resume-state", type=Path, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--eval-lock", type=Path, default=None)
    args = ap.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    if not pack.is_file():
        print(f"missing pack {pack}", file=sys.stderr)
        return 1
    tok = ZhTokenizerV1()
    rows = load_pack(pack, 8 if args.overfit_gate else (32 if args.smoke else args.limit))
    cap = 48 if args.smoke else args.seq_len
    inject = not args.smoke
    encoded, reject, seq = encode_rows(tok, rows, cap, inject)
    if not encoded:
        print(json.dumps({"error": "no masked SFT rows", "reject": reject}), file=sys.stderr)
        return 1
    valid_encoded: list[dict] = []
    if args.valid and not args.smoke and not args.overfit_gate:
        vpath = args.valid if args.valid.is_absolute() else ROOT / args.valid
        if vpath.is_file():
            valid_encoded, _, _ = encode_rows(tok, load_pack(vpath, None), seq, inject)
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    loaded = 0
    if args.init_ckpt:
        ckpt = args.init_ckpt if args.init_ckpt.is_absolute() else ROOT / args.init_ckpt
        loaded = load_params(model, ckpt, strict=not args.smoke)
    run_name = args.run_name or f"mei-1.0-51m-sft-{pack.stem}"
    out_dir = TRAIN_RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / f"{run_name}.npz"
    best_path = ckpt_out.with_name(ckpt_out.stem + "-best.npz")
    latest_path = ckpt_out.with_name(ckpt_out.stem + "-latest.npz")
    state_path = ckpt_out.with_name(ckpt_out.stem + "-state.npz")

    if args.overfit_gate:
        result = train_lm_steps(
            model, encoded, steps=min(120, max(40, len(encoded) * 8)), lr=args.lr, seed=args.seed, conf_weight=0.2
        )
        acc = teacher_acc(model, encoded, n=len(encoded))
        meta = {"overfit_gate": True, "teacher_acc": acc, "last_loss": result["last_loss"], "ok": acc >= 0.99}
        (out_dir / "overfit.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return 0 if meta["ok"] else 3

    if args.smoke or args.steps:
        steps = 6 if args.smoke else int(args.steps)
        result = train_lm_steps(
            model,
            encoded,
            steps=steps,
            lr=args.lr,
            seed=args.seed,
            conf_weight=0.2,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        save_params(model, ckpt_out)
        best_step = result["steps"]
        early_reason = "fixed_steps"
        last_loss = result["last_loss"]
        tokens_seen = result["tokens_seen"]
        opt = result["optimizer"]
        history = []
    else:
        history = []
        best_nll = float("inf")
        patience = 0
        tokens_seen = 0
        last_loss = None
        opt = None
        best_step = 0
        early_reason = "max_epochs"
        epoch_steps = max(1, (len(encoded) + args.batch_size - 1) // args.batch_size)
        for epoch in range(int(args.max_epochs)):
            result = train_lm_steps(
                model,
                encoded,
                steps=epoch_steps,
                lr=args.lr,
                seed=args.seed + epoch,
                conf_weight=0.2,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                allow_repeat=True,
                optimizer=opt,
            )
            opt = result["optimizer"]
            last_loss = result["last_loss"]
            tokens_seen = int(result["tokens_seen"])
            val_src = valid_encoded or encoded[: min(32, len(encoded))]
            nll = eval_lm_loss(model, val_src, conf_weight=0.2, max_windows=len(val_src), token_weighted=True)
            acc = teacher_acc(model, val_src, n=min(16, len(val_src)))
            rec = {"epoch": epoch + 1, "train_loss": last_loss, "valid_nll": nll, "teacher_acc": acc, "tokens_seen": tokens_seen}
            history.append(rec)
            save_params(model, latest_path)
            save_train_state(state_path, model, opt, {"epoch": epoch + 1, "tokens_seen": tokens_seen, "seq_len": seq})
            if nll < best_nll - 1e-4:
                best_nll = nll
                patience = 0
                best_step = epoch + 1
                save_params(model, best_path)
            else:
                patience += 1
                if patience >= args.patience:
                    early_reason = "early_stop"
                    break
        if best_path.is_file():
            load_params(model, best_path, strict=True)
        save_params(model, ckpt_out)

    recipe_path = (args.recipe if args.recipe.is_absolute() else ROOT / args.recipe) if args.recipe else None
    lock_path = (args.eval_lock if args.eval_lock.is_absolute() else ROOT / args.eval_lock) if args.eval_lock else None
    meta = {
        "model_family": "mei-1.0-51m",
        "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
        "base_id": "mei-1.0-51m-base-cpt300m-v1" if args.init_ckpt else "random-init-51m",
        "pack": str(pack.relative_to(ROOT)),
        "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        "recipe_hash": hashlib.sha256(recipe_path.read_bytes()).hexdigest() if recipe_path and recipe_path.is_file() else "",
        "eval_lock_hash": hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path and lock_path.is_file() else "",
        "serializer": ROUTE_SERIALIZER_ID,
        "protocol": "mei-route-protocol-v1",
        "seq_len": seq,
        "seq_len_cap": cap,
        "n_rows": len(rows),
        "n_encoded": len(encoded),
        "n_valid": len(valid_encoded),
        "reject": reject,
        "lr": args.lr,
        "seed": args.seed,
        "last_loss": last_loss,
        "tokens_seen": tokens_seen,
        "params": count_params(model),
        "loaded_tensors": loaded,
        "tokenizer_sha256": tok.model_sha256,
        "ckpt": str(ckpt_out.relative_to(ROOT)),
        "best_ckpt": str(best_path.relative_to(ROOT)) if best_path.is_file() else None,
        "best_step": best_step,
        "early_stop_reason": early_reason,
        "history": history,
        "assistant_only_mask": True,
        "schema_conditioned": True,
    }
    if not args.smoke:
        save_train_state(state_path, model, opt or result["optimizer"], meta)
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
