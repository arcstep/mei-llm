#!/usr/bin/env python3
"""Seq-len throughput sweep for Needle-zh / mei-1.0-58m pretrain.

Objective is maximum tokens/s, not GPU or memory occupancy. A layout that
uses more RAM but fewer tok/s is a failure. Floor: do not regress the 300M
run (~7270 tok/s) or the ~8k tok/s target.

Two phases per window:
  1. Fair layouts at a fixed 2048 tokens/optimizer update.
  2. Throughput search: raise batch_size only while tok/s keeps improving.

Uses a small mmap slice of zh-pretrain-v2 and optional 300M weights-only init.
Does not write checkpoints or mutate corpus releases.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

from repo_paths import (
    CORPUS_ZH_PRETRAIN_V2,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from architecture import NeedleZh, count_params  # noqa: E402
from checkpoint import load_params, load_train_state  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from data import PackedTokenSource, list_token_shards, pack_windows  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from train_common import peak_bytes, train_lm_steps  # noqa: E402

TOKENS_PER_UPDATE = 2048
DEFAULT_SEQ_LENS = (256, 512, 1024, 2048)
LAYOUTS = {
    256: ((8, 1), (4, 2), (2, 4), (1, 8)),
    512: ((4, 1), (2, 2), (1, 4)),
    1024: ((2, 1), (1, 2)),
    2048: ((1, 1),),
}
DEFAULT_INIT = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-300m.npz"
PARENT_TOKENS = 300_000_000
TARGET_1B = 1_000_000_000
REMAINING_TO_1B = TARGET_1B - PARENT_TOKENS
ACTUAL_300M_ELAPSED_S = 41256.37611222267
ACTUAL_300M_TOKENS = 300_001_280
ACTUAL_300M_TOK_S = round(ACTUAL_300M_TOKENS / ACTUAL_300M_ELAPSED_S, 1)
HOST_GPU_CORES = 40
HOST_UNIFIED_GIB = 128
FILL_IMPROVE_RATIO = 1.01
SEARCH_MAX_BATCH = {256: 16, 512: 8, 1024: 8, 2048: 2}
FLOOR_TOK_S_300M = ACTUAL_300M_TOK_S
FLOOR_TOK_S_TARGET = 8000.0
WINDOWS_PER_SEQ = 256


def parse_seq_lens(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"seq_len must be >= 1, got {n}")
        out.append(n)
    if not out:
        raise ValueError("no seq lens")
    return out


def layouts_for(seq_len: int) -> tuple[tuple[int, int], ...]:
    if seq_len in LAYOUTS:
        return LAYOUTS[seq_len]
    if TOKENS_PER_UPDATE % seq_len != 0:
        raise ValueError(
            f"seq_len {seq_len} does not divide tokens_per_update={TOKENS_PER_UPDATE}"
        )
    n = TOKENS_PER_UPDATE // seq_len
    layouts: list[tuple[int, int]] = []
    bs = n
    while bs >= 1:
        if n % bs == 0:
            layouts.append((bs, n // bs))
        bs //= 2
    if not layouts:
        layouts.append((1, n))
    return tuple(layouts)


def search_batches(seq_len: int, max_batch: int) -> list[int]:
    """Modest batch increases after the fair accum=1 layout. Stop is tok/s, not RAM."""
    fair = max(1, TOKENS_PER_UPDATE // max(seq_len, 1))
    cap = min(int(max_batch), int(SEARCH_MAX_BATCH.get(seq_len, max_batch)))
    out: list[int] = []
    bs = fair + 2
    while bs <= cap:
        out.append(bs)
        bs += 2
    doubled = fair * 2
    if fair < doubled <= cap and doubled not in out:
        out.append(doubled)
        out = sorted(set(out))
    return out


def peak_gib(peak: int | None) -> float | None:
    if peak is None:
        return None
    return round(int(peak) / (1024**3), 2)


def host_unified_gib() -> int:
    try:
        import subprocess

        raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return max(1, int(int(raw) / (1024**3)))
    except Exception:
        return HOST_UNIFIED_GIB


def peak_frac(peak: int | None, unified_gib: int = HOST_UNIFIED_GIB) -> float | None:
    if peak is None or unified_gib <= 0:
        return None
    return round(int(peak) / (unified_gib * 1024**3), 3)


def hours_for(tokens: int, tok_s: float) -> float | None:
    if tok_s <= 0:
        return None
    return round(float(tokens) / tok_s / 3600.0, 3)


def finite_loss(last) -> bool:
    if last is None:
        return False
    val = float(last)
    return val == val and abs(val) != float("inf")


def load_init_weights(model, path: Path) -> str:
    blob = mx.load(str(path))
    if any(str(k).startswith("p.") for k in blob):
        opt = optim.Adam(learning_rate=3e-4)
        load_train_state(path, model, opt, mode="weights_only")
        return "train_state_weights_only"
    load_params(model, path)
    return "params"


def release_model(model) -> None:
    del model
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def load_windows(corpus_dir: Path, seq_len: int, n_windows: int, pad_id: int) -> list[dict]:
    bins = list_token_shards(corpus_dir, "train")
    if bins:
        src = PackedTokenSource(bins, seq_len, pad_id)
        take = min(n_windows, len(src))
        if take < 1:
            raise ValueError(f"no packed windows at seq_len={seq_len}")
        return src[:take]
    print("missing token shards; using synthetic ids", file=sys.stderr)
    ids = ([2, 11, 12, 13, 14, 1] * max(8, seq_len))[: seq_len * n_windows + 1]
    return pack_windows([ids], seq_len, pad_id=pad_id)[:n_windows]


def run_one(
    windows: list[dict],
    *,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    warmup_steps: int,
    measure_steps: int,
    precision: str,
    init_ckpt: Path | None,
    lr: float,
    phase: str,
) -> dict:
    cfg = NeedleZhConfig.from_spec()
    mx.random.seed(0)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    init_mode = "scratch"
    if init_ckpt is not None:
        init_mode = load_init_weights(model, init_ckpt)
    opt = optim.Adam(learning_rate=lr)
    if not windows:
        raise ValueError("no windows")
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    warm = train_lm_steps(
        model,
        windows,
        steps=warmup_steps,
        lr=lr,
        seed=0,
        batch_size=batch_size,
        grad_accum=grad_accum,
        compile_train=False,
        precision=precision,
        allow_repeat=True,
        optimizer=opt,
        reseed=True,
    )
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    t0 = time.time()
    result = train_lm_steps(
        model,
        windows,
        steps=measure_steps,
        lr=lr,
        seed=0,
        start_step=int(warm["steps"]),
        start_tokens_seen=0,
        start_window=int(warm["window_index"]),
        batch_size=batch_size,
        grad_accum=grad_accum,
        compile_train=False,
        precision=precision,
        allow_repeat=True,
        optimizer=warm["optimizer"],
        reseed=False,
    )
    elapsed = max(1e-6, time.time() - t0)
    tokens = int(result["tokens_seen"])
    last = result["last_loss"]
    peak = peak_bytes()
    row = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_update": batch_size * grad_accum * seq_len,
        "compile": False,
        "precision": precision,
        "init_mode": init_mode,
        "warmup_steps": int(warm["steps"]),
        "steps": result["steps"],
        "tokens_seen": tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(tokens / elapsed, 1),
        "update_ms": round(1000.0 * elapsed / max(result["steps"], 1), 2),
        "last_loss": None if last is None else float(last),
        "finite": finite_loss(last),
        "peak_bytes": peak,
        "peak_gib": peak_gib(peak),
        "peak_frac_of_128gib": peak_frac(peak),
        "params": count_params(model),
        "phase": phase,
        "oom": False,
    }
    release_model(model)
    return row


def error_row(
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    precision: str,
    exc: BaseException,
    *,
    phase: str,
) -> dict:
    msg = str(exc).lower()
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_update": batch_size * grad_accum * seq_len,
        "compile": False,
        "precision": precision,
        "finite": False,
        "tok_s": 0.0,
        "phase": phase,
        "oom": (
            "out of memory" in msg
            or "oom" in type(exc).__name__.lower()
            or ("metal" in msg and "alloc" in msg)
        ),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=8),
    }


def eta_block(tok_s: float) -> dict:
    return {
        "tok_s": tok_s,
        "hours_300m_from_scratch": hours_for(PARENT_TOKENS, tok_s),
        "hours_remaining_700m": hours_for(REMAINING_TO_1B, tok_s),
        "hours_1b_from_scratch": hours_for(TARGET_1B, tok_s),
        "hours_1b_from_300m": hours_for(REMAINING_TO_1B, tok_s),
    }


def best_of(rows: list[dict]) -> dict | None:
    finite = [r for r in rows if r.get("finite") and r.get("tok_s")]
    if not finite:
        return None
    return max(finite, key=lambda r: r["tok_s"])


def summarize(seq_lens: list[int], rows: list[dict]) -> dict:
    by_len: dict[int, dict] = {}
    for seq in seq_lens:
        group = [r for r in rows if r.get("seq_len") == seq]
        fair = [r for r in group if r.get("phase") == "fair_2048"]
        search = [r for r in group if r.get("phase") == "tok_s_search"]
        best_fair = best_of(fair)
        best_search = best_of(search)
        best = best_of(group)
        by_len[seq] = {
            "seq_len": seq,
            "n_tried": len(group),
            "n_ok": len([r for r in group if r.get("finite")]),
            "best_fair_2048": best_fair,
            "best_tok_s_search": best_search,
            "best": best,
            "eta_fair": eta_block(float(best_fair["tok_s"])) if best_fair else None,
            "eta": eta_block(float(best["tok_s"])) if best else None,
            "rows": group,
        }
    base = by_len.get(256, {}).get("best")
    base_tok = float(base["tok_s"]) if base else None
    for seq, block in by_len.items():
        best = block.get("best")
        if not best or not base_tok:
            block["vs_256_tok_s"] = None
            block["vs_256_hours"] = None
            continue
        speed = float(best["tok_s"]) / base_tok
        block["vs_256_tok_s"] = round(speed, 3)
        block["vs_256_hours"] = round(1.0 / speed, 3) if speed > 0 else None
    return by_len


def emit(row: dict) -> None:
    print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)


def try_run(
    windows: list[dict],
    *,
    seq: int,
    batch_size: int,
    grad_accum: int,
    phase: str,
    args,
    init_ckpt: Path | None,
) -> dict:
    try:
        return run_one(
            windows,
            seq_len=seq,
            batch_size=batch_size,
            grad_accum=grad_accum,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            precision=args.precision,
            init_ckpt=init_ckpt,
            lr=args.lr,
            phase=phase,
        )
    except Exception as exc:
        release_model(None)
        return error_row(seq, batch_size, grad_accum, args.precision, exc, phase=phase)


def table_row(seq: int, block: dict, *, key: str = "best") -> dict:
    best = block.get(key)
    eta = block.get("eta") if key == "best" else block.get("eta_fair")
    return {
        "seq_len": seq,
        "ok": bool(best),
        "phase": None if best is None else best.get("phase"),
        "best_layout": None if best is None else f"{best['batch_size']}x{best['grad_accum']}",
        "tokens_per_update": None if best is None else best.get("tokens_per_update"),
        "tok_s": None if best is None else best["tok_s"],
        "peak_gib": None if best is None else best.get("peak_gib"),
        "peak_frac_of_128gib": None if best is None else best.get("peak_frac_of_128gib"),
        "hours_remaining_700m": None if eta is None else eta["hours_remaining_700m"],
        "hours_1b_from_scratch": None if eta is None else eta["hours_1b_from_scratch"],
        "vs_256_hours": block.get("vs_256_hours") if key == "best" else None,
        "error": None if best else next((r.get("error") for r in block["rows"] if r.get("error")), None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default=",".join(str(x) for x in DEFAULT_SEQ_LENS))
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_ZH_PRETRAIN_V2)
    ap.add_argument("--init-weights", type=Path, default=DEFAULT_INIT)
    ap.add_argument("--no-init-weights", action="store_true")
    ap.add_argument("--no-search", action="store_true", help="skip modest batch search past the fair 2048-token layout")
    ap.add_argument("--max-batch", type=int, default=16, help="cap for tok/s search batches")
    ap.add_argument("--confirm-steps", type=int, default=40, help="longer measure on the winning 256 layout")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENTS_RUNS / "needle-zh-pretrain-throughput-seqlen",
    )
    args = ap.parse_args()
    seq_lens = parse_seq_lens(args.seq_lens)
    tok = ZhTokenizerV1()
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    init_ckpt = None if args.no_init_weights else args.init_weights
    if init_ckpt is not None and not init_ckpt.is_absolute():
        init_ckpt = ROOT / init_ckpt
    if init_ckpt is not None and not init_ckpt.is_file():
        print(f"missing init weights {init_ckpt}; falling back to scratch", file=sys.stderr)
        init_ckpt = None

    dest = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest / "rows.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    def record(row: dict) -> None:
        rows.append(row)
        emit(row)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in row.items() if k != "traceback"}) + "\n")

    rows: list[dict] = []
    window_cache: dict[int, list] = {}
    for seq in seq_lens:
        layouts = layouts_for(seq)
        search = [] if args.no_search else search_batches(seq, args.max_batch)
        try:
            windows = load_windows(corpus_dir, seq, WINDOWS_PER_SEQ, tok.pad_id)
        except Exception as exc:
            record(error_row(seq, layouts[0][0], layouts[0][1], args.precision, exc, phase="fair_2048"))
            continue
        window_cache[seq] = windows
        last_ok: dict | None = None
        for bs, accum in layouts:
            if bs * accum * seq != TOKENS_PER_UPDATE:
                record(
                    {
                        "seq_len": seq,
                        "batch_size": bs,
                        "grad_accum": accum,
                        "phase": "fair_2048",
                        "finite": False,
                        "tok_s": 0.0,
                        "error": f"layout does not yield {TOKENS_PER_UPDATE} tokens/update",
                    }
                )
                continue
            row = try_run(
                windows,
                seq=seq,
                batch_size=bs,
                grad_accum=accum,
                phase="fair_2048",
                args=args,
                init_ckpt=init_ckpt,
            )
            record(row)
            if row.get("finite"):
                last_ok = row

        ok_tok = [
            float(r["tok_s"])
            for r in rows
            if r.get("seq_len") == seq and r.get("finite") and r.get("tok_s")
        ]
        best_tok = max(ok_tok) if ok_tok else 0.0
        for bs in search:
            row = try_run(
                windows,
                seq=seq,
                batch_size=bs,
                grad_accum=1,
                phase="tok_s_search",
                args=args,
                init_ckpt=init_ckpt,
            )
            record(row)
            if row.get("oom") or not row.get("finite"):
                break
            tok_s = float(row.get("tok_s") or 0.0)
            if tok_s < best_tok * FILL_IMPROVE_RATIO:
                break
            best_tok = max(best_tok, tok_s)

    confirm = None
    rec_256_pre = best_of([r for r in rows if r.get("seq_len") == 256])
    if rec_256_pre and args.confirm_steps > 0 and 256 in seq_lens:
        windows = window_cache.get(256) or load_windows(corpus_dir, 256, WINDOWS_PER_SEQ, tok.pad_id)
        confirm = try_run(
            windows,
            seq=256,
            batch_size=int(rec_256_pre["batch_size"]),
            grad_accum=int(rec_256_pre["grad_accum"]),
            phase="confirm",
            args=argparse.Namespace(
                **{
                    **vars(args),
                    "measure_steps": args.confirm_steps,
                    "warmup_steps": max(args.warmup_steps, 5),
                }
            ),
            init_ckpt=init_ckpt,
        )
        confirm["phase"] = "confirm"
        record(confirm)

    by_len = summarize(seq_lens, rows)
    best_ok = [block["best"] for block in by_len.values() if block.get("best")]
    rec_256 = by_len.get(256, {}).get("best")
    rec_eta_src = confirm if confirm and confirm.get("finite") else rec_256
    rec_eta = eta_block(float(rec_eta_src["tok_s"])) if rec_eta_src else None
    rec_fair = by_len.get(256, {}).get("best_fair_2048")
    unified = host_unified_gib()
    out = {
        "tokens_per_update_fair": TOKENS_PER_UPDATE,
        "host": {
            "gpu_cores": HOST_GPU_CORES,
            "unified_gib": unified,
            "assumed_gpu_cores": HOST_GPU_CORES,
            "note": "Apple Silicon unified memory; 40 GPU cores / 128 GiB class (request 128M treated as 128 GiB)",
        },
        "seq_lens": seq_lens,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "precision": args.precision,
        "compile": False,
        "fill_gpu": False,
        "tok_s_search": not args.no_search,
        "max_batch": args.max_batch,
        "corpus_dir": str(corpus_dir),
        "init_weights": None if init_ckpt is None else str(init_ckpt),
        "n_windows": {str(k): len(v) for k, v in window_cache.items()},
        "actual_300m": {
            "tokens": ACTUAL_300M_TOKENS,
            "elapsed_s": ACTUAL_300M_ELAPSED_S,
            "tok_s": ACTUAL_300M_TOK_S,
            "seq_len": 256,
            "batch_size": 8,
            "grad_accum": 1,
            "note": "300M used seq=256 batch=8; this is the tok/s floor to beat or match",
        },
        "rows": rows,
        "by_seq_len": {str(k): v for k, v in by_len.items()},
        "recommended_for_1b_cpt": {
            "seq_len": 256 if rec_256 else None,
            "layout": None
            if rec_256 is None
            else {
                "batch_size": rec_256["batch_size"],
                "grad_accum": rec_256["grad_accum"],
                "tokens_per_update": rec_256.get("tokens_per_update"),
                "phase": rec_256.get("phase"),
            },
            "eta": rec_eta,
            "fair_2048_256": None
            if rec_fair is None
            else {
                "batch_size": rec_fair["batch_size"],
                "grad_accum": rec_fair["grad_accum"],
                "tok_s": rec_fair["tok_s"],
                "eta": eta_block(float(rec_fair["tok_s"])),
            },
            "confirm": None
            if confirm is None
            else {
                "batch_size": confirm.get("batch_size"),
                "grad_accum": confirm.get("grad_accum"),
                "tok_s": confirm.get("tok_s"),
                "eta": eta_block(float(confirm["tok_s"])) if confirm.get("finite") else None,
            },
            "vs_300m_tok_s": None
            if rec_eta_src is None
            else round(float(rec_eta_src["tok_s"]) / FLOOR_TOK_S_300M, 3),
            "vs_8k_tok_s": None
            if rec_eta_src is None
            else round(float(rec_eta_src["tok_s"]) / FLOOR_TOK_S_TARGET, 3),
            "meets_8k_floor": None
            if rec_eta_src is None
            else bool(float(rec_eta_src["tok_s"]) >= FLOOR_TOK_S_TARGET),
            "note": (
                "Choose the 256 layout with the highest tok/s for remaining 700M CPT. "
                "Larger RAM use is only justified if tok/s rises. Longer windows are slower "
                "and are for later context extension, not the 1B unique-token target."
            ),
        },
        "best_ok": best_ok,
    }
    (dest / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    table = [table_row(seq, by_len[seq], key="best") for seq in seq_lens]
    table_fair = [table_row(seq, by_len[seq], key="best_fair_2048") for seq in seq_lens]
    eta_out = {
        "host": out["host"],
        "table_best": table,
        "table_fair_2048": table_fair,
        "recommended": out["recommended_for_1b_cpt"],
    }
    (dest / "eta.json").write_text(json.dumps(eta_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(dest), **eta_out}, indent=2))
    return 0 if rec_256 else 1


if __name__ == "__main__":
    raise SystemExit(main())
