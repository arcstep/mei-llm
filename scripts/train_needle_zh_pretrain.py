#!/usr/bin/env python3
"""Pretrain Needle-zh. --smoke is tiny/step-limited; pilots use full spec + token budget."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_PRETRAIN_PROBES,
    CORPUS_ZH_PRETRAIN,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from architecture import NeedleZh, count_params  # noqa: E402
from checkpoint import load_train_state, save_params, save_train_state  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from data import (  # noqa: E402
    PackedTokenSource,
    file_sha256,
    iter_packed_windows,
    iter_split_documents,
    list_pretrain_shards,
    list_token_shards,
    refuse_if_short,
)
from eval_needle_pretrain_probes import eval_probes, load_probes  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from train_common import eval_lm_loss, peak_bytes, segment_throughput, train_lm_steps  # noqa: E402

DEFAULTS = {
    "pilot-1m": {"target": 1_000_000, "horizon": 5_000_000, "eval_every": 250_000, "save_every": 250_000},
    "pilot-5m": {"target": 5_000_000, "horizon": 5_000_000, "eval_every": 500_000, "save_every": 500_000},
    "100m": {"target": 100_000_000, "horizon": 100_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
    "300m": {"target": 300_000_000, "horizon": 300_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
    "1b": {"target": 1_000_000_000, "horizon": 1_000_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
}


def collect_windows(shards, tok, seq: int, split: str, *, limit: int | None) -> list[dict]:
    out: list[dict] = []
    docs = iter_split_documents(shards, tok, split=split)
    for win in iter_packed_windows(docs, seq, tok.pad_id):
        out.append(win)
        if limit is not None and len(out) >= limit:
            break
    return out


def finite(x) -> bool:
    try:
        v = float(x)
        return v == v and abs(v) != float("inf")
    except (TypeError, ValueError):
        return False


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slim_probes(rep: dict | None) -> dict | None:
    if not rep:
        return None
    return {k: rep[k] for k in ("n", "mean_nll", "family_nll", "exact_rate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="100m", choices=["pilot-1m", "pilot-5m", "100m", "300m", "1b"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-params", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--target-tokens", type=int, default=None)
    ap.add_argument("--lr-horizon-tokens", type=int, default=None)
    ap.add_argument("--eval-every-tokens", type=int, default=None)
    ap.add_argument("--save-every-tokens", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--stop-at-tokens", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.count_params:
        cfg = NeedleZhConfig.from_spec()
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        n = count_params(model)
        print(json.dumps({"full_params": n, "in_band": 45_000_000 <= n <= 60_000_000}))
        return 0 if 45_000_000 <= n <= 60_000_000 else 1

    tok = ZhTokenizerV1()
    man_path = CORPUS_ZH_PRETRAIN / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    train_bins = list_token_shards(CORPUS_ZH_PRETRAIN, "train")
    valid_bins = list_token_shards(CORPUS_ZH_PRETRAIN, "valid")
    use_mmap = bool(train_bins) and not args.smoke
    shards = list_pretrain_shards(CORPUS_ZH_PRETRAIN, smoke=args.smoke)
    if not use_mmap and not shards:
        print(
            f"missing shards under {CORPUS_ZH_PRETRAIN / 'shards'} or tokens/; run build_zh_pretrain_v0.py",
            file=sys.stderr,
        )
        return 1
    hash_files = train_bins + valid_bins if use_mmap else shards
    corpus_hash = "".join(file_sha256(p) for p in hash_files)
    if man_path.is_file():
        manifest_sha = file_sha256(man_path)
        corpus_hash = corpus_hash + manifest_sha
    else:
        manifest_sha = ""
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(f"vocab mismatch cfg={cfg.vocab_size} tok={tok.vocab_size}")

    seq = args.seq_len or (32 if args.smoke else 256)
    batch_size = args.batch_size or (1 if args.smoke else 2)
    grad_accum = args.grad_accum or (1 if args.smoke else 4)
    dft = DEFAULTS[args.rung]
    target = args.target_tokens if args.target_tokens is not None else (None if args.smoke else int(dft["target"]))
    horizon_tokens = args.lr_horizon_tokens or (None if args.smoke else int(dft["horizon"]))
    eval_every = args.eval_every_tokens or (10**18 if args.smoke else int(dft["eval_every"]))
    save_every = args.save_every_tokens or (10**18 if args.smoke else int(dft["save_every"]))

    if args.stop_at_tokens is not None:
        if target is None or int(args.stop_at_tokens) < int(target):
            target = int(args.stop_at_tokens)

    if use_mmap:
        train = PackedTokenSource(train_bins, seq, tok.pad_id)
        valid = PackedTokenSource(valid_bins, seq, tok.pad_id)[:128] if valid_bins else []
        packed_tokens = int(train.n_tokens)
        if not args.allow_repeat:
            pred = int(train.n_predictable_tokens)
            if target is None or int(target) > pred:
                target = pred
    else:
        train_limit = int(target / max(seq, 1)) + 32 if target else (64 if args.smoke else None)
        train = collect_windows(shards, tok, seq, "train", limit=train_limit)
        valid = collect_windows(shards, tok, seq, "valid", limit=128)
        packed_tokens = len(train) * seq
    if not train:
        print("no train windows", file=sys.stderr)
        return 1
    corpus_tokens = int(man.get("n_train_tokens") or packed_tokens)
    try:
        refuse_if_short(
            "100m" if args.smoke else args.rung,
            corpus_tokens,
            smoke=args.smoke,
            cap=int(target) if target else None,
        )
    except ValueError as exc:
        if args.allow_repeat and str(args.rung).startswith("pilot-"):
            print(json.dumps({"warn": str(exc), "allow_repeat": True}))
        elif args.smoke:
            pass
        else:
            print(str(exc), file=sys.stderr)
            return 2
    if not args.smoke and target and packed_tokens < target and not args.allow_repeat:
        print(
            f"packed windows cover ~{packed_tokens} tokens < target {target}; pass --allow-repeat to multi-epoch",
            file=sys.stderr,
        )
        return 2

    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    steps = args.steps or (8 if args.smoke else None)
    start_step = 0
    start_tokens = 0
    start_window = 0
    opt = optim.Adam(learning_rate=args.lr)
    toks_per = max(1, batch_size * grad_accum * seq)
    total_steps = None
    if horizon_tokens:
        total_steps = max(1, (int(horizon_tokens) + toks_per - 1) // toks_per)
    expected = {
        "tokenizer_sha256": tok.model_sha256,
        "corpus_sha256": corpus_hash,
        "seq_len": seq,
        "manifest_sha256": manifest_sha,
        "lr_horizon_tokens": horizon_tokens,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "precision": args.precision,
        "shuffle_seed": 0,
    }
    if args.resume:
        opt = optim.Adam(learning_rate=args.lr)
        prev = load_train_state(args.resume, model, opt, strict=True, expected_meta=expected)
        start_step = int(prev.get("step") or 0)
        start_tokens = int(prev.get("tokens_seen") or 0)
        start_window = int(prev.get("window_index") or 0)
        total_steps = int(prev.get("total_steps") or total_steps or 1)

    run_name = "smoke" if args.smoke else args.rung
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-pretrain-{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    ckpt_dir = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"pretrain-{run_name}.npz"
    last_state = ckpt_dir / f"pretrain-{run_name}-state.npz"
    best_state = ckpt_dir / f"pretrain-{run_name}-best-state.npz"
    probes = load_probes(BANK_NEEDLE_PRETRAIN_PROBES) if BANK_NEEDLE_PRETRAIN_PROBES.is_file() else []

    state = {
        "steps": 0,
        "tokens_seen": start_tokens,
        "window_index": start_window,
        "total_steps": total_steps,
        "last_loss": None,
        "optimizer": opt,
    }
    best_valid = None
    last_eval_at = start_tokens
    last_save_at = start_tokens
    saved_milestones: set[int] = set()
    t0 = time.time()

    def copy_milestone(tokens_seen: int) -> None:
        for mark in (100_000_000, 300_000_000):
            if tokens_seen >= mark and mark not in saved_milestones:
                tag = f"{mark // 1_000_000}m"
                shutil.copy2(last_state, ckpt_dir / f"pretrain-{run_name}-{tag}-state.npz")
                meta_src = last_state.with_suffix(".meta.json")
                if meta_src.is_file():
                    shutil.copy2(meta_src, ckpt_dir / f"pretrain-{run_name}-{tag}-state.meta.json")
                saved_milestones.add(mark)

    def write_ckpt(tag: str, valid_loss, probe_rep, *, is_best: bool) -> dict:
        meta = {
            "rung": run_name,
            "requested_rung": args.rung,
            "smoke": args.smoke,
            "step": start_step + state["steps"],
            "tokens_seen": state["tokens_seen"],
            "window_index": state["window_index"],
            "seq_len": seq,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "target_tokens": target,
            "total_steps": state["total_steps"],
            "params": count_params(model),
            "tokenizer_sha256": tok.model_sha256,
            "corpus_sha256": corpus_hash,
            "manifest_sha256": manifest_sha,
            "tiny": args.smoke,
            "last_loss": state["last_loss"],
            "valid_loss": valid_loss,
            "probe_mean_nll": None if probe_rep is None else probe_rep.get("mean_nll"),
            "tag": tag,
            "epoch": int(state["window_index"]) // max(len(train), 1),
            "shuffle_seed": 0,
            "lr_horizon_tokens": horizon_tokens,
            "data_cursor": state["window_index"],
            "precision": args.precision,
            "compile_train": bool(args.compile),
            "allow_repeat": bool(args.allow_repeat),
            "mmap": use_mmap,
        }
        save_params(model, last_ckpt)
        save_train_state(last_state, model, state["optimizer"], meta)
        if is_best:
            save_params(model, ckpt_dir / f"pretrain-{run_name}-best.npz")
            save_train_state(best_state, model, state["optimizer"], meta)
        copy_milestone(int(state["tokens_seen"]))
        (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta

    def maybe_eval(tokens_seen: int, force: bool = False):
        nonlocal best_valid, last_eval_at, last_save_at
        due_eval = force or (tokens_seen - last_eval_at >= eval_every)
        due_save = force or (tokens_seen - last_save_at >= save_every)
        if not due_eval and not due_save:
            return None, None
        valid_loss = eval_lm_loss(model, valid, batch_size=min(batch_size, 4)) if valid else None
        probe_rep = eval_probes(model, tok, probes) if probes and due_eval else None
        if probe_rep is not None:
            (out_dir / "probes.json").write_text(
                json.dumps(slim_probes(probe_rep), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if due_eval:
            last_eval_at = tokens_seen
        is_best = valid_loss is not None and (best_valid is None or valid_loss < best_valid)
        if is_best:
            best_valid = valid_loss
        if due_save or is_best or force:
            write_ckpt("eval" if not force else "final", valid_loss, probe_rep, is_best=bool(is_best))
            last_save_at = tokens_seen
        return valid_loss, probe_rep

    def on_step(step: int, info: dict) -> None:
        state["steps"] = step - start_step + 1
        state["tokens_seen"] = info["tokens_seen"]
        state["window_index"] = info["window_index"]
        state["last_loss"] = info["loss"]
        elapsed = max(1e-6, time.time() - t0)
        if not finite(info["loss"]) or not finite(info["grad_norm"]):
            raise RuntimeError(f"non-finite loss/grad at step {step}: {info}")
        valid_loss, probe_rep = maybe_eval(info["tokens_seen"])
        seg_tok = int(info["tokens_seen"]) - int(start_tokens)
        tok_s = segment_throughput(info["tokens_seen"], start_tokens, elapsed)
        row = {
            "step": step,
            "tokens_seen": info["tokens_seen"],
            "loss": info["loss"],
            "grad_norm": info["grad_norm"],
            "lr": info["lr"],
            "tok_s": tok_s,
            "segment_tokens": seg_tok,
            "segment_tok_s": tok_s,
            "peak_bytes": info.get("peak_bytes") or peak_bytes(),
        }
        if valid_loss is not None:
            row["valid_loss"] = valid_loss
        if probe_rep:
            row["probe_mean_nll"] = probe_rep.get("mean_nll")
            row["probe_family_nll"] = probe_rep.get("family_nll")
        append_jsonl(metrics_path, row)

    probes_init = eval_probes(model, tok, probes) if probes and not args.smoke and args.resume is None else None
    if probes_init:
        (out_dir / "probes-init.json").write_text(
            json.dumps(slim_probes(probes_init), indent=2) + "\n", encoding="utf-8"
        )

    result = train_lm_steps(
        model,
        train,
        steps=steps,
        lr=args.lr,
        seed=0,
        start_step=start_step,
        start_tokens_seen=start_tokens,
        total_steps=total_steps,
        optimizer=opt,
        reseed=args.resume is None,
        on_step=on_step,
        batch_size=batch_size,
        grad_accum=grad_accum,
        target_tokens=None if args.smoke else target,
        start_window=start_window,
        allow_repeat=args.allow_repeat,
        compile_train=args.compile,
        precision=args.precision,
    )
    state.update(result)
    valid_loss, probe_rep = maybe_eval(result["tokens_seen"], force=True)
    meta = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    meta["valid_loss"] = valid_loss
    meta["init_probe_mean_nll"] = None if probes_init is None else probes_init.get("mean_nll")
    meta["final_probe_mean_nll"] = None if probe_rep is None else probe_rep.get("mean_nll")
    elapsed = time.time() - t0
    meta["elapsed_s"] = elapsed
    meta["ckpt"] = str(last_ckpt.relative_to(ROOT))
    meta["n_source_tokens"] = int(getattr(train, "n_tokens", packed_tokens))
    meta["n_predictable_tokens"] = int(
        getattr(train, "n_predictable_tokens", max(0, packed_tokens - 1))
    )
    meta["n_windows"] = len(train)
    meta["exhausted"] = bool(result.get("exhausted"))
    meta["segment_tokens"] = int(result["tokens_seen"]) - int(start_tokens)
    meta["segment_tok_s"] = segment_throughput(result["tokens_seen"], start_tokens, elapsed)
    meta["window_index"] = int(result.get("window_index") or meta.get("window_index") or 0)
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
