#!/usr/bin/env python3
"""Phase-1 SFT with assistant-only LM loss. Smoke uses tiny student + 2k pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx

from architecture import NeedleZh, count_params
from checkpoint import load_params, save_params, save_train_state
from config import NeedleZhConfig
from data import encode_sft_row
from tokenizer import ZhTokenizerV1
from train_common import train_lm_steps

# confidence_label is P(execute_is_correct | query) for phase-1:
# 1 = gold execute, 0 = refuse (missing/conflict/illegal/offtopic). Not slot exact-match.


def load_pack(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    ap.add_argument("--tier", default="2k")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    pack = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / f"home-sft-{args.tier}.jsonl"
    if not pack.is_file():
        print(f"missing {pack}", file=sys.stderr)
        return 1
    tok = ZhTokenizerV1()
    rows = load_pack(pack, 32 if args.smoke else None)
    seq = 48 if args.smoke else 256
    encoded = [encode_sft_row(tok, r, seq) for r in rows]
    encoded = [e for e in encoded if e["n_unmasked"] > 0]
    if not encoded:
        print("no masked SFT rows", file=sys.stderr)
        return 1
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError("vocab mismatch")
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    loaded = 0
    if args.init == "pretrained":
        tag = "smoke" if args.smoke else "100m"
        ckpt = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / f"pretrain-{tag}.npz"
        if ckpt.is_file():
            loaded = load_params(model, ckpt, strict=True)
        else:
            print(json.dumps({"warn": "missing pretrained ckpt; training from random", "ckpt": str(ckpt)}))
    steps = args.steps or (6 if args.smoke else 100)
    result = train_lm_steps(
        model,
        encoded,
        steps=steps,
        lr=1e-4,
        seed=1,
        conf_weight=0.2,
    )
    tag = f"sft-{args.init}-{args.tier}{'-smoke' if args.smoke else ''}"
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / f"{tag}.npz"
    save_params(model, ckpt_out)
    meta = {
        "init": args.init,
        "tier": args.tier,
        "smoke": args.smoke,
        "n_rows": len(rows),
        "n_encoded": len(encoded),
        "steps": steps,
        "last_loss": result["last_loss"],
        "tokens_seen": result["tokens_seen"],
        "params": count_params(model),
        "loaded_tensors": loaded,
        "tokenizer_sha256": tok.model_sha256,
        "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        "ckpt": str(ckpt_out.relative_to(ROOT)),
        "byte_fallback": False,
        "assistant_only_mask": True,
        "confidence_meaning": "P(execute_is_correct|query); 1=gold execute, 0=refuse",
        "note": "Qwen 0.8B bypass is data-teachability only; not a 24k student proxy.",
    }
    save_train_state(ckpt_out.with_name(ckpt_out.stem + "-state.npz"), model, result["optimizer"], meta)
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
