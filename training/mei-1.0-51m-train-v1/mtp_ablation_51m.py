#!/usr/bin/env python3
"""Fixed-budget training-only MTP sidecar ablation for the final 51M LM.

The control and MTP branches see identical frozen hidden states, initialization,
windows, optimizer, and budget.  The only variable is the prediction offset:
next-token (control) versus token+2 (MTP).  Both temporary heads are discarded;
neither may enter the 51,463,797 deployment identity or package v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from _repo import ROOT, architecture_contracts


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.master.is_file():
        raise RuntimeError(f"missing final LM master: {args.master}")
    corpus_files = [
        args.replay_corpus / "manifest.json",
        args.replay_corpus / "schedule-scratch.json",
    ]
    if any(not path.is_file() for path in corpus_files):
        raise RuntimeError("MTP replay corpus metadata is incomplete")
    contracts = architecture_contracts()
    immutable = {
        "master_sha256": _sha_file(args.master),
        "weight_contract_sha256": contracts["weight_contract_sha256"],
        "training_aux_sha256": contracts["training_aux_sha256"],
        "corpus": {
            str(path.relative_to(ROOT)): _sha_file(path) for path in corpus_files
        },
        "recipe": {
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "hidden_dim": 128,
            "control_offset": 1,
            "mtp_offset": 2,
            "seed": 51,
        },
    }
    return {
        "schema": "mei-mtp-ablation-plan-v1",
        "stage_fingerprint_sha256": hashlib.sha256(_canonical(immutable)).hexdigest(),
        "immutable": immutable,
        "training_only": True,
        "export": False,
    }


def _masked_loss(logits, targets, mask):
    import mlx.core as mx
    import mlx.nn as nn

    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    weights = mask.astype(mx.float32)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1.0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    if args.dry_run:
        return {"ok": True, "dry_run": True, **plan}

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from architecture import NeedleZh, count_params
    from checkpoint import load_params
    from config import NeedleZhConfig
    from data import load_scheduled_train
    from tokenizer import ZhTokenizerV1
    from train_common import stack_windows

    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, args.master, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != 51_463_797:
        raise RuntimeError("MTP ablation LM identity changed")
    tokenizer = ZhTokenizerV1()
    source = load_scheduled_train(
        args.replay_corpus,
        args.seq_len,
        tokenizer.pad_id,
        schedule_path=args.replay_corpus / "schedule-scratch.json",
        curriculum_stage="s1",
        tokens_seen=0,
    )
    if source is None:
        raise RuntimeError("MTP ablation replay source is empty")

    class AuxiliaryHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(cfg.d_model, 128, bias=False)
            self.up = nn.Linear(128, cfg.vocab_size, bias=False)

        def __call__(self, hidden):
            return self.up(nn.silu(self.down(hidden)))

    mx.random.seed(51)
    control = AuxiliaryHead()
    mx.eval(control.parameters())
    mx.random.seed(51)
    mtp = AuxiliaryHead()
    mx.eval(mtp.parameters())
    control_opt = optim.Adam(learning_rate=args.lr)
    mtp_opt = optim.Adam(learning_rate=args.lr)
    control_losses: list[float] = []
    mtp_losses: list[float] = []
    started = time.time()
    tokens = 0
    for _step in range(args.steps):
        windows = source.take_windows(args.batch_size)
        if not windows:
            raise RuntimeError("MTP ablation source exhausted")
        batch = stack_windows(windows)
        out = model(batch["x"], return_cells=False)
        hidden = mx.stop_gradient(out["hidden"][:, :-1, :]).astype(mx.float16)
        control_target = batch["y"][:, :-1]
        mtp_target = batch["y"][:, 1:]
        mask = batch["mask"][:, :-1]

        def control_loss(candidate):
            return _masked_loss(candidate(hidden), control_target, mask)

        def mtp_loss(candidate):
            return _masked_loss(candidate(hidden), mtp_target, mask)

        c_loss, c_grad = mx.value_and_grad(control_loss)(control)
        m_loss, m_grad = mx.value_and_grad(mtp_loss)(mtp)
        control_opt.update(control, c_grad)
        mtp_opt.update(mtp, m_grad)
        mx.eval(control.parameters(), mtp.parameters(), c_loss, m_loss)
        c_value = float(c_loss.item())
        m_value = float(m_loss.item())
        if not math.isfinite(c_value) or not math.isfinite(m_value):
            raise RuntimeError("MTP ablation produced non-finite loss")
        control_losses.append(c_value)
        mtp_losses.append(m_value)
        tokens += int(batch["n_tokens"])

    receipt = {
        "schema": "mei-mtp-ablation-receipt-v1",
        "stage_fingerprint_sha256": plan["stage_fingerprint_sha256"],
        "terminal_status": "passed",
        "started_unix": started,
        "finished_unix": time.time(),
        "metrics": {
            "steps": args.steps,
            "tokens_seen_each_branch": tokens,
            "control_initial_loss": control_losses[0],
            "control_final_loss": control_losses[-1],
            "mtp_initial_loss": mtp_losses[0],
            "mtp_final_loss": mtp_losses[-1],
            "only_variable": "prediction_offset_1_vs_2",
        },
        "temporary_heads_persisted": False,
        "temporary_heads_exported": False,
        "deployment_mtp_tensor_count_required": 0,
        "future_recipe_evidence_only": True,
        "not_a_score_claim": True,
    }
    _write_json(args.out, receipt)
    return {"ok": True, **receipt, "receipt": str(args.out)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--replay-corpus", type=Path, default=ROOT / "corpus/lm-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seq_len <= 2 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("invalid MTP ablation budget")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
