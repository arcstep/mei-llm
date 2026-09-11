"""Resume a bounded replay diagnostic from an existing checkpoint.

背景
----
``bounded_cpt_recovery.run`` 硬编码从 1200M 父级状态加载、且 ``output.mkdir(exist_ok=False)``
拒绝重入，因此崩溃后无法原样续跑。M4 Max + macOS 26.6.2 的 GPU 固件 bug 会让
``grad_accum=8`` 的梯度累积路径概率性触发 ``firmware-detected lockup``，训练随时可能中断。

本脚本从任意 ``checkpoints/step-NNNN-state.npz`` 精确恢复（模型 + 优化器 + 采样游标），
续跑剩余步数到 ``target_tokens``，并把 checkpoint 间隔从 512 调到 128，降低再次锁死的损失。

正确性关键点（改动前请重读）：
* ``start_step`` 传 checkpoint meta 的绝对 ``step``（如 125737），而非父级的 109865，
  这样 ``on_step`` 里 ``completed = step - 父级step + 1`` 才能从 15873 继续编号。
* ``start_tokens_seen`` 传 checkpoint meta 的 ``tokens_seen``。
* terminal checkpoint 的 ``completed`` 用 ``resume_step + result["steps"]``（``result["steps"]``
  是本次调用从 0 起的相对步数，不是绝对步数）。
* sampler 通过 ``make_sampler`` 先迁回父级，再 ``load_state_dict(checkpoint.sampler_state)``
  精确恢复到崩溃点；``consumed_windows == draw_count`` 由 load_state_dict 恢复，无需手传。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from common.paths import ROOT, ensure_formal_on_path, frozen_tokenizer_path
from common.run_lock import acquire_run_lock
from diagnostics.bounded_cpt_recovery import read

CHECKPOINT_INTERVAL = 128  # 调密：再次锁死最多丢 ~7 分钟，而非 ~半小时


def find_latest_checkpoint(checkpoints: Path) -> int:
    """返回 step 最大的 state.meta.json 的 completed 编号。"""
    best = -1
    for meta in checkpoints.glob("step-*-state.meta.json"):
        try:
            n = int(meta.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        best = max(best, n)
    if best < 0:
        raise ValueError(f"no checkpoint meta found under {checkpoints}")
    return best


def resume(config_path: Path, resume_step: int | None) -> int:
    config = read(config_path)
    metadata = read(ROOT / config["parent_metadata"])
    output = ROOT / config["output"]
    checkpoints = output / "checkpoints"

    resume_step = resume_step if resume_step is not None else find_latest_checkpoint(checkpoints)
    resume_meta = read(checkpoints / f"step-{resume_step:04d}-state.meta.json")
    resume_state = checkpoints / f"step-{resume_step:04d}-state.npz"
    if not resume_state.is_file():
        raise FileNotFoundError(resume_state)

    target_tokens = config["target_tokens"]
    if resume_meta["tokens_seen"] >= target_tokens:
        raise ValueError(f"resume step-{resume_step} already at/after target {target_tokens}")
    if resume_step >= config["steps"]:
        raise ValueError(f"resume step {resume_step} not below total {config['steps']}")

    def log_event(message: str) -> None:
        with (output / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n")

    log_event(f"RESUME step-{resume_step:04d} (step={resume_meta['step']} tokens={resume_meta['tokens_seen']})")

    lock = acquire_run_lock(output, {"scope": config["scope"], "steps": config["steps"],
                                     "resume_from": resume_step})
    started = time.monotonic()
    try:
        ensure_formal_on_path()
        import mlx.core as mx
        import mlx.optimizers as optim
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from common.checkpoint import load_train_state, save_train_state, save_params
        from common.train_common import train_lm_steps, flatten_tree
        from tokenizer import ZhTokenizerV2
        from diagnostics.cpt_batch_replay import make_sampler, evaluate_replay

        model = NeedleZh(NeedleZhConfig.from_spec())
        optimizer = optim.Adam(learning_rate=config["lr"])
        loaded = load_train_state(resume_state, model, optimizer, mode="continuation")
        if loaded != resume_meta:
            raise ValueError("resume checkpoint metadata changed during load")

        tokenizer_path = frozen_tokenizer_path()
        tokenizer = ZhTokenizerV2(tokenizer_id=tokenizer_path.stem, vocab_size=None,
                                 manifest_path=tokenizer_path.parent / f"tokenizer-{tokenizer_path.stem}-manifest.json")
        sampler = make_sampler(config, metadata, tokenizer.pad_id)
        sampler.load_state_dict(resume_meta["sampler_state"])
        if sampler.token_cursors != resume_meta["source_token_cursors"]:
            raise ValueError("sampler cursors did not match resume checkpoint")

        lr_schedule = read(ROOT / config["original_schedule"])["lr"]
        model.train()

        def checkpoint(completed: int, seen: int) -> Path:
            state = checkpoints / f"step-{completed:04d}-state.npz"
            if state.exists():
                saved = read(state.with_suffix(".meta.json"))
                if saved["tokens_seen"] != seen or saved["sampler_state"] != sampler.state_dict():
                    raise ValueError("refusing to overwrite an existing diagnostic checkpoint")
                return state
            next_meta = dict(resume_meta)
            next_meta.update({
                "step": metadata["step"] + completed,
                "tokens_seen": seen,
                "sampler_state": sampler.state_dict(),
                "source_token_cursors": dict(sampler.token_cursors),
                "source_cursors": dict(sampler.token_cursors),
                "quota_remaining": dict(sampler.quota_remaining),
                "source_tokens_drawn": dict(sampler.source_tokens_drawn),
                "stage_tokens_drawn": dict(sampler.stage_tokens_drawn),
                "window_index": sampler.consumed_windows,
            })
            save_train_state(state, model, optimizer, next_meta)
            log_event(f"CHECKPOINT step={completed} tokens={seen} file={state.name}")
            return state

        def on_step(step: int, report: dict) -> None:
            completed = step - metadata["step"] + 1
            if not math.isfinite(report["loss"]) or not math.isfinite(report["grad_norm"]):
                raise ValueError("nonfinite resume update")
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"completed_updates": completed, **report}) + "\n")
            if completed % CHECKPOINT_INTERVAL == 0 and completed < config["steps"]:
                checkpoint(completed, report["tokens_seen"])

        result = train_lm_steps(
            model, sampler, optimizer=optimizer, steps=config["steps"],
            lr=config["lr"], lr_final=lr_schedule.get("final", config["lr"]),
            batch_size=1, grad_accum=8,
            horizon_tokens=lr_schedule.get("horizon_tokens"),
            lr_token_offset=lr_schedule.get("token_offset", 0),
            target_tokens=target_tokens,
            start_step=resume_meta["step"], start_tokens_seen=resume_meta["tokens_seen"],
            start_window=resume_meta["window_index"],
            allow_repeat=False, compile_train=False, on_step=on_step,
            stop_path=output / "STOP",
        )
        result.pop("optimizer")
        final_completed = resume_step + result["steps"]
        terminal_state = checkpoint(final_completed, result["tokens_seen"])
        terminal_weights = checkpoints / f"step-{final_completed:04d}.npz"
        save_params(model, terminal_weights)

        finite = all(bool(mx.all(mx.isfinite(value)).item())
                     for tree in (model.parameters(), optimizer.state) for value in flatten_tree(tree).values())
        complete = (finite and not result["paused"]
                    and result["tokens_seen"] == target_tokens
                    and final_completed == config["steps"])
        if not complete:
            log_event(f"RESUME incomplete finite={finite} paused={result['paused']} "
                      f"tokens={result['tokens_seen']}/{target_tokens} completed={final_completed}/{config['steps']}")
            return 2

        log_event(f"TRAINING_COMPLETE tokens={result['tokens_seen']} steps={final_completed}; starting evaluation")
        del model, optimizer
        mx.clear_cache()
        code, decision = evaluate_replay(config, output, terminal_weights)
        (output / "receipt.json").write_text(
            json.dumps({"status": "complete" if code == 0 else "failed", "decision": decision,
                        "elapsed_seconds": time.monotonic() - started, "resumed_from": resume_step,
                        "automatic_parent_promotion": False, "release_eligible": False,
                        "extended_training": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log_event(f"FINISHED evaluation_exit={code} comparison={decision.get('status')}; inspect comparison.json")
        return code
    except Exception as error:
        log_event("FAILED " + traceback.format_exc())
        (output / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error), "resumed_from": resume_step,
                        "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raise
    finally:
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=int, default=None,
                        help="checkpoint completed 编号；缺省自动取最新 step-*")
    args = parser.parse_args()
    return resume(args.config, args.resume_from)


if __name__ == "__main__":
    raise SystemExit(main())
