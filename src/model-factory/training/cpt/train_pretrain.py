#!/usr/bin/env python3
"""Pretrain Needle-zh. --smoke is tiny/step-limited; pilots use full spec + token budget."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.paths import (
    ARCHITECTURE_DIR,
    ARCHITECTURE_ID,
    CORPUS_LM_V1,
    CORPUS_LM_V2,
    CORPUS_ZH_PRETRAIN,
    RECIPES_DIR,
    ROOT,
    TRAIN_RUNS,
    architecture_contracts,
    architecture_sha256,
    ensure_formal_on_path,
)

ensure_formal_on_path()
from training.cpt.pretrain_gates import refuse_non_scratch_source
from training.cpt.cpt_gates import (
    PARENT_TOKENS_SEEN,
    RESET_SOURCE_CURSORS,
    refuse_cpt_parent,
    refuse_cpt_source,
    refuse_weights_only_continuation,
)
from common.run_lock import acquire_run_lock, other_held_runs, write_heartbeat

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from architecture import NeedleZh, count_params  # noqa: E402
from common.checkpoint import load_train_state, save_params, save_train_state  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from common.data import (  # noqa: E402
    PackedTokenSource,
    classify_schedule,
    file_sha256,
    iter_packed_windows,
    iter_split_documents,
    list_pretrain_shards,
    list_token_shards,
    list_valid_set,
    load_scheduled_train,
    refuse_if_short,
    resolve_schedule_file,
    select_curriculum_stage,
)
try:
    from evaluation.base.pretrain_probes import eval_probes, load_probes  # noqa: E402
except ImportError:
    eval_probes = None  # type: ignore[assignment]
    load_probes = None  # type: ignore[assignment]
from tokenizer import ZhTokenizerV1, ZhTokenizerV2  # noqa: E402


def _frozen_tokenizer():
    """按 TOKENIZER.json 指针实例化当前冻结词表（v1 或 v2 系）。"""
    from common.paths import frozen_tokenizer_path

    path = frozen_tokenizer_path()
    tokenizer_id = path.name.replace(".model", "")
    if tokenizer_id == "zh-24k-v1":
        return ZhTokenizerV1()
    return ZhTokenizerV2(
        tokenizer_id=tokenizer_id,
        vocab_size=None,  # 由模型文件自校验
        manifest_path=path.parent / f"tokenizer-{tokenizer_id}-manifest.json",
    )
from common.train_common import eval_lm_loss, peak_bytes, segment_throughput, train_lm_steps  # noqa: E402

RECIPE_PATH = RECIPES_DIR / "pretrain-rungs.json"
RECIPE_51M_PATH = RECIPES_DIR / "pretrain-51m-rungs.json"
CPT_RECIPE_PATH = RECIPES_DIR / "cpt-1b-rungs.json"
PARENT_STATE = ROOT / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1/mei-1.0-51m-base-scratch300m-v1-state.npz"
DEFAULTS = {
    "pilot-1m": {"target": 1_000_000, "horizon": 300_000_000, "eval_every": 250_000, "save_every": 250_000},
    "pilot-5m": {"target": 5_000_000, "horizon": 300_000_000, "eval_every": 500_000, "save_every": 500_000},
    "100m": {"target": 100_000_000, "horizon": 300_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
    "300m": {"target": 300_000_000, "horizon": 300_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
    "cpt-smoke": {"target": 300_008_533, "horizon": 699_999_515, "eval_every": 10**18, "save_every": 10**18},
    "cpt-5m": {"target": 305_000_485, "horizon": 699_999_515, "eval_every": 1_000_000, "save_every": 1_000_000},
    "600m": {"target": 600_000_000, "horizon": 300_000_000, "eval_every": 5_000_000, "save_every": 5_000_000},
    "1b": {"target": 1_000_000_000, "horizon": 699_999_515, "eval_every": 10_000_000, "save_every": 10_000_000},
    "2b": {"target": 2_000_000_000, "horizon": 1_000_000_000, "eval_every": 20_000_000, "save_every": 20_000_000},
    "2b-plus": {"target": 2_000_000_001, "horizon": 1_000_000_000, "eval_every": 20_000_000, "save_every": 20_000_000},
}
MILESTONES = (
    100_000_000,
    150_000_000,
    250_000_000,
    300_000_000,
    350_000_000,
    500_000_000,
    600_000_000,
    750_000_000,
    1_000_000_000,
    2_000_000_000,
)
CPT_RUNGS = {"cpt-smoke", "cpt-5m", "600m", "1b", "2b", "2b-plus"}
RUNG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,39}$")


def defaults_for_rung(rung: str, target_tokens: int | None) -> dict:
    if rung in DEFAULTS:
        return dict(DEFAULTS[rung])
    if target_tokens is None or int(target_tokens) <= 0:
        raise ValueError(f"custom rung {rung!r} requires positive --target-tokens")
    target = int(target_tokens)
    cadence = 5_000_000 if target <= 600_000_000 else (10_000_000 if target <= 1_000_000_000 else 20_000_000)
    return {"target": target, "horizon": target, "eval_every": cadence, "save_every": cadence}


def load_recipe(kind: str = "scratch") -> dict:
    if kind == "cpt":
        path = CPT_RECIPE_PATH
    elif ARCHITECTURE_ID.startswith("mei-1.0-51m-"):
        path = RECIPE_51M_PATH
    else:
        path = RECIPE_PATH
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def expected_param_gate() -> dict:
    spec = json.loads((ARCHITECTURE_DIR / "spec/model.json").read_text(encoding="utf-8"))
    band = spec.get("expected_trainable_params") or spec.get("expected_params") or {}
    target = band.get("target", band.get("measured_mlx"))
    return {
        "min": int(band.get("min") or 45_000_000),
        "max": int(band.get("max") or 60_000_000),
        "target": int(target) if target is not None else None,
    }


def params_in_band(n: int, gate: dict) -> bool:
    if not (gate["min"] <= n <= gate["max"]):
        return False
    if ARCHITECTURE_ID.startswith("mei-1.0-51m-") and gate["target"] is not None:
        return n == gate["target"]
    return True


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


def completed_stage_tokens(
    tokens_seen: int,
    *,
    process_start_tokens: int,
    parent_tokens_seen: int,
    schedule_kind: str,
) -> int:
    """Report cumulative stage exposure while keeping process throughput local."""
    baseline = parent_tokens_seen if schedule_kind == "cpt" else process_start_tokens
    return int(tokens_seen) - int(baseline)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit_progress(path: Path, row: dict) -> None:
    extra = ""
    if row.get("valid_loss") is not None:
        extra = (
            f" valid={row['valid_loss']:.4f}"
            f" hq={row.get('valid_loss_hq')}"
            f" col={row.get('valid_loss_colloquial')}"
            f" st={row.get('valid_loss_structure')}"
        )
    line = (
        f"step={row.get('step')} tokens={row.get('tokens_seen')} "
        f"stage={row.get('curriculum_stage')} seq={row.get('seq_len')} "
        f"loss={float(row.get('loss') or 0):.4f} tok_s={float(row.get('tok_s') or 0):.1f} "
        f"lr={row.get('lr')}{extra}"
    )
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_auto_schedule_kind(corpus_dir: Path) -> str:
    if any(corpus_dir.glob("schedule-cpt*.json")):
        return "cpt"
    scratch = corpus_dir / "schedule-scratch.json"
    if scratch.is_file():
        return "scratch"
    return "none"


def refuse_schedule_mismatch(kind: str, schedule: dict) -> str | None:
    actual = classify_schedule(schedule)
    if kind == "scratch" and actual == "cpt":
        return (
            "scratch refuses an archived continuation schedule "
            "(parent_tokens_seen/skip_tokens/parent_checkpoint). "
            "Use cycles/mei-1.1-51m/exp-00300m/corpus/cpt-delta/lm-v1/schedule-scratch.json."
        )
    if kind == "cpt" and actual != "cpt":
        return "cpt schedule kind must classify as cpt"
    if kind == "scratch" and actual not in {"scratch", "none"}:
        return f"scratch schedule classified as {actual}"
    return None


def slim_probes(rep: dict | None) -> dict | None:
    if not rep:
        return None
    return {k: rep[k] for k in ("n", "mean_nll", "family_nll", "exact_rate")}


def layout_for_seq(recipe: dict, seq: int) -> tuple[int, int]:
    layouts = recipe.get("layouts") or {}
    row = layouts.get(str(seq)) or {}
    return int(row.get("batch_size") or 0), int(row.get("grad_accum") or 0)


def write_not_for_promote(out_dir: Path, reason: str) -> None:
    payload = {
        "promote": False,
        "reason": reason,
        "not_a_claim": "Pilot/smoke runs are mechanism checks, not mei-51m-base.",
    }
    (out_dir / "NOT_FOR_PROMOTE.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rung",
        default="300m",
        help="named milestone or deterministic cumulative-exposure label such as 900m",
    )
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
    ap.add_argument(
        "--resume-mode",
        choices=["auto", "strict", "curriculum", "weights_only", "continuation"],
        default="auto",
    )
    ap.add_argument("--parent", type=Path, default=None, help="scratch300m state for first CPT hop")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--lr-final", type=float, default=None)
    ap.add_argument("--lr-token-offset", type=int, default=None)
    ap.add_argument("--curriculum-stage", default=None, help="s1, s2, s3, or cpt1")
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--stop-at-tokens", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--schedule-kind",
        choices=["auto", "none", "scratch", "cpt"],
        default="auto",
        help="auto: lm-v2→cpt, lm-v1→scratch; a pack without schedule→none.",
    )
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_LM_V1,
        help="pretrain pack root (default cycles/mei-1.1-51m/exp-00300m/corpus/cpt-delta/lm-v1 four-role scratch mix)",
    )
    ap.add_argument("--exclude-source", action="append", default=[], help="Drop a scheduled mix source (ablation)")
    ap.add_argument("--run-suffix", default="", help="Append to run/ckpt name, e.g. no-colloquial")
    args = ap.parse_args()
    if not RUNG_RE.fullmatch(str(args.rung)):
        print("rung must be 2-40 lowercase letters, digits, dot, underscore or dash", file=sys.stderr)
        return 2
    if args.rung == "2b-plus" and args.target_tokens is None:
        print("2b-plus requires an explicit --target-tokens cumulative exposure", file=sys.stderr)
        return 2
    requested_cpt = args.schedule_kind == "cpt" or args.rung in CPT_RUNGS
    recipe = load_recipe("cpt" if requested_cpt else "scratch")
    if (
        args.rung == "300m"
        and not args.smoke
        and not args.count_params
        and args.curriculum_stage is None
        and args.steps is None
    ):
        print(
            "300m scratch must run through run_scratch_curriculum.py "
            "(or run_scratch_curriculum_51m.py); bare --rung 300m is ambiguous",
            file=sys.stderr,
        )
        return 2

    if args.count_params:
        cfg = NeedleZhConfig.from_spec()
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        n = count_params(model)
        gate = expected_param_gate()
        ok = params_in_band(n, gate)
        print(
            json.dumps(
                {
                    "architecture_id": ARCHITECTURE_ID,
                    "architecture_sha256": architecture_sha256(),
                    **{
                        key: value
                        for key, value in architecture_contracts().items()
                        if key.endswith("_sha256")
                    },
                    "full_params": n,
                    "expected_params": gate,
                    "in_band": ok,
                }
            )
        )
        return 0 if ok else 1

    tok = _frozen_tokenizer()
    if requested_cpt and args.corpus_dir == CORPUS_LM_V1:
        args.corpus_dir = CORPUS_LM_V2
    corpus_dir = CORPUS_ZH_PRETRAIN if args.smoke else args.corpus_dir
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    schedule_kind = "none" if args.smoke else (
        args.schedule_kind if args.schedule_kind != "auto" else resolve_auto_schedule_kind(corpus_dir)
    )
    if schedule_kind == "cpt":
        recipe = load_recipe("cpt")
    if not args.smoke:
        if schedule_kind == "cpt":
            blocked = refuse_cpt_source(corpus_dir, args.rung)
        else:
            blocked = refuse_non_scratch_source(corpus_dir, args.rung)
        if blocked:
            print(blocked, file=sys.stderr)
            return 4
    man_path = corpus_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    train_bins = list_token_shards(corpus_dir, "train")
    valid_bins = list_token_shards(corpus_dir, "valid")
    use_mmap = bool(train_bins)
    shards = [] if use_mmap else list_pretrain_shards(corpus_dir, smoke=False)
    if not use_mmap and not shards:
        print(
            f"missing tokens/*.bin under {corpus_dir}; freeze bins are required (smoke no longer uses shards/smoke.jsonl)",
            file=sys.stderr,
        )
        return 1
    cfg = NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec()
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(f"vocab mismatch cfg={cfg.vocab_size} tok={tok.vocab_size}")

    try:
        dft = defaults_for_rung(args.rung, args.target_tokens)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rung_row = ((recipe.get("rungs") or {}).get(args.rung) or {})
    compile_train = (not args.no_compile) and (
        bool(args.compile) or (bool(recipe.get("compile_train")) and not args.smoke)
    )
    schedule_path = resolve_schedule_file(corpus_dir, schedule_kind, target_tokens=args.target_tokens)
    schedule_doc = {}
    if schedule_path and schedule_path.is_file() and not args.smoke:
        schedule_doc = json.loads(schedule_path.read_text(encoding="utf-8"))
        mismatch = refuse_schedule_mismatch(schedule_kind, schedule_doc)
        if mismatch:
            print(mismatch, file=sys.stderr)
            return 2
    declared_parent_tokens = int(
        schedule_doc.get("parent_tokens_seen")
        or (PARENT_TOKENS_SEEN if schedule_kind == "cpt" else 0)
    )
    stage = {}
    if schedule_doc.get("curriculum"):
        stage = select_curriculum_stage(
            schedule_doc,
            stage_id=args.curriculum_stage,
            seq_len=args.seq_len,
            tokens_seen=declared_parent_tokens if schedule_kind == "cpt" else None,
        )
    seq = args.seq_len or (
        32 if args.smoke else int(stage.get("seq_len") or rung_row.get("seq_len") or recipe.get("seq_len") or 512)
    )
    layout_bs, layout_ga = layout_for_seq(recipe, seq)
    batch_size = args.batch_size or (
        1 if args.smoke else int(stage.get("batch_size") or rung_row.get("batch_size") or layout_bs or 8)
    )
    grad_accum = args.grad_accum or (
        1 if args.smoke else int(stage.get("grad_accum") or rung_row.get("grad_accum") or layout_ga or 1)
    )
    sched_lr = schedule_doc.get("lr") or {}
    sched_declares_cycle_lr = schedule_kind == "cpt" and sched_lr.get("token_offset") is not None
    lr = float(
        args.lr
        if args.lr is not None
        else (sched_lr.get("base") if sched_declares_cycle_lr else (recipe.get("lr") or 3e-4))
    )
    lr_final = args.lr_final if args.lr_final is not None else (
        sched_lr.get("final") if sched_declares_cycle_lr else recipe.get("lr_final")
    )
    lr_token_offset = (
        args.lr_token_offset
        if args.lr_token_offset is not None
        else int(
            sched_lr.get("token_offset")
            if sched_declares_cycle_lr
            else (recipe.get("lr_token_offset") or 0)
        )
    )
    eval_every = args.eval_every_tokens or (10**18 if args.smoke else int(rung_row.get("eval_every") or dft["eval_every"]))
    save_every = args.save_every_tokens or (10**18 if args.smoke else int(rung_row.get("save_every") or dft["save_every"]))
    schedule = {}
    extra_valids: dict[str, list] = {}
    parent_tokens = 0
    init_mode = "scratch"
    scheduled = None
    if schedule_kind in {"scratch", "cpt"}:
        if schedule_path is None:
            print(f"missing {schedule_kind} schedule under {corpus_dir}", file=sys.stderr)
            return 2
        if not args.smoke:
            scheduled = load_scheduled_train(
                corpus_dir,
                seq,
                tok.pad_id,
                exclude_sources=tuple(args.exclude_source or ()),
                schedule_path=schedule_path,
                curriculum_stage=str(stage.get("id") or args.curriculum_stage or "") or None,
                tokens_seen=declared_parent_tokens if schedule_kind == "cpt" else None,
            )
            if scheduled is None:
                print(f"schedule {schedule_path} produced no sources", file=sys.stderr)
                return 2
            mismatch = refuse_schedule_mismatch(schedule_kind, dict(scheduled.schedule or {}))
            if mismatch:
                print(mismatch, file=sys.stderr)
                return 2
    hash_files = train_bins[:1] + valid_bins[:1] if args.smoke else (train_bins + valid_bins if use_mmap else shards)
    corpus_hash = "".join(file_sha256(p) for p in hash_files)
    mix_path = corpus_dir / "mix.json"
    if mix_path.is_file() and not args.smoke:
        corpus_hash = corpus_hash + file_sha256(mix_path)
    schedule_sha = file_sha256(schedule_path) if schedule_path and schedule_path.is_file() and not args.smoke else ""
    if schedule_sha:
        corpus_hash = corpus_hash + schedule_sha
    if man_path.is_file() and not args.smoke:
        manifest_sha = file_sha256(man_path)
        corpus_hash = corpus_hash + manifest_sha
    else:
        manifest_sha = ""

    if use_mmap:
        if scheduled is not None:
            train = scheduled
            schedule = dict(scheduled.schedule or {})
            parent_tokens = int(schedule.get("parent_tokens_seen") or 0)
        else:
            train = PackedTokenSource(train_bins, seq, tok.pad_id)
        if args.smoke:
            train = train[:64]
        wiki_bins = list_valid_set(corpus_dir, "wiki") or valid_bins
        valid = PackedTokenSource(wiki_bins, seq, tok.pad_id)[: (8 if args.smoke else 128)] if wiki_bins else []
        if not args.smoke:
            for name in ("hq", "colloquial", "structure", "structure_parent", "colloquial_parent"):
                bins = list_valid_set(corpus_dir, name)
                if bins:
                    extra_valids[name] = PackedTokenSource(bins, seq, tok.pad_id)[:32]
        packed_tokens = int(getattr(train, "n_predictable_tokens", 0) or (len(train) * seq))
    else:
        target_guess = args.target_tokens if args.target_tokens is not None else (None if args.smoke else int(dft["target"]))
        train_limit = int(target_guess / max(seq, 1)) + 32 if target_guess else (64 if args.smoke else None)
        train = collect_windows(shards, tok, seq, "train", limit=train_limit)
        valid = collect_windows(shards, tok, seq, "valid", limit=128)
        packed_tokens = len(train) * seq
    if not train:
        print("no train windows", file=sys.stderr)
        return 1

    exposure_cap = int(train.exposure_cap_tokens()) if hasattr(train, "exposure_cap_tokens") else int(
        getattr(train, "n_predictable_tokens", packed_tokens)
    )
    unique_remaining = int(getattr(train, "n_predictable_tokens", packed_tokens))
    if args.target_tokens is not None:
        target = int(args.target_tokens)
    elif args.smoke:
        target = None
    else:
        target = int(rung_row.get("target") or dft["target"])
    if args.stop_at_tokens is not None:
        target = int(args.stop_at_tokens)
    elif stage.get("stop_at_tokens") and not args.smoke and target is not None:
        target = min(int(target), int(stage["stop_at_tokens"]))
    if use_mmap and not args.allow_repeat and not schedule:
        pred = int(getattr(train, "n_predictable_tokens", packed_tokens))
        if target is None or int(target) > pred:
            target = pred
    horizon_tokens = args.lr_horizon_tokens
    if horizon_tokens is None and not args.smoke:
        horizon_tokens = int(
            ((schedule or {}).get("lr") or {}).get("horizon_tokens")
            or recipe.get("lr_horizon_tokens")
            or dft["horizon"]
        )
    scheduled_exposure = int(schedule.get("exposure_tokens") or exposure_cap)
    if schedule_kind == "cpt":
        scheduled_exposure = int(schedule.get("parent_tokens_seen") or parent_tokens) + scheduled_exposure
    if schedule and not args.allow_repeat and target and scheduled_exposure < int(target):
        print(
            f"schedule exposure {scheduled_exposure} < target {target}",
            file=sys.stderr,
        )
        return 2

    corpus_tokens = int(man.get("n_unique_train_tokens") or man.get("n_train_tokens") or packed_tokens)
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
    cover = packed_tokens if not schedule else (parent_tokens + exposure_cap)
    if not args.smoke and target and cover < int(target) and not args.allow_repeat:
        print(
            f"packed windows cover ~{cover} tokens < target {target}; pass --allow-repeat to multi-epoch",
            file=sys.stderr,
        )
        return 2

    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    n_params = count_params(model)
    gate = expected_param_gate()
    if not args.smoke and not params_in_band(n_params, gate):
        print(
            json.dumps(
                {
                    "error": "trainable parameter count outside architecture gate",
                    "architecture_id": ARCHITECTURE_ID,
                    "full_params": n_params,
                    "expected_params": gate,
                }
            ),
            file=sys.stderr,
        )
        return 2
    steps = args.steps or (8 if args.smoke else None)
    start_step = 0
    start_tokens = 0
    start_window = 0
    opt = optim.Adam(learning_rate=lr)
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
    expected["architecture_id"] = ARCHITECTURE_ID
    expected["architecture_sha256"] = architecture_sha256()
    expected["weight_contract_sha256"] = architecture_contracts()["weight_contract_sha256"]
    expected["params"] = n_params
    if schedule_sha:
        expected["schedule_sha256"] = schedule_sha
    resume_mode = "scratch"
    parent_checkpoint = None
    resume_path = args.resume or args.parent
    if schedule_kind == "cpt" and resume_path is None:
        resume_path = PARENT_STATE
        args.parent = PARENT_STATE
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        meta_path = resume_path.with_suffix(".meta.json")
        prev_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        prev_seq = int(prev_meta.get("seq_len") or seq)
        is_parent_hop = schedule_kind == "cpt" and (
            str(prev_meta.get("schedule_sha256") or "") != str(schedule_sha or "")
        )
        stage_changed = prev_seq != int(seq) or str(prev_meta.get("curriculum_stage") or "") != str(
            stage.get("id") or prev_meta.get("curriculum_stage") or ""
        )
        resume_mode = args.resume_mode
        if resume_mode == "auto":
            if is_parent_hop:
                resume_mode = "continuation"
            else:
                resume_mode = "curriculum" if stage_changed else "strict"
        if schedule_kind == "cpt":
            banned = refuse_weights_only_continuation(resume_mode)
            if banned:
                print(banned, file=sys.stderr)
                return 2
            # Same-schedule pause/resume loads the mid-run state; parent-hash
            # pins apply only when hopping from the frozen parent checkpoint.
            if is_parent_hop:
                parent_err = refuse_cpt_parent(resume_path, schedule)
                if parent_err:
                    print(parent_err, file=sys.stderr)
                    return 4
        if stage_changed and resume_mode not in {"curriculum", "weights_only", "continuation"}:
            print("seq/stage change requires --resume-mode curriculum or continuation", file=sys.stderr)
            return 2
        init_mode = resume_mode
        opt = optim.Adam(learning_rate=lr)
        prev = load_train_state(resume_path, model, opt, mode=resume_mode, expected_meta=expected)
        start_step = int(prev.get("step") or 0)
        start_tokens = int(prev.get("tokens_seen") or 0)
        start_window = int(prev.get("window_index") or 0)
        total_steps = int(prev.get("total_steps") or total_steps or 1)
        sampler_state = prev.get("sampler_state")
        if not sampler_state:
            sampler_state = {
                "names": list(getattr(train, "names", [])),
                "token_cursors": dict(prev.get("source_token_cursors") or prev.get("source_cursors") or {}),
                "source_tokens_drawn": dict(prev.get("source_tokens_drawn") or {}),
            }
        if hasattr(train, "migrate_from_parent") and is_parent_hop:
            reset_sources = tuple(schedule.get("reset_source_cursors") or RESET_SOURCE_CURSORS)
            train.migrate_from_parent(sampler_state, reset_sources=reset_sources)
            parent_checkpoint = str(resume_path.relative_to(ROOT))
        elif hasattr(train, "load_state_dict"):
            if not sampler_state:
                raise ValueError("scheduled mix resume requires sampler_state in checkpoint meta")
            if stage_changed and hasattr(train, "continue_from"):
                train.continue_from(sampler_state)
            else:
                train.load_state_dict(sampler_state)
        if is_parent_hop:
            parent_checkpoint = str(resume_path.relative_to(ROOT))

    suffix = f"-{args.run_suffix}" if args.run_suffix else ""
    if schedule_kind == "scratch" and "scratch" not in suffix:
        suffix = f"-scratch{suffix}"
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        run_name = out_dir.name.removeprefix("pretrain-")
    elif args.smoke:
        run_name = "smoke"
        out_dir = TRAIN_RUNS / "pretrain-smoke"
    elif schedule_kind == "cpt" and args.rung == "1b":
        run_name = "1b-cpt-from-scratch300m"
        out_dir = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
    else:
        run_name = f"{args.rung}{suffix}"
        out_dir = TRAIN_RUNS / f"pretrain-{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if schedule_kind == "cpt":
        held = other_held_runs(except_dir=out_dir)
        if held:
            print(json.dumps({"error": "other formal run lock held", "held": held}, ensure_ascii=False), file=sys.stderr)
            return 4
        acquire_run_lock(
            out_dir,
            {
                "recipe": "cpt-cumulative-exposure-v2",
                "rung": args.rung,
                "target_exposure_tokens": target,
                "weight_contract_sha256": architecture_contracts()["weight_contract_sha256"],
                "corpus": (
                    str(corpus_dir.relative_to(ROOT))
                    if str(corpus_dir).startswith(str(ROOT))
                    else str(corpus_dir)
                ),
                "parent": parent_checkpoint,
            },
        )
    if args.smoke or str(args.rung).startswith("pilot-") or str(args.rung).startswith("cpt-"):
        write_not_for_promote(out_dir, "smoke" if args.smoke else f"{args.rung} is not a base release")
    metrics_path = out_dir / "metrics.jsonl"
    progress_path = out_dir / "progress.log"
    ckpt_dir = out_dir
    last_ckpt = ckpt_dir / f"pretrain-{run_name}.npz"
    last_state = ckpt_dir / f"pretrain-{run_name}-state.npz"
    best_state = ckpt_dir / f"pretrain-{run_name}-best-state.npz"
    probe_bank = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/needle-pretrain-probes-v0/probes-v0.jsonl"
    probes = load_probes(probe_bank) if load_probes and probe_bank.is_file() else []

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
    last_progress_at = start_tokens
    progress_every = 250_000
    saved_milestones: set[int] = set()
    t0 = time.time()
    stage_id = str(stage.get("id") or "") or None

    def copy_milestone(tokens_seen: int) -> None:
        for mark in MILESTONES:
            if tokens_seen >= mark and mark not in saved_milestones:
                if start_tokens >= mark:
                    saved_milestones.add(mark)
                    continue
                if not last_state.is_file():
                    continue
                tag = f"{mark // 1_000_000}m"
                shutil.copy2(last_state, ckpt_dir / f"pretrain-{run_name}-{tag}-state.npz")
                meta_src = last_state.with_suffix(".meta.json")
                if meta_src.is_file():
                    shutil.copy2(meta_src, ckpt_dir / f"pretrain-{run_name}-{tag}-state.meta.json")
                saved_milestones.add(mark)

    def write_ckpt(tag: str, valid_loss, probe_rep, *, is_best: bool) -> dict:
        meta = {
            "architecture_id": ARCHITECTURE_ID,
            "architecture_sha256": architecture_sha256(),
            **{
                key: value
                for key, value in architecture_contracts().items()
                if key.endswith("_sha256")
            },
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
            "compile_train": bool(compile_train),
            "allow_repeat": bool(args.allow_repeat),
            "mmap": use_mmap,
            "init_mode": init_mode,
            "resume_mode": resume_mode if resume_path else "scratch",
            "schedule_kind": schedule_kind,
            "curriculum_stage": stage_id,
            "config_source": "tiny" if args.smoke else "from_spec",
            "parent_tokens_seen": parent_tokens,
            "parent_checkpoint": parent_checkpoint,
            "schedule_sha256": schedule_sha,
            "n_unique_remaining": unique_remaining,
            "n_exposure_cap_tokens": exposure_cap,
            "source_cursors": dict(getattr(train, "cursor", {}) or {}),
            "source_token_cursors": dict(getattr(train, "token_cursors", getattr(train, "cursor", {})) or {}),
            "source_window_counts": dict(getattr(train, "window_counts", {}) or {}),
            "source_tokens_drawn": dict(getattr(train, "source_tokens_drawn", {}) or {}),
            "stage_tokens_drawn": dict(getattr(train, "stage_tokens_drawn", {}) or {}),
            "quota_remaining": dict(getattr(train, "quota_remaining", {}) or {}),
            "alignment_slack": dict(getattr(train, "alignment_slack", {}) or {}),
            "alignment_overshoot": dict(getattr(train, "alignment_overshoot", {}) or {}),
            "sampler_state": train.state_dict() if hasattr(train, "state_dict") else None,
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
            return None, None, {}
        valid_loss = eval_lm_loss(model, valid, batch_size=min(batch_size, 4)) if valid else None
        extra_losses = {
            name: eval_lm_loss(model, windows, batch_size=min(batch_size, 4))
            for name, windows in extra_valids.items()
            if windows
        }
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
        return valid_loss, probe_rep, extra_losses

    def on_step(step: int, info: dict) -> None:
        nonlocal last_progress_at
        state["steps"] = step - start_step + 1
        state["tokens_seen"] = info["tokens_seen"]
        state["window_index"] = info["window_index"]
        state["last_loss"] = info["loss"]
        elapsed = max(1e-6, time.time() - t0)
        if not finite(info["loss"]) or not finite(info["grad_norm"]):
            raise RuntimeError(f"non-finite loss/grad at step {step}: {info}")
        valid_loss, probe_rep, extra_losses = maybe_eval(info["tokens_seen"])
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
            "seq_len": seq,
            "curriculum_stage": stage_id,
        }
        if info.get("source_cursors"):
            row["source_cursors"] = info["source_cursors"]
        if info.get("source_window_counts"):
            row["source_window_counts"] = info["source_window_counts"]
        if info.get("source_tokens_drawn"):
            row["source_tokens_drawn"] = info["source_tokens_drawn"]
        if info.get("quota_remaining"):
            row["quota_remaining"] = info["quota_remaining"]
        if info.get("stage_tokens_drawn"):
            row["stage_tokens_drawn"] = info["stage_tokens_drawn"]
        if info.get("alignment_overshoot"):
            row["alignment_overshoot"] = info["alignment_overshoot"]
        if valid_loss is not None:
            row["valid_loss"] = valid_loss
        for name, loss in (extra_losses or {}).items():
            row[f"valid_loss_{name}"] = loss
        if probe_rep:
            row["probe_mean_nll"] = probe_rep.get("mean_nll")
            row["probe_family_nll"] = probe_rep.get("family_nll")
        append_jsonl(metrics_path, row)
        due_progress = int(info["tokens_seen"]) - last_progress_at >= progress_every
        if due_progress or valid_loss is not None:
            emit_progress(progress_path, row)
            last_progress_at = int(info["tokens_seen"])
        write_heartbeat(
            out_dir / "heartbeat.json",
            {
                "tokens_seen": info["tokens_seen"],
                "loss": info["loss"],
                "lr": info["lr"],
                "tok_s": tok_s,
                "step": step,
                "rung": args.rung,
            },
        )

    probes_init = eval_probes(model, tok, probes) if probes and not args.smoke and resume_path is None else None
    if probes_init:
        (out_dir / "probes-init.json").write_text(
            json.dumps(slim_probes(probes_init), indent=2) + "\n", encoding="utf-8"
        )
    write_heartbeat(out_dir / "heartbeat.json", {"tokens_seen": start_tokens, "status": "starting", "rung": args.rung})

    result = train_lm_steps(
        model,
        train,
        steps=steps,
        lr=lr,
        seed=0,
        start_step=start_step,
        start_tokens_seen=start_tokens,
        total_steps=total_steps,
        optimizer=opt,
        reseed=resume_path is None,
        on_step=on_step,
        batch_size=batch_size,
        grad_accum=grad_accum,
        target_tokens=None if args.smoke else target,
        start_window=start_window,
        allow_repeat=args.allow_repeat,
        compile_train=compile_train,
        precision=args.precision,
        horizon_tokens=None if args.smoke else horizon_tokens,
        lr_final=None if args.smoke else (float(lr_final) if lr_final is not None else None),
        lr_token_offset=0 if args.smoke else int(lr_token_offset or 0),
        stop_path=None if args.smoke else (out_dir / "STOP"),
    )
    state.update(result)
    valid_loss, probe_rep, extra_losses = maybe_eval(result["tokens_seen"], force=True)
    meta = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    meta["valid_loss"] = valid_loss
    for name, loss in (extra_losses or {}).items():
        meta[f"valid_loss_{name}"] = loss
    meta["init_probe_mean_nll"] = None if probes_init is None else probes_init.get("mean_nll")
    meta["final_probe_mean_nll"] = None if probe_rep is None else probe_rep.get("mean_nll")
    elapsed = time.time() - t0
    meta["elapsed_s"] = elapsed
    meta["ckpt"] = str(last_ckpt.relative_to(ROOT))
    meta["n_source_tokens"] = int(getattr(train, "n_tokens", packed_tokens))
    meta["n_predictable_tokens"] = unique_remaining
    meta["n_exposure_cap_tokens"] = exposure_cap
    meta["parent_tokens_seen"] = parent_tokens
    meta["init_mode"] = init_mode
    meta["schedule_kind"] = schedule_kind
    meta["curriculum_stage"] = stage_id
    meta["config_source"] = "tiny" if args.smoke else "from_spec"
    meta["schedule_sha256"] = schedule_sha
    meta["n_windows"] = len(train)
    meta["exhausted"] = bool(result.get("exhausted"))
    meta["paused"] = bool(result.get("paused"))
    meta["segment_tokens"] = completed_stage_tokens(
        int(result["tokens_seen"]),
        process_start_tokens=int(start_tokens),
        parent_tokens_seen=int(parent_tokens),
        schedule_kind=schedule_kind,
    )
    meta["segment_tok_s"] = segment_throughput(result["tokens_seen"], start_tokens, elapsed)
    meta["window_index"] = int(result.get("window_index") or meta.get("window_index") or 0)
    meta["source_tokens_drawn"] = dict(getattr(train, "source_tokens_drawn", {}) or {})
    meta["stage_tokens_drawn"] = dict(getattr(train, "stage_tokens_drawn", {}) or {})
    meta["quota_remaining"] = dict(getattr(train, "quota_remaining", {}) or {})
    meta["alignment_slack"] = dict(getattr(train, "alignment_slack", {}) or {})
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
