#!/usr/bin/env python3
"""Base-parameterized primary CQ2 v2 QAT stage for mei-1.0-51m.

Q4 remains a diagnostic branch.  This stage trains an FP32 master through the
exact portable group-128 WHT/codebook fake-quant path and never writes under a
frozen base directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils as xu

from common._repo import (
    ROOT,
    architecture_contracts,
    legacy_weight_contract_sha256,
    phase_binding_identity,
)
from common.checkpoint import flatten_params, load_params, load_train_state, save_params, save_train_state
from training.qat.cq2_policy_51m import GROUP_SIZE, QUANT_MATH_ID, lm_storage_dtype, uniform_group_bits
from training.qat.cq2_qat_51m import explicit_group_map, group_map_receipt, quantize_tree
from common.train_common import clip_grads, eval_lm_loss, masked_lm_loss, stack_windows


DEFAULT_BASE_DIR = ROOT / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1"
DEFAULT_BASE_RELEASE = DEFAULT_BASE_DIR / "RELEASE.json"
DEFAULT_BASE_WEIGHTS = DEFAULT_BASE_DIR / "mei-1.0-51m-base-scratch300m-v1.npz"
DEFAULT_CORPUS = ROOT / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/corpus/cpt-delta/lm-v1"
DEFAULT_ANCHOR = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/jobs/mei-1.0-51m/float-base-lm-anchor.json"
ARCHITECTURE_DIR = ROOT / "models/mei-1.2-51m/architecture"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def live_cpt_workers() -> list[dict[str, Any]]:
    """Use a live PID plus recent heartbeat, never a ledger alone."""

    rows = []
    run_root = ROOT / ".local/artifacts/mei-1.0-51m"
    for path in run_root.glob("exp-*/runs/**/checkpoints/*/heartbeat.json"):
        try:
            heartbeat = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = int(heartbeat.get("pid") or 0)
        age = time.time() - float(heartbeat.get("unix") or 0.0)
        if _pid_alive(pid) and age < 180:
            rows.append(
                {
                    "pid": pid,
                    "heartbeat": str(path.relative_to(ROOT)),
                    "age_seconds": age,
                    "tokens_seen": int(heartbeat.get("tokens_seen") or 0),
                }
            )
    return rows


def verify_activation_ste_contract(arch) -> dict[str, Any]:
    """Prove that activation STE is Q/DQ-forward and identity-backward.

    The previous algebra had those semantics reversed while receipts still
    declared activation int8.  Keep this executable probe in the training
    worker so a source change cannot silently recreate that false claim.
    """

    x = mx.array(
        [[[-1.0, -0.377, 0.119, 0.731], [-100.0, -37.7, 11.9, 73.1]]],
        dtype=mx.float32,
    )
    maximum = mx.max(mx.abs(x), axis=-1, keepdims=True)
    scale = mx.where(maximum > 0, maximum / 127.0, 1.0)
    expected = mx.clip(mx.round(x / scale), -128.0, 127.0) * scale
    actual = arch._ste_activation_int8(x)
    gradient = mx.grad(lambda value: mx.sum(arch._ste_activation_int8(value)))(x)
    extended = mx.concatenate(
        [x, mx.array([[[10000.0, -3770.0, 1190.0, 7310.0]]], dtype=mx.float32)],
        axis=-2,
    )
    extended_prefix = arch._ste_activation_int8(extended)[..., : int(x.shape[-2]), :]
    mx.eval(expected, actual, gradient, extended_prefix)
    forward_max_abs = float(mx.max(mx.abs(expected - actual)).item())
    backward_max_abs = float(mx.max(mx.abs(gradient - 1.0)).item())
    row_isolation_max_abs = float(mx.max(mx.abs(actual - extended_prefix)).item())
    expected_points = (
        "engram_input",
        "attention_input",
        "attention_output",
        "lm_head_input",
    )
    expected_kv_points = ("attention_key_after_rope", "attention_value")
    if tuple(arch.ACTIVATION_STE_POINTS) != expected_points:
        raise RuntimeError(f"activation QAT points changed: {arch.ACTIVATION_STE_POINTS!r}")
    if tuple(arch.KV_STE_POINTS) != expected_kv_points:
        raise RuntimeError(f"KV QAT points changed: {arch.KV_STE_POINTS!r}")
    if forward_max_abs > 1e-7 or backward_max_abs > 1e-7 or row_isolation_max_abs > 1e-7:
        raise RuntimeError(
            "activation int8 STE contract failed: "
            f"forward={forward_max_abs}, backward={backward_max_abs}, "
            f"row_isolation={row_isolation_max_abs}"
        )
    return {
        "semantics": arch.ACTIVATION_STE_SEMANTICS,
        "activation_points": list(arch.ACTIVATION_STE_POINTS),
        "kv_semantics": arch.KV_STE_SEMANTICS,
        "kv_points": list(arch.KV_STE_POINTS),
        "forward_max_abs": forward_max_abs,
        "backward_max_abs": backward_max_abs,
        "row_isolation_max_abs": row_isolation_max_abs,
    }


def validate_base(release_path: Path, weights_path: Path) -> dict[str, Any]:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("architecture_id") != "mei-1.0-51m-arch-v1":
        raise RuntimeError("base architecture_id is not the canonical 51M architecture")
    if int(release.get("params") or 0) != 51_463_797:
        raise RuntimeError("base parameter count is not 51,463,797")
    weights_sha = _sha_file(weights_path)
    if weights_sha != release.get("weights_sha256"):
        raise RuntimeError("base weights SHA does not match RELEASE.json")
    contracts = architecture_contracts()
    declared_weight = release.get("weight_contract_sha256") or legacy_weight_contract_sha256(
        str(release.get("architecture_sha256") or "")
    )
    if declared_weight != contracts["weight_contract_sha256"]:
        raise RuntimeError("base does not resolve to the canonical weight contract")
    return {
        "base_id": str(release.get("model_id") or release_path.parent.name),
        "base_release": str(release_path.resolve()),
        "base_release_sha256": _sha_file(release_path),
        "base_weights": str(weights_path.resolve()),
        "base_weights_sha256": weights_sha,
        "tokens_seen_exposure": int(release.get("tokens_seen_exposure") or 0),
        "tokenizer_sha256": str(release.get("tokenizer_sha256") or ""),
        "numeric_integrity": release.get("numeric_integrity"),
    }


def static_group_policy() -> dict[str, Any]:
    tensor_order = architecture_contracts()["weight_contract"]["tensor_order"]
    rows = []
    q2_groups = q4_groups = safe = 0
    for tensor in tensor_order:
        name = str(tensor["name"])
        shape = tuple(int(value) for value in tensor["shape"])
        dtype = lm_storage_dtype(name, shape)
        widths = uniform_group_bits(name, shape)
        groups = len(widths or ())
        q2_groups += groups if dtype == "cq2" else 0
        q4_groups += groups if dtype == "cq4" else 0
        safe += int(dtype == "f16")
        rows.append({"name": name, "shape": list(shape), "dtype": dtype, "groups": groups})
    return {
        "quant_math_id": QUANT_MATH_ID,
        "group_size": GROUP_SIZE,
        "tensor_count": len(rows),
        "q2_groups": q2_groups,
        "q4_groups": q4_groups,
        "safe_f16_tensor_count": safe,
        "tensors": rows,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    allow_live_cpt = bool(getattr(args, "allow_live_cpt", False))
    active_cpt = live_cpt_workers()
    base = validate_base(args.base_release, args.base_weights)
    contracts = architecture_contracts()
    anchor = json.loads(args.float_anchor.read_text(encoding="utf-8"))
    if anchor.get("weights_sha256") != base["base_weights_sha256"]:
        raise RuntimeError("Float Base-LM Anchor does not belong to the selected base")
    replay_corpus = args.replay_corpus.resolve()
    corpus_files = [replay_corpus / "manifest.json", replay_corpus / "schedule-scratch.json"]
    if any(not path.is_file() for path in corpus_files):
        raise RuntimeError("replay corpus manifest/schedule is missing")
    source_files = [
        Path(__file__),
        Path(__file__).with_name("cq2_qat_51m.py"),
        Path(__file__).with_name("cq2_policy_51m.py"),
        ROOT / "model-factory/common/train_common.py",
        ROOT / "model-factory/common/checkpoint.py",
        ROOT / "model-factory/common/data.py",
        ARCHITECTURE_DIR / "tokenizer.py",
        ARCHITECTURE_DIR / "architecture.py",
        ARCHITECTURE_DIR / "config.py",
    ]
    binding_identity = phase_binding_identity()
    immutable = {
        "base": base,
        "contracts": {
            key: contracts[key]
            for key in (
                "weight_contract_sha256",
                "runtime_profile_sha256",
                "training_aux_sha256",
            )
        },
        "float_anchor_sha256": _sha_file(args.float_anchor),
        "corpus_files": {str(path.relative_to(ROOT)): _sha_file(path) for path in corpus_files},
        "source_files": {str(path.relative_to(ROOT)): _sha_file(path) for path in source_files},
        "recipe": {
            "target_tokens": int(args.target_tokens),
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "gradient_accumulation": int(getattr(args, "grad_accum", 1)),
            "effective_windows_per_update": int(args.batch_size)
            * int(getattr(args, "grad_accum", 1)),
            "learning_rate": float(args.lr),
            "activation_kv_int8_ste": True,
            "activation_ste_semantics": (
                "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2"
            ),
            "kv_ste_semantics": "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1",
            "seed": 51,
        },
        "execution_policy": {
            "allow_live_cpt": allow_live_cpt,
            "live_cpt_parallel_at_plan_time": bool(active_cpt) and allow_live_cpt,
            "resource_isolation": "separate-run-directories-no-cpt-signals",
        },
        "group_policy": static_group_policy(),
    }
    if binding_identity is not None:
        immutable["phase_binding"] = binding_identity
    return {
        "kind": "mei-51m-cq2-qat-v2-plan",
        "stage_fingerprint_sha256": _sha_bytes(_canonical(immutable)),
        **immutable,
    }


def write_import_candidate(
    args: argparse.Namespace,
    plan: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
) -> Path | None:
    if receipt.get("terminal_status") != "passed":
        return None
    master_path = ROOT / str((receipt.get("outputs") or {}).get("master") or "")
    if (
        not master_path.is_file()
        or _sha_file(master_path)
        != (receipt.get("outputs") or {}).get("master_sha256")
    ):
        raise RuntimeError("CQ2 QAT master is missing or hash-drifted")
    base = plan["base"]
    candidate = {
        "schema": "mei-cq2-qat-import-candidate-receipt-v1",
        "status": "passed",
        "base": {
            "model_id": base["base_id"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
            "weights_sha256": base["base_weights_sha256"],
            "numeric_integrity": base.get("numeric_integrity"),
        },
        "quant_math_id": receipt["quant_math_id"],
        "master": {
            "path": str(master_path.relative_to(ROOT)),
            "sha256": _sha_file(master_path),
            "bytes": master_path.stat().st_size,
        },
        "worker_receipt": {
            "path": str(receipt_path.resolve().relative_to(ROOT)),
            "sha256": _sha_file(receipt_path),
        },
        "tokens_seen_qat": int((receipt.get("metrics") or {}).get("tokens_seen_qat") or 0),
        "valid_loss": (receipt.get("metrics") or {}).get("valid_loss"),
        "phase_binding": plan.get("phase_binding"),
        "reuse_policy": (
            "candidate only; SFT must revalidate Base, corpus, quant math, "
            "contracts, stage fingerprint and output SHA before import"
        ),
        "later_base_policy": "each immutable Base requires its own CQ2-QAT stage",
    }
    candidate_path = args.run_dir / "qat-import-candidate-receipt.json"
    payload = _canonical(candidate) + b"\n"
    if candidate_path.is_file():
        if candidate_path.read_bytes() != payload:
            raise RuntimeError("existing QAT import candidate disagrees with worker receipt")
    else:
        _write_json(candidate_path, candidate)
    return candidate_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    if args.dry_run:
        return {"ok": True, "dry_run": True, **plan}
    active = live_cpt_workers()
    if active and not bool(getattr(args, "allow_live_cpt", False)):
        raise RuntimeError(f"live CPT owns MLX/Metal; CQ2 QAT deferred: {active}")
    stage_dir = args.run_dir / "stages/cq2_qat_v2"
    receipt_path = stage_dir / "receipt.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        master = stage_dir / "final-master.npz"
        if (
            existing.get("stage_fingerprint_sha256") == plan["stage_fingerprint_sha256"]
            and existing.get("terminal_status") in {"passed", "degraded"}
            and master.is_file()
            and existing.get("outputs", {}).get("master_sha256") == _sha_file(master)
        ):
            candidate = write_import_candidate(args, plan, receipt_path, existing)
            return {
                "ok": True,
                "reused": True,
                "qat_import_candidate": str(candidate) if candidate else None,
                **existing,
            }
        raise RuntimeError("existing CQ2 QAT stage is not safely reusable; choose a new run ID")
    if stage_dir.exists() and any(stage_dir.iterdir()) and not args.resume:
        raise RuntimeError("CQ2 QAT stage directory is non-empty; use --resume or a new run ID")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "plan.json", plan)

    from architecture import NeedleZh, count_params
    from config import NeedleZhConfig
    from common.data import PackedTokenSource, list_valid_set, load_scheduled_train
    from tokenizer import ZhTokenizerV1

    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, args.base_weights, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != 51_463_797:
        raise RuntimeError("loaded QAT model parameter identity changed")
    group_map = explicit_group_map(model.parameters())
    map_receipt = group_map_receipt(model.parameters())
    _write_json(stage_dir / "group-map.json", map_receipt)

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
        raise RuntimeError("CQ2 replay source is empty")
    valid_bins = list_valid_set(args.replay_corpus, "wiki")
    valid = PackedTokenSource(valid_bins, args.seq_len, tokenizer.pad_id)[:128] if valid_bins else []
    optimizer = optim.Adam(learning_rate=args.lr)
    state_path = stage_dir / "latest-state.npz"
    checkpoint_meta = {
        "kind": "mei-51m-cq2-qat-v2-state",
        "stage_fingerprint_sha256": plan["stage_fingerprint_sha256"],
        "quant_math_id": QUANT_MATH_ID,
        "group_map_sha256": _sha_file(stage_dir / "group-map.json"),
        "target_tokens": int(args.target_tokens),
    }
    grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
    tokens = 0
    steps = 0
    if args.resume and state_path.is_file():
        meta = load_train_state(
            state_path, model, optimizer, mode="strict", expected_meta=checkpoint_meta
        )
        tokens = int(meta.get("tokens_seen") or 0)
        steps = int(meta.get("steps_completed") or 0)
        if meta.get("sampler_state") and hasattr(source, "load_state_dict"):
            source.load_state_dict(meta["sampler_state"])

    import architecture as arch

    arch.QAT_ACTIVATION_STE = True
    activation_ste_probe = verify_activation_ste_contract(arch)
    started = time.time()
    last_loss = None
    try:
        while tokens < int(args.target_tokens):
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_tokens = 0
            microbatches = 0
            for _ in range(grad_accum):
                windows = source.take_windows(args.batch_size)
                if not windows:
                    raise RuntimeError("CQ2 replay source exhausted before target exposure")
                stacked = stack_windows(windows)

                def loss_fn(params):
                    model.update(quantize_tree(params, group_map, ste=True))
                    logits = model(stacked["x"])["logits"].astype(mx.float32)
                    return masked_lm_loss(logits, stacked["y"], stacked["mask"])

                params = model.parameters()
                loss, grads = mx.value_and_grad(loss_fn)(params)
                model.update(params)
                n_tokens = int(stacked["n_tokens"])
                weighted = xu.tree_map(lambda value: value * float(n_tokens), grads)
                accumulated_grads = (
                    weighted
                    if accumulated_grads is None
                    else xu.tree_map(lambda left, right: left + right, accumulated_grads, weighted)
                )
                accumulated_loss += float(loss.item()) * n_tokens
                accumulated_tokens += n_tokens
                microbatches += 1
                if tokens + accumulated_tokens >= int(args.target_tokens):
                    break
            if accumulated_grads is None or accumulated_tokens <= 0:
                raise RuntimeError("CQ2 QAT accumulated an empty optimizer step")
            grads = xu.tree_map(
                lambda value: value / float(accumulated_tokens), accumulated_grads
            )
            grads, grad_norm = clip_grads(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            last_loss = accumulated_loss / float(accumulated_tokens)
            if not math.isfinite(last_loss):
                raise RuntimeError("CQ2 QAT produced non-finite loss")
            tokens += accumulated_tokens
            steps += 1
            if steps % max(1, args.checkpoint_every_steps) == 0 or tokens >= int(args.target_tokens):
                save_train_state(
                    state_path,
                    model,
                    optimizer,
                    {
                        **checkpoint_meta,
                        "tokens_seen": tokens,
                        "steps_completed": steps,
                        "last_loss": last_loss,
                        "sampler_state": source.state_dict() if hasattr(source, "state_dict") else None,
                    },
                )
            if steps == 1 or steps % 20 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "cq2_qat_v2",
                            "tokens": tokens,
                            "target_tokens": args.target_tokens,
                            "steps": steps,
                            "loss": last_loss,
                            "grad_norm": float(grad_norm.item()),
                            "microbatches": microbatches,
                            "gradient_accumulation": grad_accum,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        arch.QAT_ACTIVATION_STE = False

    master_path = stage_dir / "final-master.npz"
    save_params(model, master_path)
    master = flatten_params(model)
    model.update(quantize_tree(model.parameters(), group_map, ste=False))
    mx.eval(model.parameters())
    try:
        valid_loss = eval_lm_loss(model, valid, batch_size=1, max_windows=128) if valid else None
    finally:
        model.update(xu.tree_unflatten(list(master.items())))
        mx.eval(model.parameters())
    finite = valid_loss is None or math.isfinite(float(valid_loss))
    receipt = {
        "kind": "mei-training-receipt-v2",
        "stage_id": "cq2_qat_v2",
        "stage_fingerprint_sha256": plan["stage_fingerprint_sha256"],
        "parent_id": plan["base"]["base_id"],
        "contracts": plan["contracts"],
        "quant_math_id": QUANT_MATH_ID,
        "terminal_status": "passed" if finite else "degraded",
        "started_unix": started,
        "finished_unix": time.time(),
        "metrics": {
            "tokens_seen_qat": tokens,
            "steps": steps,
            "gradient_accumulation": grad_accum,
            "effective_windows_per_update": int(args.batch_size) * grad_accum,
            "last_train_loss": last_loss,
            "valid_loss": valid_loss,
            "activation_kv_int8_ste": True,
            "activation_ste_probe": activation_ste_probe,
        },
        "outputs": {
            "master": str(master_path.relative_to(ROOT)),
            "master_sha256": _sha_file(master_path),
            "group_map": str((stage_dir / "group-map.json").relative_to(ROOT)),
            "group_map_sha256": _sha_file(stage_dir / "group-map.json"),
            "checkpoint": str(state_path.relative_to(ROOT)),
            "checkpoint_sha256": _sha_file(state_path),
        },
        "not_a_claim": "CQ2 QAT mechanics and receipts do not claim final model usefulness.",
    }
    _write_json(receipt_path, receipt)
    candidate = write_import_candidate(args, plan, receipt_path, receipt)
    return {
        "ok": True,
        "qat_import_candidate": str(candidate) if candidate else None,
        **receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--float-anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--replay-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-live-cpt",
        action="store_true",
        help="record explicit authorization to run CQ2 QAT beside an independent live CPT",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.target_tokens <= 0
        or args.seq_len <= 1
        or args.batch_size <= 0
        or args.grad_accum <= 0
    ):
        parser.error(
            "target tokens, sequence length, batch size, and grad accumulation must be positive"
        )
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
