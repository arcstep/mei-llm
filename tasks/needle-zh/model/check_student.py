#!/usr/bin/env python3
"""Training-ready gates: tokenizer, mask, resume, operator parity, KV, grammar, short fwd/bwd."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from architecture import (
    Block,
    GroupedAttention,
    HadamardMLP,
    NeedleZh,
    ZCRMSNorm,
    apply_rope,
    count_params,
    engram_indices,
    precompute_rope,
    rms_unit,
    sinkhorn,
    walsh_matrix,
)
from checkpoint import load_params, load_train_state, save_params, save_train_state
from config import NeedleZhConfig
from data import PackedTokenSource, encode_sft_row, pack_windows, write_uint16_tokens
from decode import greedy_constrained
from grammar import dump_calls, is_legal_prefix, parse_phase1_text
from parity import (
    apply_rope_np,
    engram_indices_np,
    gqa_attn_np,
    hadamard_forward_np,
    max_abs,
    precompute_rope_np,
    rms_unit_np,
    sinkhorn_np,
    walsh_np,
    zc_rms_np,
)
from tokenizer import ZhTokenizerV1
from train_common import eval_lm_loss, segment_throughput, stratified_window_indices, train_lm_steps


def to_np(x) -> np.ndarray:
    mx.eval(x)
    return np.array(x)


def param_max_abs(a, b) -> float:
    import mlx.utils as xu

    da, db = xu.tree_flatten(a.parameters()), xu.tree_flatten(b.parameters())
    if isinstance(da, tuple):
        da = da[0]
    if isinstance(db, tuple):
        db = db[0]
    ma = {k: v for k, v in da}
    mb = {k: v for k, v in db}
    worst = 0.0
    for k, va in ma.items():
        worst = max(worst, max_abs(to_np(va), to_np(mb[k])))
    return worst


def test_grammar() -> dict:
    ok = parse_phase1_text("[]")
    assert ok["ok"]
    nod = parse_phase1_text('[{"name":"nod","arguments":{}}]')
    assert nod["ok"], nod
    bad = parse_phase1_text(
        '[{"name":"order_food","arguments":{"shop":"兰州拉面","dish":"巨无霸"}}]'
    )
    assert not bad["ok"]
    assert is_legal_prefix("[")
    assert is_legal_prefix("[]")
    assert not is_legal_prefix("请")
    dumped = dump_calls([{"name": "nod", "arguments": {}}])
    assert dumped.startswith("[")
    return {"grammar_ok": True}


def test_tokenizer() -> dict:
    tok = ZhTokenizerV1()
    assert tok.vocab_size == 24000
    assert (tok.pad_id, tok.eos_id, tok.bos_id, tok.unk_id) == (0, 1, 2, 3)
    ids = tok.encode_document("厨房灯打开")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    chat = tok.encode_chat("关灯", "[]")
    assert chat["n_prompt"] > 0
    assert chat["ids"][: chat["n_prompt"]] == chat["prompt_ids"]
    text = tok.decode(chat["answer_ids"])
    assert "[" in text
    return {"vocab_size": tok.vocab_size, "model_sha256": tok.model_sha256, "tokenizer_ok": True}


def test_sft_mask() -> dict:
    tok = ZhTokenizerV1()
    row = {
        "query": "关掉厨房灯",
        "answers": [{"name": "set_switch", "arguments": {"id": "kitchen_light", "on": False}}],
        "confidence_label": 1,
    }
    packed = encode_sft_row(tok, row, 64)
    assert packed["n_unmasked"] > 0
    prompt_positions = [i for i, _m in enumerate(packed["mask"]) if i + 1 < packed["n_prompt"]]
    assert all(packed["mask"][i] == 0.0 for i in prompt_positions)
    assert any(m == 1.0 for m in packed["mask"])
    return {"n_unmasked": packed["n_unmasked"], "n_prompt": packed["n_prompt"], "mask_ok": True}


def test_tiny_overfit() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    ids = tok.encode_document("把灯打开")
    windows = pack_windows([ids], 16, pad_id=tok.pad_id)
    result = train_lm_steps(model, windows, steps=24, lr=2e-3, seed=0)
    ok = result["last_loss"] is not None and result["last_loss"] < 8.0
    return {
        "tiny_params": count_params(model),
        "overfit_loss": result["last_loss"],
        "tokens_seen": result["tokens_seen"],
        "overfit_ok": ok,
    }


def test_resume() -> dict:
    cfg = NeedleZhConfig().tiny()
    batches = [
        {
            "x": [2, 11, 12, 13, 1, 0],
            "y": [11, 12, 13, 1, 0, 0],
            "mask": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        init_path = Path(td) / "init.npz"
        state_path = Path(td) / "state.npz"
        mx.random.seed(0)
        model_init = NeedleZh(cfg)
        mx.eval(model_init.parameters())
        save_params(model_init, init_path)

        model_ref = NeedleZh(cfg)
        load_params(model_ref, init_path, strict=True)
        ref = train_lm_steps(model_ref, batches, steps=8, lr=2e-3, seed=7, total_steps=8)

        model_a = NeedleZh(cfg)
        load_params(model_a, init_path, strict=True)
        first = train_lm_steps(model_a, batches, steps=4, lr=2e-3, seed=7, total_steps=8)
        save_train_state(
            state_path,
            model_a,
            first["optimizer"],
            {"step": 4, "tokens_seen": first["tokens_seen"]},
        )

        model_b = NeedleZh(cfg)
        opt_b = optim.Adam(learning_rate=2e-3)
        meta = load_train_state(state_path, model_b, opt_b, strict=True)
        second = train_lm_steps(
            model_b,
            batches,
            steps=4,
            lr=2e-3,
            seed=7,
            start_step=int(meta.get("step") or 4),
            start_tokens_seen=int(meta.get("tokens_seen") or 0),
            total_steps=8,
            optimizer=opt_b,
            reseed=False,
        )
        delta = param_max_abs(model_ref, model_b)
        loss_delta = abs(float(ref["last_loss"]) - float(second["last_loss"]))
    ok = delta < 5e-4 and loss_delta < 5e-4 and second["tokens_seen"] == ref["tokens_seen"]
    return {
        "resume_ok": ok,
        "param_max_abs": delta,
        "loss_delta": loss_delta,
        "tokens_seen": second["tokens_seen"],
        "ref_loss": ref["last_loss"],
        "resume_loss": second["last_loss"],
    }


def test_op_parity() -> dict:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 6, 16)).astype(np.float32)
    xm = mx.array(x)
    diffs = {}
    diffs["rms"] = max_abs(to_np(rms_unit(xm)), rms_unit_np(x))
    scale = rng.standard_normal((16,)).astype(np.float32)
    norm = ZCRMSNorm(16)
    norm.scale = mx.array(scale)
    diffs["zc_rms"] = max_abs(to_np(norm(xm)), zc_rms_np(x, scale))
    rope_m = precompute_rope(8, 8, 10000.0)
    rope_n = precompute_rope_np(8, 8, 10000.0)
    q = rng.standard_normal((1, 4, 6, 8)).astype(np.float32)
    diffs["rope"] = max_abs(
        to_np(apply_rope(mx.array(q), *rope_m, offset=2)),
        apply_rope_np(q, *rope_n, offset=2),
    )
    mlp = HadamardMLP(16)
    n = int(mlp.n)
    h = walsh_np(n)
    diffs["walsh"] = max_abs(to_np(walsh_matrix(n)), h)
    d1 = rng.standard_normal((n,)).astype(np.float32)
    d2 = rng.standard_normal((n,)).astype(np.float32)
    d3 = rng.standard_normal((n,)).astype(np.float32)
    mlp.d1, mlp.d2, mlp.d3 = mx.array(d1), mx.array(d2), mx.array(d3)
    diffs["hadamard"] = max_abs(to_np(mlp(xm)), hadamard_forward_np(x, h, d1, d2, d3, 16))
    toks = rng.integers(4, 200, size=(1, 8), dtype=np.int32)
    diffs["engram"] = max_abs(
        to_np(engram_indices(mx.array(toks), (2, 3), 2, 128)),
        engram_indices_np(toks, (2, 3), 2, 128),
    )
    logits = rng.standard_normal((1, 4, 4, 4)).astype(np.float32)
    diffs["sinkhorn"] = max_abs(to_np(sinkhorn(mx.array(logits), iters=8)), sinkhorn_np(logits, iters=8))
    qh = rng.standard_normal((1, 4, 6, 8)).astype(np.float32)
    kh = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
    vh = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
    mask = np.tril(np.ones((6, 6), dtype=bool))[None, None, :, :]
    scale_gqa = 8 ** -0.5
    repeats = 2
    qm, km, vm, mm = mx.array(qh), mx.array(kh), mx.array(vh), mx.array(mask)
    k_use_m = mx.repeat(km, repeats, axis=1)
    v_use_m = mx.repeat(vm, repeats, axis=1)
    attn_m = (qm * scale_gqa) @ mx.swapaxes(k_use_m, -1, -2)
    attn_m = mx.where(mm, attn_m, mx.array(-1e9, dtype=attn_m.dtype))
    attn_m = mx.softmax(attn_m.astype(mx.float32), axis=-1)
    out_m = attn_m @ v_use_m
    diffs["gqa"] = max_abs(to_np(out_m), gqa_attn_np(qh, kh, vh, scale_gqa, mask))
    jax_ok = False
    try:
        import jax  # noqa: F401

        jax_ok = True
    except Exception:
        jax_ok = False
    ok = all(v < 2e-4 for v in diffs.values())
    return {"parity_ok": ok, "diffs": diffs, "jax_available": jax_ok}


def test_kv_cache() -> dict:
    cfg = NeedleZhConfig().tiny()
    attn = GroupedAttention(cfg)
    mx.eval(attn.parameters())
    x = mx.random.normal((1, 8, cfg.d_model))
    rope = precompute_rope(cfg.head_dim, 8, cfg.rope_theta)
    t = 8
    mask = (mx.arange(t)[:, None] >= mx.arange(t)[None, :])[None, None, :, :]
    y_full, _ = attn(x, rope, mask=mask)
    _y1, c1 = attn(x[:, :4], rope, mask=mask[:, :, :4, :4])
    q_pos = mx.arange(4) + 4
    k_pos = mx.arange(8)
    mask2 = (q_pos[:, None] >= k_pos[None, :])[None, None, :, :]
    y2, _ = attn(x[:, 4:], rope, mask=mask2, cache=c1, rope_offset=4)
    delta = max_abs(to_np(y2), to_np(y_full[:, 4:]))
    return {"kv_ok": delta < 2e-4, "max_abs": delta}


def test_block_and_decode() -> dict:
    cfg = NeedleZhConfig().tiny()
    block = Block(cfg)
    mx.eval(block.parameters())
    x = mx.random.normal((1, 6, cfg.d_model))
    rope = precompute_rope(cfg.head_dim, 6, cfg.rope_theta)
    t = 6
    mask = (mx.arange(t)[:, None] >= mx.arange(t)[None, :])[None, None, :, :]

    def loss_fn(b):
        y, _ = b(x, rope, mask=mask)
        return y.mean()

    loss, grads = nn.value_and_grad(block, loss_fn)(block)
    mx.eval(loss, grads)
    tok = ZhTokenizerV1()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    prompt = tok.encode_chat("点头")["prompt_ids"]
    out = greedy_constrained(model, tok, prompt, max_new=16)
    return {
        "block_loss_finite": bool(np.isfinite(float(loss))),
        "decode_ran": "text" in out,
        "prefix_ok": is_legal_prefix("[") and is_legal_prefix(out["text"] or "["),
        "grads_ok": grads is not None,
    }


def test_full_short() -> dict:
    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    n = count_params(model)
    x = mx.array([[2, 11, 12, 13, 14, 15, 16, 1]], dtype=mx.int32)
    peak = None
    if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()

    def loss_fn(m):
        return m(x)["logits"].mean()

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        peak = int(mx.metal.get_peak_memory())
    return {
        "full_params": n,
        "in_band": 45_000_000 <= n <= 60_000_000,
        "loss_finite": bool(np.isfinite(float(loss))),
        "peak_bytes": peak,
        "full_ok": bool(45_000_000 <= n <= 60_000_000 and np.isfinite(float(loss))),
        "grads_ok": grads is not None,
    }


def test_token_stop() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    windows = [
        {"x": [2, 11, 12, 13], "y": [11, 12, 13, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
        {"x": [2, 14, 15, 16], "y": [14, 15, 16, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
    ]
    result = train_lm_steps(
        model,
        windows,
        lr=2e-3,
        seed=0,
        batch_size=1,
        grad_accum=1,
        target_tokens=12,
        total_steps=16,
    )
    overshoot = result["tokens_seen"] - 12
    ok = result["tokens_seen"] >= 12 and overshoot <= 4
    return {
        "token_stop_ok": ok,
        "tokens_seen": result["tokens_seen"],
        "overshoot": overshoot,
        "steps": result["steps"],
    }


def test_resume_accum() -> dict:
    cfg = NeedleZhConfig().tiny()
    batches = [
        {"x": [2, 11, 12, 13], "y": [11, 12, 13, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
        {"x": [2, 14, 15, 16], "y": [14, 15, 16, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
    ]
    with tempfile.TemporaryDirectory() as td:
        init_path = Path(td) / "init.npz"
        state_path = Path(td) / "state.npz"
        mx.random.seed(0)
        model_init = NeedleZh(cfg)
        mx.eval(model_init.parameters())
        save_params(model_init, init_path)
        model_ref = NeedleZh(cfg)
        load_params(model_ref, init_path, strict=True)
        ref = train_lm_steps(
            model_ref,
            batches,
            steps=4,
            lr=2e-3,
            seed=7,
            total_steps=8,
            batch_size=1,
            grad_accum=2,
        )
        model_a = NeedleZh(cfg)
        load_params(model_a, init_path, strict=True)
        first = train_lm_steps(
            model_a,
            batches,
            steps=2,
            lr=2e-3,
            seed=7,
            total_steps=8,
            batch_size=1,
            grad_accum=2,
        )
        save_train_state(
            state_path,
            model_a,
            first["optimizer"],
            {
                "step": first["steps"],
                "tokens_seen": first["tokens_seen"],
                "window_index": first["window_index"],
                "tokenizer_sha256": "abc",
                "corpus_sha256": "def",
                "seq_len": 4,
            },
        )
        model_b = NeedleZh(cfg)
        opt_b = optim.Adam(learning_rate=2e-3)
        meta = load_train_state(state_path, model_b, opt_b, strict=True)
        second = train_lm_steps(
            model_b,
            batches,
            steps=2,
            lr=2e-3,
            seed=7,
            start_step=int(meta.get("step") or 2),
            start_tokens_seen=int(meta.get("tokens_seen") or 0),
            start_window=int(meta.get("window_index") or 0),
            total_steps=8,
            optimizer=opt_b,
            reseed=False,
            batch_size=1,
            grad_accum=2,
        )
        delta = param_max_abs(model_ref, model_b)
        loss_delta = abs(float(ref["last_loss"]) - float(second["last_loss"]))
        bad = NeedleZh(cfg)
        opt_bad = optim.Adam(learning_rate=2e-3)
        rejected = False
        try:
            load_train_state(
                state_path,
                bad,
                opt_bad,
                strict=True,
                expected_meta={"tokenizer_sha256": "nope", "corpus_sha256": "def", "seq_len": 4},
            )
        except ValueError:
            rejected = True
    ok = delta < 5e-4 and loss_delta < 5e-4 and rejected and second["tokens_seen"] == ref["tokens_seen"]
    return {
        "resume_accum_ok": ok,
        "param_max_abs": delta,
        "loss_delta": loss_delta,
        "hash_reject_ok": rejected,
        "tokens_seen": second["tokens_seen"],
    }


def test_mmap_source() -> dict:
    seq = 4
    ids = list(range(4, 4 + 40))
    with tempfile.TemporaryDirectory() as td:
        p0 = Path(td) / "train-0000.bin"
        p1 = Path(td) / "train-0001.bin"
        write_uint16_tokens(p0, ids[:24])
        write_uint16_tokens(p1, ids[24:])
        src = PackedTokenSource([p0, p1], seq, pad_id=0)
        listed = pack_windows([ids], seq, pad_id=0)
        same = src[0] == listed[0] and src[3] == listed[3]
        crossed = src[5]
        listed5 = listed[5]
        boundary = crossed == listed5
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.bin"
        write_uint16_tokens(p, ids)
        src = PackedTokenSource([p], seq, pad_id=0)
        listed = pack_windows([ids], seq, pad_id=0)
        a = train_lm_steps(model, src, steps=2, lr=2e-3, seed=1, batch_size=1, grad_accum=1, allow_repeat=False)
        model_b = NeedleZh(cfg)
        mx.eval(model_b.parameters())
        # new random init; compare only packing equality above
    return {
        "mmap_source_ok": same and boundary and len(src) == len(listed),
        "n_windows": len(src),
        "n_listed": len(listed),
        "short_run_steps": a["steps"],
    }


def test_unique_epoch_stop() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    windows = [
        {"x": [2, 11, 12, 13], "y": [11, 12, 13, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
        {"x": [2, 14, 15, 16], "y": [14, 15, 16, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
    ]
    result = train_lm_steps(
        model,
        windows,
        lr=2e-3,
        seed=0,
        batch_size=1,
        grad_accum=1,
        target_tokens=10_000,
        total_steps=32,
        allow_repeat=False,
    )
    ok = result["exhausted"] is True and result["window_index"] == 2 and result["tokens_seen"] > 0
    return {
        "unique_epoch_ok": ok,
        "exhausted": result["exhausted"],
        "window_index": result["window_index"],
        "tokens_seen": result["tokens_seen"],
        "steps": result["steps"],
    }


def test_unique_epoch_last_window() -> dict:
    seq = 4
    ids = list(range(4, 14))
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    with tempfile.TemporaryDirectory() as td:
        p0 = Path(td) / "a.bin"
        p1 = Path(td) / "b.bin"
        write_uint16_tokens(p0, ids[:6])
        write_uint16_tokens(p1, ids[6:])
        src = PackedTokenSource([p0, p1], seq, pad_id=0)
        result = train_lm_steps(
            model,
            src,
            lr=2e-3,
            seed=0,
            batch_size=1,
            grad_accum=1,
            target_tokens=10_000,
            total_steps=32,
            allow_repeat=False,
        )
        pred = int(src.n_predictable_tokens)
        n_windows = len(src)
        ok = (
            pred == len(ids) - 1
            and result["exhausted"] is True
            and result["window_index"] == n_windows
            and int(result["tokens_seen"]) == pred
        )
        idx = stratified_window_indices(n_windows, 3)
        loss = eval_lm_loss(model, src, max_windows=None, indices=idx, batch_size=2, token_weighted=True)
        stream_ok = loss == loss and abs(float(loss)) != float("inf")
    thru = segment_throughput(100_005_710, 100_000_000, 1.0)
    thru_ok = abs(thru - 5710.0) < 1e-6
    return {
        "unique_tail_ok": ok and stream_ok and thru_ok,
        "n_predictable": pred,
        "exhausted": result["exhausted"],
        "window_index": result["window_index"],
        "n_windows": n_windows,
        "tokens_seen": result["tokens_seen"],
        "stream_loss_finite": stream_ok,
        "segment_tok_s": thru,
    }


def test_resume_strict_meta() -> dict:
    cfg = NeedleZhConfig().tiny()
    batches = [
        {"x": [2, 11, 12, 13], "y": [11, 12, 13, 1], "mask": [1.0, 1.0, 1.0, 1.0]},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.npz"
        mx.random.seed(0)
        model = NeedleZh(cfg)
        mx.eval(model.parameters())
        opt = optim.Adam(learning_rate=2e-3)
        save_train_state(
            path,
            model,
            opt,
            {
                "step": 1,
                "tokens_seen": 4,
                "window_index": 1,
                "tokenizer_sha256": "tok",
                "corpus_sha256": "corp",
                "seq_len": 4,
                "lr_horizon_tokens": 1000,
                "precision": "fp32",
            },
        )
        bad = NeedleZh(cfg)
        opt_bad = optim.Adam(learning_rate=2e-3)
        rejected_horizon = False
        try:
            load_train_state(
                path,
                bad,
                opt_bad,
                strict=True,
                expected_meta={
                    "tokenizer_sha256": "tok",
                    "corpus_sha256": "corp",
                    "seq_len": 4,
                    "lr_horizon_tokens": 9999,
                    "precision": "fp32",
                },
            )
        except ValueError:
            rejected_horizon = True
        rejected_precision = False
        try:
            load_train_state(
                path,
                bad,
                opt_bad,
                strict=True,
                expected_meta={
                    "tokenizer_sha256": "tok",
                    "corpus_sha256": "corp",
                    "seq_len": 4,
                    "lr_horizon_tokens": 1000,
                    "precision": "fp16",
                },
            )
        except ValueError:
            rejected_precision = True
        rejected_manifest = False
        try:
            load_train_state(
                path,
                bad,
                opt_bad,
                strict=True,
                expected_meta={
                    "tokenizer_sha256": "tok",
                    "corpus_sha256": "corp",
                    "seq_len": 4,
                    "manifest_sha256": "missing",
                },
            )
        except ValueError:
            rejected_manifest = True
    ok = rejected_horizon and rejected_precision and rejected_manifest
    return {
        "strict_meta_ok": ok,
        "rejected_horizon": rejected_horizon,
        "rejected_precision": rejected_precision,
        "rejected_manifest": rejected_manifest,
    }


def _run(name, fn):
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", f"{name}_ok": False}


def main() -> int:
    report = {
        "grammar": _run("grammar", test_grammar),
        "tokenizer": _run("tokenizer", test_tokenizer),
        "sft_mask": _run("sft_mask", test_sft_mask),
        "tiny": _run("tiny", test_tiny_overfit),
        "resume": _run("resume", test_resume),
        "token_stop": _run("token_stop", test_token_stop),
        "resume_accum": _run("resume_accum", test_resume_accum),
        "mmap_source": _run("mmap_source", test_mmap_source),
        "unique_epoch": _run("unique_epoch", test_unique_epoch_stop),
        "unique_tail": _run("unique_tail", test_unique_epoch_last_window),
        "strict_meta": _run("strict_meta", test_resume_strict_meta),
        "parity": _run("parity", test_op_parity),
        "kv": _run("kv", test_kv_cache),
        "block_decode": _run("block_decode", test_block_and_decode),
        "full": _run("full", test_full_short),
    }
    flags = [
        report["grammar"].get("grammar_ok"),
        report["tokenizer"].get("tokenizer_ok"),
        report["sft_mask"].get("mask_ok"),
        report["tiny"].get("overfit_ok"),
        report["resume"].get("resume_ok"),
        report["token_stop"].get("token_stop_ok"),
        report["resume_accum"].get("resume_accum_ok"),
        report["mmap_source"].get("mmap_source_ok"),
        report["unique_epoch"].get("unique_epoch_ok"),
        report["unique_tail"].get("unique_tail_ok"),
        report["strict_meta"].get("strict_meta_ok"),
        report["parity"].get("parity_ok"),
        report["kv"].get("kv_ok"),
        report["block_decode"].get("block_loss_finite"),
        report["full"].get("full_ok"),
    ]
    report["ok"] = all(bool(v) for v in flags)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
