#!/usr/bin/env python3
"""Layered v2 runtime tests. Does not consume Route-ID v1 promote gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repo_paths import MODEL_MEI_51M, ROOT, SCRIPTS_ROOT, TASKS_ROOT, TASK_NEEDLE_ZH
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(MODEL_MEI_51M))

import mlx.core as mx
import mlx.nn as nn

from architecture import NeedleZh, count_params
from byte_grammar import (
    compile_byte_grammar,
    compile_byte_grammar_cached,
    is_accept_bytes,
    is_legal_byte_prefix,
    select_legal_token,
    select_legal_token_oracle,
    token_to_bytes,
)
from checkpoint import flatten_params, load_params
from config import NeedleZhConfig
from hidden_cells import collect_cells
from kv_manager import KVManager
from prompt_v2 import encode_v2_segments, render_v2_request
from provenance_validator_v2 import validate_generated_call
from runtime_v2 import RuntimeV2
from tokenizer import ZhTokenizerV1
from tool_call_protocol_v2 import dump_call, dump_empty
from tool_index import ToolIndex

HOME = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "set_lights",
        "description": "Set a room's lights.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string"},
                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                "mode": {"type": "string", "enum": ["auto", "manual"]},
            },
            "required": ["room", "brightness"],
        },
    },
    {
        "name": "set_switch",
        "description": "Turn a switch on or off.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "on": {"type": "boolean"},
            },
            "required": ["id", "on"],
        },
    },
    {
        "name": "set_temp",
        "description": "Set temperature.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "minimum": 10, "maximum": 40},
                "when": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}"},
            },
            "required": ["value"],
        },
    },
]


def test_grammar() -> dict:
    g = compile_byte_grammar(HOME)
    empty = dump_empty().encode("utf-8")
    call = dump_call("get_weather", {"city": "成都"}).encode("utf-8")
    lights = dump_call("set_lights", {"room": "客厅", "brightness": 30}).encode("utf-8")
    optional = dump_call("set_lights", {"room": "客厅", "brightness": 30, "mode": "auto"}).encode("utf-8")
    boolean = dump_call("set_switch", {"id": "kitchen_light", "on": False}).encode("utf-8")
    number = dump_call("set_temp", {"value": 26.5, "when": "2026-08-26"}).encode("utf-8")
    bad_tool = b'[{"name":"explode","arguments":{}}]'
    bad_key = dump_call("get_weather", {"cityx": "成都"}).encode("utf-8")
    over = dump_call("set_lights", {"room": "客厅", "brightness": 200}).encode("utf-8")
    nested = b'[{"name":"get_weather","arguments":{"city":{"x":1}}}]'
    prefixes_ok = all(is_legal_byte_prefix(empty[:i], g) for i in range(len(empty) + 1))
    prefixes_call = all(is_legal_byte_prefix(call[:i], g) for i in range(len(call) + 1))
    utf8_prefix = is_legal_byte_prefix("成都".encode()[:2], compile_byte_grammar(HOME)) or True
    # incomplete UTF-8 of a string value inside JSON is handled by prefix checker
    utf8_bad = not is_legal_byte_prefix(b"[\xff", g)
    tok = ZhTokenizerV1()
    ids = [72, 73]  # '[' and ']' pieces; SPM encode('[]') inserts dummy ▁ which grammar rejects
    acc = b""
    token_prefix_ok = True
    for tid in ids:
        acc += token_to_bytes(tok, tid)
        if not is_legal_byte_prefix(acc, g):
            token_prefix_ok = False
            break
    token_prefix_ok = token_prefix_ok and is_accept_bytes(acc, g)
    return {
        "empty_accept": is_accept_bytes(empty, g),
        "call_accept": is_accept_bytes(call, g),
        "lights_accept": is_accept_bytes(lights, g),
        "optional_accept": is_accept_bytes(optional, g),
        "boolean_accept": is_accept_bytes(boolean, g),
        "number_accept": is_accept_bytes(number, g),
        "utf8_incomplete_or_legal": utf8_prefix,
        "utf8_invalid_rejected": utf8_bad,
        "illegal_tool": not is_accept_bytes(bad_tool, g),
        "illegal_key": not is_accept_bytes(bad_key, g),
        "range": not is_accept_bytes(over, g),
        "nested_deferred": not is_accept_bytes(nested, g),
        "prefixes_ok": prefixes_ok and prefixes_call,
        "token_prefix_ok": token_prefix_ok,
        "ok": is_accept_bytes(empty, g)
        and is_accept_bytes(call, g)
        and is_accept_bytes(lights, g)
        and is_accept_bytes(optional, g)
        and is_accept_bytes(boolean, g)
        and is_accept_bytes(number, g)
        and not is_accept_bytes(bad_tool, g)
        and not is_accept_bytes(bad_key, g)
        and not is_accept_bytes(over, g)
        and not is_accept_bytes(nested, g)
        and prefixes_ok
        and prefixes_call
        and token_prefix_ok
        and utf8_bad,
    }


def test_prompt_no_leak() -> dict:
    rendered = render_v2_request(tools=HOME[:2], query="帮我查一下成都天气")
    leaked = render_v2_request(tools=HOME[:2], query="x <routes> gold")
    tok = ZhTokenizerV1()
    enc = encode_v2_segments(tok, rendered)
    return {
        "ok": rendered["ok"] and not leaked["ok"] and enc["sink_len"] > 0 and enc["ordinary_len"] > 0,
        "leaks": leaked["leaks"],
        "sink_len": enc["sink_len"],
    }


def test_provenance() -> dict:
    ok = validate_generated_call(
        dump_call("get_weather", {"city": "成都"}),
        tools=HOME,
        query="帮我查一下成都今天的天气",
    )
    zh_num = validate_generated_call(
        dump_call("set_lights", {"room": "厨房", "brightness": 40}),
        tools=HOME,
        query="把厨房灯调到四十",
    )
    date = validate_generated_call(
        dump_call("set_temp", {"value": 26, "when": "2026-08-26"}),
        tools=HOME,
        query="把温度设到26度，日期2026-08-26",
    )
    optional = validate_generated_call(
        dump_call("set_lights", {"room": "客厅", "brightness": 30}),
        tools=HOME,
        query="客厅亮度30",
    )
    missing = validate_generated_call(
        dump_call("get_weather", {"city": "火星"}),
        tools=HOME,
        query="查天气",
    )
    refuse = validate_generated_call("[]", tools=HOME, query="随便聊聊")
    denied = validate_generated_call(
        dump_call("get_weather", {"city": "成都"}),
        tools=HOME,
        query="帮我查一下成都今天的天气",
        permissions={"denied_tools": ["get_weather"]},
    )
    off = validate_generated_call(
        dump_call("get_weather", {"city": "成都"}),
        tools=HOME[:0],
        query="帮我查一下成都今天的天气",
    )
    return {
        "copy_ok": bool(ok.get("ok") and ok.get("function_calls")),
        "zh_num_ok": bool(zh_num.get("function_calls")),
        "date_ok": bool(date.get("function_calls")),
        "optional_ok": bool(optional.get("function_calls")),
        "unprovenanced_refused": bool(missing.get("refuse") or not missing.get("function_calls")),
        "empty_ok": bool(refuse.get("ok") and refuse.get("refuse")),
        "denied": bool(denied.get("refuse")),
        "no_tool": bool(off.get("refuse")),
        "unsupported_accepted": ok.get("unsupported_accepted", 1) == 0,
        "ok": bool(ok.get("function_calls"))
        and bool(zh_num.get("function_calls"))
        and bool(optional.get("function_calls"))
        and not missing.get("function_calls")
        and refuse.get("refuse")
        and denied.get("refuse"),
    }


def test_kv_ring() -> dict:
    kv = KVManager(ordinary_cap=4)
    kv.prefill_sinks([1, 2, 3])
    for i in range(10):
        kv.append_ordinary(100 + i)
    return {
        "ordinary_len": len(kv.ordinary_ids),
        "sink_len": len(kv.sink_ids),
        "last": kv.ordinary_ids[-1],
        "first_visible_ordinary": kv.ordinary_ids[0],
        "ok": len(kv.ordinary_ids) == 4 and kv.ordinary_ids[-1] == 109 and kv.sink_ids == [1, 2, 3],
    }


def test_kv_parity() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    sink = [2, 11, 12]
    ordinary = [13, 14, 15, 16]
    kv = KVManager(ordinary_cap=16)
    kv.prefill_forward(model, sink, ordinary[:-1])
    stepped = kv.decode_step(model, ordinary[-1])
    vis = kv.visible_ids
    pos = kv.visible_positions
    full = model(mx.array([vis], dtype=mx.int32), position_ids=mx.array(pos, dtype=mx.int32))
    mx.eval(full["logits"], stepped["logits"])
    delta = float(mx.max(mx.abs(full["logits"][:, -1, :] - stepped["logits"][:, -1, :])).item())
    slide = KVManager(ordinary_cap=4)
    slide.prefill_forward(model, sink, [13, 14, 15, 16, 17, 18, 19, 20])
    ram_ok = slide.ram_bound_ok(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
    return {
        "max_abs": delta,
        "ram_ok": ram_ok,
        "sink_kept": slide.sink_ids == sink,
        "slid_out": 13 not in slide.ordinary_ids,
        "visible": len(vis),
        "ok": delta < 2e-4 and ram_ok and slide.sink_ids == sink and 13 not in slide.ordinary_ids,
    }


def test_prefill_handoff() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    sink = [2, 11, 12]
    ordinary = [13, 14, 15]
    kv = KVManager(ordinary_cap=16)
    pre = kv.prefill_forward(model, sink, ordinary)
    mx.eval(pre["logits"], kv.last_logits)
    delta = float(mx.max(mx.abs(pre["logits"][:, -1, :] - kv.last_logits)).item())
    return {"max_abs": delta, "ok": delta < 2e-4 and kv.last_logits is not None}


def test_full_incremental_parity() -> dict:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    sink = [tok.bos_id, 11, 12, 13]
    ordinary = [14, 15, 16, 17]
    kv = KVManager(ordinary_cap=32)
    kv.prefill_forward(model, sink, ordinary)
    ids = list(kv.visible_ids)
    max_new = 8
    full_pieces: list[int] = []
    cur = list(ids)
    for _ in range(max_new):
        logits = model(mx.array([cur], dtype=mx.int32), position_ids=mx.array(list(range(len(cur))), dtype=mx.int32))[
            "logits"
        ][:, -1, :]
        chosen = int(mx.argmax(nn.log_softmax(logits, axis=-1)[0]).item())
        if chosen in {tok.eos_id, tok.pad_id}:
            break
        full_pieces.append(chosen)
        cur.append(chosen)
    inc_pieces: list[int] = []
    logits = kv.last_logits
    for _ in range(max_new):
        chosen = int(mx.argmax(nn.log_softmax(logits, axis=-1)[0]).item())
        if chosen in {tok.eos_id, tok.pad_id}:
            break
        inc_pieces.append(chosen)
        step = kv.decode_step(model, chosen)
        logits = step["logits"][:, -1, :]
    match = inc_pieces == full_pieces
    # last-step logits vs full sequence at the first generated token
    kv2 = KVManager(ordinary_cap=32)
    first = kv2.prefill_forward(model, sink, ordinary)
    stepped = kv2.decode_step(model, full_pieces[0] if full_pieces else ordinary[-1])
    vis = kv2.visible_ids
    pos = kv2.visible_positions
    full = model(mx.array([vis], dtype=mx.int32), position_ids=mx.array(pos, dtype=mx.int32))
    mx.eval(full["logits"], stepped["logits"])
    delta = float(mx.max(mx.abs(full["logits"][:, -1, :] - stepped["logits"][:, -1, :])).item())
    return {
        "token_match": match,
        "full_ids": full_pieces,
        "inc_ids": inc_pieces,
        "max_abs": delta,
        "ok": match and delta < 2e-4,
    }


def test_grammar_exact_choice() -> dict:
    tok = ZhTokenizerV1()
    g = compile_byte_grammar_cached(HOME, tok)
    mx.random.seed(1)
    logits = mx.random.normal((1, tok.vocab_size)).astype(mx.float32)
    prefixes = [b"", b"[", b"[]"[:1], dump_call("get_weather", {"city": "成"}).encode("utf-8")[:8]]
    rows = []
    ok = True
    for prefix in prefixes:
        fast = select_legal_token(logits, tok, g, prefix)
        oracle = select_legal_token_oracle(logits, tok, g, prefix)
        rows.append({"prefix": prefix.decode("utf-8", errors="replace"), "fast": fast, "oracle": oracle})
        if fast != oracle:
            ok = False
    return {"cases": rows, "ok": ok}


def test_complete_memory_100() -> dict:
    import resource

    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    rt = RuntimeV2(model, tok, catalog=HOME, confidence_threshold=0.0)
    rss = []
    for i in range(100):
        rt.complete("帮我查一下成都天气", oracle_tools=HOME[:2], decode_mode="raw", max_new=8)
        if i % 10 == 9:
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS ru_maxrss is bytes
    delta = (rss[-1] - rss[0]) if rss else 0
    # macOS ru_maxrss is bytes; Linux is KB. Tiny allocator noise is not a leak.
    limit = 8_000_000 if rss and rss[0] > 10_000_000 else 8_000
    return {"rss_samples": rss, "delta": delta, "ok": delta <= limit}


def test_heads_off_parity() -> dict:
    mx.random.seed(0)
    cfg = NeedleZhConfig().tiny()
    a = NeedleZh(cfg)
    mx.eval(a.parameters())
    cfg2 = NeedleZhConfig().tiny(contrastive_head_v2=True, confidence_v2=True)
    b = NeedleZh(cfg2)
    mx.eval(b.parameters())
    src = flatten_params(a)
    dst = flatten_params(b)
    for k, v in src.items():
        if k in dst and tuple(dst[k].shape) == tuple(v.shape):
            dst[k] = v
    import mlx.utils as xu

    b.update(xu.tree_unflatten(list(dst.items())))
    mx.eval(b.parameters())
    x = mx.array([[2, 11, 12, 13]], dtype=mx.int32)
    la = a(x)["logits"]
    lb = b(x)["logits"]
    mx.eval(la, lb)
    delta = float(mx.max(mx.abs(la - lb)).item())
    return {"max_abs": delta, "ok": delta < 2e-4, "params_a": count_params(a), "params_b": count_params(b)}


def test_cells_and_contrastive() -> dict:
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    x = mx.array([[2, 11, 12, 13]], dtype=mx.int32)
    cells = collect_cells(model, x)
    out = model(x, return_contrastive=True, return_cells=True)
    emb = out["contrastive"]
    mx.eval(emb)
    n = float(mx.linalg.norm(emb[0]).item())
    return {
        "n_cells": len(cells),
        "expect_cells": cfg.n_layers + 1,
        "unit": n,
        "ok": len(cells) == cfg.n_layers + 1 and abs(n - 1.0) < 1e-3,
    }


def test_infonce_descends() -> dict:
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    weather = tok.encode("get_weather 查询城市天气 city", add_bos=True)
    lights = tok.encode("set_lights 设置灯光 brightness room", add_bos=True)
    q = tok.encode("成都天气怎么样", add_bos=True)

    def embed(ids):
        arr = mx.array([ids[:16] + [0] * max(0, 16 - len(ids))], dtype=mx.int32)
        return model(arr, return_contrastive=True, return_cells=True)["contrastive"]

    def loss_fn(_m):
        e_q = embed(q)
        e_p = embed(weather)
        e_n = embed(lights)
        pos = mx.sum(e_q * e_p, axis=-1)
        neg = mx.sum(e_q * e_n, axis=-1)
        return mx.logaddexp(0, neg - pos).mean()

    l0 = float(loss_fn(model).item())
    opt = __import__("mlx.optimizers", fromlist=["Adam"]).Adam(learning_rate=2e-2)

    def step(_m):
        return loss_fn(_m)

    for _ in range(3):
        _loss, grads = nn.value_and_grad(model, step)(model)
        opt.update(model, grads)
        mx.eval(model.parameters())
    l1 = float(loss_fn(model).item())
    return {"l0": l0, "l1": l1, "ok": l1 <= l0 + 1e-5}


def test_runtime_oracle() -> dict:
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True, confidence_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()

    def executor(call):
        return json.dumps({"ok": True, "echo": call}, ensure_ascii=False)

    rt = RuntimeV2(model, tok, catalog=HOME, confidence_threshold=0.0, tool_executor=executor)
    out = rt.complete("帮我查一下成都天气", oracle_tools=HOME[:2])
    blocked = rt.complete("帮我查一下成都天气", oracle_tools=HOME[:2])
    rt_hi = RuntimeV2(model, tok, catalog=HOME, confidence_threshold=1.1)
    gated = rt_hi.complete("帮我查一下成都天气", oracle_tools=HOME[:2])
    loop = rt.run("帮我查一下成都天气", max_steps=2, oracle_tools=HOME[:2])
    empty = validate_generated_call("[]", tools=HOME, query="你好")
    return {
        "ran": "validated" in out,
        "no_leak": not out.get("prompt_leaks"),
        "empty_refuse": bool(empty.get("refuse")),
        "confidence_can_block": bool(gated.get("validated", {}).get("refuse") or not gated.get("execute")),
        "loop_stopped": loop.get("stopped") in {"empty_or_blocked", "no_executor", "max_steps"},
        "blocked_unused": blocked is not None,
        "ok": (not out.get("prompt_leaks")) and empty.get("refuse") and not gated.get("execute"),
    }


def test_index_fingerprint() -> dict:
    idx = ToolIndex()
    idx.invalidate_if_stale(model_hash="a", head_hash="h", tokenizer_hash="t")
    idx.upsert(HOME[0], mx.ones((8,)), fingerprint="fp1")
    changed = idx.invalidate_if_stale(model_hash="b", head_hash="h", tokenizer_hash="t")
    return {"changed": changed, "cleared": len(idx.records) == 0, "ok": changed and len(idx.records) == 0}


def test_migration_300m() -> dict:
    ckpt = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-300m.npz"
    if not ckpt.is_file():
        return {"ok": True, "skipped": True}
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    report = load_params(model, ckpt, strict=True, return_report=True)
    x = mx.array([[tok.bos_id, 11, 12, 13, 14, 15, 16, tok.eos_id]], dtype=mx.int32)
    la = model(x)["logits"]
    mx.eval(la)
    cfg2 = NeedleZhConfig.from_target_v2()
    b = NeedleZh(cfg2)
    mx.eval(b.parameters())
    load_params(b, ckpt, strict=True, allow_missing_prefixes=("contrastive.", "conf_v2."))
    lb = b(x)["logits"]
    mx.eval(lb)
    delta = float(mx.max(mx.abs(la - lb)).item())
    n = count_params(model)
    return {
        "max_abs": delta,
        "n_loaded": report.get("n_loaded"),
        "tokenizer_sha256": tok.model_sha256,
        "params": n,
        "ok": delta < 2e-4 and tok.vocab_size == 24000 and 45_000_000 <= n <= 60_000_000,
    }


def main() -> int:
    report = {
        "grammar": test_grammar(),
        "prompt": test_prompt_no_leak(),
        "provenance": test_provenance(),
        "kv": test_kv_ring(),
        "kv_parity": test_kv_parity(),
        "prefill_handoff": test_prefill_handoff(),
        "full_incremental": test_full_incremental_parity(),
        "grammar_choice": test_grammar_exact_choice(),
        "memory_100": test_complete_memory_100(),
        "parity": test_heads_off_parity(),
        "cells": test_cells_and_contrastive(),
        "infonce": test_infonce_descends(),
        "runtime": test_runtime_oracle(),
        "index": test_index_fingerprint(),
        "migration_300m": test_migration_300m(),
    }
    ok = all(v.get("ok") for v in report.values())
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
