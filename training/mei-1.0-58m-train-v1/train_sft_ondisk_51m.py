#!/usr/bin/env python3
"""Write-disk retrieval / full-call SFT / confidence on a quantized 51M parent.

Uses clean.v2 10k JSONL (architecture-agnostic). Does not train on 58M weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from identity_51m import (
    JOBS_DIR,
    Q4_BASELINE_MAP_NAME,
    Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_DIR,
    QAT_Q4_WEIGHTS_PATH,
    ROOT,
    SFT_QAT_BASE_DIR,
    SFT_QAT_MODEL_ID,
    SFT_QAT_PACKAGE_DIR,
    SFT_QAT_WEIGHTS_PATH,
    WEIGHTS_PATH,
    fail,
    load_json,
    refuse_base_write,
    sha256_file,
    write_json,
)
from qat_replay_51m import load_bit_map, quantize_tree
from quant_pack_51m import QUANT_MATH_ID
from tokenizer import ASSISTANT_PREFIX, TURN_END, USER_PREFIX


RET_PATH = ROOT / "notebook/sft/mei-1.0-58m/train/packs/mei-retrieval-v2-10k.clean.v2.candidates.jsonl"
FC_PATH = ROOT / "notebook/sft/mei-1.0-58m/train/packs/mei-toolcall-v2-oracle-10k.clean.v2.candidates.jsonl"
UNIVERSE_PATH = ROOT / "notebook/evaluation/banks/sft-v2-eval-lock-v2/tool-universe-v1.json"
ISOLATION = ROOT / "notebook/_tooling/scripts/check_train_eval_isolation.py"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def isolation_ok() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ISOLATION), "--scope", "sft-v2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload = {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"stdout": (proc.stdout or "")[:2000], "stderr": (proc.stderr or "")[:2000]}
    payload["returncode"] = proc.returncode
    return payload


def render_tool(tool: dict) -> str:
    return json.dumps(
        {
            "name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def universe_by_name() -> dict[str, dict]:
    blob = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    return {str(t["name"]): t for t in blob.get("tools") or []}


def cells_of(runtime, text: str, max_len: int = 96):
    ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=False)[:max_len]
    out = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
    mx.eval(*out["cells"])
    return [mx.stop_gradient(c).astype(mx.float16) for c in out["cells"]]


def train_contrastive(runtime, rows: list[dict], *, steps: int, lr: float) -> dict:
    from heads import ContrastiveHead

    cfg = runtime.model.cfg
    head = ContrastiveHead(cfg.d_model, cfg.n_layers, dim=32, probes=2)
    mx.eval(head.parameters())
    tools_u = universe_by_name()
    pairs = []
    for row in rows:
        gold = str(row.get("gold_tool") or "")
        catalog = row.get("catalog_tools") or []
        if isinstance(catalog, list) and catalog and isinstance(catalog[0], str):
            catalog = [tools_u[n] for n in catalog if n in tools_u]
        by_name = {str(t.get("name")): t for t in catalog if isinstance(t, dict)}
        gold_tool = by_name.get(gold) or tools_u.get(gold)
        if not gold_tool:
            continue
        negs = [n for n in (row.get("hard_negatives") or []) if n != gold]
        if not negs:
            negs = [n for n in by_name if n != gold][:3]
        if not negs:
            continue
        pairs.append((str(row.get("query") or ""), gold_tool, by_name.get(negs[0]) or tools_u.get(negs[0])))
    pairs = [p for p in pairs if p[2] is not None]
    if len(pairs) < 8:
        raise RuntimeError(f"not enough retrieval pairs: {len(pairs)}")
    rng = random.Random(51)
    opt = optim.Adam(learning_rate=lr)
    last = None
    batch = 8
    tool_cells: dict[str, list[mx.array]] = {}

    def cached_tool(tool: dict) -> list[mx.array]:
        name = str(tool.get("name") or render_tool(tool))
        if name not in tool_cells:
            tool_cells[name] = cells_of(runtime, render_tool(tool))
        return tool_cells[name]

    for step in range(steps):
        chunk = [pairs[rng.randrange(len(pairs))] for _ in range(batch)]
        q_cells = [cells_of(runtime, q) for q, _, _ in chunk]
        p_cells = [cached_tool(g) for _, g, _ in chunk]
        n_cells = [cached_tool(n) for _, _, n in chunk]

        def loss_fn(h):
            q = mx.concatenate([h(c) for c in q_cells], axis=0)
            p = mx.concatenate([h(c) for c in p_cells], axis=0)
            n = mx.concatenate([h(c) for c in n_cells], axis=0)
            return mx.mean(nn.softplus(mx.sum(q * n, axis=-1) - mx.sum(q * p, axis=-1)))

        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
        if step == 0 or (step + 1) % 50 == 0:
            print(json.dumps({"contrastive_step": step + 1, "loss": last}), flush=True)
    runtime.contrastive = head
    return {
        "steps": steps,
        "last_loss": last,
        "n_pairs": len(pairs),
        "n_cached_tools": len(tool_cells),
        "lm_frozen_after_qat_sft": True,
        "embedding_dim": 32,
        "probes": 2,
    }


def encode_fc(tok, row: dict, *, max_prompt: int = 1152, max_ans: int = 96) -> tuple[list[int], list[int]] | None:
    prompt_text = str(row.get("prompt_text") or "").strip()
    if not prompt_text:
        prompt_text = str(row.get("query") or "")
        prompt_text = USER_PREFIX + prompt_text
    prompt = tok.encode(prompt_text + ASSISTANT_PREFIX, add_bos=True, add_eos=False)[:max_prompt]
    answers = row.get("answers")
    if answers is None:
        name = row.get("gold_name")
        if not name:
            target = "[]"
        else:
            target = json.dumps(
                [{"name": name, "arguments": row.get("gold_args") or {}}],
                ensure_ascii=False,
                separators=(",", ":"),
            )
    elif answers == [] or answers == "[]":
        target = "[]"
    else:
        target = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
    answer = tok.encode(target) + tok.encode(TURN_END) + [tok.eos_id]
    answer = answer[:max_ans]
    if len(prompt) < 4 or len(answer) < 2:
        return None
    return prompt, answer


def train_fullcall(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    lr: float,
    bit_map: dict[str, int] | None,
    activation_ste: bool,
) -> dict:
    import architecture as arch
    from train_common import clip_grads

    tok = runtime.tokenizer
    pairs = []
    for row in rows:
        enc = encode_fc(tok, row)
        if enc:
            pairs.append(enc)
    if not pairs:
        raise RuntimeError("no full-call pairs")
    opt = optim.Adam(learning_rate=lr)
    last = None
    rng = random.Random(7)
    model = runtime.model
    order = list(range(len(pairs)))
    rng.shuffle(order)
    used: set[int] = set()
    arch.QAT_ACTIVATION_STE = bool(activation_ste and bit_map)
    try:
        for step in range(steps):
            if step > 0 and step % len(order) == 0:
                rng.shuffle(order)
            pair_index = order[step % len(order)]
            used.add(pair_index)
            prompt, answer = pairs[pair_index]
            ids = prompt + answer

            def loss_fn(params):
                if bit_map:
                    model.update(quantize_tree(params, bit_map, ste=True))
                else:
                    model.update(params)
                arr = mx.array([ids], dtype=mx.int32)
                logits = model(arr)["logits"].astype(mx.float32)
                logp = nn.log_softmax(logits, axis=-1)
                total = mx.array(0.0, dtype=mx.float32)
                for i, tid in enumerate(answer):
                    pos = len(prompt) - 1 + i
                    total = total + (-logp[0, pos, int(tid)])
                return total / max(len(answer), 1)

            params = model.parameters()
            loss, grads = mx.value_and_grad(loss_fn)(params)
            grads, grad_norm = clip_grads(grads, max_norm=1.0)
            model.update(params)
            opt.update(model, grads)
            mx.eval(model.parameters(), loss)
            last = float(loss.item())
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "sft_step": step + 1,
                            "loss": last,
                            "grad_norm": float(grad_norm.item()),
                            "qat_weight_ste": bool(bit_map),
                            "activation_kv_int8_ste": bool(activation_ste and bit_map),
                        }
                    ),
                    flush=True,
                )
    finally:
        arch.QAT_ACTIVATION_STE = False
    return {
        "steps": steps,
        "last_loss": last,
        "n_pairs": len(pairs),
        "unique_rows_exposed": len(used),
        "weight_qat_ste": bool(bit_map),
        "activation_kv_int8_ste": bool(activation_ste and bit_map),
        "quant_math_id": QUANT_MATH_ID if bit_map else None,
    }


def train_fullcall_float(runtime, rows: list[dict], *, steps: int, lr: float) -> dict:
    return train_fullcall(
        runtime,
        rows,
        steps=steps,
        lr=lr,
        bit_map=None,
        activation_ste=False,
    )


def train_confidence(runtime, rows: list[dict], *, steps: int) -> dict:
    from heads import ConfidenceV2Head

    tok = runtime.tokenizer
    head = ConfidenceV2Head(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    samples = []
    positives = [row for row in rows if int(row.get("confidence_label") or 0) == 1][:64]
    negatives = [row for row in rows if int(row.get("confidence_label") or 0) == 0][:64]
    for row in positives + negatives:
        enc = encode_fc(tok, row)
        if not enc:
            continue
        prompt, _answer = enc
        label = float(int(row.get("confidence_label") or 0))
        out = runtime.model(mx.array([prompt], dtype=mx.int32), return_cells=True)
        mx.eval(*out["cells"])
        samples.append(([mx.stop_gradient(c).astype(mx.float16) for c in out["cells"]], label))
    if len(samples) < 4:
        raise RuntimeError("not enough confidence samples")
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(h):
        total = mx.array(0.0, dtype=mx.float32)
        for cells, label in samples:
            logit = h(cells)
            y = mx.array(label, dtype=mx.float32)
            p = mx.clip(mx.sigmoid(logit.reshape(())), 1e-6, 1.0 - 1e-6)
            total = total + (-(y * mx.log(p) + (1.0 - y) * mx.log(1.0 - p)))
        return (total / max(len(samples), 1)).reshape(())

    last = None
    for _ in range(steps):
        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
    runtime.conf_v2 = head
    return {
        "steps": steps,
        "last_loss": last,
        "n": len(samples),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "used_explicit_confidence_label": True,
    }


def save_sft_package(src_pkg: Path, dest: Path, runtime) -> Path:
    from checkpoint import _atomic_savez, flatten_params, save_params

    blocked = refuse_base_write(SFT_QAT_WEIGHTS_PATH)
    if blocked:
        raise RuntimeError(blocked)
    SFT_QAT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    save_params(runtime.model, SFT_QAT_WEIGHTS_PATH)

    import sys
    from pack_qat_51m import main as pack_main

    argv = sys.argv
    sys.argv = [
        "pack_qat_51m.py",
        "--master",
        str(SFT_QAT_WEIGHTS_PATH),
        "--out-dir",
        str(dest),
        "--package-id",
        SFT_QAT_MODEL_ID,
        "--parent-id",
        "mei-1.0-51m-base-scratch300m-qat-q4-v1",
        "--parent-weights-sha256",
        sha256_file(QAT_Q4_WEIGHTS_PATH),
        "--scheme",
        "q4",
        "--receipt-name",
        "sft-qat-q4-package-receipt.json",
    ]
    try:
        code = pack_main()
    finally:
        sys.argv = argv
    if code != 0:
        raise RuntimeError(f"SFT Q4 pack failed code={code}")

    arrays = {}
    if runtime.contrastive is not None:
        arrays.update(
            {f"contrastive.{k}": v.astype(mx.float16) for k, v in flatten_params(runtime.contrastive).items()}
        )
    if runtime.conf_v2 is not None:
        arrays.update({f"conf_v2.{k}": v.astype(mx.float16) for k, v in flatten_params(runtime.conf_v2).items()})
    if arrays:
        _atomic_savez(dest / "heads.npz", arrays)
    man_path = dest / "mei-model.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["package_id"] = SFT_QAT_MODEL_ID
        heads = man.setdefault("heads", {})
        heads["contrastive"] = {
            "present": runtime.contrastive is not None,
            "trained": runtime.contrastive is not None,
            "status": "ready" if runtime.contrastive is not None else "missing",
            "dim": 32,
            "probes": 2,
        }
        heads["confidence"] = {
            "present": True,
            "trained": runtime.conf_v2 is not None,
            "status": "ready" if runtime.conf_v2 is not None else "untrained",
        }
        heads["artifact"] = {
            "file": "heads.npz",
            "sha256": sha256_file(dest / "heads.npz"),
            "dtype": "float16",
        }
        man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    package_bytes = sum(path.stat().st_size for path in dest.iterdir() if path.is_file())
    if package_bytes > 30 * 1024 * 1024:
        raise RuntimeError(f"SFT Q4 package {package_bytes} bytes exceeds 30MiB")
    release = {
        "kind": "quant-aware-sft-fp32-master",
        "model_id": SFT_QAT_MODEL_ID,
        "weights": str(SFT_QAT_WEIGHTS_PATH.relative_to(ROOT)),
        "weights_sha256": sha256_file(SFT_QAT_WEIGHTS_PATH),
        "parent_package": str(src_pkg.relative_to(ROOT)),
        "parent_model_id": "mei-1.0-51m-base-scratch300m-qat-q4-v1",
        "parent_weights_sha256": sha256_file(QAT_Q4_WEIGHTS_PATH),
        "quant_math_id": QUANT_MATH_ID,
        "weight_qat_ste": True,
        "activation_kv_int8_ste": True,
        "heads_file": str((dest / "heads.npz").relative_to(ROOT)),
        "heads_dtype": "float16",
        "package_files_bytes": package_bytes,
        "package_files_mb": package_bytes / (1024 * 1024),
        "package_within_30mb": True,
        "immutable_parent": True,
    }
    blocked = write_json(SFT_QAT_BASE_DIR / "RELEASE.json", release)
    if blocked:
        raise RuntimeError(blocked)
    return dest


def _gold_text(row: dict) -> str:
    answers = row.get("answers")
    if answers is None:
        name = row.get("gold_name")
        answers = [] if not name else [{"name": name, "arguments": row.get("gold_args") or {}}]
    return json.dumps(answers or [], ensure_ascii=False, separators=(",", ":"))


def _calls_equal(left: list[dict], right: list[dict]) -> bool:
    return json.dumps(left or [], sort_keys=True, ensure_ascii=False) == json.dumps(
        right or [], sort_keys=True, ensure_ascii=False
    )


def float_task_control(rows: list[dict], steps: int, eval_n: int = 24) -> dict:
    """Small trained float 51M task control; never runs the frozen LM suite."""
    from architecture import NeedleZh
    from checkpoint import load_params
    from config import NeedleZhConfig
    from identity_51m import ARCHITECTURE_SPEC
    from mei_sdk.runtime_51m import Runtime51M, validate_call
    from tokenizer import ZhTokenizerV1

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    runtime = Runtime51M(model, tok)
    train_report = train_fullcall_float(runtime, rows, steps=steps, lr=2e-4)
    tools_u = universe_by_name()
    n_ok = 0
    text_exact = 0
    call_exact = 0
    start = min(steps, max(0, len(rows) - eval_n))
    for row in rows[start : start + eval_n]:
        enc = encode_fc(tok, row)
        if not enc:
            continue
        prompt, _answer = enc
        gold_text = _gold_text(row)
        gold_calls = row.get("answers") or []
        names = row.get("retrieved_tools") or []
        tools = [tools_u[name] for name in names if name in tools_u]
        decoded = runtime.greedy(prompt, tools=tools, max_new=96, decode_mode="constrained")
        text = (decoded.get("text") or "").strip()
        validated = validate_call(
            text,
            tools=tools,
            query=str(row.get("query") or ""),
            system_facts=str(row.get("system_facts") or ""),
        )
        n_ok += 1
        text_exact += int(text == gold_text)
        call_exact += int(_calls_equal(validated.get("function_calls") or [], gold_calls))
    return {
        "train": train_report,
        "n": n_ok,
        "text_exact": text_exact / max(n_ok, 1),
        "call_exact": call_exact / max(n_ok, 1),
        "float_weights": str(WEIGHTS_PATH),
        "same_serializer_grammar_scorer": True,
        "trained_float_control": True,
        "not_lm_eval": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--retrieval-steps", type=int, default=400)
    parser.add_argument("--sft-steps", type=int, default=4000)
    parser.add_argument("--conf-steps", type=int, default=80)
    parser.add_argument("--float-control", type=int, default=200)
    parser.add_argument("--skip-isolation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.limit = min(args.limit, 64)
        args.retrieval_steps = min(args.retrieval_steps, 2)
        args.sft_steps = min(args.sft_steps, 2)
        args.conf_steps = min(args.conf_steps, 2)
        args.float_control = min(args.float_control, 2)
    pkg_dir = args.package_dir
    if pkg_dir is None:
        pkg_dir = QAT_Q4_PACKAGE_DIR if (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file() else Q4_PACKAGE_DIR
    iso = {"ok": True, "skipped": True}
    if not args.skip_isolation:
        iso = isolation_ok()
        blocked = write_json(args.jobs_dir / "sft-v2-isolation-51m.json", iso)
        if blocked:
            return fail(blocked)
        if not iso.get("ok"):
            return fail(f"train/eval isolation failed: {iso.get('n_hits')}")

    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime
    from checkpoint import load_params
    from qat_replay_51m import apply_fake_quant_inplace, restore_master

    pkg = load_package(pkg_dir)
    runtime, loaded = load_51m_runtime(pkg)
    if not QAT_Q4_WEIGHTS_PATH.is_file():
        return fail(f"missing QAT Q4 FP32 master: {QAT_Q4_WEIGHTS_PATH}")
    load_params(runtime.model, QAT_Q4_WEIGHTS_PATH, strict=True)
    mx.eval(runtime.model.parameters())
    bit_map_path = args.jobs_dir / Q4_BASELINE_MAP_NAME
    bit_map = load_bit_map(bit_map_path)
    ret_rows = load_jsonl(RET_PATH, args.limit)
    fc_load_limit = max(args.limit, 512) if args.smoke else args.limit
    fc_rows = load_jsonl(FC_PATH, fc_load_limit)
    sft_rows = fc_rows[: args.limit]
    sft_rep = train_fullcall(
        runtime,
        sft_rows,
        steps=args.sft_steps,
        lr=2e-4,
        bit_map=bit_map,
        activation_ste=True,
    )
    master = apply_fake_quant_inplace(runtime.model, bit_map)
    try:
        ret_rep = train_contrastive(runtime, ret_rows, steps=args.retrieval_steps, lr=1e-3)
        conf_rep = train_confidence(runtime, fc_rows, steps=args.conf_steps)
    finally:
        restore_master(runtime.model, master)
    control = float_task_control(fc_rows, args.float_control)
    dest = None if args.smoke else save_sft_package(pkg_dir, SFT_QAT_PACKAGE_DIR, runtime)
    report = {
        "kind": "quant-aware-sft-smoke" if args.smoke else "quant-aware-sft-ondisk",
        "parent_package": str(pkg_dir),
        "parent_fp32_master": str(QAT_Q4_WEIGHTS_PATH),
        "sft_package": str(dest) if dest else None,
        "sft_fp32_master": str(SFT_QAT_WEIGHTS_PATH) if dest else None,
        "model_id": SFT_QAT_MODEL_ID,
        "isolation": {"ok": bool(iso.get("ok")), "n_hits": iso.get("n_hits")},
        "retrieval": ret_rep,
        "fullcall_sft": sft_rep,
        "confidence": conf_rep,
        "float_task_control": control,
        "training_order": ["qat_fullcall_sft", "freeze_lm", "retrieval_head", "confidence_head"],
        "head_representation": "q4-dequant deployment math after final QAT SFT",
        "q4_bit_map": str(bit_map_path.relative_to(ROOT)),
        "weight_qat_ste": True,
        "activation_kv_int8_ste": True,
        "quant_math_id": QUANT_MATH_ID,
        "prompt_max_tokens": 1152,
        "clean_v2": True,
        "used_58m_weights": False,
        "qat_mandatory": True,
        "not_a_claim": "On-disk SFT is not a CURRENT freeze.",
        "n_loaded": loaded.get("n_loaded"),
    }
    report_name = "sft-qat-smoke-51m.json" if args.smoke else "sft-ondisk-qat-51m.json"
    blocked = write_json(args.jobs_dir / report_name, report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
