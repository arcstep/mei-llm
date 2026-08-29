#!/usr/bin/env python3
"""51M Q4 (or mixed) STE replay against the frozen float Base-LM Anchor.

Does not re-run the float LM suite. Scores QAT weights only vs
notebook/evaluation/jobs/mei-1.0-51m/float-base-lm-anchor.json and the
already-stored greedy_float strings in q4-dequant-parity.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils as xu

from freeze_float_baseline_51m import eval_probes, greedy_continue, load_probes
from identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    EXPECTED_PARAMS,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    MODEL_ID,
    PROBE_BANK,
    Q4_BASELINE_MAP_NAME,
    QAT_MANDATORY_NOTE,
    QAT_Q4_MODEL_ID,
    QAT_Q4_RELEASE_PATH,
    QAT_Q4_WEIGHTS_PATH,
    RELEASE_PATH,
    ROOT,
    WEIGHTS_PATH,
    fail,
    load_json,
    refuse_base_write,
    sha256_file,
    validate_release,
    write_json,
)
from quant_pack_51m import QUANT_MATH_ID, mlx_fake_quant_weight, mlx_ste_quantize, sha256_bytes
from train_common import clip_grads, eval_lm_loss, masked_lm_loss, stack_windows


SCAN_PROMPTS = ("厨房灯打开", "[]", '{"name":')
RUNG_SPECS = (
    ("0p5m", 500_000, "qat-q4-rung-0p5m.json"),
    ("2m", 2_000_000, "qat-q4-rung-2m.json"),
    ("5m", 5_000_000, "qat-q4-rung-5m.json"),
)
PRE_REGISTER = {
    "kind": "qat-q4-gates-preregister",
    "registered_before_any_rung_eval": True,
    "valid_loss_delta_vs_float_anchor_max": 0.15,
    "probe_mean_nll_delta_vs_float_anchor_max": 0.40,
    "greedy_token_match_min": 2,
    "greedy_prompts_n": 3,
    "note": "Greedy is compared to stored greedy_float in q4-dequant-parity.json. Do not reload float.",
}


def flatten_tree(tree) -> dict[str, mx.array]:
    from checkpoint import flatten_tree as _flatten

    return _flatten(tree)


def load_bit_map(path: Path) -> dict[str, int]:
    payload = load_json(path)
    bits = payload.get("tensor_bits") or payload.get("bits") or {}
    out: dict[str, int] = {}
    for key, val in bits.items():
        width = int(val)
        if width not in (2, 4, 32):
            width = 4
        out[str(key)] = width
    return out


def quantize_tree(tree, bit_map: dict[str, int], *, ste: bool):
    flat = flatten_tree(tree)
    out = {}
    for key, val in flat.items():
        bits = int(bit_map.get(key, 32))
        if bits >= 32:
            out[key] = val
        elif ste:
            out[key] = mlx_ste_quantize(val, bits=bits)
        else:
            out[key] = mlx_fake_quant_weight(val, bits=bits)
    return xu.tree_unflatten(list(out.items()))


def count_quantized(flat: dict[str, mx.array], bit_map: dict[str, int]) -> dict[str, int]:
    counts = {2: 0, 4: 0, 32: 0}
    for key in flat:
        bits = int(bit_map.get(key, 32))
        if bits <= 2:
            counts[2] += 1
        elif bits <= 4:
            counts[4] += 1
        else:
            counts[32] += 1
    return counts


def load_float_parent():
    from architecture import NeedleZh, count_params
    from checkpoint import load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError(f"loaded params {count_params(model)} != {EXPECTED_PARAMS}")
    return model


def load_master(path: Path):
    from architecture import NeedleZh, count_params
    from checkpoint import load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, path, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError(f"loaded params {count_params(model)} != {EXPECTED_PARAMS}")
    return model


def apply_fake_quant_inplace(model, bit_map: dict[str, int]) -> dict[str, mx.array]:
    from checkpoint import flatten_params

    master = flatten_params(model)
    qtree = quantize_tree(model.parameters(), bit_map, ste=False)
    model.update(qtree)
    mx.eval(model.parameters())
    return master


def restore_master(model, master: dict[str, mx.array]) -> None:
    model.update(xu.tree_unflatten(list(master.items())))
    mx.eval(model.parameters())


def eval_qat_against_anchor(
    model,
    tok,
    bit_map: dict[str, int],
    *,
    valid,
    jobs_dir: Path,
    activation_ste: bool,
) -> dict:
    import architecture as arch

    anchor = load_json(jobs_dir / FLOAT_ANCHOR_NAME)
    parity = load_json(jobs_dir / "q4-dequant-parity.json")
    stored_greedy = {row["prompt"]: row.get("greedy_float") for row in (parity.get("prompts") or [])}
    prev = arch.QAT_ACTIVATION_STE
    arch.QAT_ACTIVATION_STE = False
    master = apply_fake_quant_inplace(model, bit_map)
    try:
        valid_loss = eval_lm_loss(model, valid, batch_size=1, max_windows=128) if valid else float("nan")
        probes = load_probes(PROBE_BANK)
        probe_rep = eval_probes(model, tok, probes)
        greedy_rows = []
        matches = 0
        for text in SCAN_PROMPTS:
            ids = tok.encode(text, add_bos=True, add_eos=False)[:32]
            gen = greedy_continue(model, ids, max_new=8, eos_id=tok.eos_id)
            decoded = tok.decode(gen[len(ids) :])
            gold = stored_greedy.get(text)
            match = gold is not None and decoded == gold
            matches += int(match)
            greedy_rows.append(
                {
                    "prompt": text,
                    "greedy_qat": decoded,
                    "greedy_float_stored": gold,
                    "match": match,
                }
            )
    finally:
        restore_master(model, master)
        arch.QAT_ACTIVATION_STE = prev

    float_valid = float(anchor["valid_loss"])
    float_probe = float(anchor["probe_mean_nll"])
    d_valid = float(valid_loss) - float_valid
    d_probe = float(probe_rep["mean_nll"]) - float_probe
    v_max = float(PRE_REGISTER["valid_loss_delta_vs_float_anchor_max"])
    p_max = float(PRE_REGISTER["probe_mean_nll_delta_vs_float_anchor_max"])
    g_min = int(PRE_REGISTER["greedy_token_match_min"])
    quality_ok = d_valid <= v_max and d_probe <= p_max
    greedy_ok = matches >= g_min
    return {
        "valid_loss": float(valid_loss),
        "float_anchor_valid_loss": float_valid,
        "delta_valid_loss": d_valid,
        "probe_mean_nll": probe_rep["mean_nll"],
        "probe_family_nll": probe_rep["family_nll"],
        "probe_exact_rate": probe_rep["exact_rate"],
        "probe_n": probe_rep["n"],
        "float_anchor_probe_mean_nll": float_probe,
        "delta_probe_mean_nll": d_probe,
        "greedy": greedy_rows,
        "greedy_match_n": matches,
        "greedy_match_min_preregistered": g_min,
        "greedy_ok": greedy_ok,
        "quality_ok": quality_ok,
        "product_ok": quality_ok and greedy_ok,
        "quality_fail_reasons": [
            reason
            for reason, bad in (
                (f"delta_valid_loss {d_valid:.4f} > {v_max}", d_valid > v_max),
                (f"delta_probe_mean_nll {d_probe:.4f} > {p_max}", d_probe > p_max),
            )
            if bad
        ],
        "did_not_reload_float": True,
        "eval_used_weight_fake_quant": True,
        "eval_activation_ste": False,
        "train_activation_ste": activation_ste,
        "quant_math_id": QUANT_MATH_ID,
    }


def train_qat_until(
    model,
    source,
    bit_map: dict[str, int],
    *,
    target_tokens: int,
    already: int,
    batch_size: int,
    lr: float,
    activation_ste: bool,
) -> dict:
    import architecture as arch

    arch.QAT_ACTIVATION_STE = bool(activation_ste)
    opt = optim.Adam(learning_rate=lr)
    tokens = int(already)
    step = 0
    last_loss = None
    t0 = time.perf_counter()
    while tokens < int(target_tokens):
        windows = source.take_windows(batch_size)
        if not windows:
            break
        stacked = stack_windows(windows)

        def loss_fn(params):
            qtree = quantize_tree(params, bit_map, ste=True)
            model.update(qtree)
            out = model(stacked["x"])
            return masked_lm_loss(out["logits"].astype(mx.float32), stacked["y"], stacked["mask"])

        params = model.parameters()
        loss, grads = mx.value_and_grad(loss_fn)(params)
        grads, gn = clip_grads(grads, max_norm=1.0)
        model.update(params)
        opt.update(model, grads)
        mx.eval(model.parameters(), loss)
        last_loss = float(loss.item())
        tokens += int(stacked["n_tokens"])
        step += 1
        if step == 1 or step % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        "step": step,
                        "tokens": tokens,
                        "target": target_tokens,
                        "loss": last_loss,
                        "tok_s": (tokens - already) / max(elapsed, 1e-6),
                        "grad_norm": float(gn.item()) if hasattr(gn, "item") else float(gn),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    arch.QAT_ACTIVATION_STE = False
    return {"tokens": tokens, "steps": step, "last_loss": last_loss}


def save_qat_master(
    model,
    dest: Path,
    *,
    tokens: int,
    bit_map_path: Path,
    bit_map: dict[str, int],
    model_id: str,
    release_path: Path,
) -> dict:
    from checkpoint import save_params

    blocked = refuse_base_write(dest)
    if blocked:
        raise RuntimeError(blocked)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_params(model, dest)
    parent_sha = sha256_file(WEIGHTS_PATH)
    weights_sha = sha256_file(dest)
    bit_blob = json.dumps(bit_map, sort_keys=True).encode("utf-8")
    release = {
        "kind": "qat-fp32-master",
        "model_id": model_id,
        "parent_model_id": MODEL_ID,
        "parent_weights_sha256": parent_sha,
        "architecture_id": ARCHITECTURE_ID,
        "params": EXPECTED_PARAMS,
        "weights_sha256": weights_sha,
        "quant_math_id": QUANT_MATH_ID,
        "bit_map": str(bit_map_path.relative_to(ROOT)),
        "bit_map_sha256": sha256_bytes(bit_blob),
        "tokens_seen_qat_replay": int(tokens),
        "immutable_parent": True,
        "qat_mandatory": True,
        "note": QAT_MANDATORY_NOTE,
    }
    rel_err = write_json(release_path, release)
    if rel_err:
        raise RuntimeError(rel_err)
    return release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--bit-map", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rung", default="5m", choices=["smoke", "0p5m", "2m", "5m"])
    parser.add_argument("--init-weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--start-tokens", type=int, default=0)
    parser.add_argument("--no-activation-ste", action="store_true")
    parser.add_argument("--save-master", action="store_true")
    parser.add_argument("--rung-prefix", default="qat-q4")
    args = parser.parse_args()
    error = validate_release(load_json(RELEASE_PATH), WEIGHTS_PATH)
    if error:
        return fail(error)

    jobs = args.jobs_dir
    bit_map_path = args.bit_map or (jobs / Q4_BASELINE_MAP_NAME)
    if not bit_map_path.is_file():
        return fail(f"missing bit-map: {bit_map_path}")
    bit_map = load_bit_map(bit_map_path)
    pre_path = jobs / f"{args.rung_prefix}-gates-preregister.json"
    blocked = write_json(pre_path, {**PRE_REGISTER, "bit_map": str(bit_map_path.relative_to(ROOT))})
    if blocked:
        return fail(blocked)

    from _repo import CORPUS_LM_V1
    from data import list_valid_set, load_scheduled_train, PackedTokenSource
    from tokenizer import ZhTokenizerV1

    tok = ZhTokenizerV1()
    model = load_master(args.init_weights) if args.init_weights.resolve() != WEIGHTS_PATH.resolve() else load_float_parent()
    flat = flatten_tree(model.parameters())
    qcounts = count_quantized(flat, bit_map)
    if qcounts[4] + qcounts[2] < 50:
        return fail(f"bit-map keys do not match model tensors: {qcounts}")

    schedule = CORPUS_LM_V1 / "schedule-scratch.json"
    source = load_scheduled_train(
        CORPUS_LM_V1,
        args.seq_len,
        tok.pad_id,
        schedule_path=schedule,
        curriculum_stage="s1",
        tokens_seen=0,
    )
    if source is None:
        return fail("scheduled train source is empty")
    wiki_bins = list_valid_set(CORPUS_LM_V1, "wiki")
    valid = PackedTokenSource(wiki_bins, args.seq_len, tok.pad_id)[:128] if wiki_bins else []

    activation_ste = not args.no_activation_ste
    if args.smoke or args.max_rung == "smoke":
        result = train_qat_until(
            model,
            source,
            bit_map,
            target_tokens=min(args.seq_len * args.batch_size * 4, 16_384),
            already=0,
            batch_size=min(args.batch_size, 4),
            lr=args.lr,
            activation_ste=activation_ste,
        )
        report = {
            "kind": "qat-replay-smoke",
            "architecture_id": ARCHITECTURE_ID,
            "parent": MODEL_ID,
            "quant_counts": qcounts,
            "train": result,
            "qat_mandatory": True,
            "not_a_claim": "Smoke steps are not a quality gate.",
        }
        path = jobs / f"{args.rung_prefix}-replay-smoke.json"
        blocked = write_json(path, report)
        if blocked:
            return fail(blocked)
        print(json.dumps({"ok": True, "smoke": True, "report": str(path), **result}, indent=2, ensure_ascii=False))
        return 0

    rung_limit = {"0p5m": 1, "2m": 2, "5m": 3}[args.max_rung]
    already = 0
    if int(args.start_tokens) > 0:
        skipped = 0
        while skipped < int(args.start_tokens):
            windows = source.take_windows(args.batch_size)
            if not windows:
                break
            skipped += int(sum(float(v) for w in windows for v in w["mask"]))
        already = skipped
        print(json.dumps({"skipped_unique_tokens": skipped, "requested": args.start_tokens}), flush=True)
    last_ok = False
    last_eval = None
    for index, (name, target, filename) in enumerate(RUNG_SPECS[:rung_limit]):
        out_name = filename if args.rung_prefix == "qat-q4" else f"{args.rung_prefix}-rung-{name}.json"
        if already < target:
            train = train_qat_until(
                model,
                source,
                bit_map,
                target_tokens=target,
                already=already,
                batch_size=args.batch_size,
                lr=args.lr,
                activation_ste=activation_ste,
            )
            already = int(train["tokens"])
        else:
            train = {"tokens": already, "steps": 0, "last_loss": None, "skipped": True}
        last_eval = eval_qat_against_anchor(
            model,
            tok,
            bit_map,
            valid=valid,
            jobs_dir=jobs,
            activation_ste=activation_ste,
        )
        last_ok = bool(last_eval["quality_ok"])
        report = {
            "kind": f"{args.rung_prefix}-rung",
            "rung": name,
            "target_tokens": target,
            "tokens_seen": already,
            "allow_repeat": False,
            "architecture_id": ARCHITECTURE_ID,
            "parent": MODEL_ID,
            "bit_map": str(bit_map_path.relative_to(ROOT)),
            "quant_counts": qcounts,
            "quant_math_id": QUANT_MATH_ID,
            "train": train,
            "eval": last_eval,
            "quality_ok": last_ok,
            "product_ok": bool(last_eval.get("product_ok")),
            "qat_mandatory": True,
            "gates_preregister": str(pre_path.relative_to(ROOT)),
            "not_a_claim": "QAT rung metrics are not tool-calling ability.",
        }
        blocked = write_json(jobs / out_name, report)
        if blocked:
            return fail(blocked)
        print(json.dumps({"rung": name, "quality_ok": last_ok, "product_ok": last_eval.get("product_ok"), "tokens": already, "report": str(jobs / out_name)}, indent=2, ensure_ascii=False))
        if last_ok and last_eval.get("product_ok") and args.save_master:
            if args.rung_prefix == "qat-cq2":
                from identity_51m import QAT_CQ2_BASE_DIR, QAT_CQ2_MODEL_ID, QAT_CQ2_WEIGHTS_PATH

                dest = QAT_CQ2_WEIGHTS_PATH
                model_id = QAT_CQ2_MODEL_ID
                release_path = QAT_CQ2_BASE_DIR / "RELEASE.json"
            else:
                dest = QAT_Q4_WEIGHTS_PATH
                model_id = QAT_Q4_MODEL_ID
                release_path = QAT_Q4_RELEASE_PATH
            release = save_qat_master(
                model,
                dest,
                tokens=already,
                bit_map_path=bit_map_path,
                bit_map=bit_map,
                model_id=model_id,
                release_path=release_path,
            )
            print(json.dumps({"saved_master": True, "rung": name, "release": release}, indent=2, ensure_ascii=False))
        if not last_ok:
            print(json.dumps({"stop": True, "reasons": last_eval.get("quality_fail_reasons")}, indent=2, ensure_ascii=False))
            break

    return 0 if last_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
