#!/usr/bin/env python3
"""Freeze the 51M Float Base-LM Anchor without touching base weights or CURRENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    EXPECTED_PARAMS,
    FLOAT_ANCHOR_NAME,
    JOBS_DIR,
    MODEL_ID,
    NOT_A_CLAIM,
    PROBE_BANK,
    RELEASE_PATH,
    ROOT,
    RUN_PROBES_PATH,
    SUMMARY_PATH,
    WEIGHTS_PATH,
    fail,
    load_json,
    sha256_file,
    validate_release,
    write_json,
)


def load_probes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def greedy_continue(model, ids: list[int], *, max_new: int, eos_id: int) -> list[int]:
    import mlx.core as mx

    out = list(ids)
    for _ in range(max_new):
        arr = mx.array([out], dtype=mx.int32)
        logits = model(arr)["logits"][0, -1]
        tok = int(mx.argmax(logits).item())
        if tok == eos_id:
            break
        out.append(tok)
    return out


def probe_nll(model, tok, prompt: str, target: str) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tok.encode(target, add_bos=False, add_eos=True)
    if not target_ids:
        return {"nll": float("nan"), "n_tokens": 0, "greedy": "", "exact": False}
    ids = prompt_ids + target_ids
    arr = mx.array([ids], dtype=mx.int32)
    logits = model(arr)["logits"]
    logp = nn.log_softmax(logits, axis=-1)
    nll = 0.0
    n = 0
    offset = len(prompt_ids) - 1
    for i, tid in enumerate(target_ids):
        pos = offset + i
        if pos < 0 or pos >= int(logp.shape[1]):
            continue
        nll += -float(logp[0, pos, int(tid)].item())
        n += 1
    gen = greedy_continue(model, prompt_ids, max_new=max(8, len(target_ids) + 4), eos_id=tok.eos_id)
    text = tok.decode(gen[len(prompt_ids) :])
    return {
        "nll": nll / max(n, 1),
        "n_tokens": n,
        "greedy": text,
        "exact": text.strip() == target.strip(),
    }


def eval_probes(model, tok, probes: list[dict]) -> dict:
    families: dict[str, list[float]] = {}
    rows = []
    for item in probes:
        fam = str(item.get("family") or "unknown")
        result = probe_nll(model, tok, str(item.get("input") or ""), str(item.get("target") or ""))
        result.update({"probe_id": item.get("probe_id"), "family": fam})
        rows.append(result)
        families.setdefault(fam, []).append(result["nll"])
    fam_mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in families.items()}
    nlls = [row["nll"] for row in rows if math.isfinite(row["nll"])]
    return {
        "n": len(rows),
        "mean_nll": sum(nlls) / len(nlls) if nlls else float("nan"),
        "family_nll": fam_mean,
        "exact_rate": sum(1 for row in rows if row["exact"]) / max(len(rows), 1),
    }


def load_51m_model():
    import mlx.core as mx

    from architecture import NeedleZh, count_params
    from checkpoint import load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    if cfg.architecture_id != ARCHITECTURE_ID:
        raise RuntimeError(f"from_spec loaded {cfg.architecture_id!r}, not {ARCHITECTURE_ID}")
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    params = count_params(model)
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"loaded params {params} != {EXPECTED_PARAMS}")
    return model


def build_anchor(*, probe_rep: dict, probe_source: str) -> dict:
    release = load_json(RELEASE_PATH)
    summary = load_json(SUMMARY_PATH)
    return {
        "stage": "M1.1",
        "kind": "float-base-lm-anchor",
        "model_id": MODEL_ID,
        "architecture_id": ARCHITECTURE_ID,
        "params": EXPECTED_PARAMS,
        "weights": str(WEIGHTS_PATH.relative_to(ROOT)),
        "weights_sha256": sha256_file(WEIGHTS_PATH) if WEIGHTS_PATH.is_file() else None,
        "architecture_sha256": release.get("architecture_sha256"),
        "tokenizer_sha256": release.get("tokenizer_sha256"),
        "corpus_sha256_fingerprint": (
            hashlib.sha256(str(release.get("corpus_sha256") or "").encode("utf-8")).hexdigest()
            if release.get("corpus_sha256")
            else None
        ),
        "schedule_sha256": release.get("schedule_sha256"),
        "valid_loss": release.get("valid_loss"),
        "valid_loss_hq": release.get("valid_loss_hq"),
        "valid_loss_structure": release.get("valid_loss_structure"),
        "valid_loss_colloquial": release.get("valid_loss_colloquial"),
        "valid_loss_source": "base/RELEASE.json",
        "tokens_seen_exposure": release.get("tokens_seen_exposure") or summary.get("tokens_seen"),
        "probe_bank": str(PROBE_BANK.relative_to(ROOT)) if PROBE_BANK.is_file() else None,
        "probe_source": probe_source,
        "probe_mean_nll": probe_rep.get("mean_nll"),
        "probe_family_nll": probe_rep.get("family_nll"),
        "probe_exact_rate": probe_rep.get("exact_rate"),
        "probe_n": probe_rep.get("n"),
        "task_control": None,
        "sft": None,
        "runtime": None,
        "qat_mandatory": True,
        "not_a_claim": NOT_A_CLAIM,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=JOBS_DIR)
    parser.add_argument(
        "--reuse-run-probes",
        action="store_true",
        help="transcribe training-run probes.json instead of re-evaluating",
    )
    args = parser.parse_args()
    release = load_json(RELEASE_PATH)
    error = validate_release(release, WEIGHTS_PATH)
    if error:
        return fail(error)
    dest = args.out_dir / FLOAT_ANCHOR_NAME
    if args.reuse_run_probes:
        probe_rep = load_json(RUN_PROBES_PATH)
        if not probe_rep:
            return fail(f"missing run probes: {RUN_PROBES_PATH}")
        probe_source = "training/runs summary probes.json (transcribed; hashes verified)"
    else:
        from tokenizer import ZhTokenizerV1

        model = load_51m_model()
        tok = ZhTokenizerV1()
        probes = load_probes(PROBE_BANK)
        probe_rep = eval_probes(model, tok, probes)
        probe_source = "re-evaluated on mei-1.0-51m-arch-v1 + immutable base weights"
    anchor = build_anchor(probe_rep=probe_rep, probe_source=probe_source)
    written = write_json(dest, anchor)
    if written:
        return fail(written)
    print(json.dumps(anchor, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
