#!/usr/bin/env python3
"""Held-out pretrain probe runner: target NLL + greedy continuation diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from repo_paths import BANK_NEEDLE_PRETRAIN_PROBES, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from architecture import NeedleZh  # noqa: E402
from checkpoint import load_params  # noqa: E402
from config import NeedleZhConfig  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402


def load_probes(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def greedy_continue(model, ids: list[int], *, max_new: int, eos_id: int) -> list[int]:
    out = list(ids)
    for _ in range(max_new):
        arr = mx.array([out], dtype=mx.int32)
        logits = model(arr)["logits"][0, -1]
        tok = int(mx.argmax(logits).item())
        if tok == eos_id:
            break
        out.append(tok)
    return out


def probe_nll(model, tok: ZhTokenizerV1, prompt: str, target: str) -> dict:
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


def eval_probes(model, tok: ZhTokenizerV1, probes: list[dict]) -> dict:
    families: dict[str, list[float]] = {}
    rows = []
    for p in probes:
        fam = str(p.get("family") or "unknown")
        inp = str(p.get("input") or "")
        tgt = str(p.get("target") or "")
        r = probe_nll(model, tok, inp, tgt)
        r.update({"probe_id": p.get("probe_id"), "family": fam, "input": inp, "target": tgt})
        rows.append(r)
        families.setdefault(fam, []).append(r["nll"])
    fam_mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in families.items()}
    nlls = [r["nll"] for r in rows if math.isfinite(r["nll"])]
    return {
        "n": len(rows),
        "mean_nll": sum(nlls) / len(nlls) if nlls else float("nan"),
        "family_nll": fam_mean,
        "exact_rate": sum(1 for r in rows if r["exact"]) / max(len(rows), 1),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--probes", type=Path, default=BANK_NEEDLE_PRETRAIN_PROBES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig().tiny() if args.tiny else NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    if args.ckpt:
        load_params(model, args.ckpt, strict=True)
    probes = load_probes(args.probes)
    report = eval_probes(model, tok, probes)
    report["ckpt"] = str(args.ckpt) if args.ckpt else None
    report["tiny"] = args.tiny
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
