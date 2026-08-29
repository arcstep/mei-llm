#!/usr/bin/env python3
"""Promote the finished 1B CPT run into base/. Refuses pilots and failed gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ROOT, TRAIN_RUNS, ensure_formal_on_path

ensure_formal_on_path()
from cpt_gates import (
    ALIGNMENT_TOLERANCE,
    CPT_CUMULATIVE_EXPOSURE,
    CPT_CUMULATIVE_QUOTAS,
    CPT_INCREMENTAL_QUOTAS,
    PARENT_TOKENS_SEEN,
    REQUIRED_ROLES,
    refuse_cpt_parent,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    run_dir = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
    summary = load(run_dir / "summary.json")
    if (run_dir / "NOT_FOR_PROMOTE.json").is_file():
        print("run is marked NOT_FOR_PROMOTE", file=sys.stderr)
        return 2
    parent_err = refuse_cpt_parent()
    if parent_err:
        print(parent_err, file=sys.stderr)
        return 2
    if int(summary.get("tokens_seen") or 0) < CPT_CUMULATIVE_EXPOSURE:
        print(f"tokens_seen {summary.get('tokens_seen')} < {CPT_CUMULATIVE_EXPOSURE}", file=sys.stderr)
        return 2
    if int(summary.get("seq_len") or 0) != 2048:
        print("final seq_len must be 2048", file=sys.stderr)
        return 2
    if summary.get("allow_repeat"):
        print("repeat epochs cannot be promoted", file=sys.stderr)
        return 2
    if int(summary.get("parent_tokens_seen") or 0) != PARENT_TOKENS_SEEN:
        print("parent_tokens_seen drifted", file=sys.stderr)
        return 2
    drawn = summary.get("source_tokens_drawn") or {}
    for name, quota in CPT_CUMULATIVE_QUOTAS.items():
        got = int(drawn.get(name) or 0)
        if abs(got - quota) > ALIGNMENT_TOLERANCE:
            print(f"{name} drawn {got} outside cumulative quota {quota} ±{ALIGNMENT_TOLERANCE}", file=sys.stderr)
            return 2
    stage = summary.get("stage_tokens_drawn") or {}
    for name, quota in CPT_INCREMENTAL_QUOTAS.items():
        got = int(stage.get(name) or 0)
        if got and abs(got - quota) > ALIGNMENT_TOLERANCE and abs(int(drawn.get(name) or 0) - CPT_CUMULATIVE_QUOTAS[name]) > ALIGNMENT_TOLERANCE:
            print(f"{name} incremental drawn {got} outside quota {quota}", file=sys.stderr)
            return 2
    parent = load(ROOT / "base/mei-1.0-58m-base-scratch300m-v1/RELEASE.json")
    wiki_now = summary.get("valid_loss")
    hq_now = summary.get("valid_loss_hq")
    if wiki_now is None or hq_now is None:
        print("missing wiki/HQ valid losses", file=sys.stderr)
        return 2
    if float(wiki_now) > float(parent.get("valid_loss") or 99) and float(hq_now) > float(parent.get("valid_loss_hq") or 99):
        print("1B benefit gate failed: wiki and HQ valid both worse than 300M parent", file=sys.stderr)
        return 2
    for name in REQUIRED_ROLES:
        key = "valid_loss" if name == "wiki" else f"valid_loss_{name}"
        if summary.get(key) is None:
            print(f"missing {key}", file=sys.stderr)
            return 2
    weights = run_dir / "pretrain-1b-cpt-from-scratch300m.npz"
    state = run_dir / "pretrain-1b-cpt-from-scratch300m-state.npz"
    if not weights.is_file() or not state.is_file():
        print("missing final weights or train state", file=sys.stderr)
        return 2

    dest = ROOT / "base/mei-1.0-58m-base-cpt1b-v1"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dest / "pretrain-1b-cpt-from-scratch300m.npz")
    shutil.copy2(state, dest / "pretrain-1b-cpt-from-scratch300m-state.npz")
    meta_src = state.with_suffix(".meta.json")
    if meta_src.is_file():
        shutil.copy2(meta_src, dest / "pretrain-1b-cpt-from-scratch300m-state.meta.json")
    shutil.copy2(run_dir / "summary.json", dest / "summary.json")
    release = {
        "model_id": "mei-1.0-58m-base-cpt1b-v1",
        "kind": "base-cpt",
        "params": int(summary.get("params") or 0),
        "tokens_seen_exposure": int(summary["tokens_seen"]),
        "exposure_tokens": CPT_CUMULATIVE_EXPOSURE,
        "incremental_tokens": int(summary.get("segment_tokens") or 0),
        "source_tokens_drawn": drawn,
        "stage_tokens_drawn": stage,
        "max_trained_seq": 2048,
        "seq_len": 2048,
        "corpus": "corpus/lm-v2",
        "schedule": "corpus/lm-v2/schedule-cpt-1b.json",
        "parent_release": "base/mei-1.0-58m-base-scratch300m-v1",
        "parent_checkpoint": "base/mei-1.0-58m-base-scratch300m-v1/pretrain-300m-scratch-state.npz",
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "corpus_sha256": summary.get("corpus_sha256"),
        "schedule_sha256": summary.get("schedule_sha256"),
        "parent_schedule_sha256": "e97b8c9ec1d5930369cb18204217c59ac0e3a9729ace0012439599ce9ed39628",
        "valid_loss": summary.get("valid_loss"),
        "valid_loss_hq": summary.get("valid_loss_hq"),
        "valid_loss_structure": summary.get("valid_loss_structure"),
        "valid_loss_colloquial": summary.get("valid_loss_colloquial"),
        "valid_loss_structure_parent": summary.get("valid_loss_structure_parent"),
        "valid_loss_colloquial_parent": summary.get("valid_loss_colloquial_parent"),
        "final_probe_mean_nll": summary.get("final_probe_mean_nll"),
        "public_distribution_clearance_asserted": False,
        "not_a_claim": "LM loss/probes are not tool-calling ability.",
        "weights_sha256": sha256(dest / "pretrain-1b-cpt-from-scratch300m.npz"),
        "state_sha256": sha256(dest / "pretrain-1b-cpt-from-scratch300m-state.npz"),
        "init_mode": "continuation",
        "allow_repeat": False,
    }
    (dest / "RELEASE.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    current = load(ROOT / "CURRENT.json")
    current["base"] = "base/mei-1.0-58m-base-cpt1b-v1"
    current["stage"] = "cpt-1b-promoted"
    (ROOT / "CURRENT.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "base": current["base"], "tokens_seen": summary["tokens_seen"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
