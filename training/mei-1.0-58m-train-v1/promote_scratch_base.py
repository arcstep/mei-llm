#!/usr/bin/env python3
"""Promote the finished 300M scratch run into base/. Refuses pilots and incomplete ledgers."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import CORPUS_LM_V1, ROOT, TRAIN_RUNS, ensure_formal_on_path

ensure_formal_on_path()
from pretrain_gates import REQUIRED_ROLES, SCRATCH_EXPOSURE_TOKENS, SCRATCH_SOURCE_QUOTAS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    run_dir = TRAIN_RUNS / "pretrain-300m-scratch"
    summary = load(run_dir / "summary.json")
    if (run_dir / "NOT_FOR_PROMOTE.json").is_file():
        print("run is marked NOT_FOR_PROMOTE", file=sys.stderr)
        return 2
    if int(summary.get("tokens_seen") or 0) < SCRATCH_EXPOSURE_TOKENS:
        print(f"tokens_seen {summary.get('tokens_seen')} < {SCRATCH_EXPOSURE_TOKENS}", file=sys.stderr)
        return 2
    if int(summary.get("seq_len") or 0) != 2048:
        print("final seq_len must be 2048", file=sys.stderr)
        return 2
    if summary.get("curriculum_stage") != "s3":
        print("final curriculum_stage must be s3", file=sys.stderr)
        return 2
    if summary.get("allow_repeat"):
        print("repeat epochs cannot be promoted", file=sys.stderr)
        return 2
    if summary.get("parent_checkpoint") not in (None, ""):
        print("scratch base cannot have a parent checkpoint", file=sys.stderr)
        return 2
    drawn = summary.get("source_tokens_drawn") or {}
    for name, quota in SCRATCH_SOURCE_QUOTAS.items():
        got = int(drawn.get(name) or 0)
        if abs(got - quota) > 2048:
            print(f"{name} drawn {got} outside quota {quota} ±2048", file=sys.stderr)
            return 2
    for name in REQUIRED_ROLES:
        if summary.get(f"valid_loss_{name}") is None and name != "wiki":
            print(f"missing valid_loss_{name}", file=sys.stderr)
            return 2
    if summary.get("valid_loss") is None:
        print("missing wiki valid_loss", file=sys.stderr)
        return 2
    weights = run_dir / "pretrain-300m-scratch.npz"
    state = run_dir / "pretrain-300m-scratch-state.npz"
    if not weights.is_file() or not state.is_file():
        print("missing final weights or train state", file=sys.stderr)
        return 2

    dest = ROOT / "base/mei-1.0-58m-base-scratch300m-v1"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dest / "pretrain-300m-scratch.npz")
    shutil.copy2(state, dest / "pretrain-300m-scratch-state.npz")
    meta_src = state.with_suffix(".meta.json")
    if meta_src.is_file():
        shutil.copy2(meta_src, dest / "pretrain-300m-scratch-state.meta.json")
    shutil.copy2(run_dir / "summary.json", dest / "summary.json")
    release = {
        "model_id": "mei-1.0-58m-base-scratch300m-v1",
        "kind": "base-scratch",
        "params": int(summary.get("params") or 0),
        "tokens_seen_exposure": int(summary["tokens_seen"]),
        "exposure_tokens": SCRATCH_EXPOSURE_TOKENS,
        "source_tokens_drawn": drawn,
        "seq_curriculum": [512, 1024, 2048],
        "max_trained_seq": 2048,
        "corpus": "corpus/lm-v1",
        "schedule": "corpus/lm-v1/schedule-scratch.json",
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "corpus_sha256": summary.get("corpus_sha256"),
        "schedule_sha256": summary.get("schedule_sha256"),
        "valid_loss": summary.get("valid_loss"),
        "valid_loss_hq": summary.get("valid_loss_hq"),
        "valid_loss_structure": summary.get("valid_loss_structure"),
        "valid_loss_colloquial": summary.get("valid_loss_colloquial"),
        "final_probe_mean_nll": summary.get("final_probe_mean_nll"),
        "public_distribution_clearance_asserted": False,
        "not_a_claim": "LM loss/probes are not tool-calling ability.",
        "weights_sha256": sha256(dest / "pretrain-300m-scratch.npz"),
        "state_sha256": sha256(dest / "pretrain-300m-scratch-state.npz"),
        "colloquial_consumed": int(drawn.get("colloquial") or 0),
        "parent_checkpoint": None,
        "init_mode": "scratch",
    }
    (dest / "RELEASE.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    current = load(ROOT / "CURRENT.json")
    current["base"] = "base/mei-1.0-58m-base-scratch300m-v1"
    current["stage"] = "scratch-300m-promoted"
    (ROOT / "CURRENT.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "base": current["base"], "tokens_seen": summary["tokens_seen"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
