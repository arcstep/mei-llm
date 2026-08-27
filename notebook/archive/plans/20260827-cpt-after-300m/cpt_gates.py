#!/usr/bin/env python3
"""Refuse dirty zh-pretrain-v2 structure as a 1B CPT source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ROOT

CORPUS_ZH_PRETRAIN_V2 = ROOT / "notebook/archive/corpus/zh-pretrain-v2"


def refuse_dirty_v2_for_1b(corpus_dir: Path, rung: str) -> str | None:
    if rung != "1b":
        return None
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    blocked = corpus_dir / "BLOCKED-for-1b.json"
    if corpus_dir.resolve() == CORPUS_ZH_PRETRAIN_V2.resolve() or blocked.is_file():
        return (
            "1B CPT blocked: zh-pretrain-v2 structure is dirty GitHub signature/comment extract. "
            "Rebuild clean structure in zh-pretrain-v3 and pass audit/probes before 1B. "
            "Do not --resume v2; 1B must --init-weights from 300M and skip already-seen wiki."
        )
    release = corpus_dir / "RELEASE.json"
    if not release.is_file():
        return f"1B CPT blocked: missing {release}"
    data = json.loads(release.read_text(encoding="utf-8"))
    if not data.get("promote_structure"):
        return "1B CPT blocked: RELEASE.promote_structure is not true"
    ledger = data.get("unique_train_tokens") or data.get("n_unique_train_tokens")
    if ledger is not None and int(ledger) < 1_000_000_000:
        # unique < 1B is allowed to start only if caller names the rung honestly;
        # repeating epochs to pretend 2B/10B is still forbidden at naming time.
        pass
    if data.get("repeat_epochs_named_as_new_unique"):
        return "1B CPT blocked: cannot name repeated epochs as new unique tokens"
    return None


def write_v2_block() -> Path:
    path = CORPUS_ZH_PRETRAIN_V2 / "BLOCKED-for-1b.json"
    payload = {
        "blocked_for": ["1b", "2b", "10b"],
        "reason": "structure ~39.7M tokens, mostly over-wide GitHub signature/comment extract",
        "rewrite_release": False,
        "successor": "corpus/lm-v1/structure/zh-pretrain-v3",
        "init_from": "mei-1.0-58m-base-cpt300m-v1",
        "init_mode": "weights_only",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
