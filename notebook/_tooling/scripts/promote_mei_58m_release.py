#!/usr/bin/env python3
"""Promote mei-1.0-58m only when grounded protocol + Qwen3.5 9B + product gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

GATES = TASKS_ROOT / TASK_NEEDLE_ZH / "spec" / "gates.json"
REG = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "registry"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        type=Path,
        default=EXPERIMENTS_RUNS / "mei-1.0-58m-matrix-grounded" / "matrix.json",
    )
    args = ap.parse_args()
    matrix = load(args.matrix)
    gates = load(GATES)
    cells = {c.get("id"): c for c in matrix.get("cells") or [] if c.get("id")}
    cand = cells.get("mei-1.0-58m-route-cpt300m-sft10k-v1") or cells.get("mei-1.0-58m-route-cpt300m-sft2k-v1")
    qwen9 = cells.get("qwen35-9b")
    reasons = []
    if not cand or not cand.get("summary"):
        reasons.append("missing student cell")
    if not cand.get("recipe_hash") or not cand.get("eval_lock_hash"):
        reasons.append("missing recipe/eval lock hash")
    s = (cand or {}).get("summary") or {}
    q9 = (qwen9 or {}).get("summary") or {}
    if not qwen9 or qwen9.get("skipped") or not q9:
        reasons.append("qwen 9B same-protocol baseline missing; cannot claim reach")
    student = float(s.get("accepted_call_exact") or s.get("exact_match") or 0)
    qwen = float(q9.get("accepted_call_exact") or q9.get("exact_match") or 0)
    hard = float((gates.get("exact_match") or {}).get("overall") or 0.9)
    if student < max(hard, qwen):
        reasons.append(f"accepted_call_exact {student:.4f} < max(0.90, 9B={qwen:.4f})")
    by_kind = s.get("by_kind") or {}
    for fam in ("execute", "refuse"):
        need = float((gates.get("exact_match") or {}).get(f"{fam}_family") or 0.9)
        got = float((by_kind.get(fam) or {}).get("exact") or 0)
        if got < need:
            reasons.append(f"{fam} exact {got:.4f} < {need}")
    gnd = gates.get("grounding") or {}
    if float(s.get("unsupported_accepted_rate") or 0) > float(gnd.get("unsupported_accepted_max") or 0):
        reasons.append("unsupported accepted rate")
    if float(s.get("unprovenanced_accepted_rate") or 0) > float(gnd.get("unprovenanced_accepted_max") or 0):
        reasons.append("unprovenanced accepted rate")
    p50 = ((s.get("latency") or {}).get("p50_ms"))
    if p50 is not None and p50 > float((gates.get("latency") or {}).get("p50_ms_max") or 100):
        reasons.append("p50 latency gate")
    if (qwen9 or {}).get("skipped") or s.get("invalid_protocol"):
        reasons.append("protocol preflight")
    promote = not reasons
    dest = REG / ("mei-1.0-58m.json" if promote else "mei-1.0-58m.NOT_PROMOTED.json")
    payload = {
        "release_alias": "mei-1.0-58m",
        "promote": promote,
        "reasons": reasons,
        "candidate": cand,
        "qwen_9b": qwen9,
        "protocol": "mei-route-protocol-v1",
        "ssot": "SSOT:mei-llm distilled 2026-08-25 Route-ID",
        "note": "Publish bar is zero-shot Qwen3.5 9B on mei-tool-grounded-v1. v1 schema matrix is invalid_for_publish.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promote": promote, "path": str(dest.relative_to(ROOT)), "reasons": reasons}, indent=2))
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
