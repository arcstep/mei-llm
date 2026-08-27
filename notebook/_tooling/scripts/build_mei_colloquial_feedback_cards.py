#!/usr/bin/env python3
"""Build abstract colloquial feedback cards from existing slice reports.

Cards never copy holdout queries. They only raise scene/style quotas for the
next synth release. ME/HD/SF leaf names are not redefined here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_MW,
    EXPERIMENTS_RUNS,
    ROOT,
    SCHEMA_MW_GOVERNANCE,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from zh_pretrain_ingest import dump_json, sha256_text

FAMILY_TO_STYLE = {
    "gesture": ["ellipsis", "deixis", "short_reply"],
    "paraphrase": ["filler", "repair"],
    "home": ["deixis", "number_date_unit"],
    "order": ["repair", "interrupt"],
    "missing": ["negation", "ellipsis"],
    "scene_conflict": ["negation"],
    "illegal_pair": ["negation"],
    "offtopic": ["filler"],
}
FAMILY_TO_SCENE = {
    "home": ["home", "repair"],
    "order": ["restaurant", "delivery"],
    "gesture": ["home", "workplace"],
    "paraphrase": ["phone_call", "family"],
}
SLICE_TO_STYLE = {
    "S0": ["filler"],
    "S1": ["ellipsis"],
    "S2": ["repair"],
    "S3": ["deixis"],
    "S4": ["negation"],
    "S5": ["interrupt"],
    "S6": ["short_reply"],
    "S7": ["number_date_unit"],
}
ACT_TO_LAYER = {
    "execute": "sft",
    "expand": "sft",
    "shape": "sft",
    "escalate": "runtime",
    "stop": "runtime",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def weak_families(report: dict) -> list[tuple[str, float, int]]:
    out = []
    by = ((report.get("v1_eval") or {}).get("always_refuse") or {}).get("by_family") or {}
    if not by:
        by = report.get("families") or {}
    for fam, row in by.items():
        n = int(row.get("n") or 0)
        exact = float(row.get("exact_match") if row.get("exact_match") is not None else row.get("always_refuse_exact_match") or 0)
        if n >= 3 and exact <= 0.15:
            out.append((str(fam), exact, n))
    return out


def cards_from_report(report: dict, *, source: str) -> list[dict]:
    cards = []
    for fam, exact, n in weak_families(report):
        cards.append(
            {
                "card_id": sha256_text(f"{source}|family|{fam}")[:16],
                "cluster": f"family:{fam}",
                "layer": "cpt" if fam in {"paraphrase", "home"} else "sft",
                "evidence_runs": [source],
                "target_scenes": FAMILY_TO_SCENE.get(fam, ["home", "family"]),
                "target_styles": FAMILY_TO_STYLE.get(fam, ["filler", "ellipsis"]),
                "quota_docs": min(8000, max(400, n * 40)),
                "forbidden_strings": [],
                "stop_conditions": ["no_holdout_copy", "no_gold_answers", "no_student_recycle"],
                "metrics": {"family": fam, "exact_match": exact, "n": n},
            }
        )
    by_act = (report.get("mw") or report.get("by_act") or {})
    if isinstance(by_act, dict):
        for act, row in by_act.items():
            if not isinstance(row, dict):
                continue
            acc = float(row.get("accuracy") or row.get("exact") or 1.0)
            if acc < 0.5:
                cards.append(
                    {
                        "card_id": sha256_text(f"{source}|act|{act}")[:16],
                        "cluster": f"mw_act:{act}",
                        "layer": ACT_TO_LAYER.get(str(act), "sft"),
                        "evidence_runs": [source],
                        "target_scenes": ["home", "phone_call", "workplace"],
                        "target_styles": ["ellipsis", "deixis", "repair"],
                        "quota_docs": 1200,
                        "forbidden_strings": [],
                        "stop_conditions": ["no_holdout_copy"],
                        "metrics": {"act": act, "accuracy": acc},
                    }
                )
    by_slice = report.get("by_slice") or {}
    for sl, row in by_slice.items():
        if not isinstance(row, dict):
            continue
        n = int(row.get("n") or 0)
        n_pass = int(row.get("n_pass") or 0)
        if n >= 4 and (n_pass / n) < 0.4:
            cards.append(
                {
                    "card_id": sha256_text(f"{source}|slice|{sl}")[:16],
                    "cluster": f"schema_slice:{sl}",
                    "layer": "sft",
                    "evidence_runs": [source],
                    "target_scenes": ["workplace", "home"],
                    "target_styles": SLICE_TO_STYLE.get(str(sl), ["filler"]),
                    "quota_docs": 800,
                    "forbidden_strings": [],
                    "stop_conditions": ["no_holdout_copy"],
                    "metrics": {"slice": sl, "n": n, "n_pass": n_pass},
                }
            )
    return cards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slice-json",
        type=Path,
        action="append",
        default=[],
        help="Optional extra eval reports with families/by_act/by_slice",
    )
    ap.add_argument("--out-dir", type=Path, default=EXPERIMENTS_RUNS / "colloquial-feedback")
    args = ap.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "notebook/evaluation/banks/needle-vrm-mw-v0/baselines/no-train-baselines.json",
        ROOT / "notebook/evaluation/banks/needle-vrm-mw-v0/baselines/isolation-all.json",
    ]
    sources.extend(args.slice_json)
    cards: list[dict] = []
    seen = set()
    used = []
    for path in sources:
        p = path if path.is_absolute() else ROOT / path
        if not p.is_file():
            continue
        used.append(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        for card in cards_from_report(load_json(p), source=used[-1]):
            if card["card_id"] in seen:
                continue
            seen.add(card["card_id"])
            cards.append(card)
    if not cards:
        cards.append(
            {
                "card_id": "seed-coverage",
                "cluster": "coverage_floor",
                "layer": "cpt",
                "evidence_runs": [],
                "target_scenes": ["home", "commute", "family", "workplace"],
                "target_styles": ["ellipsis", "repair", "short_reply", "number_date_unit"],
                "quota_docs": 2000,
                "forbidden_strings": [],
                "stop_conditions": ["no_holdout_copy"],
            }
        )
    jsonl = out / "cards-v0.jsonl"
    jsonl.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards), encoding="utf-8")
    dump_json(
        out / "cards-v0.manifest.json",
        {
            "n_cards": len(cards),
            "sources": used,
            "mw_schema": str(SCHEMA_MW_GOVERNANCE.relative_to(ROOT)),
            "mw_bank": str(BANK_NEEDLE_VRM_MW.relative_to(ROOT)),
            "forbid_holdout_copy": True,
            "taxonomy_note": "ME/HD/SF leaves stay in private mei-world SSOT; cards only carry product clusters.",
        },
    )
    print(json.dumps({"ok": True, "n_cards": len(cards), "out": str(jsonl)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
