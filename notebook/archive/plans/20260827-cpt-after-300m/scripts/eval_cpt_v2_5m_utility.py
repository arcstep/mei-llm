#!/usr/bin/env python3
"""cpt-v2-5m utility gate vs 300M parent and no-colloquial ablation.

Does not launch 20M. Failure writes NOT_PROMOTED plus abstract feedback cards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    CORPUS_ZH_PRETRAIN_V4,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zh_pretrain_ingest import dump_json  # noqa: E402

PARENT = ROOT / "notebook/archive/base/mei-1.0-58m-checkpoints/pretrain-300m.npz"
TRAIN = SCRIPTS_ROOT / "train_needle_zh_pretrain.py"
EVAL = SCRIPTS_ROOT / "eval_needle_zh_pretrain_valid.py"
ISOLATION = SCRIPTS_ROOT / "check_train_eval_isolation.py"
CARDS = SCRIPTS_ROOT / "build_mei_colloquial_feedback_cards.py"
VALID_SETS = ("wiki", "hq", "structure", "colloquial")
CKPT_5M = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-cpt-v2-5m.npz"
CKPT_ABL = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-cpt-v2-5m-no-colloquial.npz"
REG = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "registry"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT))


def eval_ckpt(ckpt: Path, tag: str, out_dir: Path) -> dict:
    scores = {}
    for vs in VALID_SETS:
        dest = out_dir / f"{tag}-{vs}.json"
        proc = run(
            [
                sys.executable,
                str(EVAL),
                "--ckpt",
                str(ckpt),
                "--corpus-dir",
                str(CORPUS_ZH_PRETRAIN_V4),
                "--valid-set",
                vs,
                "--mode",
                "sentinel",
                "--out",
                str(dest),
            ]
        )
        payload = json.loads(dest.read_text(encoding="utf-8")) if dest.is_file() else {}
        scores[vs] = {
            "returncode": proc.returncode,
            "loss": payload.get("loss") or payload.get("valid_loss") or payload.get("nll"),
            "payload": payload,
        }
    return scores


def nll(row: dict) -> float | None:
    payload = row.get("payload") or {}
    val = payload.get("valid_loss") or row.get("loss") or payload.get("loss") or payload.get("mean_nll")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    out_dir = EXPERIMENTS_RUNS / "mei-1.0-58m-cpt-v2-5m-utility"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = json.loads((CORPUS_ZH_PRETRAIN_V4 / "RELEASE.json").read_text(encoding="utf-8"))
    if not rel.get("roles_complete"):
        dump_json(out_dir / "FAIL_CLOSED.json", {"ok": False, "reason": "roles_complete_false"})
        print("v4 roles_complete=false", file=sys.stderr)
        return 4
    if not PARENT.is_file():
        dump_json(out_dir / "MISSING_PARENT.json", {"ok": False, "path": str(PARENT)})
        print(f"missing parent weights {PARENT}", file=sys.stderr)
        return 2
    parent_scores = eval_ckpt(PARENT, "parent", out_dir)
    dump_json(out_dir / "parent.json", parent_scores)
    if not args.skip_train:
        proc = run(
            [
                sys.executable,
                str(TRAIN),
                "--rung",
                "cpt-v2-5m",
                "--corpus-dir",
                str(CORPUS_ZH_PRETRAIN_V4),
                "--init-weights",
                str(PARENT),
            ]
        )
        if proc.returncode != 0:
            dump_json(out_dir / "TRAIN_FAILED.json", {"ok": False, "returncode": proc.returncode})
            return proc.returncode
    if not CKPT_5M.is_file():
        print(f"missing {CKPT_5M}", file=sys.stderr)
        return 2
    five_scores = eval_ckpt(CKPT_5M, "cpt-v2-5m", out_dir)
    dump_json(out_dir / "cpt-v2-5m.json", five_scores)
    abl_scores = {}
    if not args.skip_ablation:
        proc = run(
            [
                sys.executable,
                str(TRAIN),
                "--rung",
                "cpt-v2-5m",
                "--run-suffix",
                "no-colloquial",
                "--exclude-source",
                "colloquial",
                "--corpus-dir",
                str(CORPUS_ZH_PRETRAIN_V4),
                "--init-weights",
                str(PARENT),
            ]
        )
        if proc.returncode != 0:
            dump_json(out_dir / "ABLATION_TRAIN_FAILED.json", {"ok": False, "returncode": proc.returncode})
            return proc.returncode
        if CKPT_ABL.is_file():
            abl_scores = eval_ckpt(CKPT_ABL, "no-colloquial", out_dir)
            dump_json(out_dir / "no-colloquial.json", abl_scores)
    iso_cpt = run([sys.executable, str(ISOLATION), "--scope", "cpt-v2"])
    iso_sft = run([sys.executable, str(ISOLATION), "--scope", "sft-v2"])
    col_parent = nll(parent_scores.get("colloquial") or {})
    col_five = nll(five_scores.get("colloquial") or {})
    col_abl = nll((abl_scores.get("colloquial") or {}) if abl_scores else {})
    colloquial_improved = col_five is not None and col_parent is not None and col_five < col_parent - 1e-4
    if col_abl is not None and col_five is not None:
        colloquial_improved = colloquial_improved and col_five < col_abl - 1e-4
    regressions = []
    for vs in ("wiki", "hq", "structure"):
        p = nll(parent_scores.get(vs) or {})
        f = nll(five_scores.get(vs) or {})
        if p is None or f is None:
            continue
        if f > p * 1.02 + 0.02:
            regressions.append({"valid_set": vs, "parent": p, "cpt5m": f})
    systematic = len(regressions) >= 2
    ok = colloquial_improved and not systematic and iso_cpt.returncode == 0
    report = {
        "ok": ok,
        "colloquial_improved": colloquial_improved,
        "systematic_non_spoken_regression": systematic,
        "regressions": regressions,
        "parent": {k: nll(v) for k, v in parent_scores.items()},
        "cpt_v2_5m": {k: nll(v) for k, v in five_scores.items()},
        "no_colloquial": {k: nll(v) for k, v in abl_scores.items()} if abl_scores else {},
        "isolation_cpt_v2": iso_cpt.returncode,
        "isolation_sft_v2": iso_sft.returncode,
        "promoted": ok,
        "next_rung_20m_allowed": ok,
    }
    dump_json(out_dir / ("PASS.json" if ok else "NOT_PROMOTED.json"), report)
    REG.mkdir(parents=True, exist_ok=True)
    if ok:
        dump_json(
            REG / "mei-1.0-58m-cpt-v2-5m.json",
            {
                "model_id": "mei-1.0-58m-cpt-v2-5m",
                "parent": "mei-1.0-58m-base-cpt300m-v1",
                "corpus": "zh-pretrain-v4",
                "promoted": True,
                "weights": str(CKPT_5M.relative_to(ROOT)),
            },
        )
    else:
        dump_json(
            REG / "mei-1.0-58m-cpt-v2-5m.NOT_PROMOTED.json",
            {
                "model_id": "mei-1.0-58m-cpt-v2-5m",
                "status": "NOT_PROMOTED",
                "reason": "utility_gate",
                "colloquial_improved": colloquial_improved,
                "regressions": regressions,
            },
        )
        failed_scenes = [r["valid_set"] for r in regressions] or ["colloquial"]
        run(
            [
                sys.executable,
                str(CARDS),
                "--out-dir",
                str(out_dir / "feedback"),
            ]
        )
        dump_json(
            out_dir / "qwen-v2-increment.json",
            {
                "parent_pack": "zh-pretrain-colloquial-synth-qwen-v1",
                "next_pack": "zh-pretrain-colloquial-synth-qwen-v2",
                "failed_buckets": failed_scenes,
                "copy_holdout": False,
            },
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
