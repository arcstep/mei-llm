#!/usr/bin/env python3
"""Minimal CPT×SFT matrix: cpt300m × sft2k/10k + random-init + always-refuse.

Default Qwen cell is zero-shot 9B (publish bar). 0.8B is optional --also-qwen-0p8b.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from eval_mei_tool_schema_v1 import aggregate, score_item
from eval_needle_toolcall_v0 import load_jsonl, norm_calls
from repo_paths import EVAL_BANKS_ROOT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

PY = sys.executable
BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"
LOCK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/holdout-schema-v1.lock.json"
CKPT_300 = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-300m.npz"
PACK2 = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / "mei-tool-sft-v1-2k.jsonl"
PACK10 = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / "mei-tool-sft-v1-10k.jsonl"
REC2 = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / "mei-tool-sft-v1-2k.json"
REC10 = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / "mei-tool-sft-v1-10k.json"


def run(cmd: list[str]) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def always_refuse(bank: Path, out_dir: Path) -> dict:
    rows = load_jsonl(bank)
    preds = [{"item_id": r.get("item_id"), "function_calls": [], "text": "[]"} for r in rows]
    scored = [score_item(r, [], raw_text="[]") for r in rows]
    summary = aggregate(rows, scored)
    summary["system"] = "always-refuse"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--also-qwen-0p8b", action="store_true", help="Optional side cell; 9B is the publish bar")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--limit-eval", type=int, default=None)
    args = ap.parse_args()
    matrix_dir = EXPERIMENTS_RUNS / "mei-1.0-58m-matrix-cpt300m"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    cells = []

    refuse = always_refuse(BANK, matrix_dir / "always-refuse")
    cells.append({"id": "always-refuse", "summary": refuse})

    jobs = [
        {
            "id": "mei-1.0-58m-tool-cpt300m-sft2k-v1",
            "init": CKPT_300,
            "pack": PACK2,
            "recipe": REC2,
            "base_id": "mei-1.0-58m-base-cpt300m-v1",
            "cpt_unique": 300001280,
            "cpt_exposure": 300001280,
            "sft_pack": "mei-tool-sft-v1-2k",
        },
        {
            "id": "mei-1.0-58m-tool-cpt300m-sft10k-v1",
            "init": CKPT_300,
            "pack": PACK10,
            "recipe": REC10,
            "base_id": "mei-1.0-58m-base-cpt300m-v1",
            "cpt_unique": 300001280,
            "cpt_exposure": 300001280,
            "sft_pack": "mei-tool-sft-v1-10k",
        },
        {
            "id": "mei-1.0-58m-tool-random-sft2k-v1",
            "init": None,
            "pack": PACK2,
            "recipe": REC2,
            "base_id": "random-init-58m",
            "cpt_unique": 0,
            "cpt_exposure": 0,
            "sft_pack": "mei-tool-sft-v1-2k",
        },
    ]
    if not args.skip_train:
        for job in jobs:
            cmd = [
                PY,
                str(ROOT / "scripts" / "train_mei_58m_sft.py"),
                "--pack",
                str(job["pack"]),
                "--recipe",
                str(job["recipe"]),
                "--eval-lock",
                str(LOCK),
                "--steps",
                str(args.steps),
                "--run-name",
                job["id"],
            ]
            if job["init"]:
                cmd += ["--init-ckpt", str(job["init"])]
            rc = run(cmd)
            if rc != 0:
                return rc
            ckpt = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / f"{job['id']}.npz"
            eval_dir = matrix_dir / job["id"]
            ecmd = [
                PY,
                str(ROOT / "scripts" / "run_eval_mei_tool_schema_student_v0.py"),
                "--ckpt",
                str(ckpt),
                "--out-dir",
                str(eval_dir),
            ]
            if args.limit_eval:
                ecmd += ["--limit", str(args.limit_eval)]
            rc = run(ecmd)
            if rc != 0:
                return rc
            summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
            recipe_hash = hashlib.sha256(job["recipe"].read_bytes()).hexdigest()
            eval_lock_hash = hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.is_file() else ""
            pack_hash = hashlib.sha256(job["pack"].read_bytes()).hexdigest()
            if not recipe_hash or not eval_lock_hash or not pack_hash:
                print(json.dumps({"error": "missing hash", "job": job["id"]}), file=sys.stderr)
                return 2
            cells.append(
                {
                    "id": job["id"],
                    "base_id": job["base_id"],
                    "cpt_unique": job["cpt_unique"],
                    "cpt_exposure": job["cpt_exposure"],
                    "sft_pack": job["sft_pack"],
                    "quality_tag": "v1-program-gold",
                    "recipe_hash": recipe_hash,
                    "eval_lock_hash": eval_lock_hash,
                    "pack_sha256": pack_hash,
                    "summary": summary,
                }
            )

    if not args.skip_qwen:
        q9dir = matrix_dir / "qwen35-9b"
        q9cmd = [
            PY,
            str(ROOT / "scripts" / "run_eval_mei_tool_schema_qwen_v0.py"),
            "--backend",
            "ollama",
            "--model",
            "qwen3.5:9b-mlx",
            "--out-dir",
            str(q9dir),
        ]
        if args.limit_eval:
            q9cmd += ["--limit", str(args.limit_eval)]
        rc = run(q9cmd)
        if rc == 0:
            cells.append({"id": "qwen35-9b", "summary": json.loads((q9dir / "summary.json").read_text())})
        else:
            cells.append({"id": "qwen35-9b", "error": f"exit {rc}", "skipped": True})
        if args.also_qwen_0p8b:
            qdir = matrix_dir / "qwen35-0.8b"
            qcmd = [
                PY,
                str(ROOT / "scripts" / "run_eval_mei_tool_schema_qwen_v0.py"),
                "--out-dir",
                str(qdir),
            ]
            if args.limit_eval:
                qcmd += ["--limit", str(args.limit_eval)]
            rc = run(qcmd)
            if rc == 0:
                cells.append({"id": "qwen35-0.8b", "summary": json.loads((qdir / "summary.json").read_text())})
            else:
                cells.append({"id": "qwen35-0.8b", "error": f"exit {rc}", "skipped": True})

    report = {
        "matrix": "cpt300m × sft2k/10k",
        "publish_bar": "qwen35-9b same-protocol zero-shot",
        "expand_1b_only_after_pass": True,
        "cells": cells,
    }
    (matrix_dir / "matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_cells": len(cells), "out": str(matrix_dir.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
