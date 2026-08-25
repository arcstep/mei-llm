#!/usr/bin/env python3
"""Grounded Route-ID matrix: Qwen3.5 9B vs mei-1.0-58m only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from eval_mei_tool_grounded_v1 import BANK, LOCK, aggregate, score_item
from eval_needle_toolcall_v0 import load_jsonl
from repo_paths import EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

PY = sys.executable
CKPT_300 = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "pretrain-300m.npz"
PACK2 = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / "mei-tool-route-sft-v1-2k.jsonl"
PACK10 = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / "mei-tool-route-sft-v1-10k.jsonl"
VALID = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs" / "mei-tool-route-sft-v1-valid.jsonl"
REC2 = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / "mei-tool-route-sft-v1-2k.json"
REC10 = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / "mei-tool-route-sft-v1-10k.json"


def run(cmd: list[str]) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def always_refuse(bank: Path, out_dir: Path) -> dict:
    rows = load_jsonl(bank)
    preds = []
    scored = []
    for r in rows:
        item = score_item(r, raw_text="[]")
        scored.append(item)
        preds.append(
            {
                "item_id": r.get("item_id"),
                "text": "[]",
                "raw_route_output": "[]",
                "function_calls": [],
                "protocol_id": "mei-route-protocol-v1",
                "serializer": "mei-route-serializer-v1",
                "validator_id": "mei-provenance-validator-v1",
                "scorer_id": "mei-grounded-scorer-v1",
            }
        )
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
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--qwen-model", default="qwen3.5:9b-mlx")
    args = ap.parse_args()
    matrix_dir = EXPERIMENTS_RUNS / "mei-1.0-58m-matrix-grounded"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    refuse = always_refuse(BANK, matrix_dir / "always-refuse")
    cells.append({"id": "always-refuse", "summary": refuse, "role": "floor"})

    if not args.skip_overfit:
        rc = run(
            [
                PY,
                str(ROOT / "scripts" / "train_mei_58m_sft.py"),
                "--init-ckpt",
                str(CKPT_300),
                "--pack",
                str(PACK2),
                "--overfit-gate",
                "--run-name",
                "mei-1.0-58m-route-overfit",
            ]
        )
        overfit = json.loads((EXPERIMENTS_RUNS / "mei-1.0-58m-route-overfit" / "overfit.json").read_text()) if rc == 0 else {"ok": False, "returncode": rc}
        cells.append({"id": "overfit-gate", "summary": overfit})
        if not overfit.get("ok"):
            (matrix_dir / "matrix.json").write_text(
                json.dumps({"ok": False, "reason": "overfit_gate_failed", "cells": cells}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"ok": False, "reason": "overfit_gate_failed"}, indent=2))
            return 3

    qwen_dir = matrix_dir / "qwen35-9b"
    if not args.skip_qwen:
        qcmd = [
            PY,
            str(ROOT / "scripts" / "run_eval_mei_tool_grounded_qwen_v0.py"),
            "--backend",
            "ollama",
            "--model",
            args.qwen_model,
            "--out-dir",
            str(qwen_dir),
        ]
        if args.limit_eval:
            qcmd += ["--limit", str(args.limit_eval)]
        rc = run(qcmd)
        qsum = json.loads((qwen_dir / "summary.json").read_text()) if (qwen_dir / "summary.json").is_file() else {"skipped": True, "returncode": rc}
        cells.append({"id": "qwen35-9b", "summary": qsum, "skipped": bool(qsum.get("skipped") or rc != 0)})
    else:
        cells.append({"id": "qwen35-9b", "skipped": True, "summary": {}})

    jobs = [
        {
            "id": "mei-1.0-58m-route-cpt300m-sft2k-v1",
            "pack": PACK2,
            "recipe": REC2,
            "sft_pack": "mei-tool-route-sft-v1-2k",
        },
        {
            "id": "mei-1.0-58m-route-cpt300m-sft10k-v1",
            "pack": PACK10,
            "recipe": REC10,
            "sft_pack": "mei-tool-route-sft-v1-10k",
        },
    ]
    lock = LOCK
    if not args.skip_train:
        for job in jobs:
            tcmd = [
                PY,
                str(ROOT / "scripts" / "train_mei_58m_sft.py"),
                "--init-ckpt",
                str(CKPT_300),
                "--pack",
                str(job["pack"]),
                "--valid",
                str(VALID),
                "--recipe",
                str(job["recipe"]),
                "--eval-lock",
                str(lock),
                "--run-name",
                job["id"],
            ]
            rc = run(tcmd)
            ckpt = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / f"{job['id']}.npz"
            eval_dir = matrix_dir / job["id"]
            ecmd = [
                PY,
                str(ROOT / "scripts" / "run_eval_mei_tool_grounded_student_v0.py"),
                "--ckpt",
                str(ckpt),
                "--out-dir",
                str(eval_dir),
            ]
            if args.limit_eval:
                ecmd += ["--limit", str(args.limit_eval)]
            if rc == 0 and ckpt.is_file():
                run(ecmd)
            summary = json.loads((eval_dir / "summary.json").read_text()) if (eval_dir / "summary.json").is_file() else {}
            train_meta = json.loads((EXPERIMENTS_RUNS / job["id"] / "summary.json").read_text()) if (EXPERIMENTS_RUNS / job["id"] / "summary.json").is_file() else {}
            cells.append(
                {
                    "id": job["id"],
                    "base_id": "mei-1.0-58m-base-cpt300m-v1",
                    "sft_pack": job["sft_pack"],
                    "recipe_hash": train_meta.get("recipe_hash"),
                    "eval_lock_hash": train_meta.get("eval_lock_hash") or (hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else ""),
                    "pack_sha256": train_meta.get("pack_sha256"),
                    "summary": summary,
                    "train": train_meta,
                }
            )
    matrix = {
        "id": "mei-1.0-58m-matrix-grounded",
        "protocol": "mei-route-protocol-v1",
        "eval_bank": "mei-tool-grounded-v1",
        "eval_lock_hash": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else "",
        "systems": ["qwen35-9b", "mei-1.0-58m"],
        "cells": cells,
    }
    (matrix_dir / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "path": str((matrix_dir / "matrix.json").relative_to(ROOT)), "n_cells": len(cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
