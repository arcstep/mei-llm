#!/usr/bin/env python3
"""Canonical paths for the multi-task mei-llm layout.

Shared corpora and eval banks live at the repo root. Per-model recipes live
under tasks/<id>/. Scripts should import this module instead of hardcoding
legacy train/ / mlx/ / data/eval/ paths.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "tasks"
CORPORA_ROOT = ROOT / "corpora"
CORPUS_ZH_VOCAB = CORPORA_ROOT / "zh-vocab-v0"
CORPUS_ZH_PRETRAIN = CORPORA_ROOT / "zh-pretrain-v0"
CORPUS_SHARED_TRACES = CORPORA_ROOT / "shared-tool-traces-v0"
EVAL_ROOT = ROOT / "eval"
EVAL_BANKS_ROOT = EVAL_ROOT / "banks"
EVAL_SHARED_ROOT = EVAL_ROOT / "shared"
EXPERIMENTS_RUNS = ROOT / "experiments/runs"
TASK_INDEX = TASKS_ROOT / "index.json"

TASK_MEI_EXPERT = "mei-expert-qwen35-0p8b"
TASK_NEEDLE_ZH = "needle-zh"

BANK_MEI_EXPERT = EVAL_BANKS_ROOT / "mei-expert-v0/eval-bank-v0.pending.jsonl"
HELDOUT_MEI_EXPERT = EVAL_BANKS_ROOT / "mei-expert-v0/heldout-task-ids.v0.json"
SEED_MEI_EXPERT = TASKS_ROOT / TASK_MEI_EXPERT / "train/seed/sft-smoke-v0.jsonl"
MLX_DIR_MEI_EXPERT = TASKS_ROOT / TASK_MEI_EXPERT / "mlx"
MLX_EXPORT_MEI_EXPERT = MLX_DIR_MEI_EXPERT / "exports/sft-smoke-v0"
MLX_ADAPTER_MEI_EXPERT = MLX_DIR_MEI_EXPERT / "adapters/qwen35-0.8b-sft-smoke"

BANK_NEEDLE_TOOLCALL = EVAL_BANKS_ROOT / "needle-toolcall-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_AGENT = EVAL_BANKS_ROOT / "needle-vrm-agent-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_AGENT_EN = EVAL_BANKS_ROOT / "needle-vrm-agent-en-v0/eval-bank-v0.jsonl"
SEED_NEEDLE_ZH = TASKS_ROOT / TASK_NEEDLE_ZH / "train/seed/sft-phase1-v0.jsonl"
SPEC_NEEDLE_ZH = TASKS_ROOT / TASK_NEEDLE_ZH / "spec"
TOKENIZER_ZH_V1 = CORPUS_ZH_VOCAB / "zh-24k-v1.model"
BANK_NEEDLE_PRETRAIN_PROBES = EVAL_BANKS_ROOT / "needle-pretrain-probes-v0/probes-v0.jsonl"


def load_task_index() -> dict:
    if not TASK_INDEX.is_file():
        return {"version": 1, "tasks": []}
    return json.loads(TASK_INDEX.read_text(encoding="utf-8"))


def task_record(task_id: str) -> dict:
    for row in load_task_index().get("tasks") or []:
        if row.get("id") == task_id:
            return row
    raise KeyError(f"unknown task_id={task_id}")


def repo_file(rel: str | Path) -> Path:
    return ROOT / rel


def all_eval_jsonl() -> list[Path]:
    if not EVAL_BANKS_ROOT.is_dir():
        return []
    return sorted(p for p in EVAL_BANKS_ROOT.rglob("*.jsonl") if p.is_file())


def all_train_seeds() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for row in load_task_index().get("tasks") or []:
        rel = row.get("train_seed")
        if rel:
            paths.append(ROOT / rel)
            seen.add(ROOT / rel)
        for extra in row.get("train_seeds") or []:
            p = ROOT / extra
            if p not in seen:
                paths.append(p)
                seen.add(p)
    for p in sorted((TASKS_ROOT).glob("*/train/seed/*.jsonl")):
        if p not in seen:
            paths.append(p)
            seen.add(p)
    for p in sorted((TASKS_ROOT).glob("*/train/packs/*.jsonl")):
        if p not in seen:
            paths.append(p)
            seen.add(p)
    return paths
