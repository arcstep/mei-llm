#!/usr/bin/env python3
"""No-training baselines for frozen v1/v2 and MW v0. Does not create checkpoints."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    BANK_NEEDLE_VRM_AGENT_V2,
    BANK_NEEDLE_VRM_MW,
    BANK_NEEDLE_VRM_MW_LOCK,
    EXPERIMENTS_RUNS,
    PACK_NEEDLE_MW_SFT_2K,
    ROOT,
    SCHEMA_MW_GOVERNANCE,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_needle_mw_v0 import constant_preds, gold_oracle_phase1, load_jsonl, schema_errors, score_rows  # noqa: E402
from eval_needle_toolcall_v0 import gold_baselines, schema_errors as vrm_schema_errors  # noqa: E402
from needle_home_sft_lib import sha256_file  # noqa: E402
from needle_mw_governance_lib import V1_SHA, V2_SHA, frozen_upstream_hashes  # noqa: E402


def _sha(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _ckpts() -> list[dict]:
    d = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints"
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.npz")):
        if p.name.endswith("-state.npz"):
            continue
        kind = "sft-smoke" if "sft" in p.name and "smoke" in p.name else (
            "pretrain" if p.name.startswith("pretrain-") else "other"
        )
        product = kind == "sft-smoke"  # smoke is not a product router
        out.append(
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": _sha(p),
                "bytes": p.stat().st_size,
                "kind": kind,
                "usable_as_phase1_router": False,
                "note": "smoke SFT or CPT weights; no product phase-1 tool-router checkpoint yet",
            }
        )
    return out


def score_vrm(bank: Path, split: str) -> dict:
    rows = [r for r in load_jsonl(bank) if r.get("split") == split]
    errors = vrm_schema_errors(rows)
    return {
        "bank": str(bank.relative_to(ROOT)),
        "split": split,
        "n": len(rows),
        "schema_ok": not errors,
        "sha256": _sha(bank),
        "always_refuse": gold_baselines(rows),
    }


def score_mw(strategy: str, split: str) -> dict:
    rows = [r for r in load_jsonl(BANK_NEEDLE_VRM_MW) if r.get("split") == split]
    if strategy == "oracle_phase1":
        preds = gold_oracle_phase1(rows)
        projection = "phase1"
    else:
        preds = constant_preds(rows, strategy)
        projection = "phase1" if strategy == "always_refuse" else None
    scored = score_rows(rows, preds, projection=projection)
    scored.pop("details", None)
    return {
        "strategy": strategy,
        "split": split,
        "n": len(rows),
        "score": scored,
    }


def main() -> int:
    frozen = frozen_upstream_hashes()
    if frozen["v1_bank_sha256"] != V1_SHA or frozen["v2_bank_sha256"] != V2_SHA:
        print(json.dumps({"error": "upstream hash drift", "frozen": frozen}, ensure_ascii=False))
        return 1
    mw_rows = load_jsonl(BANK_NEEDLE_VRM_MW)
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "spine": "needle-mw-no-train-baselines",
        "did_not_train": True,
        "did_not_modify_v1_v2_home_sft_cpt": True,
        "runner": "scripts/run_needle_mw_no_train_baselines.py",
        "runner_sha256": _sha(Path(__file__).resolve()),
        "schema_sha256": _sha(SCHEMA_MW_GOVERNANCE),
        "mw_bank_sha256": _sha(BANK_NEEDLE_VRM_MW),
        "mw_sft_sha256": _sha(PACK_NEEDLE_MW_SFT_2K),
        "mw_lock_sha256": _sha(BANK_NEEDLE_VRM_MW_LOCK),
        "frozen_upstream": frozen,
        "v1_eval": score_vrm(BANK_NEEDLE_VRM_AGENT, "eval"),
        "v2_eval": score_vrm(BANK_NEEDLE_VRM_AGENT_V2, "eval"),
        "mw_schema_ok": not schema_errors(mw_rows, kind="eval"),
        "mw_strategies": {
            "always_stop": score_mw("always_stop", "eval"),
            "always_execute": score_mw("always_execute", "eval"),
            "always_refuse_phase1": score_mw("always_refuse", "eval"),
            "oracle_phase1_upper_bound": score_mw("oracle_phase1", "eval"),
        },
        "needle_student_checkpoint": {
            "status": "pending_parallel_training_task",
            "reason": "No non-smoke phase-1 tool-router checkpoint. Pretrain npz is CPT, not a closed-set router. Smoke SFT is tiny/6-step and is not scored as a product model.",
            "found": _ckpts(),
        },
        "qwen_behavior_baseline": {
            "status": "attempted_below",
            "note": "Qwen is not a needle-zh proxy. Five-act closed prompt lives in run_eval_needle_mw_qwen_v0.py.",
        },
    }

    qwen_dir = EXPERIMENTS_RUNS / "needle-mw-qwen-baseline"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    qwen_summary = {
        "status": "skipped",
        "reason": "backend unavailable or probe failed",
    }
    try:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_eval_needle_mw_qwen_v0.py"),
            "--backend",
            "ollama",
            "--model",
            "qwen3.5:0.8b-mlx",
            "--split",
            "eval",
            "--limit",
            "8",
            "--timeout",
            "20",
            "--out-dir",
            str(qwen_dir / "probe"),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=90)
        if proc.returncode == 0:
            summary_path = qwen_dir / "probe" / "summary.json"
            qwen_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
                "status": "ok_no_summary"
            }
            qwen_summary["probe_n"] = 8
            qwen_summary["status"] = "probe_only"
            qwen_summary["note"] = (
                "8-item Ollama probe. Full-bank Qwen run is optional and not required to freeze data. "
                "Qwen is not a needle-zh proxy."
            )
        else:
            qwen_summary = {
                "status": "skipped",
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[-800:],
                "note": "Ollama/Qwen unavailable. Fixed-strategy baselines still stand. Parallel task may re-run full Qwen.",
            }
    except (OSError, subprocess.TimeoutExpired) as exc:
        qwen_summary = {"status": "skipped", "error": str(exc)}
    report["qwen_behavior_baseline"] = qwen_summary

    out_dir = BANK_NEEDLE_VRM_MW.parent / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "no-train-baselines.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_path.relative_to(ROOT)), "v1_always_refuse": report["v1_eval"]["always_refuse"]["always_refuse_exact_match"], "v2_always_refuse": report["v2_eval"]["always_refuse"]["always_refuse_exact_match"], "mw_always_stop_act_macro_f1": report["mw_strategies"]["always_stop"]["score"]["act_macro_f1"], "mw_always_refuse_unsafe": report["mw_strategies"]["always_refuse_phase1"]["score"]["unsafe_execute_rate"], "qwen": report["qwen_behavior_baseline"].get("status"), "ckpt": report["needle_student_checkpoint"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
