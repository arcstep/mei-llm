#!/usr/bin/env python3
"""Independent CPT staircase from 300M base. Fail-closed without a formal colloquial source.

When roles_complete, --dry-run writes planned commands; --launch runs each rung
in order and refuses to overwrite the parent 300M registry entry.
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
    REGISTRY_MEI_51M_CPT300M,
    ROOT,
    SCRIPTS_ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    TRAINING_V1,
)

RUNGS = (
    ("cpt-v2-5m", 5_000_000),
    ("cpt-v2-20m", 20_000_000),
    ("cpt-v2-50m", 50_000_000),
    ("cpt-v2-300m", 300_000_000),
)
REG = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / "registry"
TRAIN = TRAINING_V1 / "train_pretrain.py"
EVAL_VALID = SCRIPTS_ROOT / "eval_needle_zh_pretrain_valid.py"
PARENT_WEIGHTS = "notebook/archive/base/mei-1.0-51m-checkpoints/pretrain-300m.npz"
VALID_SETS = ("wiki", "hq", "structure", "colloquial")


def rung_ckpt(rung: str) -> Path:
    return TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints" / f"{rung}.npz"


def planned_cmd(rung: str, init: str) -> list[str]:
    return [
        sys.executable,
        str(TRAIN),
        "--rung",
        rung,
        "--corpus-dir",
        str(CORPUS_ZH_PRETRAIN_V4),
        "--init-weights",
        str(ROOT / init if not Path(init).is_absolute() else init),
    ]


def eval_cmds(rung: str) -> list[list[str]]:
    ckpt = rung_ckpt(rung)
    cmds = []
    for vs in VALID_SETS:
        cmds.append(
            [
                sys.executable,
                str(EVAL_VALID),
                "--ckpt",
                str(ckpt),
                "--corpus-dir",
                str(CORPUS_ZH_PRETRAIN_V4),
                "--valid-set",
                vs,
                "--mode",
                "sentinel",
            ]
        )
    return cmds


def write_fail_closed(out_dir: Path, report: dict) -> None:
    (out_dir / "FAIL_CLOSED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REG.mkdir(parents=True, exist_ok=True)
    (REG / "mei-1.0-51m-cpt-v2.FAIL_CLOSED.json").write_text(
        json.dumps(
            {
                "model_id": "mei-1.0-51m-cpt-v2",
                "status": "fail_closed",
                "parent": "mei-1.0-51m-base-cpt300m-v1",
                "corpus": "zh-pretrain-v4",
                "reason": report["stop_reason"],
                "rungs_not_started": [r for r, _ in RUNGS],
                "immutable_parent_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-train", action="store_true", help="ignored unless roles_complete")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-rung", choices=[r for r, _ in RUNGS], default="cpt-v2-300m")
    ap.add_argument(
        "--ablation",
        choices=["none", "no-colloquial", "prev-release"],
        default="none",
        help="Record ablation intent. no-colloquial/prev-release never mutate the parent 300M weights.",
    )
    args = ap.parse_args()
    release_path = CORPUS_ZH_PRETRAIN_V4 / "RELEASE.json"
    if not release_path.is_file():
        print("missing zh-pretrain-v4 RELEASE.json; run build_zh_pretrain_v4.py", file=sys.stderr)
        return 2
    release = json.loads(release_path.read_text(encoding="utf-8"))
    out_dir = EXPERIMENTS_RUNS / "mei-1.0-51m-cpt-v2-staircase"
    out_dir.mkdir(parents=True, exist_ok=True)
    REG.mkdir(parents=True, exist_ok=True)
    roles_complete = bool(release.get("roles_complete"))
    max_idx = [r for r, _ in RUNGS].index(args.max_rung)
    planned = []
    init = PARENT_WEIGHTS
    for i, (rung, ntok) in enumerate(RUNGS[: max_idx + 1]):
        planned.append(
            {
                "rung": rung,
                "unique_tokens": ntok,
                "init_weights": init,
                "train": planned_cmd(rung, init),
                "eval": eval_cmds(rung),
                "allow_repeat": False,
                "overwrite_parent_300m": False,
            }
        )
        init = str(rung_ckpt(rung).relative_to(ROOT))
    report = {
        "ok": False,
        "parent": "mei-1.0-51m-base-cpt300m-v1",
        "parent_registry": str(REGISTRY_MEI_51M_CPT300M.relative_to(ROOT)),
        "corpus": "zh-pretrain-v4",
        "rungs_planned": [r for r, _ in RUNGS[: max_idx + 1]],
        "mtp": False,
        "retrieval_loss": False,
        "sft_loss": False,
        "confidence_loss": False,
        "allow_repeat": False,
        "roles_complete": roles_complete,
        "excludes_cwt2": True,
        "excludes_v2_dirty_structure": True,
        "ran_training": False,
        "stop_reason": None,
        "ablation": args.ablation,
        "commands": planned,
        "rung_reports": [],
    }
    (out_dir / "PLAN.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not roles_complete:
        report["stop_reason"] = "fail_closed_missing_license_cleared_colloquial"
        report["ok"] = True
        report["fail_closed"] = True
        report["note"] = (
            "Formal independent CPT did not start. Wiki/HQ must not impersonate a four-role mix. "
            "Register a frozen qwen-plus approved-colloquial.json (≥30M unique, audit ok) "
            "and rebuild v4 before 5M→20M→50M→300M unique."
        )
        write_fail_closed(out_dir, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.ablation != "none":
        report["stop_reason"] = f"ablation_{args.ablation}_recorded_not_launched"
        report["ok"] = True
        report["note"] = (
            "Ablation is a comparison plan only. no-colloquial uses v4 without spoken shards; "
            "prev-release keeps the previous synth unique mix. Parent 300M stays immutable."
        )
        (out_dir / f"ABLATION_{args.ablation}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if not args.launch:
        report["stop_reason"] = "roles_complete_dry_run"
        report["ok"] = True
        report["dry_run"] = True
        (out_dir / "PENDING.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if not (ROOT / PARENT_WEIGHTS).is_file():
        report["stop_reason"] = "missing_parent_weights"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    for step in planned:
        cmd = step["train"]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        step_rep = {"rung": step["rung"], "returncode": proc.returncode, "train": cmd}
        if proc.returncode != 0:
            report["stop_reason"] = f"train_failed_{step['rung']}"
            report["rung_reports"].append(step_rep)
            (out_dir / "FAILED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return proc.returncode
        eval_ok = True
        eval_reps = []
        for ecmd in step["eval"]:
            eproc = subprocess.run(ecmd, cwd=str(ROOT))
            eval_reps.append({"cmd": ecmd, "returncode": eproc.returncode})
            if eproc.returncode not in (0, 2):
                eval_ok = False
        step_rep["eval"] = eval_reps
        step_rep["eval_ok_or_missing_valid"] = eval_ok
        report["rung_reports"].append(step_rep)
        (REG / f"mei-1.0-51m-{step['rung']}.json").write_text(
            json.dumps(
                {
                    "model_id": f"mei-1.0-51m-{step['rung']}",
                    "parent": "mei-1.0-51m-base-cpt300m-v1",
                    "rung": step["rung"],
                    "corpus": "zh-pretrain-v4",
                    "allow_repeat": False,
                    "overwrites_parent_300m": False,
                    "weights": str(rung_ckpt(step["rung"]).relative_to(ROOT)),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    report["ok"] = True
    report["ran_training"] = True
    report["stop_reason"] = None
    (out_dir / "DONE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
