#!/usr/bin/env python3
"""Offline Job → accepted/rejected → validate → draft receipt for the three SFT lines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    FLEET_SFT_SYNTH,
    JOB_FULLCALL_V2,
    JOB_MW_DISPOSITION_V1,
    JOB_RETRIEVAL_V2,
    ROOT,
    SCRIPTS_ROOT,
    SFT_TRAIN,
    TOPIC_TOOLCALL_SFT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from jobs.create_job import ensure_topic  # noqa: E402
from jobs.paths import job_card, job_dir, job_scratch, job_work, receipt_path, rel_to_root, topic_dir  # noqa: E402
from jobs.schema import validate_job, validate_receipt  # noqa: E402
from jobs.registry import upsert_job, upsert_receipt_pointer  # noqa: E402
from sft_canonical_lib import dump_jsonl, load_jsonl  # noqa: E402
from sft_synth_lib import dump_json, fleet_contract, load_fleet, sha256_text  # noqa: E402

LINES = (
    {
        "job": JOB_RETRIEVAL_V2,
        "task": "retrieval",
        "builder": "build_mei_retrieval_v2.py",
        "validator": "validate_mei_retrieval_v2_pack.py",
        "pack": SFT_TRAIN / "packs/mei-retrieval-v2-smoke.jsonl",
    },
    {
        "job": JOB_FULLCALL_V2,
        "task": "fullcall",
        "builder": "build_mei_toolcall_v2.py",
        "validator": "validate_mei_toolcall_v2_pack.py",
        "pack": SFT_TRAIN / "packs/mei-toolcall-v2-oracle-smoke.jsonl",
        "validator_args": ["--pack", str(SFT_TRAIN / "packs/mei-toolcall-v2-oracle-smoke.jsonl"), "--require-compiler"],
    },
    {
        "job": JOB_MW_DISPOSITION_V1,
        "task": "mw",
        "builder": "build_mei_mw_disposition_v2.py",
        "validator": "validate_mei_mw_disposition_v2.py",
        "pack": SFT_TRAIN / "packs/mei-mw-disposition-v2-smoke.jsonl",
    },
)


def run(script: str, argv: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / script), *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload: dict = {"returncode": proc.returncode, "stderr": proc.stderr.strip()}
    if proc.stdout.strip():
        try:
            payload["stdout"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload["stdout_text"] = proc.stdout[-1000:]
    return payload


JOB_SPECS = {
    JOB_RETRIEVAL_V2: "RetrievalHead pairs: positives, same-catalog hard negatives, no-match, seen/unseen schema families.",
    JOB_FULLCALL_V2: "v2 oracle-top5 full-call. Home 2k/10k are raw material; compiler writes gold.",
    JOB_MW_DISPOSITION_V1: "MWDispositionHead reason_code only. Runtime maps act/cell. Holdout stays isolated.",
}


def ensure_jobs() -> None:
    ensure_topic(TOPIC_TOOLCALL_SFT, root=ROOT)
    topic = topic_dir(TOPIC_TOOLCALL_SFT, root=ROOT)
    fleet = load_fleet(FLEET_SFT_SYNTH)
    readme = topic / "README.md"
    readme.write_text(
        "# toolcall-sft\n\n"
        "唯一落点：`notebook/jobs/toolcall-sft/`。三账三 Job：retrieval / full-call / MW。\n\n"
        "- fleet：`notebook/sft/mei-1.0-51m/recipes/sft-synth-fleet-v1.json`\n"
        "- 计量单位：accepted unique row\n"
        "- 教师只改写 query；gold 由 compiler/validator 确定\n"
        "- 付费 bake-off / 扩量必须另确认预算和 model snapshot\n",
        encoding="utf-8",
    )
    inbox = topic / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    fleet_src = FLEET_SFT_SYNTH.read_text(encoding="utf-8")
    (inbox / "sft-synth-fleet-v1.json").write_text(fleet_src, encoding="utf-8")
    for job_id, intent in JOB_SPECS.items():
        work = job_work(TOPIC_TOOLCALL_SFT, job_id, root=ROOT)
        work.mkdir(parents=True, exist_ok=True)
        job_scratch(TOPIC_TOOLCALL_SFT, job_id, root=ROOT).mkdir(parents=True, exist_ok=True)
        for name in ("raw", "state", "reviews", "tokens"):
            (work / name).mkdir(exist_ok=True)
        card = {
            "job_id": job_id,
            "topic": TOPIC_TOOLCALL_SFT,
            "kind": "produce",
            "status": "running",
            "task_id": "mei-1.0-51m",
            "work_dir": rel_to_root(work, root=ROOT),
            "receipt": None,
            "publish_corpus_id": None,
            "publish_path": None,
            "parents": [],
            "entry": rel_to_root(job_card(TOPIC_TOOLCALL_SFT, job_id, root=ROOT), root=ROOT),
            "intent": intent,
            **fleet_contract(fleet),
        }
        card["status"] = "running"
        validate_job(card)
        path = job_card(TOPIC_TOOLCALL_SFT, job_id, root=ROOT)
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        job_readme = job_dir(TOPIC_TOOLCALL_SFT, job_id, root=ROOT) / "README.md"
        job_readme.write_text(
            f"# {job_id}\n\n- topic: `{TOPIC_TOOLCALL_SFT}`\n- work: `{card['work_dir']}`\n- intent: {intent}\n",
            encoding="utf-8",
        )
        upsert_job(card, root=ROOT)


def write_draft_receipt(job_id: str, work: Path, extra: dict) -> Path:
    receipt = {
        "job_id": job_id,
        "topic": TOPIC_TOOLCALL_SFT,
        "status": "draft",
        "publish_policy": "promote_only",
        "work_dir": rel_to_root(work, root=ROOT),
        "gates": {
            "offline_template": True,
            "paid_api": False,
            "confidence_qat": False,
        },
        "unique_train_tokens": None,
        "n_accepted_unique": extra.get("n_accepted_unique"),
        "spend_cny": extra.get("spend_cny") or 0,
        "note": extra.get("note") or "offline template engineering_smoke",
        "engineering_smoke": True,
        "fleet_id": "sft-synth-fleet-v1",
        "model_snapshot": "template",
        "prompt_version": extra.get("prompt_version") or "sft-teacher-v2-query-only.1",
    }
    validate_receipt(receipt)
    path = receipt_path(TOPIC_TOOLCALL_SFT, job_id, "draft", root=ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card_path = job_card(TOPIC_TOOLCALL_SFT, job_id, root=ROOT)
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["status"] = "draft"
        card["receipt"] = rel_to_root(path, root=ROOT)
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        upsert_job(card, root=ROOT)
    upsert_receipt_pointer(receipt, root=ROOT)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()
    fleet = load_fleet(FLEET_SFT_SYNTH)
    ensure_jobs()
    reports = []
    ok = True
    for line in LINES:
        work = job_work(TOPIC_TOOLCALL_SFT, line["job"], root=ROOT) / "engineering-smoke"
        work.mkdir(parents=True, exist_ok=True)
        builder = run(line["builder"], ["--limit", str(args.limit), "--work-dir", str(work)])
        teacher = run(
            "produce_sft_teacher.py",
            [
                "--task",
                line["task"],
                "--cases",
                str(work / "canonical.jsonl"),
                "--work-dir",
                str(work),
                "--lane",
                "template",
                "--phase",
                "smoke",
            ],
        )
        accepted = load_jsonl(work / "raw" / "accepted.jsonl")
        if accepted:
            dump_jsonl(line["pack"], accepted)
            dump_jsonl(work / "accepted.jsonl", accepted)
        validator_args = line.get("validator_args") or ["--pack", str(line["pack"])]
        validator = run(line["validator"], validator_args)
        bakeoff = run(
            "run_sft_synth_bakeoff.py",
            [
                "--task",
                line["task"],
                "--cases",
                str(work / "canonical.jsonl"),
                "--lanes",
                "template",
                "--out",
                str(work / "bakeoff-template.json"),
            ],
        )
        isolation = run("check_train_eval_isolation.py", ["--scope", "sft-v2"])
        n_unique = len({sha256_text(str(r.get("query") or "")) for r in accepted})
        receipt = write_draft_receipt(
            line["job"],
            work,
            {
                "n_accepted_unique": n_unique,
                "spend_cny": 0,
                "note": f"offline template engineering_smoke for {line['task']}",
            },
        )
        dump_json(
            work / "manifest.json",
            {
                "job_id": line["job"],
                "task": line["task"],
                "fleet_id": fleet.get("id"),
                "model_snapshot": "template",
                "pack": str(line["pack"].relative_to(ROOT)),
                "n_accepted_unique": n_unique,
                "source_license": "schema-program + optional CrossWOZ/KdConv style only",
            },
        )
        item = {
            "job": line["job"],
            "task": line["task"],
            "builder": builder,
            "teacher": teacher,
            "validator": validator,
            "bakeoff": bakeoff,
            "isolation": {"returncode": isolation["returncode"]},
            "receipt": rel_to_root(receipt, root=ROOT),
            "n_accepted_unique": n_unique,
        }
        item["ok"] = (
            builder.get("returncode") == 0
            and teacher.get("returncode") == 0
            and validator.get("returncode") == 0
            and isolation.get("returncode") == 0
            and n_unique > 0
        )
        reports.append(item)
        ok = ok and item["ok"]
    stair = run("run_sft_synth_staircase.py", ["--tier", "smoke", "--out", str(topic_dir(TOPIC_TOOLCALL_SFT, root=ROOT) / "outbox" / "draft" / "staircase-smoke.json")])
    promote = run("promote_sft_v2_releases.py", ["--state", "engineering_smoke"])
    out = {
        "ok": ok and stair.get("returncode") == 0 and promote.get("returncode") == 0,
        "fleet": str(FLEET_SFT_SYNTH.relative_to(ROOT)),
        "lines": reports,
        "staircase": stair,
        "promote": promote,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
