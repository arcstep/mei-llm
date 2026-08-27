#!/usr/bin/env python3
"""Build template-only 2k candidate packs for retrieval / full-call / MW. Spend stays 0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
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

from jobs.paths import job_work, rel_to_root  # noqa: E402
from sft_canonical_lib import load_jsonl  # noqa: E402
from sft_synth_lib import dump_json, sha256_text  # noqa: E402


LINES = (
    {
        "job": JOB_RETRIEVAL_V2,
        "task": "retrieval",
        "builder": "build_mei_retrieval_v2.py",
        "validator": "validate_mei_retrieval_v2_pack.py",
        "pack": SFT_TRAIN / "packs/mei-retrieval-v2-2k.candidates.jsonl",
        "subdir": "candidates-2k",
    },
    {
        "job": JOB_FULLCALL_V2,
        "task": "fullcall",
        "builder": "build_mei_toolcall_v2.py",
        "validator": "validate_mei_toolcall_v2_pack.py",
        "pack": SFT_TRAIN / "packs/mei-toolcall-v2-oracle-2k.candidates.jsonl",
        "subdir": "candidates-2k",
        "validator_args": lambda pack: ["--pack", str(pack), "--require-compiler"],
    },
    {
        "job": JOB_MW_DISPOSITION_V1,
        "task": "mw",
        "builder": "build_mei_mw_disposition_v2.py",
        "validator": "validate_mei_mw_disposition_v2.py",
        "pack": SFT_TRAIN / "packs/mei-mw-disposition-v2-2k.candidates.jsonl",
        "subdir": "candidates-2k",
    },
)


def run(script: str, argv: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / script), *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload: dict = {"returncode": proc.returncode, "stderr": proc.stderr.strip()[-2000:]}
    if proc.stdout.strip():
        try:
            payload["stdout"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload["stdout_text"] = proc.stdout[-1000:]
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()
    reports = []
    ok = True
    for line in LINES:
        work = job_work(TOPIC_TOOLCALL_SFT, line["job"], root=ROOT) / line["subdir"]
        work.mkdir(parents=True, exist_ok=True)
        for stale in ("queue.sqlite", "queue.sqlite-wal", "queue.sqlite-shm"):
            p = work / stale
            if p.is_file():
                p.unlink()
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
                "2k",
            ],
        )
        accepted = load_jsonl(work / "raw" / "accepted.jsonl")
        if accepted:
            from sft_canonical_lib import dump_jsonl

            dump_jsonl(line["pack"], accepted)
        vargs = line.get("validator_args")
        validator_args = vargs(line["pack"]) if callable(vargs) else ["--pack", str(line["pack"])]
        validator = run(line["validator"], validator_args)
        n_unique = len({sha256_text(str(r.get("query") or "")) for r in accepted})
        item = {
            "job": line["job"],
            "task": line["task"],
            "builder": builder,
            "teacher": teacher,
            "validator": validator,
            "n": len(accepted),
            "n_accepted_unique": n_unique,
            "pack": rel_to_root(line["pack"], root=ROOT),
            "ok": (
                builder.get("returncode") == 0
                and teacher.get("returncode") == 0
                and validator.get("returncode") == 0
                and n_unique >= min(args.limit, 2000) * 0.9
            ),
        }
        reports.append(item)
        ok = ok and item["ok"]
    isolation = run("check_train_eval_isolation.py", ["--scope", "sft-v2"])
    promote = run("promote_sft_v2_releases.py", ["--state", "candidate"])
    out = {
        "ok": ok and isolation.get("returncode") == 0 and promote.get("returncode") == 0,
        "limit": args.limit,
        "spend_cny": 0,
        "lines": reports,
        "isolation": isolation,
        "promote": promote,
        "note": "Template 2k candidates only. Paid 2k waits for canary gates. accepted semantic pack waits for 0303.",
    }
    dest = ROOT / "notebook/jobs/toolcall-sft/outbox/draft/2k-template-report.json"
    dump_json(dest, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
