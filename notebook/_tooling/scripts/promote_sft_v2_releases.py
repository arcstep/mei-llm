#!/usr/bin/env python3
"""Promote retrieval → oracle-top5 full-call → MW releases. Eval banks never become train packs.

Confidence sampling and QAT token packing stay post-gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    PACK_MEI_MW_DISPOSITION_V2_2K,
    PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
    PACK_MEI_MW_DISPOSITION_V2_SMOKE,
    PACK_MEI_RETRIEVAL_V2_2K,
    PACK_MEI_RETRIEVAL_V2_2K_PAID,
    PACK_MEI_RETRIEVAL_V2_10K_PAID,
    PACK_MEI_RETRIEVAL_V2_SMOKE,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_SMOKE,
    ROOT,
    SCRIPTS_ROOT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from sft_canonical_lib import has_route_id_gold, load_jsonl  # noqa: E402

SMOKE_ORDER = (
    ("retrieval", PACK_MEI_RETRIEVAL_V2_SMOKE),
    ("fullcall-oracle-top5", PACK_MEI_TOOLCALL_V2_ORACLE_SMOKE),
    ("mw-disposition", PACK_MEI_MW_DISPOSITION_V2_SMOKE),
)
CANDIDATE_ORDER = (
    ("retrieval", PACK_MEI_RETRIEVAL_V2_2K),
    ("fullcall-oracle-top5", PACK_MEI_TOOLCALL_V2_ORACLE_2K),
    ("mw-disposition", PACK_MEI_MW_DISPOSITION_V2_2K),
)
CANDIDATE_PAID_ORDER = (
    ("retrieval", PACK_MEI_RETRIEVAL_V2_2K_PAID),
    ("fullcall-oracle-top5", PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID),
    ("mw-disposition", PACK_MEI_MW_DISPOSITION_V2_2K_PAID),
)
CANDIDATE_PAID_10K_ORDER = (
    ("retrieval", PACK_MEI_RETRIEVAL_V2_10K_PAID),
    ("fullcall-oracle-top5", PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID),
    ("mw-disposition", PACK_MEI_MW_DISPOSITION_V2_10K_PAID),
)
POST_GATES = [
    "learned-top5 E2E data waits for retrieval Recall@5 gate",
    "joint tool+MW data waits for pure-tool baseline",
    "Confidence labels wait for the final QAT model generations",
    "tool-index rebuild waits for quant-aware base",
    "park browser mei-51m manifest stays released=false until QAT + confidence",
]


def refuse_reason(path: Path, state: str) -> str | None:
    rel = path.as_posix()
    if "evaluation/banks" in rel:
        return "eval_bank"
    if not path.is_file():
        return "missing"
    if state == "accepted" and "smoke" in path.name:
        return "smoke_not_accepted"
    rows = load_jsonl(path)
    if not rows:
        return "empty"
    if state == "accepted" and any(row.get("engineering_smoke") for row in rows):
        return "engineering_smoke_not_accepted"
    if any(has_route_id_gold(row) for row in rows):
        return "route_id"
    blob = path.read_text(encoding="utf-8")
    if "<routes>" in blob or "gold_route_id" in blob:
        return "route_id"
    if state == "accepted":
        current = json.loads((ROOT / "CURRENT.json").read_text(encoding="utf-8"))
        if current.get("base") is None or current.get("sft") is None:
            return "0303_base_sft_null"
        if current.get("stage") != "sft-ready":
            return "0303_not_sft_ready"
    return None


def register(pack: Path, state: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "jobs" / "promote_sft_pack.py"),
            "--pack",
            str(pack.relative_to(ROOT)),
            "--task",
            "mei-1.0-51m",
            "--state",
            state,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"stdout": proc.stdout[-500:]}
    return {"returncode": proc.returncode, "stderr": proc.stderr.strip(), **payload}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="smoke", choices=["smoke", "engineering_smoke", "draft", "candidate", "accepted"])
    ap.add_argument("--allow-accepted", action="store_true")
    ap.add_argument("--paid", action="store_true", help="register paid candidate packs instead of template")
    ap.add_argument("--tier", choices=["2k", "10k"], default="2k")
    args = ap.parse_args()
    state = args.state
    if state == "accepted" and not args.allow_accepted:
        print(json.dumps({"ok": False, "error": "refusing accepted without --allow-accepted"}), indent=2)
        return 2
    if state in {"smoke", "engineering_smoke"}:
        order = SMOKE_ORDER
    elif args.paid and args.tier == "10k":
        order = CANDIDATE_PAID_10K_ORDER
    elif args.paid:
        order = CANDIDATE_PAID_ORDER
    else:
        order = CANDIDATE_ORDER
    results = []
    ok = True
    for kind, pack in order:
        reason = refuse_reason(pack, state)
        if reason:
            results.append({"kind": kind, "pack": str(pack.relative_to(ROOT)), "ok": False, "reason": reason})
            ok = False
            continue
        registered = register(pack, state)
        item = {
            "kind": kind,
            "pack": str(pack.relative_to(ROOT)),
            "state": state,
            "ok": registered.get("returncode") == 0,
            "register": registered,
        }
        results.append(item)
        ok = ok and item["ok"]
    report = {
        "ok": ok,
        "order": [k for k, _ in order],
        "packs": results,
        "post_gates": POST_GATES,
        "note": "Eval banks are never registered. Confidence/QAT remain post-gates. accepted needs 0303 + float base.",
        "tier": args.tier if args.paid or state not in {"smoke", "engineering_smoke"} else "smoke",
        "paid": bool(args.paid),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
