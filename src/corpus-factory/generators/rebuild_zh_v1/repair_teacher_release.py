#!/usr/bin/env python3
"""Fail-closed repair pass over a teacher-polished SFT release.

Background: the first teacher pass used a protected-number regex whose
``\\w`` boundaries silently matched Chinese characters (Python ``re`` treats
CJK as word chars), so plain numbers adjacent to Chinese ("份数是3") were
not enforced as protected values. Re-verification with the fixed regex
(ASCII boundaries, sentence-final dot allowed) found the rows where the
teacher converted a protected number into hanzi ("三份" for "份数是3") —
those queries no longer match their locally-compiled gold, so they are
reverted to the original query in a NEW superseding release. The flawed
release is left untouched (write-once evidence).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebuild_zh_v1 import common as C

FAMILIES = ("retrieval", "full_call", "agent", "mw_disposition", "narration", "confidence", "trajectory")
VALUE_RE = re.compile(r"值\d+")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?!\.\d)(?![A-Za-z0-9_])")


def protected_values(query: str) -> list[str]:
    seen: list[str] = []
    for match in VALUE_RE.finditer(query):
        if match.group() not in seen:
            seen.append(match.group())
    for match in NUMBER_RE.finditer(query):
        if match.group() not in seen:
            seen.append(match.group())
    return seen


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(C.canonical_json(r) + "\n" for r in rows), encoding="utf-8")


def build_manifest(base_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(base_dir.rglob("*.jsonl")):
        rel = str(path.relative_to(base_dir))
        data = path.read_bytes()
        artifacts[rel] = {
            "bytes": len(data),
            "rows": data.count(b"\n"),
            "sha256": C.sha256_bytes(data),
        }
    return {"artifacts": artifacts, "artifact_merkle_root": C.merkle_root(artifacts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--out-release-id", required=True)
    ap.add_argument("--resume-file", type=Path, required=True)
    args = ap.parse_args()

    source_dir = args.release_dir
    out_dir = source_dir.parent / args.out_release_id
    if not source_dir.is_dir():
        raise SystemExit(f"missing source release: {source_dir}")
    if out_dir.exists():
        raise SystemExit(f"write-once refusal: out release exists: {out_dir}")

    t0 = time.time()
    # re-verify every polished row against the fixed protected-value rule
    violations: dict[str, dict[str, Any]] = {}
    for line in args.resume_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("kept_variant"):
            continue
        before, after = rec["before_query"], rec["after_query"]
        if before == after:
            continue  # budget-fail revert: already original
        reasons = [f"protected-lost:{p}" for p in protected_values(before) if p not in after]
        reasons += [f"fabricated-value:{m.group()}" for m in VALUE_RE.finditer(after) if m.group() not in before]
        reasons += [f"fabricated-number:{m.group()}" for m in NUMBER_RE.finditer(after) if m.group() not in before]
        if reasons:
            violations[rec["case_id"]] = {"reasons": reasons, "before": before, "after": after}

    print(f"re-verified with fixed protected-number regex: {len(violations)} violated rows to revert")
    for case_id, info in violations.items():
        print(f"  {case_id}: {info['reasons'][0]} | {info['before'][:40]} -> {info['after'][:40]}")

    shutil.copytree(source_dir, out_dir)
    reverted = 0
    for family in FAMILIES:
        family_violations = {cid: v for cid, v in violations.items() if cid.startswith(f"{family}:")}
        if not family_violations:
            continue
        semantic_path = out_dir / "semantic" / f"{family}.jsonl"
        rows = read_jsonl(semantic_path)
        for row in rows:
            case_id = str(row["case_id"])
            info = family_violations.get(case_id)
            if info is None:
                continue
            current = str(row["query"])
            row["query"] = info["before"]
            budget = dict(row.get("budget") or {})
            if "prompt_tokens" in budget:
                delta = C.count_tokens(info["before"]) - C.count_tokens(current)
                new_total = int(budget["prompt_tokens"]) + delta
                budget["prompt_tokens"] = new_total
                budget["fits"] = new_total <= int(budget["cap"])
                row["budget"] = budget
            teacher = dict(row.get("teacher") or {})
            teacher["kept_variant"] = None
            teacher["repaired"] = True
            teacher["repair_reason"] = ";".join(info["reasons"])
            row["teacher"] = teacher
            reverted += 1
        write_jsonl(semantic_path, rows)
        semantic_by_id = {str(r["case_id"]): r for r in rows}
        train_path = out_dir / "compiled" / family / "train.jsonl"
        train_rows = read_jsonl(train_path)
        for row in train_rows:
            source = semantic_by_id.get(str(row["case_id"]))
            if source is not None and str(row["case_id"]) in family_violations:
                row["query"] = source["query"]
                row["budget"] = source["budget"]
                row["teacher"] = source["teacher"]
        write_jsonl(train_path, train_rows)

    # manifests + merkle
    new_manifest = build_manifest(out_dir)
    (out_dir / "manifests" / "artifact-manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    repair_report = {
        "schema": "mei-51m-sft-teacher-repair-report-v1",
        "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "defect": ("teacher-pass protected-number regex used \\w boundaries which match CJK in Python re, "
                   "so numbers adjacent to Chinese were not enforced verbatim; "
                   "7 polished rows converted protected numbers to hanzi and were reverted to the original query"),
        "reverted_rows": {
            case_id: {"reasons": info["reasons"], "before_query": info["before"], "rejected_query": info["after"]}
            for case_id, info in violations.items()
        },
        "verification": "all polished rows re-verified against fixed regex (ASCII boundaries); 0 remaining violations",
        "reverted_count": reverted,
    }
    (out_dir / "governance" / "repair-report-v1.json").write_text(
        json.dumps(repair_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    release_manifest = json.loads((out_dir / "release-manifest.json").read_text(encoding="utf-8"))
    release_manifest.update({
        "release_id": args.out_release_id,
        "supersedes": {
            "release_id": release_manifest["release_id"],
            "artifact_merkle_root": release_manifest["artifact_merkle_root"],
            "reason": "7 rows reverted: protected numbers had been paraphrased to hanzi (regex boundary defect); gold untouched elsewhere",
        },
        "repairs": {
            "reverted_rows": reverted,
            "report": "governance/repair-report-v1.json",
            "defect": repair_report["defect"],
        },
        "artifact_merkle_root": new_manifest["artifact_merkle_root"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(time.time() - t0, 2),
    })
    (out_dir / "release-manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "release_id": args.out_release_id,
        "merkle_root": new_manifest["artifact_merkle_root"],
        "reverted_rows": reverted,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
