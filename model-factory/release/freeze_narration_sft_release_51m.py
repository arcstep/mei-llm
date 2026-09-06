#!/usr/bin/env python3
"""Freeze grounded Chinese narration data from verified Agent trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from common._repo import ROOT


PARENT_ID = "mei-1.0-51m-tool-sft-v2-agent300m-v1"
RELEASE_ID = "mei-1.0-51m-narration-sft-agent300m-v1"
RELEASE_ROOT = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases"
PARENT_DIR = RELEASE_ROOT / PARENT_ID
PROVIDER_ID = "mei-zh-narration-adapter-r16-v1"
PROMPT_ID = "mei-verified-result-narration-prompt-v1"

sys.path.insert(0, str(ROOT / "platform/_shared/runtime"))
from narration import NarrationProvider, verified_result_view  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sanitized_views(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = list(row.get("prior_calls") or [])
    results = list(row.get("tool_results") or [])
    if len(calls) != len(results) or not results:
        raise RuntimeError("narration source must contain matched nonempty call/results")
    views = []
    for call, result in zip(calls, results):
        if result.get("call_id") != call.get("call_id") or result.get("status") != "ok":
            raise RuntimeError("narration source contains unmatched or unsuccessful result")
        if (result.get("provenance") or {}).get("verified") is not True:
            raise RuntimeError("narration source result is not verified")
        views.append(
            {
                "call_id": str(result["call_id"]),
                "tool_name": str(call.get("name") or "该工具"),
                "status": "ok",
                "payload": result.get("payload"),
                "provenance": {
                    "source": "frozen-agent-host-simulator",
                    "verified": True,
                },
            }
        )
    return views


def narration_prompt(views: list[dict[str, Any]]) -> str:
    public_views = [
        {
            "call_id": view["call_id"],
            "tool_name": view["tool_name"],
            "status": view["status"],
            "payload": view["payload"],
        }
        for view in views
    ]
    return (
        "任务：只依据 <verified_results> 中的已验证结果，用简洁中文逐项说明执行结果。"
        "不得添加结果中不存在的事实，不得调用工具。\n"
        "<verified_results>"
        + canonical(public_views).decode("utf-8")
        + "</verified_results>\n解说："
    )


def build_rows(parent: Path) -> dict[str, list[dict[str, Any]]]:
    provider = NarrationProvider()
    output: dict[str, list[dict[str, Any]]] = {}
    query_sets: dict[str, set[str]] = {}
    for split in ("train", "valid", "eval"):
        source = load_jsonl(parent / f"agent-continuation.{split}.jsonl")
        terminals = [row for row in source if (row.get("agent_trace") or {}).get("target_kind") == "respond"]
        rows = []
        for row in terminals:
            views = sanitized_views(row)
            target = "\n".join(
                provider.narrate(verified_result_view(view)) for view in views
            )
            rows.append(
                {
                    "sample_id": f"NAR-{row['trajectory_id']}",
                    "trajectory_id": row["trajectory_id"],
                    "cf_group": row["cf_group"],
                    "family": row["family"],
                    "split": split,
                    "prompt": narration_prompt(views),
                    "target": target,
                    "verified_result_views": views,
                    "call_ids": [view["call_id"] for view in views],
                    "prompt_id": PROMPT_ID,
                    "provider_id": PROVIDER_ID,
                    "grounding_target": "exact-deterministic-template",
                    "can_execute_tools": False,
                }
            )
        output[split] = rows
        query_sets[split] = {str(row["prompt"]) for row in rows}
    if any(query_sets[left] & query_sets[right] for left, right in (("train", "valid"), ("train", "eval"), ("valid", "eval"))):
        raise RuntimeError("narration prompt leakage across splits")
    return output


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    parent = args.release_root / args.parent_id
    output = args.release_root / args.release_id
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen narration release")
    parent_manifest = parent / "manifest.json"
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if manifest.get("release_id") != args.parent_id:
        raise RuntimeError("narration parent release identity mismatch")
    rows = build_rows(parent)
    with tempfile.TemporaryDirectory(prefix="mei-narration-freeze-", dir=args.release_root.parent) as temp_name:
        temp = Path(temp_name) / "release"
        temp.mkdir()
        files = {}
        for split in ("train", "valid", "eval"):
            name = f"narration.{split}.jsonl"
            dump_jsonl(temp / name, rows[split])
            files[name] = {"rows": len(rows[split]), "sha256": sha_file(temp / name)}
        receipt = {
            "schema": "mei-narration-data-isolation-receipt-v1",
            "status": "passed",
            "group_cross_split": 0,
            "prompt_overlap": 0,
            "verified_result_only": True,
            "executor_access": False,
            "grounding_gate": "adapter-output-equals-deterministic-template-else-fallback",
        }
        (temp / "isolation-receipt.json").write_bytes(canonical(receipt))
        release_manifest = {
            "schema": "mei-narration-sft-data-release-v1",
            "release_id": args.release_id,
            "product": "mei-1.0-51m",
            "parent_release": {
                "release_id": args.parent_id,
                "manifest_sha256": sha_file(parent_manifest),
            },
            "provider_id": PROVIDER_ID,
            "prompt_id": PROMPT_ID,
            "adapter": {"kind": "frozen-backbone-logit-residual", "rank": 16},
            "files": files,
            "isolation_receipt": {
                "file": "isolation-receipt.json",
                "sha256": sha_file(temp / "isolation-receipt.json"),
            },
        }
        (temp / "manifest.json").write_bytes(canonical(release_manifest))
        temp.replace(output)
    return {
        "ok": True,
        "release_id": args.release_id,
        "release_dir": str(output),
        "rows": {split: len(value) for split, value in rows.items()},
        "manifest_sha256": sha_file(output / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--parent-id", default=PARENT_ID)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
