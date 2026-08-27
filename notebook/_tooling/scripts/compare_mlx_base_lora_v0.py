#!/usr/bin/env python3
"""Compare base vs adapter MLX predictions (heuristic trend only)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import ROOT

RUNNER = ROOT / "notebook/_tooling/scripts/run_eval_mlx_v0.py"


def run_one(tag: str, adapter: str | None, face: str, limit: int) -> Path:
    out = (
        ROOT
        / "notebook/archive/runs"
        / f"{tag}-{'lora' if adapter else 'base'}-{face.replace('.', '')}-n{limit}"
    )
    cmd = [
        sys.executable,
        str(RUNNER),
        "--face",
        face,
        "--limit",
        str(limit),
        "--judge",
        "heuristic",
        "--max-tokens",
        "256",
        "--out-dir",
        str(out),
    ]
    if adapter:
        cmd.extend(["--adapter", adapter])
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    return out


def load_preds(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        by[o["item_id"]] = o
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    from repo_paths import MLX_ADAPTER_MEI_EXPERT

    ap.add_argument("--adapter", default=str(MLX_ADAPTER_MEI_EXPERT))
    ap.add_argument("--faces", default="face.edge,face.dev")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    faces = [f.strip() for f in args.faces.split(",") if f.strip()]
    rows: list[dict] = []

    for face in faces:
        base_dir = run_one(tag, None, face, args.limit)
        lora_dir = run_one(tag, args.adapter, face, args.limit)
        base = load_preds(base_dir)
        lora = load_preds(lora_dir)
        for iid in sorted(set(base) & set(lora)):
            b, a = base[iid], lora[iid]
            bj = (b.get("judgment") or {}).get("pass_guess")
            aj = (a.get("judgment") or {}).get("pass_guess")
            rows.append(
                {
                    "item_id": iid,
                    "face": face,
                    "polarity": b.get("polarity"),
                    "topic": b.get("topic"),
                    "base_pass_guess": bj,
                    "lora_pass_guess": aj,
                    "delta": None
                    if bj is None or aj is None
                    else (1 if aj and not bj else (-1 if bj and not aj else 0)),
                    "base_preview": (b.get("answer") or "")[:160],
                    "lora_preview": (a.get("answer") or "")[:160],
                }
            )

    improved = sum(1 for r in rows if r["delta"] == 1)
    regressed = sum(1 for r in rows if r["delta"] == -1)
    same = sum(1 for r in rows if r["delta"] == 0)
    summary = {
        "created_utc": tag,
        "adapter": args.adapter,
        "n": len(rows),
        "improved": improved,
        "regressed": regressed,
        "same": same,
        "note": "heuristic only; not acceptance KPI",
    }
    out_dir = ROOT / "notebook/archive/runs" / f"{tag}-compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
