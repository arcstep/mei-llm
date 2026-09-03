#!/usr/bin/env python3
"""Compare independent retrieval/MW/confidence heads across MLX and Rust."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from common._repo import ROOT


sys.path.insert(0, str(ROOT / "platform/python-sdk"))

from mei_sdk.package import load_package  # noqa: E402
from mei_sdk.runtime_51m import load_51m_runtime  # noqa: E402


def _confidence_bucket(logit: float) -> str:
    probability = 1.0 / (1.0 + math.exp(-float(logit)))
    if probability >= 0.70:
        return "execute"
    if probability >= 0.35:
        return "escalate"
    return "refuse"


def compare(package_dir: Path, rust_cli: Path, text: str) -> dict[str, Any]:
    import mlx.core as mx

    package = load_package(package_dir)
    runtime, _load = load_51m_runtime(package, backend="mlx-reference")
    if (
        runtime.contrastive is None
        or runtime.mw_disposition is None
        or runtime.conf_v2 is None
        or runtime.narration_adapter is None
    ):
        raise RuntimeError("Python runtime did not load all independent sidecars")
    ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=False)
    out = runtime._forward(ids, return_confidence=True, return_cells=True)
    cells = out.get("cells")
    if cells is None:
        raise RuntimeError("MLX runtime did not return cell states")
    embedding = runtime.contrastive(cells)[0].astype(mx.float32)
    stacked = mx.stack(list(cells), axis=2)
    pooled = mx.mean(stacked.astype(mx.float32)[:, -1, :, :], axis=1)
    mw_logits = (
        pooled @ runtime.mw_disposition.weight.astype(mx.float32).T
        + runtime.mw_disposition.bias.astype(mx.float32)
    )
    mw_probabilities = mx.softmax(mw_logits, axis=-1)[0]
    confidence_logit = runtime.conf_v2(cells)[0].astype(mx.float32)
    narration_residual = runtime.narration_adapter(
        out["hidden"][:, -1, :]
    )[0].astype(mx.float32)
    mx.eval(embedding, mw_probabilities, confidence_logit, narration_residual)
    python_embedding = [float(value) for value in embedding.tolist()]
    python_mw = [float(value) for value in mw_probabilities.tolist()]
    python_confidence = float(confidence_logit.item())
    python_narration = [float(value) for value in narration_residual.tolist()]
    index_row = next(
        row for row in package.manifest["files"] if row.get("role") == "tool_index"
    )
    catalog = [
        row["schema"]
        for row in json.loads((package_dir / index_row["path"]).read_text(encoding="utf-8"))[
            "records"
        ]
    ]
    python_top5 = [row["name"] for row in runtime.search_top_k(text, catalog, k=5)]

    process = subprocess.run(
        [str(rust_cli), "diagnose-heads", str(package_dir), text],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Rust head diagnostic failed: {process.stderr[-4000:]}")
    rust = json.loads(process.stdout)
    rust_embedding = [float(value) for value in rust["contrastive_embedding"]]
    rust_mw = [float(value) for value in rust["mw_probabilities"]]
    rust_narration = [float(value) for value in rust["narration_residual"]]
    if (
        len(rust_embedding) != len(python_embedding)
        or len(rust_mw) != len(python_mw)
        or len(rust_narration) != len(python_narration)
    ):
        raise RuntimeError("portable head output dimensions disagree")
    dot = sum(left * right for left, right in zip(python_embedding, rust_embedding))
    left_norm = math.sqrt(sum(value * value for value in python_embedding))
    right_norm = math.sqrt(sum(value * value for value in rust_embedding))
    cosine = dot / max(left_norm * right_norm, 1e-12)
    mw_max_abs = max(abs(left - right) for left, right in zip(python_mw, rust_mw))
    confidence_abs = abs(python_confidence - float(rust["confidence_logit"]))
    narration_dot = sum(
        left * right for left, right in zip(python_narration, rust_narration)
    )
    narration_left_norm = math.sqrt(sum(value * value for value in python_narration))
    narration_right_norm = math.sqrt(sum(value * value for value in rust_narration))
    narration_cosine = narration_dot / max(
        narration_left_norm * narration_right_norm, 1e-12
    )
    narration_max_abs = max(
        abs(left - right)
        for left, right in zip(python_narration, rust_narration)
    )
    python_narration_argmax = max(
        range(len(python_narration)),
        key=lambda index: (python_narration[index], -index),
    )
    rust_narration_argmax = max(
        range(len(rust_narration)),
        key=lambda index: (rust_narration[index], -index),
    )
    python_reason = max(range(20), key=lambda index: (python_mw[index], -index))
    checks = {
        "token_ids_exact": rust["token_ids"] == ids,
        "contrastive_cosine": cosine >= 0.98,
        "retrieval_top5_exact": rust["retrieval_top5"] == python_top5,
        "mw_reason_exact": int(rust["mw_reason_code"]) == python_reason,
        "mw_probability_max_abs": mw_max_abs <= 0.15,
        "confidence_bucket_exact": _confidence_bucket(rust["confidence_logit"])
        == _confidence_bucket(python_confidence),
        "confidence_logit_abs": confidence_abs <= 1.0,
        "narration_residual_cosine": narration_cosine >= 0.98,
        "narration_residual_max_abs": narration_max_abs <= 0.15,
        "narration_residual_argmax_exact": (
            rust_narration_argmax == python_narration_argmax
        ),
    }
    report = {
        "schema": "mei-portable-head-parity-v1",
        "text": text,
        "semantic_boundaries": {
            "retrieval": "independent-contrastive-head",
            "mw_disposition": "independent-20class-sidecar",
            "confidence": "independent-calibrated-sidecar",
            "narration": "independent-frozen-backbone-rank16-generation-sidecar",
            "mw_deviation": "deterministic-governance-only-no-tensor",
        },
        "checks": checks,
        "all_passed": all(checks.values()),
        "metrics": {
            "contrastive_cosine": cosine,
            "mw_probability_max_abs": mw_max_abs,
            "confidence_logit_abs": confidence_abs,
            "narration_residual_cosine": narration_cosine,
            "narration_residual_max_abs": narration_max_abs,
        },
        "python": {
            "token_ids": ids,
            "retrieval_top5": python_top5,
            "mw_reason_code": python_reason,
            "confidence_logit": python_confidence,
            "confidence_bucket": _confidence_bucket(python_confidence),
            "narration_residual_argmax": python_narration_argmax,
        },
        "rust": rust,
    }
    report["rust"]["narration_residual_argmax"] = rust_narration_argmax
    report["rust"].pop("narration_residual", None)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--rust-cli", type=Path, required=True)
    parser.add_argument("--text", default="打开厨房灯并确认状态")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.package, args.rust_cli, args.text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
