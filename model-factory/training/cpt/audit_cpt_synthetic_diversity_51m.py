#!/usr/bin/env python3
"""Hash-bound CPT synthetic-role diversity audit.

The audit deliberately uses training evidence as well as raw-corpus identity.
An exact-unique corpus can still be a collapsed template renderer: changing an
ID in every row makes hashes unique while the model predicts the row almost
perfectly.  Source-specific loss therefore acts as a fail-closed collapse
signal for synthetic structure and colloquial roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


ROLES = ("wiki", "hq", "structure", "colloquial")
SYNTHETIC_ROLES = ("structure", "colloquial")
LOW_LOSS_THRESHOLD = 0.1
LOW_LOSS_RATE_LIMIT = 0.5
MIN_ROLE_WINDOWS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"metrics line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("metrics input is empty")
    return rows


def losses_by_source(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped = {role: [] for role in ROLES}
    previous: dict[str, Any] | None = None
    for row in rows:
        counters = row.get("source_tokens_drawn")
        if not isinstance(counters, dict):
            continue
        if previous is not None:
            changed = [
                role
                for role in ROLES
                if int(counters.get(role) or 0) > int(previous.get(role) or 0)
            ]
            loss = row.get("loss")
            if (
                len(changed) == 1
                and isinstance(loss, (int, float))
                and math.isfinite(float(loss))
            ):
                grouped[changed[0]].append(float(loss))
        previous = counters
    return grouped


def raw_identity(path: Path) -> dict[str, Any]:
    rows = 0
    text_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}: line {line_number} has no text")
            rows += 1
            text_hashes.add(hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
    if rows == 0:
        raise ValueError(f"{path}: no corpus rows")
    return {
        "rows": rows,
        "exact_unique_texts": len(text_hashes),
        "exact_unique_ratio": len(text_hashes) / rows,
    }


def loss_evidence(losses: list[float]) -> dict[str, Any]:
    ordered = sorted(losses)
    if not ordered:
        return {
            "windows": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "below_threshold": 0,
            "below_threshold_rate": None,
        }

    def quantile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

    below = sum(value < LOW_LOSS_THRESHOLD for value in ordered)
    return {
        "windows": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p05": quantile(0.05),
        "p95": quantile(0.95),
        "below_threshold": below,
        "below_threshold_rate": below / len(ordered),
    }


def parse_role_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or role not in SYNTHETIC_ROLES or not raw_path:
            raise ValueError("--role-input must be structure=PATH or colloquial=PATH")
        if role in result:
            raise ValueError(f"duplicate role input: {role}")
        result[role] = Path(raw_path).resolve()
    if set(result) != set(SYNTHETIC_ROLES):
        raise ValueError("both structure and colloquial role inputs are required")
    return result


def build_receipt(metrics_path: Path, role_inputs: dict[str, Path]) -> dict[str, Any]:
    metrics_path = metrics_path.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    for path in role_inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    grouped = losses_by_source(load_metrics(metrics_path))
    roles: dict[str, Any] = {}
    for role in SYNTHETIC_ROLES:
        loss = loss_evidence(grouped[role])
        enough = int(loss["windows"]) >= MIN_ROLE_WINDOWS
        rate = loss["below_threshold_rate"]
        collapsed = enough and isinstance(rate, float) and rate >= LOW_LOSS_RATE_LIMIT
        status = "degraded" if collapsed or not enough else "passed"
        roles[role] = {
            "status": status,
            "input": {
                "path": str(role_inputs[role]),
                "sha256": sha256_file(role_inputs[role]),
            },
            "raw_identity": raw_identity(role_inputs[role]),
            "training_loss": loss,
            "reasons": (
                ["source_loss_template_collapse"]
                if collapsed
                else (["insufficient_source_windows"] if not enough else [])
            ),
        }
    degraded = any(row["status"] != "passed" for row in roles.values())
    return {
        "schema": "mei-synthetic-corpus-diversity-receipt-v1",
        "terminal_status": "degraded" if degraded else "passed",
        "corpus_diversity_degraded": degraded,
        "policy": {
            "low_loss_threshold": LOW_LOSS_THRESHOLD,
            "low_loss_rate_limit": LOW_LOSS_RATE_LIMIT,
            "minimum_role_windows": MIN_ROLE_WINDOWS,
            "exact_uniqueness_is_not_sufficient": True,
        },
        "training_evidence": {
            "path": str(metrics_path),
            "sha256": sha256_file(metrics_path),
            "rows": len(load_metrics(metrics_path)),
        },
        "roles": roles,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return
        raise FileExistsError(f"write-once audit receipt already exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--role-input", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.metrics, parse_role_inputs(args.role_input))
    atomic_write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
