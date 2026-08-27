#!/usr/bin/env python3
"""Run 1M (with mid-run resume) then promote to 5M. Does not start 100M."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    CORPUS_ZH_PRETRAIN,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

PY = sys.executable
CKPT_DIR = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints"


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or ROOT, check=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def valid_not_twice_worse(rows: list[dict], *, frac: float = 0.20) -> bool:
    valids = [float(r["valid_loss"]) for r in rows if r.get("valid_loss") is not None]
    consec = 0
    for prev, cur in zip(valids, valids[1:]):
        if prev > 0 and cur > prev * (1.0 + frac):
            consec += 1
            if consec >= 2:
                return False
        else:
            consec = 0
    return True


def preflight() -> int:
    checks = [
        [PY, str(SCRIPTS_ROOT / "report_zh_vocab_coverage.py"), "--v1"],
        [PY, str(SCRIPTS_ROOT / "check_train_eval_isolation.py"), "--all"],
        [PY, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model" / "check_student.py")],
        [PY, str(SCRIPTS_ROOT / "train_needle_zh_pretrain.py"), "--count-params"],
    ]
    for cmd in checks:
        proc = run(cmd)
        if proc.returncode != 0:
            print(f"preflight failed: {cmd[-1]} exit {proc.returncode}", file=sys.stderr)
            return proc.returncode
    return 0


def ensure_wiki_shards() -> int:
    man = CORPUS_ZH_PRETRAIN / "manifest.json"
    shards = CORPUS_ZH_PRETRAIN / "shards"
    has_wiki = any(shards.glob("train-*.jsonl")) if shards.is_dir() else False
    if has_wiki and man.is_file():
        print(json.dumps({"wiki_ready": True, **{k: load_json(man).get(k) for k in ("n_tokens", "n_docs", "gap_to_100m")}}))
        return 0
    proc = run([PY, str(SCRIPTS_ROOT / "build_zh_pretrain_v0.py"), "--rung", "100m"])
    # exit 2 means under 100M but shards/manifest were still written
    if proc.returncode not in (0, 2):
        return proc.returncode
    return 0


def train(rung: str, extra: list[str]) -> int:
    cmd = [PY, str(SCRIPTS_ROOT / "train_needle_zh_pretrain.py"), "--rung", rung, *extra]
    return run(cmd).returncode


def promote_1m() -> dict:
    out_dir = EXPERIMENTS_RUNS / "needle-zh-pretrain-pilot-1m"
    summary = load_json(out_dir / "summary.json")
    metrics = load_jsonl(out_dir / "metrics.jsonl")
    state = CKPT_DIR / "pretrain-pilot-1m-state.npz"
    tokens = int(summary.get("tokens_seen") or 0)
    valid = summary.get("valid_loss")
    gates = {
        "tokens_ok": tokens >= 1_000_000,
        "finite_valid": valid is not None and valid == valid and abs(float(valid)) != float("inf"),
        "valid_not_twice_worse": valid_not_twice_worse(metrics),
        "ckpt_ok": state.is_file() and state.with_suffix(".meta.json").is_file(),
        "hashes_ok": bool(summary.get("tokenizer_sha256") and summary.get("corpus_sha256")),
        "no_nan_train": all(
            (r.get("loss") is None) or (float(r["loss"]) == float(r["loss"])) for r in metrics
        ),
    }
    gates["promote"] = all(gates.values())
    gates["tokens_seen"] = tokens
    gates["valid_loss"] = valid
    (out_dir / "promote-1m.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))
    return gates


def main() -> int:
    rc = preflight()
    if rc != 0:
        return rc
    rc = ensure_wiki_shards()
    if rc != 0:
        return rc
    man = load_json(CORPUS_ZH_PRETRAIN / "manifest.json")
    n_train = int(man.get("n_train_tokens") or 0)
    allow = ["--allow-repeat"] if n_train < 5_000_000 else []

    state_1m = CKPT_DIR / "pretrain-pilot-1m-state.npz"
    rc = train(
        "pilot-1m",
        ["--stop-at-tokens", "250000", "--lr-horizon-tokens", "5000000"],
    )
    if rc != 0:
        return rc
    rc = train(
        "pilot-1m",
        ["--resume", str(state_1m), "--lr-horizon-tokens", "5000000"],
    )
    if rc != 0:
        return rc
    gates = promote_1m()
    if not gates.get("promote"):
        print("1M promotion failed; not starting 5M", file=sys.stderr)
        return 3
    rc = train(
        "pilot-5m",
        ["--resume", str(state_1m), "--lr-horizon-tokens", "5000000", *allow],
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
