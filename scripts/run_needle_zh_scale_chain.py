#!/usr/bin/env python3
"""Random-init unique-token scale chain: 100M review, then 300M, then full train tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

PY = sys.executable
CKPT_DIR = TASKS_ROOT / TASK_NEEDLE_ZH / "checkpoints"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


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


def probe_families_ok(rep: dict | None) -> bool:
    if not rep:
        return True
    fam = rep.get("family_nll") or {}
    return all(v == v and abs(float(v)) != float("inf") for v in fam.values())


def throughput_flags() -> list[str]:
    summary = load_json(EXPERIMENTS_RUNS / "needle-zh-pretrain-throughput" / "summary.json")
    chosen = summary.get("chosen") or {}
    flags: list[str] = []
    if chosen.get("batch_size"):
        flags += ["--batch-size", str(chosen["batch_size"])]
    if chosen.get("grad_accum"):
        flags += ["--grad-accum", str(chosen["grad_accum"])]
    if chosen.get("compile"):
        flags.append("--compile")
    prec = chosen.get("precision") or "fp32"
    if prec != "fp32":
        flags += ["--precision", prec]
    return flags


def preflight() -> int:
    man = load_json(CORPUS_ZH_PRETRAIN / "manifest.json")
    if man.get("unk_gate_ok") is False:
        print("UNK gate failed; refuse official scale", file=sys.stderr)
        return 3
    checks = [
        [PY, str(ROOT / "scripts" / "report_zh_vocab_coverage.py"), "--v1"],
        [PY, str(ROOT / "scripts" / "check_train_eval_isolation.py"), "--all"],
        [PY, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model" / "check_student.py")],
        [PY, str(ROOT / "scripts" / "train_needle_zh_pretrain.py"), "--count-params"],
    ]
    for cmd in checks:
        proc = run(cmd)
        if proc.returncode != 0:
            print(f"preflight failed: {cmd[-1]} exit {proc.returncode}", file=sys.stderr)
            return proc.returncode
    return 0


def train(rung: str, extra: list[str]) -> int:
    cmd = [PY, str(ROOT / "scripts" / "train_needle_zh_pretrain.py"), "--rung", rung, *extra]
    return run(cmd).returncode


def review_gate(
    rung: str,
    min_tokens: int,
    prev_valid: float | None = None,
    *,
    coverage: bool = False,
) -> dict:
    out_dir = EXPERIMENTS_RUNS / f"needle-zh-pretrain-{rung}"
    summary = load_json(out_dir / "summary.json")
    metrics = load_jsonl(out_dir / "metrics.jsonl")
    probes = load_json(out_dir / "probes.json")
    state = CKPT_DIR / f"pretrain-{rung}-state.npz"
    tokens = int(summary.get("tokens_seen") or 0)
    valid = summary.get("valid_loss")
    gates = {
        "tokens_ok": tokens >= min_tokens,
        "finite_valid": valid is not None and valid == valid and abs(float(valid)) != float("inf"),
        "valid_not_twice_worse": valid_not_twice_worse(metrics),
        "ckpt_ok": state.is_file() and state.with_suffix(".meta.json").is_file(),
        "hashes_ok": bool(summary.get("tokenizer_sha256") and summary.get("corpus_sha256")),
        "no_nan_train": all((r.get("loss") is None) or (float(r["loss"]) == float(r["loss"])) for r in metrics),
        "probe_families_ok": probe_families_ok(probes),
    }
    if prev_valid is not None and valid is not None:
        gates["learning_signal"] = float(valid) < float(prev_valid) * 1.05
    if coverage:
        n_windows = int(summary.get("n_windows") or 0)
        gates["exhausted"] = bool(summary.get("exhausted"))
        gates["unique_epoch"] = (not summary.get("allow_repeat")) and int(summary.get("epoch") or 0) == 0
        if n_windows > 0:
            gates["cursor_at_end"] = int(summary.get("window_index") or 0) >= n_windows
    gates["promote"] = all(bool(v) for v in gates.values())
    gates["tokens_seen"] = tokens
    gates["valid_loss"] = valid
    gates["probe_mean_nll"] = probes.get("mean_nll") if probes else summary.get("probe_mean_nll")
    dest = out_dir / f"promote-{rung}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))
    return gates


def disk_ok(path: Path, need_gb: float = 8.0) -> bool:
    import shutil

    free = shutil.disk_usage(path).free
    if free < need_gb * (1 << 30):
        print(f"need >= {need_gb} GiB free under {path}, have {free / (1 << 30):.2f}", file=sys.stderr)
        return False
    return True


def eval_valid(rung: str, mode: str) -> int:
    ckpt = CKPT_DIR / f"pretrain-{rung}.npz"
    out = EXPERIMENTS_RUNS / f"needle-zh-pretrain-{rung}" / f"valid-{mode}.json"
    cmd = [
        PY,
        str(ROOT / "scripts" / "eval_needle_zh_pretrain_valid.py"),
        "--ckpt",
        str(ckpt),
        "--mode",
        mode,
        "--out",
        str(out),
    ]
    return run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--stop-after", choices=["100m", "300m", "full"], default="100m")
    ap.add_argument("--from-rung", choices=["100m", "300m", "1b"], default="100m")
    args = ap.parse_args()
    if not args.skip_preflight:
        rc = preflight()
        if rc != 0:
            return rc
    man = load_json(CORPUS_ZH_PRETRAIN / "manifest.json")
    n_train = int(man.get("n_train_tokens") or 0)
    if n_train <= 0:
        print("manifest missing n_train_tokens", file=sys.stderr)
        return 2
    n_predictable = max(0, n_train - 1)
    flags = throughput_flags()
    horizon = ["--lr-horizon-tokens", str(n_train)]
    state_100 = CKPT_DIR / "pretrain-100m-state.npz"
    g100: dict = {}
    g300: dict = {}

    if args.from_rung == "100m":
        rc = train("100m", ["--stop-at-tokens", "100000000", *horizon, *flags])
        if rc != 0:
            return rc
        g100 = review_gate("100m", 100_000_000)
        if not g100.get("promote"):
            print("100M gate failed; stop for diagnosis", file=sys.stderr)
            return 3
        if args.stop_after == "100m":
            return 0
    else:
        g100 = load_json(EXPERIMENTS_RUNS / "needle-zh-pretrain-100m" / "promote-100m.json")

    if args.from_rung in {"100m", "300m"}:
        rc = train("300m", ["--resume", str(state_100), "--stop-at-tokens", "300000000", *horizon, *flags])
        if rc != 0:
            return rc
        g300 = review_gate("300m", 300_000_000, prev_valid=g100.get("valid_loss"))
        if not g300.get("promote"):
            print("300M gate failed; stop for diagnosis", file=sys.stderr)
            return 3
        eval_valid("300m", "stratified")
        if args.stop_after == "300m":
            return 0
    else:
        g300 = load_json(EXPERIMENTS_RUNS / "needle-zh-pretrain-300m" / "promote-300m.json")

    if not disk_ok(CKPT_DIR):
        return 5
    rc = train(
        "1b",
        [
            "--resume",
            str(CKPT_DIR / "pretrain-300m-state.npz"),
            "--stop-at-tokens",
            str(n_predictable),
            *horizon,
            *flags,
        ],
    )
    if rc != 0:
        return rc
    gfull = review_gate("1b", n_predictable, prev_valid=g300.get("valid_loss"), coverage=True)
    gfull["unique_train_tokens"] = n_train
    gfull["prediction_targets"] = n_predictable
    gfull["reached_1b"] = n_train >= 1_000_000_000
    gfull["gap_to_1b"] = max(0, 1_000_000_000 - n_train)
    full_dir = EXPERIMENTS_RUNS / "needle-zh-pretrain-1b"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "promote-full.json").write_text(json.dumps(gfull, indent=2) + "\n", encoding="utf-8")
    if gfull.get("promote"):
        eval_valid("1b", "full")
    return 0 if gfull.get("promote") else 4


if __name__ == "__main__":
    raise SystemExit(main())
