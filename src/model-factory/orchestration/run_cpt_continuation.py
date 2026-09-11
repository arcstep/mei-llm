#!/usr/bin/env python3
"""CPT continuation 编排器（固化版，进 git）。

把「流水线上下游契约」的 P2–P4 不变量固化为可执行 assert。只暴露「该变」的参数
（parent state / target rung / 冻结池 / 语料配额），其余（batch=1、grad_accum=1、
seq=2048、increment 算术、六角色公式配额、schedule 名、cumulative==rung、
parent_state_sha256 pin、唯一 .venv Python）全部 assert 锁死。

流程：读 parent terminal state → 算 increment → plan-mix → build layout →
assert 产物契约 → init + run。--dry-run 止步于 build+assert，不真正训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
VENV_PYTHON = ROOT / ".venv/bin/python"

# 六角色固定公式配额（不含 fineweb2_hq，hq 吸收增量差额）。
FIXED_QUOTA = {
    "code": 20_000_000,
    "dialogue": 40_000_000,
    "structured": 25_000_000,
    "wiki_en": 10_000_000,
    "wiki_zh": 80_000_000,
}
FIXED_SUM = sum(FIXED_QUOTA.values())  # 175_000_000

# P4 训练不变量（effective batch 契约的锚点，任何一项都不许动）。
BATCH_SIZE = 1
GRAD_ACCUM = 1
SEQ_LEN = 2048

# P4 LR 契约。
LR_BASE = 3e-4
LR_FINAL = 3e-5


def _fail(message: str) -> None:
    print(f"ASSERT FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH="src")
    print("$ " + " ".join(str(a) for a in args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env)


def must_succeed(proc: subprocess.CompletedProcess, step: str) -> None:
    if proc.returncode != 0:
        print(f"FAILED at {step} (exit {proc.returncode})", file=sys.stderr)
        sys.exit(proc.returncode)


def mei(*args: str) -> subprocess.CompletedProcess:
    return run([str(VENV_PYTHON), "-m", "mei_llm", *args])


def assert_environment() -> None:
    """锁死唯一 Python 入口 + mlx 可用（否则续训会静默失败）。"""
    _check(VENV_PYTHON.is_file(), f".venv python missing: {VENV_PYTHON}")
    probe = subprocess.run(
        [str(VENV_PYTHON), "-c", "import mlx; print(mlx.__version__)"],
        capture_output=True,
        text=True,
    )
    _check(probe.returncode == 0, f".venv python lacks mlx: {probe.stderr.strip()}")


def read_parent_state(parent_state: Path) -> tuple[int, dict[str, int]]:
    """读 parent terminal state 的 tokens_seen 与 source_token_cursors。"""
    _check(parent_state.is_file(), f"parent state missing: {parent_state}")
    meta_path = parent_state.with_suffix(".meta.json")
    _check(meta_path.is_file(), f"parent meta missing: {meta_path}")
    meta = load_json(meta_path)
    tokens_seen = int(meta["tokens_seen"])
    cursors = {k: int(v) for k, v in meta["source_token_cursors"].items()}
    return tokens_seen, cursors


def assert_increment(*, parent_tokens_seen: int, target: int) -> int:
    """锁死 rung 算术：increment = target − parent 真实落地 exposure。"""
    _check(target > parent_tokens_seen, f"target {target} <= parent {parent_tokens_seen}")
    increment = target - parent_tokens_seen
    _check(increment > 0, "increment must be positive")
    return increment


def assert_quota(*, increment: int, quota: dict[str, int]) -> dict[str, int]:
    """锁死六角色公式配额；fineweb2_hq 吸收差额。"""
    for role, amount in FIXED_QUOTA.items():
        _check(quota.get(role) == amount, f"quota[{role}]={quota.get(role)} != {amount}")
    hq = increment - FIXED_SUM
    _check(quota.get("fineweb2_hq") == hq, f"quota[fineweb2_hq]={quota.get('fineweb2_hq')} != {hq}")
    _check(hq > 0, f"increment {increment} too small to cover fixed {FIXED_SUM}")
    return quota


def plan_mix(*, increment: int, consumed: dict[str, int], quota: dict[str, int],
             pool_release: Path, candidate_id: str, out: Path) -> None:
    capacity = load_json(pool_release)["tokens_by_role"]
    for role, amount in consumed.items():
        _check(amount <= capacity.get(role, 0), f"consumed[{role}]={amount} > capacity")
    args = [
        "corpus", "source", "plan-mix",
        "--target-tokens", str(increment),
    ]
    for role, amount in capacity.items():
        args += ["--capacity", f"{role}={amount}"]
    for role, amount in consumed.items():
        args += ["--consumed", f"{role}={amount}"]
    for role, amount in quota.items():
        args += ["--quota", f"{role}={amount}"]
    args += ["--candidate-id", candidate_id,
             "--reason", f"续训链：增量 {increment}，hq 吸收差额 {increment - FIXED_SUM}",
             "--out", str(out)]
    must_succeed(mei(*args), f"plan-mix {candidate_id}")
    candidate = load_json(out)
    _check(candidate.get("status") == "passed", f"candidate not passed: {candidate.get('shortages')}")
    _check(int(candidate["target_increment_tokens"]) == increment,
           f"candidate target_increment {candidate['target_increment_tokens']} != {increment}")


def build_layout(*, target: int, increment: int, parent_tokens_seen: int,
                 parent_state: Path, pool_release: Path, candidate: Path,
                 cycle_id: str, out: Path) -> None:
    args = [
        str(VENV_PYTHON),
        "src/model-factory/training/cpt/build_v2_corpus_layout.py",
        "--pool-release", str(pool_release),
        "--candidate", str(candidate),
        "--out", str(out),
        "--kind", "cpt",
        "--parent-tokens-seen", str(parent_tokens_seen),
        "--parent-checkpoint", str(parent_state.relative_to(ROOT)),
        "--cpt-batch-size", str(BATCH_SIZE),
        "--cycle-id", cycle_id,
        "--valid-fraction", "0.004",
    ]
    must_succeed(run(args), f"build {cycle_id}")


def assert_layout(*, target: int, increment: int, parent_tokens_seen: int,
                  parent_state: Path, out: Path) -> None:
    """锁死 P3/P4 产物不变量。"""
    schedule_name = f"schedule-cpt-{target // 1_000_000}m.json"
    schedule_path = out / schedule_name
    _check(schedule_path.is_file(), f"schedule missing: {schedule_path}")
    schedule = load_json(schedule_path)

    # cumulative == rung 目标（否则 refuse_cpt_source 拒收）。
    cumulative = int(schedule["cumulative_exposure_tokens"])
    _check(cumulative == target, f"cumulative {cumulative} != target {target}")
    _check(int(schedule["parent_tokens_seen"]) == parent_tokens_seen,
           f"parent_tokens_seen {schedule['parent_tokens_seen']} != {parent_tokens_seen}")

    # 单一 2048 阶段，batch/grad_accum 不变量。
    curriculum = schedule["curriculum"]
    _check(len(curriculum) == 1, f"continuation expects single stage, got {len(curriculum)}")
    stage = curriculum[0]
    _check(int(stage["seq_len"]) == SEQ_LEN, f"seq_len {stage['seq_len']} != {SEQ_LEN}")
    _check(int(stage["batch_size"]) == BATCH_SIZE, f"batch {stage['batch_size']} != {BATCH_SIZE}")
    _check(int(stage["grad_accum"]) == GRAD_ACCUM, f"grad_accum {stage['grad_accum']} != {GRAD_ACCUM}")

    # LR 契约（cosine，base/final/horizon/token_offset）。
    lr = schedule["lr"]
    _check(lr["kind"] == "cosine_tokens", f"lr kind {lr['kind']} != cosine_tokens")
    _check(abs(float(lr["base"]) - LR_BASE) < 1e-12, f"lr base {lr['base']} != {LR_BASE}")
    _check(abs(float(lr["final"]) - LR_FINAL) < 1e-12, f"lr final {lr['final']} != {LR_FINAL}")
    _check(int(lr["horizon_tokens"]) == increment, f"lr horizon {lr['horizon_tokens']} != {increment}")
    _check(int(lr["token_offset"]) == parent_tokens_seen,
           f"lr token_offset {lr['token_offset']} != {parent_tokens_seen}")

    # parent 是未注册 state 文件时必须 pin parent_state_sha256（否则 refuse_cpt_parent 拒收）。
    _check(schedule.get("parent_state_sha256") == sha256_file(parent_state),
           "parent_state_sha256 not pinned to parent state file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-state", type=Path, required=True,
                        help="父轮 terminal state（pretrain-cpt-state.npz）")
    parser.add_argument("--target", type=int, required=True,
                        help="本轮 rung 目标累计 token（如 2100000000）")
    parser.add_argument("--pool-release", type=Path, required=True,
                        help="冻结池 RELEASE.json（该变的语料入口）")
    parser.add_argument("--quota", action="append", default=None, metavar="role=amount",
                        help="覆盖六角色配额（默认走固定公式 + hq 吸收差额）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只 plan-mix + build + assert，不 init/run 训练")
    args = parser.parse_args()

    assert_environment()
    target = args.target
    parent_tokens_seen, cursors = read_parent_state(args.parent_state)
    increment = assert_increment(parent_tokens_seen=parent_tokens_seen, target=target)

    if args.quota:
        quota = dict(FIXED_QUOTA)
        for item in args.quota:
            role, _, amount = item.partition("=")
            quota[role] = int(amount)
    else:
        quota = dict(FIXED_QUOTA)
    quota = assert_quota(increment=increment, quota=quota)

    n_m = target // 1_000_000
    cycle_id = f"exp-{target:07d}-v1"
    run_id = f"cpt-{n_m}m-v1-c01"
    candidate_id = f"mix-zhv2-{n_m}m-c01-v1"
    pool_root = Path(args.pool_release).parent.parent  # pools/<release>/RELEASE.json
    candidate = pool_root / "candidates" / f"{candidate_id}.json"
    layout = pool_root / "layouts" / f"zh-v2-layout-{cycle_id}-v1"

    plan_mix(increment=increment, consumed=cursors, quota=quota,
             pool_release=Path(args.pool_release), candidate_id=candidate_id, out=candidate)
    build_layout(target=target, increment=increment, parent_tokens_seen=parent_tokens_seen,
                 parent_state=args.parent_state, pool_release=Path(args.pool_release),
                 candidate=candidate, cycle_id=cycle_id, out=layout)
    assert_layout(target=target, increment=increment, parent_tokens_seen=parent_tokens_seen,
                  parent_state=args.parent_state, out=layout)

    print(f"contract OK: increment={increment} cumulative={target} layout={layout}")

    if args.dry_run:
        print("dry-run: 停在 build+assert，未 init/run 训练")
        return 0

    must_succeed(
        mei("cpt", "init", "--run-id", run_id, "--cycle-id", cycle_id,
            "--corpus-dir", str(layout.relative_to(ROOT)),
            "--target-exposure", str(target)),
        f"init {run_id}",
    )
    must_succeed(
        mei("cpt", "run", "--run-id", run_id, "--until", "cpt", "--confirm-training"),
        f"run {run_id}",
    )
    print(f"DONE: {run_id} completed cumulative={target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
