#!/usr/bin/env python3
"""最小训练闭环：加载 1800M base + 公开工具目录，训 fullcall + retrieval 头。

这是 0406 五段流水线之外的最小能力/链路验证入口，只回答一个问题：
「44 条语义 review 样板 → 训练 row → 训几十步 → loss 是否下降」。
不走 productize 的 16-stage，直接调 train_lm_sft_v3 + train_retrieval_v3 两个训练原语。

工具目录是公开语料的（public_* / ToolACE），不是 Mei 147 设备工具；base 用本机
1800M（mei-1.2-51m-base-cpt1800m-v1，参数 51.5M，sft=null）。

关键约束（现役训练器对 row 字段的要求，compaction 已探明）：
- train_lm_sft_v3 的 FULLCALL_SAMPLER_ID 要求 execute+refuse 混合 + kind/candidate_tool，
  而 compile 产出的 32 条 fullcall rows 全是 execute、无 refuse。故此处用
  AGENT_SAMPLER_ID（每条样板 = 独立单步轨迹：trajectory_id=sample_id, step=0），
  语义自洽且不合成假 refuse 数据。
- train_retrieval_v3 用 gold_tool + 4 hard_negatives（22 工具去重 >= batch_size 8）。

用法（主仓库 .venv python，master 用主仓库绝对路径，rows-dir/out-dir 用 worktree 路径）：
    .venv/bin/python src/model-factory/train_public_sft_minimal.py \
      --master <1800M npz 绝对路径> \
      --rows-dir corpus/pools/task-trials/2026-09-15-public-sft-compile-r01 \
      --out-dir cycles/mei-1.2-51m/exp-corpus-survey/runs/2026-09-15-public-sft-minimal-r01 \
      --steps 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# 先定位 worktree 根并设置 sys.path（扁平平铺 import，同 _ensure_python_paths），
# 必须发生在任何 model-factory import 之前，否则 common.paths.find_root() 会误定位。
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT / "src/architecture/mei-1.2-51m",
    _ROOT / "src/model-factory",
    _ROOT / "src/platform/python-sdk",
    _ROOT / "src/platform/_shared/runtime",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orchestration.productize_51m import _load_runtime  # noqa: E402
from training.tool_use.sft_v3_training_51m import (  # noqa: E402
    AGENT_SAMPLER_ID,
    train_lm_sft_v3,
    train_retrieval_v3,
)
from common.checkpoint import save_params  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _add_agent_trajectory_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给纯 execute 的 fullcall rows 补 agent sampler 所需字段。

    每条样板 = 独立单步轨迹（trajectory_id=sample_id 唯一，trajectory_step=0）。
    FULLCALL_SAMPLER_ID 需要 execute+refuse 混合分组，纯 execute 样板不满足，故用
    AGENT_SAMPLER_ID，其 agent_epoch_order 只要求 trajectory_id/cf_group 非空 +
    trajectory_step 可排序（execute_repeat=1 时无需 kind 字段）。
    """
    for row in rows:
        row["trajectory_id"] = str(row.get("sample_id") or row.get("case_id") or "")
        row["trajectory_step"] = 0
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="公开语料最小训练闭环（fullcall + retrieval）")
    parser.add_argument("--master", required=True, help="1800M base 权重 npz 绝对路径")
    parser.add_argument("--rows-dir", required=True, help="compile 产出目录（fullcall/retrieval rows + tools）")
    parser.add_argument("--out-dir", required=True, help="run 产出目录（fullcall.npz / retrieval.npz / manifest）")
    parser.add_argument("--steps", type=int, default=40, help="训练步数（默认 40）")
    parser.add_argument("--lr-lm", type=float, default=2e-4, help="fullcall LM 学习率")
    parser.add_argument("--lr-retrieval", type=float, default=1e-3, help="retrieval 头学习率")
    parser.add_argument("--holdout", type=int, default=4, help="随机留出 fullcall 样本数（不进训练，供生成验证）")
    parser.add_argument("--seed", type=int, default=42, help="holdout 随机留出的固定种子（可复现）")
    args = parser.parse_args()

    rows_dir = Path(args.rows_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 runtime（1800M base + 冻结 tokenizer，不量化）
    runtime = _load_runtime(Path(args.master).resolve(), quantized=False)

    # 2. 加载 compile 产出
    fullcall_rows = _load_jsonl(rows_dir / "fullcall_rows.jsonl")
    retrieval_rows = _load_jsonl(rows_dir / "retrieval_rows.jsonl")
    tools = json.loads((rows_dir / "tools.json").read_text(encoding="utf-8"))["tools"]
    tools_by_name = {str(tool["name"]): tool for tool in tools}

    # 3. 随机留出 holdout fullcall 样本（固定 seed 可复现，覆盖各 source），其余补 agent 轨迹字段
    if args.holdout > 0:
        shuffled = list(fullcall_rows)
        random.Random(args.seed).shuffle(shuffled)
        holdout_rows = shuffled[: args.holdout]
        train_fullcall = shuffled[args.holdout:]
    else:
        holdout_rows = []
        train_fullcall = list(fullcall_rows)
    _add_agent_trajectory_fields(train_fullcall)

    # 4. 训 fullcall LM（AGENT_SAMPLER_ID，走普通 FP row 路径）
    fullcall_report = train_lm_sft_v3(
        runtime,
        train_fullcall,
        tools_by_name,
        steps=args.steps,
        lr=args.lr_lm,
        group_map=None,
        activation_ste=False,
        sampler=AGENT_SAMPLER_ID,
        checkpoint_dir=None,
        resume=False,
        stage_id="public-fullcall-minimal",
        execute_repeat=1,
    )
    fullcall_npz = out_dir / "fullcall.npz"
    save_params(runtime.model, fullcall_npz)

    # 5. 训 retrieval 头（LM 冻结，只更新 contrastive head）
    retrieval_report = train_retrieval_v3(
        runtime,
        retrieval_rows,
        tools_by_name,
        steps=args.steps,
        lr=args.lr_retrieval,
        batch_size=8,
        checkpoint_dir=None,
        resume=False,
        stage_id="public-retrieval-minimal",
    )
    retrieval_npz = out_dir / "retrieval.npz"
    save_params(runtime.contrastive, retrieval_npz)

    # 6. 落盘 holdout + manifest
    (out_dir / "holdout_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in holdout_rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "mei-public-sft-minimal-train-v1",
        "master": str(Path(args.master).resolve()),
        "rows_dir": str(rows_dir),
        "steps": args.steps,
        "lr_lm": args.lr_lm,
        "lr_retrieval": args.lr_retrieval,
        "fullcall_train_rows": len(train_fullcall),
        "fullcall_holdout_rows": len(holdout_rows),
        "retrieval_rows": len(retrieval_rows),
        "tools": len(tools),
        "fullcall_sampler": AGENT_SAMPLER_ID,
        "fullcall_holdout_sample_ids": [str(row.get("sample_id")) for row in holdout_rows],
        "fullcall_report": fullcall_report,
        "retrieval_report": retrieval_report,
        "fullcall_npz": str(fullcall_npz),
        "retrieval_npz": str(retrieval_npz),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
