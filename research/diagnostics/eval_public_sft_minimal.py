#!/usr/bin/env python3
"""Step 4：最小闭环生成验证 —— 对 holdout fullcall 样本生成调用，判正误。

加载某个权重（1800M base 或训后 fullcall.npz）作为 runtime，对留出的 holdout
样本渲染 prompt → 约束解码生成 → parse_call_text 解析 → 与 gold 精确比对
（name + arguments 完全一致才算对）。跑两次（--master 分别指 base 与 fullcall.npz）
即可对比「训练前 vs 训练后」的正确调用率。

判正误口径：解析结果 ok 且非 refuse 且恰好 1 个调用，且 name/arguments 与
answers[0] 完全一致 → 正确。其余（refuse / grammar 错 / schema 错 / 调用不匹配）
一律算错。

用法（主仓库 .venv python）：
    .venv/bin/python src/model-factory/eval_public_sft_minimal.py \
      --master <base 或 fullcall.npz 绝对路径> \
      --holdout cycles/.../holdout_rows.jsonl \
      --tools corpus/pools/task-trials/2026-09-15-public-sft-compile-r01/tools.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 先定位 worktree 根并设置 sys.path（扁平平铺 import），必须在 model-factory import 之前。
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT / "src/architecture/mei-1.2-51m",
    _ROOT / "src/model-factory",
    _ROOT / "src/platform/python-sdk",
    _ROOT / "src/platform/_shared/runtime",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orchestration.productize_controller import _load_runtime  # noqa: E402
from training.tool_use.sft_v3_training_51m import encode_fullcall_row  # noqa: E402
from mei_sdk.shared import parse_call_text  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="公开语料最小闭环生成验证（holdout fullcall）")
    parser.add_argument("--master", required=True, help="权重 npz 绝对路径（base 或 fullcall.npz）")
    parser.add_argument("--holdout", required=True, help="holdout_rows.jsonl 路径")
    parser.add_argument("--tools", required=True, help="tools.json 路径")
    parser.add_argument("--out", help="可选：判定明细输出 jsonl 路径")
    args = parser.parse_args()

    runtime = _load_runtime(Path(args.master).resolve(), quantized=False)
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))["tools"]
    tools_by_name = {str(tool["name"]): tool for tool in tools}
    holdout = _load_jsonl(Path(args.holdout).resolve())

    details: list[dict[str, Any]] = []
    exact = 0
    for row in holdout:
        prompt, _answer, _stats = encode_fullcall_row(runtime.tokenizer, row, tools_by_name)
        selected = [tools_by_name[str(name)] for name in (row.get("retrieved_tools") or [])[:5]]
        decoded = runtime.greedy(prompt, tools=selected, max_new=128, decode_mode="constrained")
        text = str(decoded.get("text") or "")
        parsed = parse_call_text(text, selected)
        gold = (row.get("answers") or [{}])[0]
        ok = bool(
            parsed.get("ok")
            and not parsed.get("refuse")
            and len(parsed.get("function_calls") or []) == 1
        )
        call = (parsed.get("function_calls") or [{}])[0] if ok else {}
        matched = bool(
            ok
            and call.get("name") == gold.get("name")
            and call.get("arguments") == gold.get("arguments")
        )
        exact += int(matched)
        detail = {
            "sample_id": row.get("sample_id"),
            "gold_name": gold.get("name"),
            "pred_name": call.get("name") if ok else None,
            "exact": matched,
            "parse_ok": bool(parsed.get("ok")),
            "refuse": bool(parsed.get("refuse")),
            "parse_error": parsed.get("error"),
            "n_out": int(decoded.get("n_out") or 0),
            "generated_text": text,
        }
        details.append(detail)
        print(
            json.dumps(
                {k: v for k, v in detail.items() if k != "generated_text"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = {
        "master": str(Path(args.master).resolve()),
        "holdout_rows": len(holdout),
        "exact_correct": exact,
        "exact_rate": exact / max(len(holdout), 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
