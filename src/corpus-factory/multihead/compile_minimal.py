#!/usr/bin/env python3
"""最小闭环转换器：语义样板 → fullcall + retrieval 训练 row。

这是 0406 五段流水线的「compile」最小实现，只覆盖最可信的两个头（工具LM + 检索），
用于打通「公开语料 → 训练 row → 训练 → 生成正确调用」这条链路的最小假设验证。

输入：semantic-specimens.jsonl（语义 review 过的样板，字段训练友好）：
    id / group_id / calls[{name,arguments}] / input.messages[{role,content,...}]
    / source_catalog[{name,description,parameters}] / semantic_label_masks

转换（忠实、不引入映射错误）：
- fullcall row（仅单调用样板，serialize_tool_target 只允许恰好一次调用）：
    answers = calls；target_text = contracts.serialize_tool_target(answers)（复用现成
    serializer，保证与训练器 drift 校验一致）
    query = 最后一条 user 的 content；history = 它之前的所有 message（{role,content}）
    retrieved_tools = gold + 4 干扰工具（复用 selected_tools 的 oracle-top5 视图）
- retrieval row（全部样板，gold_tool = calls[0].name）：
    hard_negatives = 4 个干扰工具（复用 hard_negatives），catalog_tools = 5 个 compact_tool

边界（明确不做什么）：
- 多调用样板（calls 数量 != 1）不进 fullcall（serialize_tool_target 契约限制），后续单独做。
- 不做语义审核、不做参数归一化，只忠实搬运 review 过的样板。
- 工具目录是公开语料的（public_* / ToolACE），不是 Mei 147 设备工具。

用法（脚本目录自动进 sys.path，扁平平铺 import）：
    python3 compile_minimal.py --specimens <semantic-specimens.jsonl> --out <out_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# 复用 model-factory 的现成 serializer/sampler，保证 target_text 与训练器 drift 校验一致。
# contracts.sft_v3_contract_51m 无 MLX import，可安全在 corpus-factory 侧加载。
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src" / "model-factory"))
from contracts.sft_v3_contract_51m import (  # noqa: E402
    compact_tool,
    hard_negatives,
    selected_tools,
    serialize_tool_target,
)


def _last_user_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return -1


def _history_and_query(spec: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """最后一条 user → query，它之前的所有 message → history（{role, content}）。"""
    messages = spec.get("input", {}).get("messages", [])
    last_user = _last_user_index(messages)
    if last_user < 0:
        return [], ""
    query = str(messages[last_user].get("content") or "")
    history = [
        {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
        for message in messages[:last_user]
        if (message.get("content") or "").strip()
    ]
    return history, query


def collect_tools(specimens: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """跨样板收集 source_catalog 去重，作为 training_catalog（工具目录）。"""
    by_name: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for spec in specimens:
        for tool in spec.get("source_catalog", []):
            name = str(tool.get("name") or "")
            if not name or name in by_name:
                continue
            by_name[name] = compact_tool(tool)
            ordered.append(name)
    return by_name, [by_name[name] for name in ordered]


def build_fullcall_rows(
    specimens: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
    all_tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """单调用样板 → fullcall row。返回 (rows, excluded_result_dependency)。

    排除 verified_result_dependency=true 的样板：其参数依赖前置工具结果，正确表示需要
    prior_calls + tool_results（结果配对派生通道），不属「query → 单调用」的纯 fullcall。
    这类样板留待后续专门处理，不在此处生成参数凭空（provenance 缺失）的缺陷样本。
    """
    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    for variant, spec in enumerate(specimens):
        calls = spec.get("calls") or []
        if len(calls) != 1:
            continue  # 多调用后续单独做（serialize_tool_target 只允许一次调用）
        if spec.get("behavior", {}).get("verified_result_dependency"):
            excluded.append(str(spec["id"]))
            continue
        gold_name = str(calls[0].get("name") or "")
        gold_tool = tools_by_name.get(gold_name)
        if gold_tool is None:
            continue
        history, query = _history_and_query(spec)
        answers = calls
        target_text = serialize_tool_target(answers)
        catalog_5, _negatives = selected_tools(gold_tool, all_tools, variant)
        rows.append({
            "sample_id": str(spec["id"]),
            "case_id": str(spec.get("group_id") or spec["id"]),
            "task": "fullcall",
            "query": query,
            "history": history,
            "retrieved_tools": [str(tool["name"]) for tool in catalog_5],
            "answers": answers,
            "target_text": target_text,
        })
    return rows, excluded


def build_retrieval_rows(
    specimens: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
    all_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, spec in enumerate(specimens):
        calls = spec.get("calls") or []
        if not calls:
            continue
        gold_name = str(calls[0].get("name") or "")
        gold_tool = tools_by_name.get(gold_name)
        if gold_tool is None:
            continue
        history, query = _history_and_query(spec)
        catalog_5, negatives = selected_tools(gold_tool, all_tools, variant)
        rows.append({
            "sample_id": str(spec["id"]) + ":retrieval",
            "case_id": str(spec.get("group_id") or spec["id"]),
            "task": "retrieval",
            "kind": "hard_positive",
            "query": query,
            "gold_tool": gold_name,
            "hard_negatives": list(negatives),
            "catalog_tools": catalog_5,
            "seen_schema": True,
        })
    return rows


def _verify(fullcall_rows: list[dict[str, Any]]) -> None:
    """serializer drift 校验：target_text 必须与 serialize_tool_target(answers) 逐条一致。"""
    for row in fullcall_rows:
        recomputed = serialize_tool_target(row["answers"])
        if row["target_text"] != recomputed:
            raise RuntimeError(f"target serializer drift: {row['sample_id']}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="语义样板 → fullcall + retrieval 训练 row（最小闭环）")
    parser.add_argument("--specimens", required=True, help="semantic-specimens.jsonl 路径")
    parser.add_argument("--out", required=True, help="产出目录")
    args = parser.parse_args()

    specimens_path = Path(args.specimens).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specimens = [json.loads(line) for line in specimens_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tools_by_name, all_tools = collect_tools(specimens)

    fullcall_rows, excluded_result_dependency = build_fullcall_rows(specimens, tools_by_name, all_tools)
    retrieval_rows = build_retrieval_rows(specimens, tools_by_name, all_tools)
    _verify(fullcall_rows)

    (out_dir / "fullcall_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in fullcall_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "retrieval_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "tools.json").write_text(
        json.dumps({"tools": all_tools}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "mei-public-sft-compile-minimal-v1",
        "specimens_path": str(specimens_path),
        "specimens_sha256": _sha256(specimens_path),
        "specimens_total": len(specimens),
        "fullcall_rows": len(fullcall_rows),
        "retrieval_rows": len(retrieval_rows),
        "tools": len(all_tools),
        "serializer_id": "mei-tool-call-serializer-v2",
        "fullcall_only_single_call": True,
        "retrieval_gold_from_first_call": True,
        "excluded_result_dependency": excluded_result_dependency,
        "excluded_result_dependency_reason": (
            "参数依赖前置工具结果（verified_result_dependency），正确表示需 prior_calls+tool_results，"
            "不属 query→单调用 纯 fullcall，留待结果配对派生通道处理"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
