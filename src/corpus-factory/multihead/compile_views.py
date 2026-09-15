#!/usr/bin/env python3
"""扩量转换器：pipeline 产出的 views → fullcall + retrieval 训练 row。

这是 compile_minimal 的扩量版：compile_minimal 吃人工 review 过的 semantic-specimens
（44 条，semantic_label_masks 已置 1），本脚本吃未 review 的 views（pipeline 的
intake→derive→audit 产出，语义标签仍 masked），用「结构可信」替代「语义 review」作准入：

fullcall 准入（更严格，保证 target 忠实）：
- calls 恰好 1 个（serialize_tool_target 单调用限制）
- schema_errors 空（schema 检查通过）
- behavior.verified_result_dependency == false
- behavior.arguments_with_prior_result_literal == 0（排除参数依赖前置结果字面量）
- fits_five_tool_catalog == true（恰好 5 工具目录）
- encoding.fits_2048 == true（prompt 不超预算）
- 5 工具 sink 不超 stable prefix 预算（STABLE_PREFIX_TOKENS_MAX=1024，预算感知选干扰工具）
- target 序列化后不超 128 token（encode_fullcall_row 的 max_answer 契约）

retrieval 准入（更宽松，检索只看 gold 工具）：
- schema_errors 空 + calls 非空（build_retrieval_rows 内部再跳过无调用）

关于「预算感知选干扰工具」：compile_minimal 用 selected_tools（相似度优先），在 44 个
小 schema 样板下 sink 不超预算。扩量到 1028 工具后，ToolACE/MOSS 里有 schema 巨大的
工具，相似度排名会把它们排进 top-48 窗口，5 工具 sink 渲染后超 1024 token，训练期
encode_fullcall_row 抛 ToolSchemaBudgetExceeded（实测 335/1447 条触发）。修复：fullcall
的 4 个干扰工具改为「相似度优先 + sink 预算约束」的贪心选择——干扰工具本就是非 gold 的
候选占位，换 schema 小的不影响训练语义，但保住 335 条数据不丢。

定位：能力/链路验证的扩量。语义标签仍 masked，训练可能引入少量错误 gold，属
「用规模换泛化」的探索，不是最终 SFT 语料。

用法（需 .venv 的 mlx + sentencepiece，因加载 tokenizer 做预算校验）：
    .venv/bin/python src/corpus-factory/multihead/compile_views.py \
      --views <views.jsonl> --out <out_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# 复用 model-factory 的现成 serializer/sampler + tokenizer（同 _ensure_python_paths 的
# 扁平平铺 import），保证 sink 预算校验与训练器 encode_fullcall_row 零 drift。
_ROOT = Path(__file__).resolve().parents[3]
for _p in (
    _ROOT / "src/architecture/mei-1.2-51m",
    _ROOT / "src/model-factory",
    _ROOT / "src/platform/python-sdk",
    _ROOT / "src/platform/_shared/runtime",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compile_minimal import (  # noqa: E402
    _history_and_query,
    _verify,
    build_retrieval_rows,
    collect_tools,
)
from contracts.sft_v3_contract_51m import (  # noqa: E402
    STABLE_PREFIX_TOKENS_MAX,
    TURN_END,
    compact_tool,
    ranked_hard_negatives,
    serialize_tool_target,
)
from training.tool_use.sft_v3_training_51m import render_fullcall_prompt_parts  # noqa: E402
from training.cpt.train_pretrain import _frozen_tokenizer  # noqa: E402

# encode_fullcall_row 的 max_answer 默认值（train_lm_sft_v3 未覆盖，走默认 128）。
ANSWER_TOKEN_MAX = 128


def _no_prior_result_literal(view: dict[str, Any]) -> bool:
    behavior = view.get("behavior") or {}
    if behavior.get("verified_result_dependency"):
        return False
    return int(behavior.get("arguments_with_prior_result_literal") or 0) == 0


def _compact_len(tool: dict[str, Any]) -> int:
    return len(json.dumps(compact_tool(tool), ensure_ascii=False, separators=(",", ":")))


def _sink_tokens(tools_5: list[dict[str, Any]], tokenizer: Any) -> int:
    """5 工具渲染 sink 的 token 数（与 encode_fullcall_row 的 stable_ids 计算一致）。"""
    sink = render_fullcall_prompt_parts({"query": ""}, tools_5)["sink"]
    return len(tokenizer.encode(sink + "\n", add_bos=True, add_eos=False))


def _budget_aware_catalog(
    gold_tool: dict[str, Any],
    all_tools: list[dict[str, Any]],
    variant: int,
    tokenizer: Any,
    smallest_names: list[str],
) -> list[str] | None:
    """相似度优先 + sink 预算约束选 4 个干扰工具，返回 5 工具名（gold 在 variant%5 位）。

    返回 None 表示无法在该 gold 工具下凑出 sink <= 预算的 5 工具目录（gold 本身太大）。
    """
    by_name = {str(item["name"]): item for item in all_tools}
    gold_name = str(gold_tool["name"])
    budget = STABLE_PREFIX_TOKENS_MAX

    # 1. 下界判断：gold + 4 个全局最小 schema 工具是否已超预算（gold 太大则无解）。
    smallest_neg = [name for name in smallest_names if name != gold_name][:4]
    if len(smallest_neg) < 4:
        return None
    if _sink_tokens([by_name[n] for n in [gold_name] + smallest_neg], tokenizer) > budget:
        return None

    # 2. 贪心：按相似度降序遍历，接受「加入后仍能用最小工具补齐到 5 且不超预算」的候选。
    ranked = ranked_hard_negatives(gold_tool, all_tools)
    negatives: list[str] = []
    for name in ranked:
        if len(negatives) == 4:
            break
        if name == gold_name or name in negatives:
            continue
        remaining = 4 - len(negatives) - 1
        pad = [
            n for n in smallest_names
            if n != gold_name and n != name and n not in negatives
        ][:remaining]
        test = negatives + [name] + pad
        names = list(test)
        names.insert(variant % 5, gold_name)
        if _sink_tokens([by_name[n] for n in names], tokenizer) <= budget:
            negatives.append(name)

    # 3. 贪心凑不满 4 个（早期候选过大卡住）时，兜底用全局最小 4 个。
    if len(negatives) < 4:
        negatives = smallest_neg[:4]

    # 4. 最终精确校验（理论不应失败，兜底后仍超则放弃该 view）。
    names = list(negatives)
    names.insert(variant % 5, gold_name)
    if _sink_tokens([by_name[n] for n in names], tokenizer) > budget:
        return None
    return names


def build_fullcall_rows_from_views(
    views: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
    all_tools: list[dict[str, Any]],
    tokenizer: Any,
    smallest_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """结构可信的单调用 view → fullcall row。返回 (rows, excluded_counts)。"""
    rows: list[dict[str, Any]] = []
    excluded: dict[str, int] = {
        "multi_call": 0,
        "schema_error": 0,
        "prior_result_literal": 0,
        "budget": 0,
        "answer_over": 0,
        "missing_tool": 0,
    }
    for variant, view in enumerate(views):
        calls = view.get("calls") or []
        if len(calls) != 1:
            excluded["multi_call"] += 1
            continue
        if view.get("schema_errors"):
            excluded["schema_error"] += 1
            continue
        if not _no_prior_result_literal(view):
            excluded["prior_result_literal"] += 1
            continue
        if not bool(view.get("fits_five_tool_catalog")) or not bool(
            (view.get("encoding") or {}).get("fits_2048")
        ):
            excluded["budget"] += 1
            continue
        gold_name = str(calls[0].get("name") or "")
        gold_tool = tools_by_name.get(gold_name)
        if gold_tool is None:
            excluded["missing_tool"] += 1
            continue
        history, query = _history_and_query(view)
        answers = calls
        target_text = serialize_tool_target(answers)
        answer_tokens = (
            len(tokenizer.encode(target_text, add_bos=False, add_eos=False))
            + len(tokenizer.encode(TURN_END, add_bos=False, add_eos=False))
            + 1
        )
        if answer_tokens > ANSWER_TOKEN_MAX:
            excluded["answer_over"] += 1
            continue
        names = _budget_aware_catalog(gold_tool, all_tools, variant, tokenizer, smallest_names)
        if names is None:
            excluded["budget"] += 1
            continue
        rows.append({
            "sample_id": str(view["id"]),
            "case_id": str(view.get("group_id") or view["id"]),
            "task": "fullcall",
            "query": query,
            "history": history,
            "retrieved_tools": names,
            "answers": answers,
            "target_text": target_text,
        })
    return rows, excluded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="views → fullcall + retrieval 训练 row（扩量）")
    parser.add_argument("--views", required=True, help="views.jsonl 路径（pipeline audit 产出）")
    parser.add_argument("--out", required=True, help="产出目录")
    args = parser.parse_args()

    views_path = Path(args.views).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [json.loads(line) for line in views_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    schema_clean = [view for view in views if not view.get("schema_errors")]
    tools_by_name, all_tools = collect_tools(schema_clean)

    tokenizer = _frozen_tokenizer()
    smallest_names = [str(tool["name"]) for tool in sorted(all_tools, key=_compact_len)]

    fullcall_rows, excluded = build_fullcall_rows_from_views(
        views, tools_by_name, all_tools, tokenizer, smallest_names
    )
    retrieval_rows = build_retrieval_rows(schema_clean, tools_by_name, all_tools)
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
        "schema": "mei-public-sft-compile-views-v1",
        "views_path": str(views_path),
        "views_sha256": _sha256(views_path),
        "views_total": len(views),
        "schema_clean_views": len(schema_clean),
        "fullcall_rows": len(fullcall_rows),
        "fullcall_excluded": excluded,
        "retrieval_rows": len(retrieval_rows),
        "tools": len(all_tools),
        "serializer_id": "mei-tool-call-serializer-v2",
        "admission": "structural_trust_not_semantic_review",
        "fullcall_only_single_call": True,
        "retrieval_gold_from_first_call": True,
        "fullcall_negative_selection": "similarity_ranked_with_sink_budget_greedy",
        "stable_prefix_tokens_max": STABLE_PREFIX_TOKENS_MAX,
        "answer_token_max": ANSWER_TOKEN_MAX,
        "excluded_prior_result_literal_reason": (
            "参数依赖前置工具结果字面量（arguments_with_prior_result_literal>0），"
            "纯 query→单调用 会产出参数凭空样本，留待结果配对派生通道"
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
