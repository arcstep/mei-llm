#!/usr/bin/env python3
"""检索头派生器：从 TaskEvidence 派生正工具集合（oracle / learned 分离）。

规则（0405 第 4 节，忠实、无据不编造）：
- 正工具集合 = 该 case 行为侧 original_calls 的工具名去重（oracle gold）。
- 无调用（original_calls 为空）的 case 不派生检索视图——没有正例就不编造。
- 负工具不自动标注，记 negative_tools=[] 与 candidate_construction=unknown，
  留待 compile 阶段按「同 catalog 抽样 + 语义审核」补齐（learned 视图）。
"""

from __future__ import annotations

from evidence import TaskEvidence
from derive_base import HeadDeriver, HeadView


class RetrievalDeriver:
    """检索头：正工具集合（任务级，一个 case 一个视图）。"""
    head = "retrieval"

    def derive(self, evidence: TaskEvidence) -> list[HeadView]:
        names = sorted({c.name for c in evidence.behavior.original_calls})
        if not names:
            return []
        construction = evidence.tool_env.candidate_construction or "unknown"
        return [HeadView(
            view_id=f"{evidence.identity.case_id}:retrieval",
            case_id=evidence.identity.case_id,
            head=self.head,
            target={
                "positive_tools": names,
                "negative_tools": [],
            },
            label_mask={"lm": 0, "retrieval": 1},
            status="pending_semantic_review",
            evidence={
                "candidate_construction": construction,
                "gold_source": "behavior.original_calls（oracle 视图，源发布不等于 gold 正确）",
            },
        )]
