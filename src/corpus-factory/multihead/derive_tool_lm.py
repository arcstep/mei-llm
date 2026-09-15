#!/usr/bin/env python3
"""工具LM 头派生器：从 TaskEvidence 派生调用 batch 与参数证据。

规则（0405 第 4~5 节，忠实、无据不编造）：
- 只从 assistant 事件里已解析出的 tool_calls 派生，不凭空生成调用。
- 调用前的可见历史 = timeline[:该事件位置]；未来工具结果绝不进入调用前输入。
- 参数证据做三通道之一：字面定位。参数值能在可见历史文本里找到精确子串，
  记为 literal_location_only；否则记为 semantic_or_normalization_review，
  留待语义审核或归一化映射（枚举/单位/搜索改写），不在此处猜值。
"""

from __future__ import annotations

from typing import Any

from evidence import TaskEvidence, Event
from derive_base import HeadDeriver, HeadView, leaves


def _parameter_evidence(calls: list[dict[str, Any]], visible: list[Event]) -> list[dict[str, Any]]:
    """对调用 batch 的每个参数叶子，在可见历史里做字面定位。"""
    evidence: list[dict[str, Any]] = []
    for ci, call in enumerate(calls):
        for path, value in leaves(call.get("arguments", {})):
            if not isinstance(value, str) or len(value) < 2:
                continue
            spans = [
                {"event_id": e.event_id, "role": e.role}
                for e in visible
                if value in (e.content or "")
            ]
            evidence.append({
                "call_index": ci,
                "path": list(path),
                "value": value,
                "prior_exact_spans": spans,
                "status": "literal_location_only" if spans else "semantic_or_normalization_review",
            })
    return evidence


class ToolLmDeriver:
    """工具LM 头：调用 batch + 参数来源证据。"""
    head = "tool_lm"

    def derive(self, evidence: TaskEvidence) -> list[HeadView]:
        views: list[HeadView] = []
        catalog_names = {t["name"] for t in evidence.tool_env.catalog}
        for position, event in enumerate(evidence.timeline):
            if event.role != "assistant" or not event.tool_calls:
                continue
            visible = evidence.timeline[:position]
            calls = [c.to_dict() for c in event.tool_calls]
            argument_evidence = _parameter_evidence(calls, visible)
            # 结构门：调用工具必须在 catalog 内。不在 catalog 的调用不派生（无据）。
            unknown = [c["name"] for c in calls if c["name"] not in catalog_names]
            if unknown:
                continue
            target = {
                "calls": calls,
                "anchor_event_id": event.event_id,
                "visible_history_events": [e.event_id for e in visible],
            }
            views.append(HeadView(
                view_id=f"{evidence.identity.case_id}:tool_lm:{event.event_id}",
                case_id=evidence.identity.case_id,
                head=self.head,
                target=target,
                label_mask={"lm": 1, "retrieval": 0},
                status="pending_semantic_review",
                evidence={
                    "argument_provenance": argument_evidence,
                    "unresolved_parameters": [
                        p for p in argument_evidence
                        if p["status"] == "semantic_or_normalization_review"
                    ],
                },
            ))
        return views
