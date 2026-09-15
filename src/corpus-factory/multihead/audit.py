#!/usr/bin/env python3
"""结构自动门：派生视图的结构审计（不替代语义审核）。

只做「能被规则自动判定」的结构检查，语义正确性由后续 reviewer 阶段负责。
检查项（对 0405 第 9 节的结构门，复用 validate + 派生契约）：
- 工具可见性：调用/正工具必须在 catalog 内。
- 参数字面量：target 可 JSON 序列化，无内部对象泄漏。
- 事件顺序：调用前可见历史只含调用前事件，无未来结果混入。
- 可重建：behavior.original_calls 与 timeline 展开的 tool_calls 一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evidence import TaskEvidence
from derive_base import HeadView


@dataclass
class AuditReport:
    """单个「样本×头」视图的结构门结论。"""
    case_id: str
    view_id: str
    head: str
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "view_id": self.view_id,
            "head": self.head,
            "passed": self.passed,
            "issues": self.issues,
        }


def _json_roundtrip(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


def structural_gate(evidence: TaskEvidence, view: HeadView) -> AuditReport:
    """对派生视图做结构门检查，返回 AuditReport。"""
    report = AuditReport(case_id=view.case_id, view_id=view.view_id, head=view.head)
    catalog_names = {t["name"] for t in evidence.tool_env.catalog}

    if not _json_roundtrip(view.target):
        report.issues.append({"reason": "target_not_json_serializable"})

    if view.head == "tool_lm":
        for call in view.target.get("calls", []):
            name = call.get("name")
            if name not in catalog_names:
                report.issues.append({"reason": "tool_not_visible", "tool": name})
            if not isinstance(call.get("arguments"), dict):
                report.issues.append({"reason": "arguments_not_dict", "tool": name})
        # 可见历史必须严格是调用事件之前的子序列（无未来事件混入），顺序也一致。
        anchor = view.target.get("anchor_event_id")
        timeline_ids = [e.event_id for e in evidence.timeline]
        if anchor not in timeline_ids:
            report.issues.append({"reason": "anchor_event_not_in_timeline"})
        else:
            prefix = timeline_ids[:timeline_ids.index(anchor)]
            if view.target.get("visible_history_events") != prefix:
                report.issues.append({"reason": "visible_history_not_prefix"})

    elif view.head == "retrieval":
        positives = view.target.get("positive_tools", [])
        if not positives:
            report.issues.append({"reason": "empty_positive_tools"})
        for name in positives:
            if name not in catalog_names:
                report.issues.append({"reason": "positive_tool_not_visible", "tool": name})

    return report


def timeline_structural_checks(evidence: TaskEvidence) -> list[dict[str, Any]]:
    """轨迹级结构检查：可重建（original_calls 与 timeline 展开一致）。"""
    issues: list[dict[str, Any]] = []
    expanded = [c for e in evidence.timeline if e.role == "assistant" for c in e.tool_calls]
    original = evidence.behavior.original_calls
    if [c.name for c in expanded] != [c.name for c in original]:
        issues.append({"reason": "original_calls_mismatch_timeline"})
    return issues
