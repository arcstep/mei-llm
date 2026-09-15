#!/usr/bin/env python3
"""处置头派生器：从 TaskEvidence 派生处置动作（execute/clarify/wait/refuse/complete/review）。

规则（0405 第 6 节，忠实、无据不编造）：
- 优先用 behavior.source_disposition（adapter 从源数据提取的动作/原因/来源标注）。
- 无源标注时，有调用 → 推断 execute（reason=inferred_from_tool_call）；无调用 → 不派生。
  「无调用」不能混成一个标签（可能是澄清/等待/拒绝/完成，需要源标注区分），
  所以宁缺毋滥，不伪造。
- execute 过剩不得掩盖 clarify/wait/refuse/complete 零覆盖——缺失动作保持未标注。
"""

from __future__ import annotations

from evidence import TaskEvidence
from derive_base import HeadDeriver, HeadView

# 标准处置动作类别（0405 第 6 节）。adapter 可产出 review（约束放宽等需复核）。
ACTIONS: tuple[str, ...] = ("execute", "clarify", "wait", "refuse", "complete", "review")


class DispositionDeriver:
    """处置头：动作码 + 分离的原因/对象/缺失字段。"""
    head = "disposition"

    def derive(self, evidence: TaskEvidence) -> list[HeadView]:
        case_id = evidence.identity.case_id
        sd = evidence.behavior.source_disposition
        if not isinstance(sd, dict):
            sd = {}
        # 多轮源（如 Nemotron 的 policy 驱动对话）：每个带 action 的轮派生一个视图。
        turns = sd.get("turns")
        if isinstance(turns, list):
            views: list[HeadView] = []
            for turn in turns:
                action = turn.get("action")
                if not action:
                    # 无调用回复等未标注轮：禁止默认值，不派生。
                    continue
                views.append(HeadView(
                    view_id=f"{case_id}:disposition:{turn.get('turn_index', '?')}",
                    case_id=case_id,
                    head=self.head,
                    target={
                        "action": action,
                        "reason": turn.get("reason"),
                        "missing_fields": turn.get("missing_fields", []),
                    },
                    label_mask={"disposition": 1},
                    status="pending_semantic_review",
                    evidence={
                        "source_kind": turn.get("source_kind", sd.get("source_kind")),
                        "source": "behavior.source_disposition.turns",
                    },
                ))
            return views
        action = sd.get("action")
        if action:
            return [HeadView(
                view_id=f"{case_id}:disposition",
                case_id=case_id,
                head=self.head,
                target={
                    "action": action,
                    "reason": sd.get("reason"),
                    "missing_fields": sd.get("missing_fields", []),
                },
                label_mask={"disposition": 1},
                status="pending_semantic_review",
                evidence={
                    "source_kind": sd.get("source_kind"),
                    "source": "behavior.source_disposition",
                },
            )]
        # 无源标注：仅当有调用时推断 execute，其余动作不猜。
        if evidence.behavior.original_calls:
            return [HeadView(
                view_id=f"{case_id}:disposition",
                case_id=case_id,
                head=self.head,
                target={"action": "execute", "reason": "inferred_from_tool_call"},
                label_mask={"disposition": 1},
                status="pending_semantic_review",
                evidence={"source": "inferred_from_original_calls"},
            )]
        return []
