#!/usr/bin/env python3
"""解说头派生器：从 TaskEvidence 派生「结果—回复对」候选。

规则（0405 第 8 节，忠实、无据不编造）：
- 结果—回复对先作候选，逐事实核对后才可 grounding。
- 有结果缺回复 → 不派生（可先备事实清单，但不擅自编解说）。
- 未来结果不能进入调用前输入（这里只按 timeline 顺序配对，tool 结果在调用之后）。
"""

from __future__ import annotations

from evidence import TaskEvidence
from derive_base import HeadDeriver, HeadView


class NarrationDeriver:
    """解说头：结果—回复对候选（结果可验证后逐事实核对）。"""
    head = "narration"

    def derive(self, evidence: TaskEvidence) -> list[HeadView]:
        case_id = evidence.identity.case_id
        views: list[HeadView] = []
        for i, event in enumerate(evidence.timeline):
            if event.role != "tool" or not (event.content or "").strip():
                continue
            reply = None
            for later in evidence.timeline[i + 1:]:
                if later.role == "assistant" and (later.content or "").strip():
                    reply = later
                    break
            if reply is None:
                continue
            views.append(HeadView(
                view_id=f"{case_id}:narration:{event.event_id}",
                case_id=case_id,
                head=self.head,
                target={
                    "result_event_id": event.event_id,
                    "reply_event_id": reply.event_id,
                    "reply": reply.content,
                },
                label_mask={"narration": 1},
                status="pending_semantic_review",
                evidence={
                    "result_content": event.content,
                    "grounding": "result_reply_pair_requires_fact_check",
                },
            ))
        return views
