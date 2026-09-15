#!/usr/bin/env python3
"""样本×头 独立状态机。

每个「样本×头」独立维护一个状态 + issue 列表。`admitted` 只表示该头在指定
binding 与验证范围下准入，不代表整个任务五头齐备。多问题用 issue 列表保存，
不互相覆盖。状态语义对 0405 第 9 节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 五个模型职责
HEADS: tuple[str, ...] = (
    "retrieval",
    "tool_lm",
    "disposition",
    "confidence",
    "narration",
)

# 状态全集（0405 第 9 节）
CANDIDATE = "candidate"
PENDING_EVIDENCE = "pending_evidence"
PENDING_SEMANTIC_REVIEW = "pending_semantic_review"
PENDING_RUNTIME = "pending_runtime"
PENDING_MODEL = "pending_model"
ADMITTED = "admitted"
EXCLUDED = "excluded"

STATUSES: tuple[str, ...] = (
    CANDIDATE,
    PENDING_EVIDENCE,
    PENDING_SEMANTIC_REVIEW,
    PENDING_RUNTIME,
    PENDING_MODEL,
    ADMITTED,
    EXCLUDED,
)

# 合法转移：主推进链 + 任意态可排除。admitted/excluded 为终态。
_TRANSITIONS: dict[str, set[str]] = {
    CANDIDATE: {
        PENDING_EVIDENCE,
        PENDING_SEMANTIC_REVIEW,
        PENDING_RUNTIME,
        PENDING_MODEL,
        ADMITTED,
        EXCLUDED,
    },
    PENDING_EVIDENCE: {
        PENDING_SEMANTIC_REVIEW,
        PENDING_RUNTIME,
        PENDING_MODEL,
        ADMITTED,
        EXCLUDED,
    },
    PENDING_SEMANTIC_REVIEW: {PENDING_RUNTIME, PENDING_MODEL, ADMITTED, EXCLUDED},
    PENDING_RUNTIME: {PENDING_MODEL, ADMITTED, EXCLUDED},
    PENDING_MODEL: {ADMITTED, EXCLUDED},
    ADMITTED: set(),
    EXCLUDED: set(),
}


class StatusError(RuntimeError):
    pass


@dataclass
class HeadStatus:
    """单个样本某个头的状态与问题列表。"""
    head: str
    status: str = CANDIDATE
    issues: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, to: str, *, issue: dict[str, Any] | None = None) -> None:
        """按合法转移表推进。非法转移抛错；excluded 需附原因。"""
        if to not in STATUSES:
            raise StatusError(f"未知状态: {to}")
        if self.status not in _TRANSITIONS:
            raise StatusError(f"未知当前状态: {self.status}")
        if to not in _TRANSITIONS[self.status]:
            raise StatusError(f"非法转移: {self.head} {self.status} -> {to}")
        if to == EXCLUDED and not issue:
            raise StatusError("excluded 必须附替代/隔离原因 issue")
        if issue:
            self.issues.append(issue)
        self.status = to

    def is_terminal(self) -> bool:
        return self.status in (ADMITTED, EXCLUDED)


@dataclass
class SampleHeads:
    """一个样本的五头状态集合。"""
    case_id: str
    heads: dict[str, HeadStatus] = field(default_factory=dict)

    def ensure(self) -> None:
        """补齐五个头的默认 candidate 状态。"""
        for head in HEADS:
            self.heads.setdefault(head, HeadStatus(head=head))

    def admitted(self, head: str) -> bool:
        entry = self.heads.get(head)
        return bool(entry and entry.status == ADMITTED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "heads": {
                head: {"status": entry.status, "issues": entry.issues}
                for head, entry in self.heads.items()
            },
        }
