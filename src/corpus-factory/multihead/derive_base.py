#!/usr/bin/env python3
"""五头派生协议与视图契约。

派生器从统一中间表示（TaskEvidence）派生该头的视图，不做语义审核。输入构造器
（serializer/tokenizer/runtime/scorer）由 compile/freeze 阶段绑定，不在派生阶段产生。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evidence import TaskEvidence


@dataclass
class HeadView:
    """一个「样本×头」派生视图。target 用可 JSON 序列化的 Python 对象。"""
    view_id: str
    case_id: str
    head: str                    # retrieval / tool_lm / disposition / confidence / narration
    target: Any = None           # 该头目标（调用 batch / 工具集合 / 动作码 / 解说文本）
    label_mask: dict[str, Any] = field(default_factory=dict)  # 监督 mask（与 token loss mask 分开）
    status: str = "candidate"    # 见 status.STATUSES
    evidence: dict[str, Any] = field(default_factory=dict)    # 标签证据引用

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "case_id": self.case_id,
            "head": self.head,
            "target": self.target,
            "label_mask": self.label_mask,
            "status": self.status,
            "evidence": self.evidence,
        }


@runtime_checkable
class HeadDeriver(Protocol):
    """五头派生器协议：从统一中间表示派生该头视图。"""
    head: str

    def derive(self, evidence: TaskEvidence) -> list[HeadView]:
        """从 TaskEvidence 派生该头的视图。不做语义审核，不伪造缺失标签。"""
        ...


def leaves(value: Any, path: tuple = ()) -> list[tuple[tuple, Any]]:
    """遍历嵌套 dict/list 的叶子值，返回 (路径, 值) 列表。"""
    if isinstance(value, dict):
        out: list[tuple[tuple, Any]] = []
        for k, v in value.items():
            out.extend(leaves(v, path + (k,)))
        return out
    if isinstance(value, list):
        out = []
        for i, v in enumerate(value):
            out.extend(leaves(v, path + (i,)))
        return out
    return [(path, value)]
