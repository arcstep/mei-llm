#!/usr/bin/env python3
"""Canonical Task Evidence：公开工具调用数据集的统一中间表示。

这是「适配器产出、派生消费」的唯一契约，字段语义对 0405 第 3 节六组信息：
身份与来源、用户任务、工具环境、时点证据、行为与结果、评审与用途。

职责边界：
- 适配器只做「忠实映射到契约」，不做五头派生、不做语义审核。
- 派生器从这里读证据，按 0405 第 4~8 节派生五头视图。
- 未来工具结果不能进入调用前输入；每个派生视图只注入当时已可见的事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# 事件流最小单元
# ---------------------------------------------------------------------------

@dataclass
class Call:
    """一次工具调用（工具名 + 命名参数）。"""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class Result:
    """一次工具调用的返回结果。"""
    call_id: str | None = None
    name: str = ""
    results: Any = None
    status: str | None = None  # success / failure / timeout / ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "results": self.results,
            "status": self.status,
        }


@dataclass
class Event:
    """轨迹中的一个事件。user / assistant / tool 各算一个。"""
    event_id: str
    role: str                    # user / assistant / tool
    content: str | None = None   # 该角色可见文本
    tool_calls: list[Call] = field(default_factory=list)
    tool_results: list[Result] = field(default_factory=list)
    visible_at: str = ""         # 顺序指针，标识该事件在轨迹中的位置
    provider: str = "source"     # user / assistant / system / source
    verified: bool = False       # 源发布不等于 gold 正确

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "role": self.role,
            "content": self.content,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "visible_at": self.visible_at,
            "provider": self.provider,
            "verified": self.verified,
        }


# ---------------------------------------------------------------------------
# 任务真源六组信息
# ---------------------------------------------------------------------------

@dataclass
class Identity:
    """身份与来源。治理字段不进入模型 prompt。"""
    case_id: str = ""
    source_id: str = ""
    source_record_id: str = ""
    family: str = ""
    origin_path: str = ""
    origin_sha256: str = ""
    lang: str = ""
    license: str = ""
    split: str = "candidate"     # train / dev / calibration / locked-test / candidate
    producer_identity: str = ""  # 发布方生成身份（人工 / 合成）


@dataclass
class Task:
    """用户任务。模型应自行理解的内容放目标侧，不提前注入输入。"""
    query: str = ""
    subtasks: list[str] = field(default_factory=list)
    completion_condition: str = ""
    allowed_alternatives: list[str] = field(default_factory=list)
    plan_version: str = ""


@dataclass
class ToolEnv:
    """工具环境：catalog 及 schema、权限副作用、可见候选构造方式、描述语言。"""
    catalog: list[dict[str, Any]] = field(default_factory=list)  # 工具 schema 列表
    permissions_side_effects: list[str] = field(default_factory=list)
    candidate_construction: str = ""  # oracle / learned
    description_lang: str = ""


@dataclass
class Behavior:
    """行为与结果：原始/可接受调用、参数证据、依赖、call_id↔result 关联。"""
    original_calls: list[Call] = field(default_factory=list)
    acceptable_calls: list[list[Call]] = field(default_factory=list)  # 多组可接受
    subtask_assignment: dict[str, Any] = field(default_factory=dict)
    argument_provenance: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    call_to_result: dict[str, str] = field(default_factory=dict)  # call_id -> result_id
    failure_unexecuted: list[str] = field(default_factory=list)


@dataclass
class Review:
    """评审与用途：split 登记、验证结论、审核者方法范围、每头准入状态。"""
    split_registration: str = ""
    validation_conclusions: list[dict[str, Any]] = field(default_factory=list)
    reviewer: str = ""
    method: str = ""
    scope: str = ""
    per_head_admission: dict[str, str] = field(default_factory=dict)
    supersede_isolate_reason: str = ""


@dataclass
class TaskEvidence:
    """任务真源：适配器产出的统一中间表示。"""
    identity: Identity = field(default_factory=Identity)
    task: Task = field(default_factory=Task)
    tool_env: ToolEnv = field(default_factory=ToolEnv)
    timeline: list[Event] = field(default_factory=list)
    behavior: Behavior = field(default_factory=Behavior)
    review: Review = field(default_factory=Review)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        """基本完整性校验，返回问题列表。不替代语义审核。"""
        problems: list[str] = []
        if not self.identity.case_id:
            problems.append("identity.case_id 为空")
        if not self.identity.source_id:
            problems.append("identity.source_id 为空")
        if not self.timeline:
            problems.append("timeline 为空")
        roles = [e.role for e in self.timeline]
        if roles and roles[0] != "user":
            problems.append("timeline 首个事件不是 user")
        seen: set[str] = set()
        for event in self.timeline:
            if not event.event_id:
                problems.append("存在空 event_id")
            if event.event_id in seen:
                problems.append(f"event_id 重复: {event.event_id}")
            seen.add(event.event_id)
        return problems
