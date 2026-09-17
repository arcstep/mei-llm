#!/usr/bin/env python3
"""CrossWOZ 适配器：中文多轮对话状态标注 → 统一中间表示。

源格式：zip 内 ``train.json``（顶层 dict，key 为对话 id）。每个对话的 ``messages``
是 ``usr``/``sys`` 交替列表。``usr`` 有 ``content``（用户话术）与 ``dialog_act``
（[intent, domain, slot, value] 四元组）；``sys`` 有 ``content``（系统回复）、
``sys_state_init``（查询前状态）与 ``sys_state``（查询后状态），状态为
``{domain: {slot: value, ..., selectedResults}}``。

映射（调用从对话状态标注**重建**，这是有依据的派生，记录派生规则）：
- usr.content → timeline[user].content
- dialog_act 的 Inform/Request + sys_state_init → 重建调用 → timeline[assistant].tool_calls
- sys_state 的 selectedResults（结果名）→ timeline[tool].content
- sys.content → timeline[assistant].content（回复）
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from evidence import (
    TaskEvidence, Event, Call, Identity, Task, ToolEnv, Behavior,
)
from adapter_base import Adapter, AdapterLimits, AdapterManifest, register


DOMAIN_SLUGS = {
    "景点": "attraction", "旅游景点": "attraction", "餐馆": "restaurant", "餐厅": "restaurant",
    "酒店": "hotel", "地铁": "metro", "出租": "taxi", "医院": "hospital", "天气": "weather",
    "汽车": "car", "火车": "train", "电影": "movie", "电脑": "computer", "电视剧": "tv_series",
    "辅导班": "tutoring", "飞机": "flight",
}

DEFAULT_DOMAINS = ["景点", "餐馆", "酒店", "地铁", "出租"]


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _domain_catalog(prefix: str, domains: list[str]) -> list[dict[str, Any]]:
    result = []
    for domain in sorted(domains):
        name = prefix + DOMAIN_SLUGS.get(domain, re.sub(r"[^a-zA-Z0-9]+", "_", domain).strip("_"))
        result.append(_tool(name, f"查询{domain}数据库；约束与请求字段来自当前已验证状态。",
            {"constraints": {"type": "object"},
             "requested_fields": {"type": "array", "items": {"type": "string"}},
             "selected_entities": {"type": "array", "items": {"type": "string"}}},
            ["constraints", "requested_fields"]))
    return result


def _reconstruct_calls(acts: list[Any], state: dict[str, Any], domains: list[str],
                       tool_by_domain: dict[str, dict[str, Any]]) -> list[Call]:
    """从 dialog_act + sys_state_init 重建数据库查询调用（有依据的派生）。"""
    active: list[str] = []
    for act in acts:
        if len(act) >= 2 and act[1] in domains and act[1] not in active:
            active.append(act[1])
    calls: list[Call] = []
    for domain in active:
        dstate = state.get(domain, {}) or {}
        constraints = {k: v for k, v in dstate.items() if k != "selectedResults" and v not in (None, "", [])}
        for act in acts:
            if len(act) >= 4 and act[0] == "Inform" and act[1] == domain and act[2] and act[3] not in (None, "", "none"):
                constraints[act[2]] = act[3]
        requested = [a[2] for a in acts if len(a) >= 4 and a[0] == "Request" and a[1] == domain and a[2]]
        arguments: dict[str, Any] = {"constraints": constraints, "requested_fields": requested}
        if dstate.get("selectedResults"):
            arguments["selected_entities"] = dstate["selectedResults"]
        calls.append(Call(name=tool_by_domain[domain]["name"], arguments=arguments))
    return calls


@register("crosswoz")
class CrosswozAdapter:
    source_id = "crosswoz"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """raw_path 为 zip 文件；member 名 train.json。按对话 id 顺序取前 N 条。"""
        evidences: list[TaskEvidence] = []
        with zipfile.ZipFile(raw_path) as archive:
            member = "train.json" if "train.json" in archive.namelist() else archive.namelist()[0]
            with archive.open(member) as stream:
                raw = json.load(stream)
        domains = DEFAULT_DOMAINS
        catalog = _domain_catalog("public_crosswoz_", domains)
        tool_by_domain = dict(zip(sorted(domains), catalog))
        for group_id in sorted(raw):
            if len(evidences) >= limits.max_records:
                break
            evidence = self._to_evidence(group_id, raw[group_id], domains, tool_by_domain)
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _to_evidence(self, group_id: str, row: dict[str, Any], domains: list[str],
                     tool_by_domain: dict[str, dict[str, Any]]) -> TaskEvidence | None:
        messages = row.get("messages", [])
        timeline: list[Event] = []
        original_calls: list[Call] = []
        turns: list[dict[str, Any]] = []

        for index in range(0, len(messages) - 1, 2):
            user, system = messages[index], messages[index + 1]
            if user.get("role") != "usr" or system.get("role") != "sys":
                continue
            acts = user.get("dialog_act", [])
            state_init = system.get("sys_state_init", {}) or {}
            state_after = system.get("sys_state", {}) or {}

            calls = _reconstruct_calls(acts, state_init, domains, tool_by_domain)

            selected = {d: (state_after.get(d, {}) or {}).get("selectedResults", []) for d in {a[1] for a in acts if len(a) >= 2}}
            selected = {k: v for k, v in selected.items() if v}
            relaxed = False
            for domain in {a[1] for a in acts if len(a) >= 2 and a[1] in domains}:
                before = {k: v for k, v in (state_init.get(domain, {}) or {}).items() if k != "selectedResults" and v not in (None, "", [])}
                after = {k: v for k, v in (state_after.get(domain, {}) or {}).items() if k != "selectedResults" and v not in (None, "", [])}
                if before != after:
                    relaxed = True
            is_bye = any(len(a) > 0 and a[0] == "General" and len(a) > 1 and a[1] == "bye" for a in acts)

            base = f"{group_id}:{index // 2}"
            timeline.append(Event(event_id=f"{base}:user", role="user",
                                  content=user.get("content", ""), visible_at=f"{base}:user", provider="source"))
            if calls:
                timeline.append(Event(event_id=f"{base}:call", role="assistant",
                                      tool_calls=calls, visible_at=f"{base}:call", provider="source"))
            if selected:
                timeline.append(Event(event_id=f"{base}:result", role="tool",
                                      content=json.dumps(selected, ensure_ascii=False), visible_at=f"{base}:result", provider="source"))
            if system.get("content"):
                timeline.append(Event(event_id=f"{base}:reply", role="assistant",
                                      content=system.get("content", ""), visible_at=f"{base}:reply", provider="source"))

            turn: dict[str, Any] = {"turn_index": index // 2, "action": None, "reason": None}
            if calls:
                original_calls.extend(calls)
                if relaxed:
                    turn.update(action="review", reason="source_relaxed_constraints",
                                source_kind="crosswoz_dialogue_state_annotation")
                else:
                    turn.update(action="execute", reason="annotated_database_query",
                                source_kind="crosswoz_dialogue_state_annotation")
            elif is_bye:
                turn.update(action="complete", reason="annotated_bye",
                            source_kind="crosswoz_dialogue_state_annotation")
            turns.append(turn)

        if not timeline:
            return None

        query = next((e.content for e in timeline if e.role == "user"), "") or ""
        return TaskEvidence(
            identity=Identity(
                case_id=f"crosswoz:{group_id}",
                source_id="crosswoz",
                source_record_id=str(group_id),
                origin_path=f"train.json@{group_id}",
                lang="zh",
                split="candidate",
                producer_identity="human_dialogue_with_annotations",
            ),
            task=Task(query=query),
            tool_env=ToolEnv(
                catalog=[tool_by_domain[d] for d in sorted(tool_by_domain)],
                candidate_construction="oracle",
                description_lang="zh",
            ),
            timeline=timeline,
            behavior=Behavior(
                original_calls=original_calls,
                source_disposition={"multi_turn": True, "turns": turns},
            ),
        )

    def manifest(self) -> AdapterManifest:
        return AdapterManifest(
            source_id="crosswoz",
            source_format="CrossWOZ：zip 内 train.json（dict，key=对话 id），messages 为 usr/sys 交替，含 dialog_act/sys_state_init/sys_state",
            field_mapping={
                "usr.content": "timeline[user].content",
                "dialog_act(Inform/Request) + sys_state_init": "重建调用 → timeline[assistant].tool_calls（public_crosswoz_* 域名归一化）",
                "sys_state.selectedResults（结果名）": "timeline[tool].content",
                "sys.content": "timeline[assistant].content（回复）",
                "sys_state_init vs sys_state 约束差异（relaxed）": "behavior.source_disposition.action=review",
            },
            unrecoverable=[
                "调用从对话状态标注重建，非源数据直接给调用；selectedResults 只有结果名，非完整执行结果",
                "约束放宽（relaxed）不得当静默执行 gold，标 review 待政策决策",
                "domains 用默认五域（景点/餐馆/酒店/地铁/出租）归一化到 public_crosswoz_*",
            ],
            examples=[{"case_id": "crosswoz:391", "tool": "public_crosswoz_attraction"}],
            counterexamples=[
                "relaxed（系统改了约束再选结果）→ disposition=review，不做静默 execute",
                "无 dialog_act 的轮不重建调用（不编造）",
            ],
            verifier_version="crosswoz-adapter-v1",
            trial_report="对标 public_sft_multihead_trial.py 的 _crosswoz_groups（_domain_catalog/调用重建/disposition 分支）",
        )
