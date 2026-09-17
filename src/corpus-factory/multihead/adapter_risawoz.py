#!/usr/bin/env python3
"""RiSAWOZ 适配器：中文任务导向对话（belief state + DB 结果 + 系统动作）→ 统一中间表示。

源格式：zip 内 ``source-data/all_train10000.json``（顶层 list，每条一个对话）。每个对话
``dialogue`` 是 turn 列表；每个 turn 有 ``user_utterance``（用户话术）、``turn_domain``、
``belief_state``（``inform slot-values`` 槽位约束 + ``turn request`` 请求字段）、
``db_results``（数据库结果实体）、``system_actions``（系统动作列表）、``system_utterance``。

映射（调用从 belief state + DB 结果关联重建，有依据的派生）：
- user_utterance → timeline[user].content
- belief_state(domain- 前缀剥壳) + db_results 非空 → 重建调用 → timeline[assistant].tool_calls
- db_results → timeline[tool].content
- system_utterance → timeline[assistant].content（回复）
- system_actions 的 NoOffer/Request/Select/Bye → behavior.source_disposition.action
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


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _domain_catalog(prefix: str, domains: list[str]) -> list[dict[str, Any]]:
    result = []
    for domain in sorted(domains):
        slug = DOMAIN_SLUGS.get(domain, re.sub(r"[^a-zA-Z0-9]+", "_", domain).strip("_"))
        result.append(_tool(prefix + slug, f"查询{domain}数据库；约束与请求字段来自当前已验证状态。",
            {"constraints": {"type": "object"},
             "requested_fields": {"type": "array", "items": {"type": "string"}}},
            ["constraints", "requested_fields"]))
    return result


def _collect_domains(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({domain for row in rows for domain in row.get("domains", [])})


def _strip_domain_prefix(key: str, domain: str) -> str:
    prefix = domain + "-"
    return key[len(prefix):] if key.startswith(prefix) else key


@register("risawoz")
class RisAwozAdapter:
    source_id = "risawoz"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """raw_path 为 zip；member 名 source-data/all_train10000.json。按顺序取前 N 条。"""
        with zipfile.ZipFile(raw_path) as archive:
            members = archive.namelist()
            member = next((m for m in members if m.endswith(".json")), members[0])
            with archive.open(member) as stream:
                rows = json.load(stream)
        domains = _collect_domains(rows)
        catalog = _domain_catalog("public_risawoz_", domains)
        tool_by_domain = dict(zip(sorted(domains), catalog))

        evidences: list[TaskEvidence] = []
        for row in rows:
            if len(evidences) >= limits.max_records:
                break
            evidence = self._to_evidence(row, domains, tool_by_domain)
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any], domains: list[str],
                     tool_by_domain: dict[str, dict[str, Any]]) -> TaskEvidence | None:
        dialogue_id = row.get("dialogue_id", "?")
        timeline: list[Event] = []
        original_calls: list[Call] = []
        turns: list[dict[str, Any]] = []

        for turn in row.get("dialogue", []):
            turn_id = turn.get("turn_id")
            active = [d for d in turn.get("turn_domain", []) if d in tool_by_domain]
            belief = turn.get("belief_state", {}) or {}
            constraints = belief.get("inform slot-values", {}) or {}
            requested = belief.get("turn request", []) or []
            results = turn.get("db_results", []) or []
            actions = turn.get("system_actions", []) or []

            calls: list[Call] = []
            if results:
                for domain in active:
                    prefix = domain + "-"
                    domain_constraints = {_strip_domain_prefix(k, domain): v
                                          for k, v in constraints.items() if k.startswith(prefix)}
                    calls.append(Call(name=tool_by_domain[domain]["name"],
                                      arguments={"constraints": domain_constraints,
                                                 "requested_fields": list(requested)}))

            no_offer = any(a and a[0] == "NoOffer" for a in actions)
            asks = any(a and a[0] in ("Request", "Select") for a in actions)
            bye = any(a and a[0] == "Bye" for a in actions)

            base = f"{dialogue_id}:{turn_id}"
            timeline.append(Event(event_id=f"{base}:user", role="user",
                                  content=turn.get("user_utterance", ""), visible_at=f"{base}:user", provider="source"))
            if calls:
                timeline.append(Event(event_id=f"{base}:call", role="assistant",
                                      tool_calls=calls, visible_at=f"{base}:call", provider="source"))
            if results:
                timeline.append(Event(event_id=f"{base}:result", role="tool",
                                      content=json.dumps(results, ensure_ascii=False), visible_at=f"{base}:result", provider="source"))
            if turn.get("system_utterance"):
                timeline.append(Event(event_id=f"{base}:reply", role="assistant",
                                      content=turn.get("system_utterance", ""), visible_at=f"{base}:reply", provider="source"))

            turn_rec: dict[str, Any] = {"turn_index": turn_id, "action": None, "reason": None}
            if calls:
                original_calls.extend(calls)
                if no_offer:
                    turn_rec.update(action="review", reason="published_no_offer",
                                    source_kind="risawoz_belief_state_annotation")
                else:
                    turn_rec.update(action="execute", reason="annotated_database_query",
                                    source_kind="risawoz_belief_state_annotation")
            elif asks:
                turn_rec.update(action="clarify", reason="published_system_request",
                                source_kind="risawoz_belief_state_annotation")
            elif bye:
                turn_rec.update(action="complete", reason="published_bye",
                                source_kind="risawoz_belief_state_annotation")
            turns.append(turn_rec)

        if not timeline:
            return None

        query = next((e.content for e in timeline if e.role == "user"), "") or ""
        return TaskEvidence(
            identity=Identity(
                case_id=f"risawoz:{dialogue_id}",
                source_id="risawoz",
                source_record_id=str(dialogue_id),
                origin_path=f"all_train10000.json@{dialogue_id}",
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
            source_id="risawoz",
            source_format="RiSAWOZ：zip 内 source-data/all_train10000.json（list），dialogue 为 turn 列表，含 belief_state/db_results/system_actions",
            field_mapping={
                "user_utterance": "timeline[user].content",
                "belief_state(inform slot-values, domain- 前缀剥壳) + db_results 非空": "重建调用 → timeline[assistant].tool_calls（public_risawoz_*）",
                "db_results": "timeline[tool].content",
                "system_utterance": "timeline[assistant].content（回复）",
                "system_actions NoOffer → review / Request·Select → clarify / Bye → complete / 有调用 → execute": "behavior.source_disposition.action",
            },
            unrecoverable=[
                "调用仅在 db_results 非空时重建（无结果的轮不编造调用）",
                "domains 从全体对话 domains 字段去重排序，归一化到 public_risawoz_*",
                "db_results 保留原文 json 序列化（实体列表，非 selected 结果名）",
            ],
            examples=[{"case_id": "risawoz:0", "tool": "public_risawoz_restaurant"}],
            counterexamples=[
                "无 db_results 的轮不重建调用，asks → clarify、bye → complete、否则 unlabelled（无 action，不伪造 execute）",
                "NoOffer（系统无结果）→ disposition=review，不做静默 execute gold",
            ],
            verifier_version="risawoz-adapter-v1",
            trial_report="对标 public_sft_multihead_trial.py 的 _risawoz_groups（domain 前缀剥壳/仅结果轮重建/NoOffer-Request-Bye 处置分支）",
        )
