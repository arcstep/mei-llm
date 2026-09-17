#!/usr/bin/env python3
"""Nemotron interactive-agent 适配器：policy 驱动客服对话 → 统一中间表示。

源格式：jsonl，每条一个对话。顶层键 ``uuid``/``messages``/``license``/``tools``。
``messages[0]`` 是 system，``content`` 在 ``<policy>``/``</policy>`` 标签之间给出
行为约束全文；后续是 user/assistant/tool 交替。``tools`` 是 OpenAI 函数格式
（``{"type":"function","function":{"name","description","parameters"}}``）。
``assistant`` 的 ``tool_calls`` 也是 OpenAI 格式，``function.arguments`` 是 JSON 字符串。

映射（忠实；调用是源数据显式给出，非重建）：
- <policy> 正文 → behavior.source_policy_text（处置监督输入依据）
- tools → tool_env.catalog（name 归一化 public_nemotron_*）
- user/assistant/tool 事件 → timeline（assistant.tool_calls 的 arguments JSON 解析为 dict）
- 每轮处置 → behavior.source_disposition.turns（有调用=execute；无调用回复=unlabelled，禁止默认值）

处置原则（禁止默认值）：无调用的 assistant 回复**不推断** action——它可能是
clarify/refuse/complete/wait，需要与 policy 条款对齐后由语义审核判定，宁缺毋滥。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evidence import (
    TaskEvidence, Event, Call, Identity, Task, ToolEnv, Behavior,
)
from adapter_base import Adapter, AdapterLimits, AdapterManifest, register

POLICY_RE = re.compile(r"<policy>\s*\n(.*?)\n\s*</policy>", re.S)


def _alias(name: str) -> str:
    return "public_nemotron_" + re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()


def _extract_policy(system_content: str) -> str:
    # 说明文字里有 "between the <policy> and </policy> tags" 的示例标签（同一行），
    # 真正的 policy 标签独占一行：用 <policy> 后紧跟换行锚定，避开示例。
    match = POLICY_RE.search(system_content or "")
    return match.group(1).strip() if match else ""


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    """OpenAI function.arguments 是 JSON 字符串；解析为 dict，失败返回 None（不编造）。"""
    if raw is None or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_tool_calls(tool_calls: Any, alias: dict[str, str]) -> list[Call]:
    calls: list[Call] = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue
        args = _parse_arguments(fn.get("arguments", ""))
        if args is None:
            continue
        calls.append(Call(name=alias.get(name, name), arguments=args))
    return calls


def _to_catalog(tools: Any, alias: dict[str, str]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for item in tools or []:
        fn = item.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue
        catalog.append({
            "name": alias.get(name, name),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return catalog


@register("nemotron")
class NemotronAdapter:
    source_id = "nemotron"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """raw_path 为 jsonl；逐行读，取前 N 条。"""
        evidences: list[TaskEvidence] = []
        with raw_path.open(encoding="utf-8") as stream:
            for line in stream:
                if len(evidences) >= limits.max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evidence = self._to_evidence(row)
                if evidence is not None:
                    evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any]) -> TaskEvidence | None:
        messages = row.get("messages", [])
        if not messages:
            return None

        system_content = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
        policy_text = _extract_policy(system_content)

        tools = row.get("tools", [])
        alias = {fn.get("name", ""): _alias(fn.get("name", ""))
                 for item in tools for fn in [item.get("function", {})] if fn.get("name")}
        catalog = _to_catalog(tools, alias)

        uuid = row.get("uuid", "?")
        timeline: list[Event] = []
        original_calls: list[Call] = []
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "system":
                continue
            if role == "user":
                if current is not None:
                    turns.append(current)
                current = {"turn_index": len(turns), "action": None, "reason": None, "reply_text": None}
                timeline.append(Event(
                    event_id=f"{uuid}:{idx}:user", role="user",
                    content=msg.get("content", "") or "", visible_at=f"{uuid}:{idx}", provider="source",
                ))
            elif role == "assistant":
                tool_calls = _parse_tool_calls(msg.get("tool_calls"), alias)
                content = (msg.get("content") or "").strip()
                if tool_calls:
                    original_calls.extend(tool_calls)
                    timeline.append(Event(
                        event_id=f"{uuid}:{idx}:call", role="assistant",
                        tool_calls=tool_calls, visible_at=f"{uuid}:{idx}", provider="source",
                    ))
                    if current is not None:
                        current["action"] = "execute"
                        current["reason"] = f"published_tool_call:{tool_calls[0].name}"
                elif content:
                    timeline.append(Event(
                        event_id=f"{uuid}:{idx}:reply", role="assistant",
                        content=content, visible_at=f"{uuid}:{idx}", provider="source",
                    ))
                    if current is not None:
                        # 无调用回复：禁止默认值，不推断 action，仅记录原文供语义审核对齐 policy。
                        current["reply_text"] = content
            elif role == "tool":
                timeline.append(Event(
                    event_id=f"{uuid}:{idx}:result", role="tool",
                    content=msg.get("content", "") or "", visible_at=f"{uuid}:{idx}", provider="source",
                ))
        if current is not None:
            turns.append(current)

        if not timeline:
            return None

        first_user = next((e for e in timeline if e.role == "user"), None)
        query = first_user.content if first_user else ""

        source_disposition: dict[str, Any] = {"multi_turn": True, "turns": turns}

        return TaskEvidence(
            identity=Identity(
                case_id=f"nemotron:{uuid}",
                source_id="nemotron",
                source_record_id=str(uuid),
                origin_path=f"data__interactive_agent.jsonl@{uuid}",
                lang="en",
                license=row.get("license", ""),
                split="candidate",
                producer_identity="publisher_synthetic",
            ),
            task=Task(query=query),
            tool_env=ToolEnv(
                catalog=catalog,
                candidate_construction="oracle",
                description_lang="en",
            ),
            timeline=timeline,
            behavior=Behavior(
                original_calls=original_calls,
                source_disposition=source_disposition,
                source_policy_text=policy_text,
            ),
        )

    def manifest(self) -> AdapterManifest:
        return AdapterManifest(
            source_id="nemotron",
            source_format="Nemotron interactive-agent：jsonl，messages[0] system 含 <policy> 标签，tools/tool_calls 为 OpenAI 函数格式",
            field_mapping={
                "<policy> 正文": "behavior.source_policy_text（处置监督输入依据）",
                "tools（OpenAI 函数格式）": "tool_env.catalog（name 归一化 public_nemotron_*）",
                "user/assistant/tool 事件": "timeline（assistant.tool_calls 的 function.arguments JSON 解析为 dict）",
                "assistant 有 tool_calls → execute / 无调用回复 → unlabelled": "behavior.source_disposition.turns（每轮一项，禁止默认值）",
            },
            unrecoverable=[
                "function.arguments 是 JSON 字符串，解析失败（非 dict）的调用跳过不编造",
                "无调用回复不推断 action（可能是 clarify/refuse/complete/wait），保留 reply_text 待语义审核对齐 policy",
                "纯 reasoning 的 assistant（content 空且无 tool_calls）不生成事件",
            ],
            examples=[{"case_id": "nemotron:ff6ab2b0", "tool": "public_nemotron_authenticate_user"}],
            counterexamples=[
                "无调用回复 ≠ refuse ≠ complete：不设 action，宁缺毋滥",
                "arguments JSON 解析失败的调用丢弃，不保留半解析参数",
            ],
            verifier_version="nemotron-adapter-v1",
            trial_report="全新 policy→处置证据挖掘（policy span 提取 + 轮级多标签 + 禁止默认值）；调用为源数据显式给出",
        )
