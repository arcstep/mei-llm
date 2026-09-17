#!/usr/bin/env python3
"""MSAgentBench 适配器：ModelScope 插件 agent 轨迹 → 统一中间表示。

源格式：``dev.jsonl``（每行 ``{id, conversations}``）。``conversations`` 是
``[{from, value}]`` 列表，``from`` 取 system/user/assistant。

- system value：嵌入 ModelScope 插件声明的中文提示，形如
  ``1. {"plugin_name": ..., "plugin_schema_for_model": {...}}\\n\\n2. ...``。
  plugin_schema_for_model 含 ``description`` 与 ``paths[].parameters[]``
  （参数只有 name/description/required，**无 type**）。
- user value：用户话术。
- assistant value：自包含 agent 轨迹，含三段（可能有多个 think/exec 段）：
  1. ``<|startofthink|>```JSON\\n{"api_name","url","parameters"}\\n````<|endofthink|>`` → 工具调用
  2. ``<|startofexec|>```JSON\\n{结果}\\n````<|endofexec|>`` → 工具结果
  3. 最后一个 ``<|endofexec|>`` 之后的纯文本 → 最终回复

映射（调用从 think 段直接取，有依据，非重建）：
- plugin_name（全局去重）→ tool catalog name=plugin_name
- paths[0].parameters[{name,description,required}] → OpenAI properties（type 补 string，
  required "True"/"False" 字符串 → bool）
- think JSON 的 api_name → timeline[assistant].tool_calls[].name
- think JSON 的 parameters → timeline[assistant].tool_calls[].arguments
- exec JSON → timeline[tool].content（保留原文）
- user value → timeline[user].content
- reply 文本 → timeline[assistant].content

归一化派生规则（有依据）：
1. 参数补 type=string：ModelScope 插件格式无 type，实测全部参数值均为 string
   （1018/1018 无嵌套），byte_grammar 平铺只支持 scalar。
2. url 忽略：同一 plugin_name 有多个 url 部署实例（同工具冗余副本），工具标识按
   plugin_name 归一，调用里的 url 不进入 arguments。
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

_THINK_RE = re.compile(r"<\|startofthink\|>```JSON\n(.*?)\n```<\|endofthink\|>", re.DOTALL)
_EXEC_RE = re.compile(r"<\|startofexec\|>```JSON\n(.*?)\n```<\|endofexec\|>", re.DOTALL)


def _extract_plugin_jsons(text: str) -> list[dict[str, Any]]:
    """从 system value 抽取所有 ``{"plugin_name": ...}`` 顶级 JSON 对象（括号匹配）。"""
    out: list[dict[str, Any]] = []
    i = 0
    while True:
        i = text.find('{"plugin_name"', i)
        if i == -1:
            break
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < len(text):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1
        try:
            out.append(json.loads(text[i:j]))
        except (json.JSONDecodeError, ValueError):
            pass
        i = j
    return out


def _model_scope_to_tool(plug: dict[str, Any]) -> dict[str, Any]:
    """ModelScope 插件声明 → OpenAI function 工具（参数 type 补 string）。"""
    name = plug.get("plugin_name") or plug.get("plugin_schema_for_model", {}).get("name")
    sm = plug.get("plugin_schema_for_model", {}) or {}
    paths = sm.get("paths", []) or []
    params = paths[0].get("parameters", []) if paths else []
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pp in params:
        pname = pp.get("name")
        if not pname:
            continue
        properties[pname] = {"type": "string", "description": pp.get("description", "") or ""}
        if str(pp.get("required", "")).strip().lower() == "true":
            required.append(pname)
    return {
        "name": name,
        "description": sm.get("description", "") or plug.get("description", "") or name,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _reply_text(val: str) -> str:
    """最后一个 <|endofexec|> 之后的纯文本作为最终回复。"""
    parts = val.split("<|endofexec|>")
    return parts[-1].strip() if len(parts) > 1 else ""


@register("msagent_bench")
class MsAgentBenchAdapter:
    source_id = "msagent_bench"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """raw_path 为 dev.jsonl；全局收集插件 schema（9 工具），按行序取前 N 条。"""
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        catalog: dict[str, dict[str, Any]] = {}
        for row in rows:
            for turn in row.get("conversations", []):
                if turn.get("from") != "system":
                    continue
                for plug in _extract_plugin_jsons(turn.get("value") or ""):
                    tool = _model_scope_to_tool(plug)
                    if tool["name"] and tool["name"] not in catalog:
                        catalog[tool["name"]] = tool
        tools = [catalog[name] for name in sorted(catalog)]

        evidences: list[TaskEvidence] = []
        for row in rows:
            if len(evidences) >= limits.max_records:
                break
            evidence = self._to_evidence(row, tools)
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any], tools: list[dict[str, Any]]) -> TaskEvidence | None:
        row_id = row.get("id", "?")
        timeline: list[Event] = []
        original_calls: list[Call] = []
        turns: list[dict[str, Any]] = []

        for turn_idx, turn in enumerate(row.get("conversations", [])):
            frm = turn.get("from")
            val = turn.get("value") or ""
            base = f"{row_id}:{turn_idx}"
            if frm == "user":
                timeline.append(Event(event_id=f"{base}:user", role="user",
                                      content=val, visible_at=f"{base}:user", provider="source"))
            elif frm == "assistant":
                thinks = []
                for m in _THINK_RE.finditer(val):
                    try:
                        obj = json.loads(m.group(1))
                    except (json.JSONDecodeError, ValueError):
                        continue
                    thinks.append(Call(name=obj.get("api_name", ""),
                                       arguments=obj.get("parameters", {}) or {}))
                execs = [m.group(1).strip() for m in _EXEC_RE.finditer(val)]
                for k, call in enumerate(thinks):
                    if not call.name:
                        continue
                    original_calls.append(call)
                    timeline.append(Event(event_id=f"{base}:call:{k}", role="assistant",
                                          tool_calls=[call], visible_at=f"{base}:call:{k}", provider="source"))
                    if k < len(execs):
                        timeline.append(Event(event_id=f"{base}:result:{k}", role="tool",
                                              content=execs[k], visible_at=f"{base}:result:{k}", provider="source"))
                reply = _reply_text(val)
                if reply:
                    timeline.append(Event(event_id=f"{base}:reply", role="assistant",
                                          content=reply, visible_at=f"{base}:reply", provider="source"))

            action = None
            reason = None
            if frm == "assistant" and any(e.event_id.startswith(f"{base}:call") for e in timeline):
                action, reason = "execute", "annotated_agent_tool_call"
            turns.append({"turn_index": turn_idx, "action": action, "reason": reason})

        if not timeline:
            return None

        query = next((e.content for e in timeline if e.role == "user"), "") or ""
        return TaskEvidence(
            identity=Identity(
                case_id=f"msagent_bench:{row_id}",
                source_id="msagent_bench",
                source_record_id=str(row_id),
                origin_path=f"dev.jsonl@{row_id}",
                lang="zh",
                split="dev",
                producer_identity="human_agent_trajectory_with_plugin_annotations",
            ),
            task=Task(query=query),
            tool_env=ToolEnv(
                catalog=tools,
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
            source_id="msagent_bench",
            source_format="MSAgentBench：dev.jsonl（{id, conversations:[{from,value}]}），system 嵌 ModelScope 插件 JSON，assistant 含 think/exec 段",
            field_mapping={
                "plugin_name（全局去重）": "tool catalog name",
                "paths[0].parameters[{name,description,required}]": "OpenAI properties（type 补 string）",
                "think JSON api_name/parameters": "timeline[assistant].tool_calls",
                "exec JSON": "timeline[tool].content",
                "user value": "timeline[user].content",
                "reply 文本": "timeline[assistant].content",
            },
            unrecoverable=[
                "url 冗余部署实例忽略，工具标识按 plugin_name 归一",
                "参数 type 隐含 string（ModelScope 无 type，实测全 string）",
            ],
            examples=[{"case_id": "msagent_bench:MS_Agent_Bench_Test_0", "tool": "modelscope_text-address"}],
            counterexamples=[
                "think JSON 解析失败（格式变体）跳过该调用，不编造",
                "无 think 段的 assistant turn 不重建调用",
            ],
            verifier_version="msagent-bench-adapter-v1",
            trial_report="9 工具全部标量参数，1018/1018 参数值为 string，符合约束解码平铺",
        )
