#!/usr/bin/env python3
"""ToolACE 适配器：函数调用对话 → 统一中间表示。

源格式：本地 JSON 数组，每个记录含 ``system``（系统提示词，尾部内嵌工具定义的
JSON 数组）与 ``conversations``（``[{from, value}, ...]``，from 为
user / assistant / tool）。工具名可含空格（如 ``Market Trends API``）。

映射（忠实，不做五头派生、不做语义审核）：
- conversations[from=user]      → timeline[user].content
- conversations[from=assistant] → timeline[assistant]（value 以 ``[`` 开头则解析为 tool_calls）
- conversations[from=tool]      → timeline[tool].content（保留原始结果文本）
"""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from evidence import (
    TaskEvidence, Event, Call, Identity, Task, ToolEnv, Behavior, Review,
)
from adapter_base import Adapter, AdapterLimits, AdapterManifest, register


def map_schema_types(schema: Any) -> Any:
    """类型别名映射（dict→object、float→number 等），递归到子字段。"""
    result = deepcopy(schema)
    if not isinstance(result, dict):
        return result
    aliases = {"dict": "object", "float": "number", "int": "integer",
               "bool": "boolean", "list": "array"}
    t = result.get("type")
    if isinstance(t, str):
        result["type"] = aliases.get(t, t)
    if isinstance(result.get("properties"), dict):
        result["properties"] = {k: map_schema_types(v) for k, v in result["properties"].items()}
    if isinstance(result.get("items"), dict):
        result["items"] = map_schema_types(result["items"])
    return result


def _literal(node: ast.AST) -> Any:
    """受限字面量读取器：只读常量/容器，绝不执行来源代码。"""
    if isinstance(node, ast.Dict):
        keys = [_literal(k) for k in node.keys]
        if any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("nonstring or duplicate object keys")
        return dict(zip(keys, (_literal(v) for v in node.values)))
    if isinstance(node, ast.List):
        return [_literal(v) for v in node.elts]
    if isinstance(node, ast.Constant) and (
        node.value is None or type(node.value) in (str, int, float, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
            and isinstance(node.operand, ast.Constant) \
            and type(node.operand.value) in (int, float):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    raise ValueError("nonliteral argument")


def parse_literal_tool_calls(text: str) -> list[dict[str, Any]]:
    """限制语法读取器：只读 ``[func(a=..., b=...)]`` 字面调用列表。"""
    tree = ast.parse(text, mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("not a call list")
    calls: list[dict[str, Any]] = []
    for item in tree.elts:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
            raise ValueError("non-simple named call")
        keys = [k.arg for k in item.keywords]
        if None in keys or len(set(keys)) != len(keys):
            raise ValueError("spread or duplicate keyword")
        calls.append({"name": item.func.id, "arguments": dict(zip(keys, (_literal(k.value) for k in item.keywords)))})
    return calls


def parse_declared_tool_calls(text: str, names: list[str]) -> list[Call]:
    """读取含空格的声明工具名，不改写 quoted 参数文本。"""
    value = text.strip()
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("call list required")
    body = value[1:-1]
    position = 0
    calls: list[Call] = []
    while position < len(body):
        while position < len(body) and body[position].isspace():
            position += 1
        if position == len(body):
            break
        hits = [n for n in names if isinstance(n, str) and body.startswith(n, position)
                and body[position + len(n):].lstrip().startswith("(")]
        if not hits:
            raise ValueError("undeclared function name")
        name = max(hits, key=len)
        start = position + len(name)
        while body[start].isspace():
            start += 1
        depth = 0
        quote = None
        escaped = False
        end = None
        for i in range(start, len(body)):
            char = body[i]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ('"', "'"):
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("unclosed function call")
        parsed = parse_literal_tool_calls("[f" + body[start:end] + "]")[0]
        parsed["name"] = name
        calls.append(Call(name=name, arguments=parsed["arguments"]))
        position = end
        while position < len(body) and body[position].isspace():
            position += 1
        if position < len(body):
            if body[position] != ",":
                raise ValueError("unexpected call suffix")
            position += 1
            if not body[position:].strip():
                raise ValueError("trailing call delimiter")
    return calls


def toolace_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    """从 system 提示词内嵌的 JSON 数组提取工具 catalog。"""
    system = row["system"]
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in system:
        raise ValueError("alternate_catalog_format_needs_adapter")
    start = system.index("[", system.index(marker) + len(marker))
    raw, _ = json.JSONDecoder().raw_decode(system, start)
    tools = [{"name": t["name"], "description": t.get("description", ""),
              "parameters": map_schema_types(t["parameters"])} for t in raw]
    if len({t["name"] for t in tools}) != len(tools):
        raise ValueError("duplicate_tool_name")
    return tools


@register("toolace")
class ToolaceAdapter:
    source_id = "toolace"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        evidences: list[TaskEvidence] = []
        for index, row in enumerate(rows):
            if len(evidences) >= limits.max_records:
                break
            evidence = self._to_evidence(row, index=index)
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any], *, index: int) -> TaskEvidence | None:
        try:
            tools = toolace_tools(row)
        except (ValueError, KeyError, TypeError):
            return None
        names = [t["name"] for t in tools]
        timeline: list[Event] = []
        for i, turn in enumerate(row.get("conversations", [])):
            role = turn.get("from")
            value = turn.get("value") or ""
            if role not in ("user", "assistant", "tool"):
                continue
            event = Event(
                event_id=f"{index}:{i}",
                role=role,
                content=value,
                visible_at=str(i),
                provider="source",
            )
            if role == "assistant" and value.lstrip().startswith("["):
                try:
                    event.tool_calls = parse_declared_tool_calls(value, names)
                except (ValueError, SyntaxError, TypeError, RecursionError):
                    event.tool_calls = []  # 保留原文，解析失败不丢
            timeline.append(event)

        if not timeline:
            return None

        original_calls = [c for e in timeline if e.role == "assistant" for c in e.tool_calls]
        query = next((e.content for e in timeline if e.role == "user"), "") or ""
        lang = "zh_or_mixed" if any("㐀" <= ch <= "鿿" for ch in query) else "en_or_other"

        return TaskEvidence(
            identity=Identity(
                case_id=f"toolace:{index}",
                source_id="toolace",
                source_record_id=str(index),
                origin_path=f"row@{index}",
                lang=lang,
                split="candidate",
                producer_identity="publisher_synthetic",
            ),
            task=Task(query=query),
            tool_env=ToolEnv(
                catalog=tools,
                candidate_construction="oracle",
                description_lang="en",
            ),
            timeline=timeline,
            behavior=Behavior(original_calls=original_calls),
        )

    def manifest(self) -> AdapterManifest:
        return AdapterManifest(
            source_id="toolace",
            source_format="ToolACE：JSON 数组，每记录含 system（内嵌工具定义 JSON 数组）与 conversations（[{from, value}]）",
            field_mapping={
                "system 内嵌工具定义": "tool_env.catalog（map_schema_types 归一化类型别名）",
                "conversations[from=user]": "timeline[user].content",
                "conversations[from=assistant]（value 以 [ 开头）": "timeline[assistant].tool_calls",
                "conversations[from=assistant]（其余）": "timeline[assistant].content",
                "conversations[from=tool]": "timeline[tool].content（保留原始结果文本）",
            },
            unrecoverable=[
                "工具返回结果的结构化解析（call_id↔result 关联）留待 derive 阶段",
                "未来结果与调用前的可见性边界由 derive 阶段按 timeline 顺序判定",
                "assistant 调用解析失败时 tool_calls 置空，原文保留在 event.content",
            ],
            examples=[{"case_id": "toolace:0", "tool": "Market Trends API"}],
            counterexamples=[
                "工具名含空格（Market Trends API）需 parse_declared_tool_calls 按声明名匹配",
                "多调用（SEC Filings + United States Away from Home Mobility API）保留原批，不做无据拆分",
            ],
            verifier_version="toolace-adapter-v1",
            trial_report="对标 public_sft_scale_trial.py 的 ToolACE 分支（toolace_tools/events_for）与 local_diagnostics.parse_declared_tool_calls",
        )
