#!/usr/bin/env python3
"""API-Bank 适配器：工具定义 + 问题 → API 调用 → 结果 → 回复 → 统一中间表示。

源格式：`training-data__lv{1,2,3}-api-train.json`（工具定义 + 问题 + 调用输出）与
`training-data__lv{1,2,3}-response-train.json`（结果 + 回复）。每个文件是 JSON 数组，
每条记录含 ``instruction``/``input``/``output``（均为字符串）。

``input`` 为换行分隔：先每行一个工具定义 JSON（``apiCode``/``description``/
``parameters``/``response``），再对话轮 ``User: ...``/``AI: ...``，末行
``Generate API Request: ``（api-train）或 ``Generate AI Response: ``（response-train）。
``output`` 为 ``API-Request: [ApiName(key='...')]``（api-train）或 ``AI: <文本>``
（response-train）。结果以 ``API-Request: [Call(...)]->{...}`` 形式嵌在 response 的
input 历史里，结果右段可能是 Python repr（单引号），保留原文不做 json 解析。

映射（忠实，调用别名是有依据的归一化，记录派生规则）：
- input 工具定义 JSON → tool_env.catalog（apiCode 归一化为 public_apibank_*）
- input 的 ``User:`` 后文本 → timeline[user].content（query）
- output 的调用 → timeline[assistant].tool_calls（别名到 public_apibank_*）
- response 配对（按 ``Generate API Request:`` 前的对话前缀精确匹配）的结果 → timeline[tool].content
- response 的 output → timeline[assistant].content（回复）
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from evidence import (
    TaskEvidence, Event, Call, Identity, Task, ToolEnv, Behavior,
)
from adapter_base import Adapter, AdapterLimits, AdapterManifest, register


_TYPE_MAP = {"str": "string", "float": "number", "int": "integer", "bool": "boolean",
             "list": "array", "dict": "object"}


def _leading_json_objects(value: str) -> list[dict[str, Any]]:
    """读 input 开头的工具定义行（每行一个 JSON 对象，含 apiCode）。"""
    result: list[dict[str, Any]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict) and item.get("apiCode"):
            result.append(item)
    return result


def _alias(api_code: str) -> str:
    return "public_apibank_" + re.sub(r"[^a-zA-Z0-9_]+", "_", api_code).strip("_").lower()


def _api_tool(item: dict[str, Any]) -> dict[str, Any]:
    """工具定义 → catalog 项（apiCode 归一化 + 类型别名映射）。"""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in (item.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            spec = {"type": "string", "description": str(spec)}
        entry = dict(spec)
        entry["type"] = _TYPE_MAP.get(entry.get("type"), entry.get("type", "string"))
        if entry.pop("required", False):
            required.append(name)
        properties[name] = entry
    return {
        "name": _alias(item["apiCode"]),
        "description": item.get("description", ""),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _parse_api_calls(value: str) -> list[Call]:
    """解析 ``API-Request: [ApiName(key='...')]`` 形式的调用，不执行来源代码。"""
    text = value.strip()
    if text.startswith("API-Request:"):
        text = text.split(":", 1)[1].strip()
    tree = ast.parse(text, mode="eval").body
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls: list[Call] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("non_literal_call")
        arguments = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("expanded_keyword")
            arguments[kw.arg] = ast.literal_eval(kw.value)
        calls.append(Call(name=node.func.id, arguments=arguments))
    return calls


@register("api_bank")
class ApiBankAdapter:
    source_id = "api_bank"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """raw_path 为 api-train 所在目录；扫描 lv1/2/3 的 api-train + response-train。"""
        evidences: list[TaskEvidence] = []
        for level in (1, 2, 3):
            api_path = raw_path / f"training-data__lv{level}-api-train.json"
            response_path = raw_path / f"training-data__lv{level}-response-train.json"
            if not api_path.exists():
                continue
            evidences.extend(self._load_level(api_path, response_path, level, limits))
            if len(evidences) >= limits.max_records:
                break
        return evidences[:limits.max_records]

    def _load_level(self, api_path: Path, response_path: Path, level: int, limits: AdapterLimits) -> list[TaskEvidence]:
        api_rows = json.loads(api_path.read_text(encoding="utf-8"))
        response_rows = json.loads(response_path.read_text(encoding="utf-8")) if response_path.exists() else []
        response_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in response_rows:
            response_index[row["input"].rsplit("\nAPI-Request:", 1)[0]].append(row)

        evidences: list[TaskEvidence] = []
        for index, row in enumerate(api_rows):
            if len(evidences) >= limits.max_records:
                break
            evidence = self._to_evidence(row, index, level, response_index)
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any], index: int, level: int,
                     response_index: dict[str, list[dict[str, Any]]]) -> TaskEvidence | None:
        definitions = _leading_json_objects(row.get("input", ""))
        if not definitions:
            return None
        try:
            raw_calls = _parse_api_calls(row.get("output", ""))
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None
        if not raw_calls or len(raw_calls) > 5:
            return None
        declared = {item["apiCode"] for item in definitions}
        if any(c.name not in declared for c in raw_calls):
            return None

        tools = [_api_tool(item) for item in definitions]
        alias = {item["apiCode"]: _alias(item["apiCode"]) for item in definitions}
        calls = [Call(name=alias[c.name], arguments=c.arguments) for c in raw_calls]
        query = row["input"].rsplit("\nUser:", 1)[-1].rsplit("\nGenerate API Request:", 1)[0].strip()

        prefix = row["input"].rsplit("\nGenerate API Request:", 1)[0]
        response = response_index[prefix][0] if response_index.get(prefix) else None

        timeline: list[Event] = [Event(
            event_id=f"lv{level}:{index}:user", role="user", content=query,
            visible_at="0", provider="source",
        )]
        timeline.append(Event(
            event_id=f"lv{level}:{index}:call", role="assistant",
            tool_calls=calls, visible_at="1", provider="source",
        ))
        result_text = ""
        narration = ""
        if response is not None:
            result_suffix = response["input"].rsplit("\nAPI-Request:", 1)[-1]
            result_text = result_suffix.split("->", 1)[1] if "->" in result_suffix else result_suffix
            narration = (response.get("output", "") or "").strip()
            if narration.startswith("AI:"):
                narration = narration[3:].strip()
            if result_text.strip():
                timeline.append(Event(
                    event_id=f"lv{level}:{index}:result", role="tool",
                    content=result_text, visible_at="2", provider="source",
                ))
            if narration.strip():
                timeline.append(Event(
                    event_id=f"lv{level}:{index}:reply", role="assistant",
                    content=narration, visible_at="3", provider="source",
                ))

        return TaskEvidence(
            identity=Identity(
                case_id=f"api_bank:lv{level}:{index}",
                source_id="api_bank",
                source_record_id=str(index),
                origin_path=f"lv{level}-api-train@{index}",
                lang="en",
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
            behavior=Behavior(
                original_calls=calls,
                source_disposition={"action": "execute", "reason": "published_api_call",
                                    "source_kind": "publisher_api_annotation"},
            ),
        )

    def manifest(self) -> AdapterManifest:
        return AdapterManifest(
            source_id="api_bank",
            source_format="API-Bank：training-data__lv{1,2,3}-{api,response}-train.json，JSON 数组，每条 {instruction,input,output}",
            field_mapping={
                "input 开头的工具定义 JSON": "tool_env.catalog（apiCode 归一化 public_apibank_*）",
                "input 的 User: 后文本": "timeline[user].content（query）",
                "output 的 API-Request 调用": "timeline[assistant].tool_calls（别名 public_apibank_*）",
                "response 配对（按 Generate API Request: 前对话前缀精确匹配）的结果": "timeline[tool].content",
                "response 的 output": "timeline[assistant].content（回复）",
            },
            unrecoverable=[
                "结果右段可能是 Python repr（单引号），保留原文不做 json 解析",
                "无精确 response 配对时，只有调用无结果/回复（narration 监督不可用）",
                "lv3 User: 行可带 TIME: 内联后缀，query 提取会保留",
            ],
            examples=[{"case_id": "api_bank:lv1:0", "tool": "public_apibank_get_all_sessions"}],
            counterexamples=[
                "ToolSearcher 调用是 Mei 内部检索（非可执行 LM 工具），忠实映射为普通调用，retrieval-only 判定留待 derive/audit",
                "调用数 >5 或调用名不在声明工具集内 → 跳过（不编造）",
            ],
            verifier_version="api-bank-adapter-v1",
            trial_report="对标 public_sft_multihead_trial.py 的 _api_bank_groups（_leading_json_objects/_api_tool/parse_python_calls）",
        )
