#!/usr/bin/env python3
"""MOSS with-tools 适配器：中文工具调用对话 → 统一中间表示。

源格式：MOSS-003 with-tools，zip 内单个 jsonl，每行一个多轮对话。字段为
``chat.turn_N``（含 Human / Commands / Tool Responses / MOSS 四字段）、
``meta_instruction``（工具启用清单）、``conversation_id``（原生 ID）。

映射（忠实，不做五头派生、不做语义审核）：
- Human          → timeline[user].content
- Commands       → timeline[assistant].tool_calls（别名到 public_*）
- Tool Responses → timeline[tool].content（保留原始结果文本，结构化解析留 derive）
- MOSS           → timeline[assistant].content（文本回复）
"""

from __future__ import annotations

import ast
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from evidence import (
    TaskEvidence, Event, Call, Identity, Task, ToolEnv, Behavior, Review,
)
from adapter_base import Adapter, AdapterLimits, AdapterManifest, register


MOSS_POSITIONAL = {
    "Search": ["query"],
    "Calculate": ["expression"],
    "Solve": ["equation"],
    "Text2Image": ["description"],
}

MOSS_ALIASES = {
    "Search": "public_search",
    "Calculate": "public_calculate",
    "Solve": "public_solve",
    "Text2Image": "public_text_to_image",
}


def _clean_marker(value: str | None, prefix: str, suffix: str) -> str:
    value = (value or "").strip()
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.endswith(suffix):
        value = value[:-len(suffix)]
    return value.strip()


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def moss_tools(meta: str) -> list[dict[str, Any]]:
    """从 meta_instruction 提取启用的工具 catalog（public_* 别名）。"""
    catalog = {
        "Search": _tool("public_search", "搜索与当前问题相关的资料。", {"query": {"type": "string"}}, ["query"]),
        "Calculate": _tool("public_calculate", "计算给定表达式。", {"expression": {"type": "string"}}, ["expression"]),
        "Solve": _tool("public_solve", "求解给定方程。", {"equation": {"type": "string"}}, ["equation"]),
        "Text2Image": _tool("public_text_to_image", "根据文字描述生成图片。", {"description": {"type": "string"}}, ["description"]),
    }
    enabled: list[dict[str, Any]] = []
    for line in meta.splitlines():
        if "enabled" not in line or "API:" not in line:
            continue
        for name in catalog:
            if name + "(" in line:
                enabled.append(catalog[name])
    return enabled


def moss_calls(text: str) -> list[Call]:
    """解析 Commands 标记里的 Python 调用表达式，别名到 public_*。"""
    value = _clean_marker(text, "<|Commands|>:", "<eoc>")
    if not value or value in ("None", "null", "[]"):
        return []
    tree = ast.parse(value, mode="eval").body
    nodes = tree.elts if isinstance(tree, (ast.Tuple, ast.List)) else [tree]
    calls: list[Call] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("non_literal_call")
        name = node.func.id
        if name not in MOSS_POSITIONAL:
            raise ValueError("unknown_function")
        keys = [k.arg for k in node.keywords]
        pos = MOSS_POSITIONAL[name]
        if len(node.args) > len(pos) or None in keys or len(keys) != len(set(keys)):
            raise ValueError("spread_or_duplicate_argument")
        args = dict(zip(pos, map(ast.literal_eval, node.args)))
        if set(args) & set(keys):
            raise ValueError("duplicate_argument")
        args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords})
        calls.append(Call(name=MOSS_ALIASES[name], arguments=args))
    return calls


def _turn_number(name: str) -> int:
    match = re.fullmatch(r"turn_(\d+)", name)
    if not match:
        raise ValueError("unknown_turn_order")
    return int(match.group(1))


@register("moss")
class MossAdapter:
    source_id = "moss"

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        evidences: list[TaskEvidence] = []
        with zipfile.ZipFile(raw_path) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as stream:
                for line_number, line in enumerate(stream):
                    if len(evidences) >= limits.max_records:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    evidence = self._to_evidence(row, member=member, line_number=line_number)
                    if evidence is not None:
                        evidences.append(evidence)
        return evidences

    def _to_evidence(self, row: dict[str, Any], *, member: str, line_number: int) -> TaskEvidence | None:
        tools = moss_tools(row.get("meta_instruction", ""))
        chat = row.get("chat") or {}
        try:
            turn_names = sorted(chat, key=_turn_number)
        except (ValueError, KeyError, TypeError):
            return None

        timeline: list[Event] = []
        for turn_name in turn_names:
            turn = chat[turn_name]
            for field, role, begin, end in (
                ("Human", "user", "<|Human|>:", "<eoh>"),
                ("Commands", "assistant", "<|Commands|>:", "<eoc>"),
                ("Tool Responses", "tool", "<|Results|>:", "<eor>"),
                ("MOSS", "assistant", "<|MOSS|>:", "<eom>"),
            ):
                content = _clean_marker(turn.get(field, ""), begin, end)
                if not content:
                    continue
                event = Event(
                    event_id=f"{line_number}:{turn_name}:{field}",
                    role=role,
                    content=content,
                    visible_at=f"{turn_name}:{field}",
                    provider="source",
                )
                if field == "Commands":
                    try:
                        event.tool_calls = moss_calls(content)
                    except (ValueError, SyntaxError):
                        # 保留原始文本，结构化解析失败不丢原文（记入不能恢复的信息）。
                        event.tool_calls = []
                timeline.append(event)

        if not timeline:
            return None

        original_calls = [c for e in timeline if e.role == "assistant" for c in e.tool_calls]
        query = next((e.content for e in timeline if e.role == "user"), "") or ""
        lang = "zh_or_mixed" if re.search(r"[㐀-鿿]", query) else "en_or_other"
        case_id = str(row.get("conversation_id") or f"moss:{line_number}")

        return TaskEvidence(
            identity=Identity(
                case_id=case_id,
                source_id="moss",
                source_record_id=str(row.get("conversation_id", "")),
                origin_path=f"{member}@{line_number}",
                lang=lang,
                split="candidate",
                producer_identity="publisher_synthetic",
            ),
            task=Task(query=query),
            tool_env=ToolEnv(
                catalog=tools,
                candidate_construction="oracle",
                description_lang="zh",
            ),
            timeline=timeline,
            behavior=Behavior(original_calls=original_calls),
        )

    def manifest(self) -> AdapterManifest:
        return AdapterManifest(
            source_id="moss",
            source_format="MOSS-003 with-tools：zip 内单 jsonl，每行一个多轮对话（chat.turn_N 含 Human/Commands/Tool Responses/MOSS）",
            field_mapping={
                "chat.turn_N.Human": "timeline[user].content",
                "chat.turn_N.Commands": "timeline[assistant].tool_calls（别名 public_search/public_calculate/public_solve/public_text_to_image）",
                "chat.turn_N.Tool Responses": "timeline[tool].content（保留原始结果文本）",
                "chat.turn_N.MOSS": "timeline[assistant].content",
                "meta_instruction": "tool_env.catalog",
                "conversation_id": "identity.case_id / source_record_id",
            },
            unrecoverable=[
                "工具返回结果的结构化解析（call_id↔result 关联）留待 derive 阶段",
                "未来结果与调用前的可见性边界由 derive 阶段按 timeline 顺序判定",
                "Commands 解析失败时调用置空，原文保留在 event.content",
            ],
            examples=[{"case_id": "moss:0", "tool": "public_search", "argument": "query"}],
            counterexamples=[
                "Commands 含多行/非字面调用时 moss_calls 抛 non_literal_call，事件保留但 tool_calls 置空",
            ],
            verifier_version="moss-adapter-v1",
            trial_report="对标 public_sft_scale_trial.py 的 MOSS 分支（moss_calls/events_for/moss_tools），固化别名映射",
        )
