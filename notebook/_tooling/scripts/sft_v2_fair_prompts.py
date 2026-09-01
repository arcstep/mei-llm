"""Fair Full-call / MW prompts. Tune only on DEV; freeze before TEST."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repo_paths import MW_REASON_DEFINITIONS_V1, SFT_V2_BASELINE_CONTRACT_V2
from sft_canonical_lib import compact_tools, dumps_canonical, render_v2_prompt

PROMPT_VERSION = "sft-v2-fair-prompt-v1"
FULLCALL_ENVELOPE_EXAMPLE = '[{"name":"set_lights","arguments":{"room":"客厅","brightness":80}}]'
REFUSE_EXAMPLE = "[]"


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_mw_definitions() -> dict[str, Any]:
    return json.loads(MW_REASON_DEFINITIONS_V1.read_text(encoding="utf-8"))


def mw_codebook_block() -> str:
    data = load_mw_definitions()
    lines = [
        "你是闭集处置分类器。只根据 query、允许事实和当前可见的至多 5 个工具 schema，输出一个 reason_code。",
        "禁止输出解释、JSON 工具调用、HD/SF 文本或多个码。",
        "码义：",
    ]
    for row in data["codes"]:
        neigh = "、".join(row.get("neighbors") or [])
        lines.append(
            f"- {row['reason_code']}: {row['definition']} 易混：{neigh}。"
        )
    lines.append("输出示范：只输出一行，例如 ready_to_execute")
    lines.append("错误示范：不要输出“因为缺温度所以 missing_slot”。")
    return "\n".join(lines)


def fullcall_system() -> str:
    return (
        "你是闭集工具调用器，不是聊天助手。\n"
        "只能调用当前提示里给出的工具之一，或输出空数组 []。禁止调用未列出的工具。\n"
        "每个调用必须是 {\"name\": 工具名, \"arguments\": 对象}。不要用 parameters 代替 arguments。\n"
        "缺少必填证据、场景冲突、越权或无法在这 5 个工具内完成时输出 []。\n"
        "禁止解释、禁止提问、禁止 Markdown 围栏。\n"
        f"格式示例（内容与评测题无关）：{FULLCALL_ENVELOPE_EXAMPLE}\n"
        f"拒绝示例：{REFUSE_EXAMPLE}"
    )


def tools_to_ollama(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in compact_tools(tools):
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def fullcall_json_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        },
    }


def render_fullcall_user(row: dict, tools: list[dict[str, Any]]) -> str:
    rendered = render_v2_prompt(
        tools,
        str(row.get("query") or ""),
        scene=row.get("scene"),
        system_facts=row.get("system_facts") or row.get("system_facts_text"),
        history=row.get("history"),
        prior_tool_results=row.get("prior_tool_results"),
        entities=row.get("entities"),
        selected_entity=row.get("selected_entity"),
        permissions=row.get("permissions"),
    )
    return rendered["text"]


def render_mw_user(row: dict, tools: list[dict[str, Any]]) -> str:
    facts = str(row.get("system_facts") or row.get("facts") or "").strip()
    query = str(row.get("query") or "").strip()
    block = "<tools>" + dumps_canonical(compact_tools(tools)) + "</tools>"
    parts = [f"query：{query}"]
    if facts:
        parts.append(f"允许事实：{facts}")
    parts.append("可见工具：")
    parts.append(block)
    parts.append("只输出一个 reason_code。")
    return "\n".join(parts)


def prompt_asset() -> dict[str, Any]:
    full_sys = fullcall_system()
    mw_sys = mw_codebook_block()
    return {
        "prompt_version": PROMPT_VERSION,
        "fullcall_system": full_sys,
        "fullcall_system_sha256": sha256_text(full_sys),
        "mw_system": mw_sys,
        "mw_system_sha256": sha256_text(mw_sys),
        "envelope_example": FULLCALL_ENVELOPE_EXAMPLE,
        "refuse_example": REFUSE_EXAMPLE,
        "contract": str(SFT_V2_BASELINE_CONTRACT_V2.name),
        "notes": [
            "Examples are independent stubs, not TEST items.",
            "Do not tune on TEST.",
            "51M SFT (not this round) may drop few-shot; eval still supplies schema/query/facts.",
        ],
    }


def dump_prompt_asset(path: Path) -> dict[str, Any]:
    asset = prompt_asset()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return asset
