#!/usr/bin/env python3
"""Canonical cases and v2 compilers for retrieval / full-call / MW disposition.

Gold (tool, arguments, provenance, reason_code) is program-determined.
Teachers may only rewrite the visible query.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repo_paths import (
    CODEBOOK_MW_DISPOSITION_V1,
    EVAL_SHARED_ROOT,
    SCHEMA_MW_GOVERNANCE,
    ROOT,
)

SERIALIZER_V2 = "mei-tool-call-serializer-v2"
PROMPT_VERSION = "sft-teacher-v2-query-only.1"
GENERATOR_VERSION = "sft-synth-v2"
TASK_CONTRACT_V2 = (
    "任务：只输出一个 schema 合法的工具 JSON 数组，或 []。"
    "最多一次调用。缺少 required 证据时输出 []。禁止输出解释或 route_id。"
)
FORBIDDEN_PROMPT_MARKERS = (
    "<routes>",
    "gold_route_id",
    "gold_provenance",
    "compiled_call_candidates",
    "gold_entity_link",
    "holdout_only_relation",
    "scenario_id",
)
FAKE_VARIANT_RE = re.compile(r"（说法\s*\d+）|\(说法\s*\d+\)")
DIGIT_SUFFIX_RE = re.compile(
    r"(?:可以吗|好吗|谢谢|一下)\d+$|"
    r"[吗吧呀呢]\d{1,4}$|"
    r"（扩[^）]*）|（样例[^）]*）|"
    r"\(扩[^)]*\)|\(样例[^)]*\)"
)
DOUBLE_QING_RE = re.compile(r"请请")
EVAL_RE = re.compile(r"\bEVAL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+\b")
PII_RE = re.compile(
    r"(?:\+?86[-\s]?)?1[3-9]\d{9}"
    r"|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
    re.I,
)

RETRIEVAL_EXTRA_TOOLS = [
    {
        "name": "get_weather_status",
        "description": "Get weather API health status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "create_invoice",
        "description": "Create an invoice document.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
    {
        "name": "check_inventory",
        "description": "Check on-hand inventory for a SKU.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Open a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_toolset(toolset_id: str) -> dict:
    path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw.get("tools"):
        raise ValueError(f"toolset {toolset_id} has no tools")
    return raw


def compact_tools(tools: list[dict] | dict) -> list[dict]:
    seq = tools.get("tools") if isinstance(tools, dict) else tools
    out = []
    for tool in seq or []:
        out.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def schema_fingerprint(tools: list[dict] | dict) -> str:
    return hashlib.sha256(dumps_canonical(compact_tools(tools)).encode("utf-8")).hexdigest()


def catalog_tools(toolset_ids: list[str], *, extra: list[dict] | None = None) -> list[dict]:
    tools: list[dict] = []
    seen: set[str] = set()
    for tid in toolset_ids:
        for tool in load_toolset(tid).get("tools") or []:
            name = str(tool.get("name") or "")
            if name and name not in seen:
                tools.append(tool)
                seen.add(name)
    for tool in extra or []:
        name = str(tool.get("name") or "")
        if name and name not in seen:
            tools.append(tool)
            seen.add(name)
    return tools


def tools_by_name(tools: list[dict]) -> dict[str, dict]:
    return {str(t["name"]): t for t in tools if t.get("name")}


def select_top5(
    catalog: list[dict],
    *,
    gold_name: str | None,
    hard_negatives: list[str] | None = None,
    rng: random.Random,
) -> list[dict]:
    by_name = tools_by_name(catalog)
    picked: list[dict] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if not name or name in seen or name not in by_name or len(picked) >= 5:
            return
        picked.append(by_name[name])
        seen.add(name)

    add(gold_name)
    for name in hard_negatives or []:
        add(name)
    others = [t for t in catalog if str(t.get("name")) not in seen]
    rng.shuffle(others)
    for tool in others:
        if len(picked) >= 5:
            break
        picked.append(tool)
        seen.add(str(tool.get("name")))
    rng.shuffle(picked)
    return picked[:5]


def split_for_key(key: str, *, valid_frac: float = 0.09) -> str:
    cut = max(1, int(valid_frac * 10_000))
    bucket = int(sha256_text(key)[:8], 16) % 10_000
    return "valid" if bucket < cut else "train"


def ensure_train_valid_split(cases: list[dict], *, min_valid: int = 1) -> list[dict]:
    """Small banks must not collapse to train-only (24-row smoke bug)."""
    if not cases:
        return cases
    n_valid = sum(1 for c in cases if c.get("split") == "valid")
    n_train = sum(1 for c in cases if c.get("split") == "train")
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[str(case.get("cf_group") or case.get("case_id") or id(case))].append(case)
    keys = list(groups)
    if n_valid < min_valid and keys:
        for case in groups[keys[-1]]:
            case["split"] = "valid"
        n_valid = sum(1 for c in cases if c.get("split") == "valid")
        n_train = len(cases) - n_valid
    if n_train < 1 and len(keys) > 1:
        for case in groups[keys[0]]:
            case["split"] = "train"
    return cases


def query_banned(text: str) -> str | None:
    if not text or not str(text).strip():
        return "empty"
    if FAKE_VARIANT_RE.search(text):
        return "fake_variant"
    if DIGIT_SUFFIX_RE.search(text):
        return "digit_suffix"
    if DOUBLE_QING_RE.search(text):
        return "double_qing"
    if EVAL_RE.search(text):
        return "eval_id"
    if PII_RE.search(text):
        return "pii"
    if "�" in text:
        return "garbled"
    return None


def leak_markers(text: str) -> list[str]:
    return [m for m in FORBIDDEN_PROMPT_MARKERS if m in (text or "")]


def has_route_id_gold(row: dict) -> bool:
    if row.get("gold_route_id") is not None:
        return True
    answers = row.get("answers")
    if isinstance(answers, dict) and "route_id" in answers:
        return True
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict) and "route_id" in item and "name" not in item:
                return True
    blob = json.dumps({k: row.get(k) for k in ("prompt_text", "serializer", "protocol") if k in row}, ensure_ascii=False)
    return "<routes>" in blob or "mei-route-serializer" in blob


def render_v2_prompt(
    tools: list[dict],
    query: str,
    *,
    scene: str | None = None,
    system_facts: str | None = None,
    history: list[dict] | None = None,
    prior_tool_results: list | None = None,
    entities: dict | None = None,
    selected_entity: str | None = None,
    permissions: list[str] | None = None,
) -> dict:
    """Delegate to runtime prompt_v2 so train/student/Qwen stay isomorphic."""
    import sys

    model_dir = ROOT / "notebook/_tooling/model/mei-1.0-51m"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from prompt_v2 import render_v2_request  # noqa: E402

    facts = system_facts
    if not facts and scene and str(scene).strip():
        facts = "场景摘要：" + str(scene).strip()
    rendered = render_v2_request(
        tools=tools,
        query=query,
        system_facts=facts,
        history=history,
        prior_tool_results=[
            r if isinstance(r, str) else dumps_canonical(r) for r in (prior_tool_results or [])
        ],
        entities=entities,
        selected_entity=selected_entity,
        permissions=permissions,
    )
    rendered.setdefault("serializer", SERIALIZER_V2)
    return rendered


def slot_provenance(query: str, zh_slots: dict | None, gold_args: dict | None) -> list[dict]:
    out = []
    slots = zh_slots or {}
    for arg, span in slots.items():
        if span and str(span) in (query or ""):
            out.append({"arg": arg, "span": span, "source": "query"})
    for arg, value in (gold_args or {}).items():
        if arg in slots:
            continue
        if isinstance(value, str) and value and value in (query or ""):
            out.append({"arg": arg, "span": value, "source": "query"})
    return out


def protected_slots_ok(case: dict, query: str) -> bool:
    kind = case.get("kind")
    slots = case.get("zh_slots") or {}
    if kind in {"missing", "offtopic", "no_match"} or case.get("gold_tool") is None and case.get("task") == "retrieval":
        if case.get("task") == "retrieval" and case.get("family") == "no_match":
            return True
    if kind in {"missing", "offtopic"}:
        return True
    for value in slots.values():
        if value and str(value) not in (query or ""):
            return False
    return True


def project_cell(act: str, reason_code: str) -> str:
    if act == "execute":
        return "MW.OK"
    if act == "expand":
        return "SF.PAR"
    if act == "shape":
        return "SF.AMB"
    if act == "escalate":
        return "Unknown"
    if act == "stop":
        return "SF.PRE" if reason_code == "scene_conflict" else "Unknown"
    raise ValueError(f"unknown act {act}")


def freeze_mw_codebook(gov: dict | None = None) -> dict:
    schema = gov or json.loads(SCHEMA_MW_GOVERNANCE.read_text(encoding="utf-8"))
    classes = []
    for act, reasons in (schema.get("reason_codes") or {}).items():
        for reason in reasons:
            classes.append(
                {
                    "class_id": len(classes),
                    "reason_code": reason,
                    "act": act,
                    "cell": project_cell(act, reason),
                }
            )
    return {
        "id": "mw-disposition-codebook-v1",
        "source": schema.get("id"),
        "n_classes": len(classes),
        "classes": classes,
        "rules": [
            "Head target is reason_code only.",
            "Runtime maps act/cell from this codebook.",
            "Teachers must not edit reason_code/act/cell/gaps/function_calls.",
        ],
    }


def load_mw_codebook(path: Path | None = None) -> dict:
    p = path or CODEBOOK_MW_DISPOSITION_V1
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return freeze_mw_codebook()


def codebook_by_reason(codebook: dict | None = None) -> dict[str, dict]:
    book = codebook or load_mw_codebook()
    return {str(row["reason_code"]): row for row in book.get("classes") or []}


def lexical_rank(query: str, tools: list[dict]) -> list[str]:
    scored = []
    for tool in tools:
        blob = str(tool.get("name") or "") + " " + str(tool.get("description") or "")
        score = sum(1 for ch in query if ch in blob)
        scored.append((score, str(tool.get("name"))))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in scored]


def retrieval_metrics(ranks: list[int]) -> dict:
    n = max(1, len(ranks))
    return {
        "recall_at_1": sum(1 for r in ranks if r == 0) / n,
        "recall_at_5": sum(1 for r in ranks if 0 <= r < 5) / n,
        "mrr": sum((1.0 / (r + 1) if r >= 0 else 0.0) for r in ranks) / n,
        "n": len(ranks),
    }


def stratified_retrieval_metrics(rows: list[dict], catalog: list[dict]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        gold = row.get("gold_tool")
        order = lexical_rank(str(row.get("query") or ""), catalog)
        rank = order.index(gold) if gold in order else -1
        groups["all"].append(rank)
        groups[f"split:{row.get('split')}"].append(rank)
        groups[f"family:{row.get('family')}"].append(rank)
    return {key: retrieval_metrics(vals) for key, vals in groups.items()}


def _case_id(prefix: str, payload: dict) -> str:
    return prefix + sha256_text(dumps_canonical(payload))[:16]


def retrieval_case_bank() -> list[dict]:
    """Frozen canonical retrieval cases. Eval bank queries must not appear here."""
    seen_home = catalog_tools(["needle-home-v0"], extra=RETRIEVAL_EXTRA_TOOLS)
    office = catalog_tools(["mei-office-v0"])
    vrm = catalog_tools(["needle-vrm-agent-v0"])
    retail = catalog_tools(["mei-retail-v0"])
    invoice = catalog_tools(["needle-invoice-v0"])
    merged_seen = catalog_tools(
        ["needle-home-v0", "mei-office-v0", "needle-vrm-agent-v0"],
        extra=RETRIEVAL_EXTRA_TOOLS,
    )
    specs = [
        {
            "stem": "麻烦看下南京现在热不热",
            "gold_tool": "get_weather",
            "family": "seen_schema",
            "toolset_id": "needle-home-v0",
            "catalog": seen_home,
            "zh_slots": {"city": "南京"},
            "hard_negatives": ["get_weather_status", "set_lights"],
            "kind": "positive",
        },
        {
            "stem": "把次卧的灯调到四十五",
            "gold_tool": "set_lights",
            "family": "seen_schema",
            "toolset_id": "needle-home-v0",
            "catalog": seen_home,
            "zh_slots": {"room": "次卧"},
            "hard_negatives": ["set_switch", "get_weather"],
            "kind": "positive",
        },
        {
            "stem": "建一个叫晨会的日历",
            "gold_tool": "create_event",
            "family": "seen_schema",
            "toolset_id": "mei-office-v0",
            "catalog": office,
            "zh_slots": {"title": "晨会"},
            "hard_negatives": ["lookup_price", "set_volume"],
            "kind": "positive",
        },
        {
            "stem": "音量给我调到七",
            "gold_tool": "set_volume",
            "family": "seen_schema",
            "toolset_id": "mei-office-v0",
            "catalog": office,
            "zh_slots": {},
            "hard_negatives": ["lookup_price", "create_event"],
            "kind": "positive",
        },
        {
            "stem": "纸杯十个一共多少钱",
            "gold_tool": "lookup_price",
            "family": "seen_schema",
            "toolset_id": "mei-office-v0",
            "catalog": office,
            "zh_slots": {"item": "纸杯"},
            "hard_negatives": ["create_event", "set_volume"],
            "kind": "positive",
        },
        {
            "stem": "请你去厨房一趟",
            "gold_tool": "go_to",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {"place": "厨房"},
            "hard_negatives": ["open_door", "take_out_trash"],
            "kind": "positive",
        },
        {
            "stem": "请打开客厅灯开关",
            "gold_tool": "set_switch",
            "family": "same_action_diff_object",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": merged_seen,
            "zh_slots": {"light": "客厅灯"},
            "hard_negatives": ["set_lights", "open_door"],
            "kind": "hard",
        },
        {
            "stem": "厨房灯再暗两档",
            "gold_tool": "set_lights",
            "family": "same_object_diff_action",
            "toolset_id": "needle-home-v0",
            "catalog": merged_seen,
            "zh_slots": {"room": "厨房"},
            "hard_negatives": ["set_switch", "get_weather"],
            "kind": "hard",
        },
        {
            "stem": "查一下天气接口通不通",
            "gold_tool": "get_weather_status",
            "family": "similar_name",
            "toolset_id": "needle-home-v0",
            "catalog": merged_seen,
            "zh_slots": {},
            "hard_negatives": ["get_weather", "lookup_price"],
            "kind": "hard",
        },
        {
            "stem": "订一张四人的室内桌",
            "gold_tool": "book_table",
            "family": "unseen_schema",
            "toolset_id": "mei-retail-v0",
            "catalog": retail,
            "zh_slots": {},
            "hard_negatives": ["charge_card", "print_label"],
            "kind": "positive",
        },
        {
            "stem": "刷卡收二百人民币",
            "gold_tool": "charge_card",
            "family": "unseen_schema",
            "toolset_id": "mei-retail-v0",
            "catalog": retail,
            "zh_slots": {},
            "hard_negatives": ["book_table", "print_label"],
            "kind": "positive",
        },
        {
            "stem": "打印 SKU-B200 的条码",
            "gold_tool": "print_label",
            "family": "unseen_schema",
            "toolset_id": "mei-retail-v0",
            "catalog": retail,
            "zh_slots": {"sku": "SKU-B200"},
            "hard_negatives": ["book_table", "charge_card"],
            "kind": "positive",
        },
        {
            "stem": "从这段话里抽出供应商和金额",
            "gold_tool": "invoice",
            "family": "unseen_schema",
            "toolset_id": "needle-invoice-v0",
            "catalog": invoice,
            "zh_slots": {},
            "hard_negatives": [],
            "kind": "positive",
        },
        {
            "stem": "解释一下光合作用给小孩听",
            "gold_tool": None,
            "family": "no_match",
            "toolset_id": "needle-home-v0",
            "catalog": seen_home,
            "zh_slots": {},
            "hard_negatives": ["get_weather", "set_lights"],
            "kind": "no_match",
        },
        {
            "stem": "帮我写一封生日祝福",
            "gold_tool": None,
            "family": "no_match",
            "toolset_id": "mei-office-v0",
            "catalog": office,
            "zh_slots": {},
            "hard_negatives": ["create_event"],
            "kind": "no_match",
        },
        {
            "stem": "太阳系有几颗行星来着",
            "gold_tool": None,
            "family": "no_match",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": [],
            "kind": "no_match",
        },
        {
            "stem": "把垃圾袋拿出去丢掉",
            "gold_tool": "take_out_trash",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["go_to", "open_door"],
            "kind": "positive",
        },
        {
            "stem": "麻烦把前门打开",
            "gold_tool": "open_door",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {"door": "前门"},
            "hard_negatives": ["close_door", "go_to"],
            "kind": "positive",
        },
        {
            "stem": "点一下头就行",
            "gold_tool": "nod",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["shake_head", "bow"],
            "kind": "positive",
        },
        {
            "stem": "摇摇头表示不行",
            "gold_tool": "shake_head",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["nod", "wave"],
            "kind": "positive",
        },
        {
            "stem": "过来我这边一下",
            "gold_tool": "come_here",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["stop", "go_to"],
            "kind": "positive",
        },
        {
            "stem": "先停住别走",
            "gold_tool": "stop",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["come_here", "sit"],
            "kind": "positive",
        },
        {
            "stem": "开一张标题为月结的单据",
            "gold_tool": "create_invoice",
            "family": "similar_name",
            "toolset_id": "needle-home-v0",
            "catalog": merged_seen,
            "zh_slots": {"title": "月结"},
            "hard_negatives": ["invoice", "lookup_price"],
            "kind": "hard",
        },
        {
            "stem": "把后门关上",
            "gold_tool": "close_door",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {"door": "后门"},
            "hard_negatives": ["open_door", "go_to"],
            "kind": "positive",
        },
        {
            "stem": "杭州现在热不热",
            "gold_tool": "get_weather",
            "family": "seen_schema",
            "toolset_id": "needle-home-v0",
            "catalog": seen_home,
            "zh_slots": {"city": "杭州"},
            "hard_negatives": ["get_weather_status", "check_inventory"],
            "kind": "positive",
        },
        {
            "stem": "把书房的灯调到三十",
            "gold_tool": "set_lights",
            "family": "seen_schema",
            "toolset_id": "needle-home-v0",
            "catalog": seen_home,
            "zh_slots": {"room": "书房"},
            "hard_negatives": ["set_switch", "control_room_devices"],
            "kind": "positive",
        },
        {
            "stem": "建一个叫复盘的日历",
            "gold_tool": "create_event",
            "family": "seen_schema",
            "toolset_id": "mei-office-v0",
            "catalog": office,
            "zh_slots": {"title": "复盘"},
            "hard_negatives": ["create_ticket", "create_invoice"],
            "kind": "positive",
        },
        {
            "stem": "查一下货架上矿泉水还剩多少",
            "gold_tool": "check_inventory",
            "family": "similar_name",
            "toolset_id": "needle-home-v0",
            "catalog": merged_seen,
            "zh_slots": {"item": "矿泉水"},
            "hard_negatives": ["lookup_price", "order_food"],
            "kind": "hard",
        },
        {
            "stem": "开一张标题为年会的单据",
            "gold_tool": "create_invoice",
            "family": "similar_name",
            "toolset_id": "needle-home-v0",
            "catalog": merged_seen,
            "zh_slots": {"title": "年会"},
            "hard_negatives": ["create_ticket", "create_event"],
            "kind": "hard",
        },
        {
            "stem": "去厨房那边一下",
            "gold_tool": "go_to",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {"place": "厨房"},
            "hard_negatives": ["come_here", "open_door"],
            "kind": "positive",
        },
        {
            "stem": "把侧门打开",
            "gold_tool": "open_door",
            "family": "seen_schema",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {"door": "侧门"},
            "hard_negatives": ["close_door", "go_to"],
            "kind": "positive",
        },
        {
            "stem": "今天太阳系有几颗行星",
            "gold_tool": None,
            "family": "no_match",
            "toolset_id": "needle-vrm-agent-v0",
            "catalog": vrm,
            "zh_slots": {},
            "hard_negatives": ["get_weather", "create_event"],
            "kind": "no_match",
        },
    ]
    cases = []
    for spec in specs:
        catalog = spec.pop("catalog")
        payload = {k: spec[k] for k in ("stem", "gold_tool", "family", "toolset_id", "kind")}
        case_id = _case_id("RET-", payload)
        cf_group = "CFG-" + sha256_text(spec["stem"] + "|" + str(spec["gold_tool"]))[:12]
        cases.append(
            {
                "case_id": case_id,
                "task": "retrieval",
                "query": spec["stem"],
                "stem": spec["stem"],
                "gold_tool": spec["gold_tool"],
                "hard_negatives": spec["hard_negatives"],
                "zh_slots": spec["zh_slots"],
                "family": spec["family"],
                "kind": spec["kind"],
                "toolset_id": spec["toolset_id"],
                "catalog_tools": compact_tools(catalog),
                "seen_schema": spec["family"] not in {"unseen_schema"},
                "split": split_for_key(cf_group),
                "cf_group": cf_group,
                "source_role": "schema-program",
                "prefer_template": True,
            }
        )
    return cases


def compile_retrieval_row(case: dict, query: str, *, teacher_model: str = "template") -> dict:
    banned = query_banned(query)
    if banned:
        raise ValueError(banned)
    if not protected_slots_ok(case, query):
        raise ValueError("protected_slot")
    gold = case.get("gold_tool")
    catalog_names = [str(t.get("name")) for t in case.get("catalog_tools") or []]
    if gold and gold not in catalog_names:
        raise ValueError("gold_not_in_catalog")
    row = {
        "sample_id": "RET-" + sha256_text(case["case_id"] + "\n" + query)[:16],
        "case_id": case["case_id"],
        "task": "retrieval",
        "query": query,
        "gold_tool": gold,
        "hard_negatives": list(case.get("hard_negatives") or []),
        "catalog_tools": list(case.get("catalog_tools") or []),
        "toolset_id": case.get("toolset_id"),
        "family": case.get("family"),
        "kind": case.get("kind"),
        "split": case.get("split") or split_for_key(case["case_id"]),
        "seen_schema": bool(case.get("seen_schema")),
        "cf_group": case.get("cf_group"),
        "zh_slots": dict(case.get("zh_slots") or {}),
        "source_role": "schema-program",
        "teacher_model": teacher_model,
        "prompt_version": PROMPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "slot_provenance": slot_provenance(query, case.get("zh_slots"), None),
        "catalog_id": case.get("catalog_id") or case.get("toolset_id"),
        "toolset_hash": case.get("toolset_hash") or schema_fingerprint(case.get("catalog_tools") or []),
        "intent_id": case.get("intent_id"),
        "engineering_smoke": bool(case.get("engineering_smoke")),
    }
    if has_route_id_gold(row):
        raise ValueError("route_id_leak")
    return row


def compile_fullcall_row(
    case: dict,
    query: str,
    *,
    rng: random.Random,
    teacher_model: str = "template",
) -> dict:
    banned = query_banned(query)
    if banned:
        raise ValueError(banned)
    if not protected_slots_ok(case, query):
        raise ValueError("protected_slot")
    catalog = case.get("catalog_tools") or compact_tools(load_toolset(str(case.get("toolset_id"))))
    top5 = select_top5(
        catalog,
        gold_name=case.get("gold_name"),
        hard_negatives=case.get("hard_negatives") or [],
        rng=rng,
    )
    rendered = render_v2_prompt(
        top5,
        query,
        scene=case.get("scene"),
        system_facts=case.get("system_facts_text"),
        history=case.get("history"),
        prior_tool_results=case.get("prior_tool_results"),
        entities=case.get("entities"),
        selected_entity=case.get("selected_entity"),
        permissions=case.get("permissions"),
    )
    if not rendered["ok"]:
        raise ValueError("prompt_leak:" + ",".join(rendered["leaks"]))
    answers = list(case.get("answers") or [])
    if case.get("kind") in {"missing", "scene_conflict", "illegal_pair", "offtopic", "fault", "noop", "no_rsv"}:
        answers = []
    row = {
        "sample_id": "FC-" + sha256_text(case["case_id"] + "\n" + query)[:16],
        "case_id": case["case_id"],
        "task": "fullcall",
        "query": query,
        "scene": case.get("scene") or "",
        "system_facts": case.get("system_facts_text") or "",
        "selected_entity": case.get("selected_entity"),
        "entities": dict(case.get("entities") or {}),
        "permissions": list(case.get("permissions") or []),
        "history": list(case.get("history") or []),
        "prior_tool_results": list(case.get("prior_tool_results") or []),
        "park_layer": case.get("park_layer"),
        "intent_id": case.get("intent_id"),
        "template_id": case.get("template_id"),
        "trigger": case.get("trigger"),
        "counterfactual_group": case.get("counterfactual_group") or case.get("cf_group"),
        "catalog_id": case.get("catalog_id") or case.get("toolset_id"),
        "toolset_hash": case.get("toolset_hash") or rendered.get("schema_fingerprint"),
        "gold_rank": case.get("gold_rank", 0),
        "provenance_transform": case.get("provenance_transform"),
        "normalizer_version": case.get("normalizer_version") or "park-normalizer-v1",
        "review_status": case.get("review_status") or "canonical",
        "toolset_id": case.get("toolset_id"),
        "retrieved_tools": [str(t.get("name")) for t in top5],
        "answers": answers,
        "serializer": SERIALIZER_V2,
        "schema_fingerprint": rendered["schema_fingerprint"],
        "prompt_text": rendered["text"],
        "prompt_n_chars": rendered.get("n_chars"),
        "prompt_tokens_est": rendered.get("prompt_tokens_est"),
        "kind": case.get("kind"),
        "family": case.get("family"),
        "split": case.get("split") or split_for_key(str(case.get("cf_group") or case["case_id"])),
        "cf_group": case.get("cf_group"),
        "gold_name": case.get("gold_name"),
        "gold_args": dict(case.get("gold_args") or {}),
        "zh_slots": dict(case.get("zh_slots") or {}),
        "slot_provenance": slot_provenance(query, case.get("zh_slots"), case.get("gold_args")),
        "source_role": "schema-program",
        "teacher_model": teacher_model,
        "prompt_version": PROMPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "act": None,
        "confidence_label": 0 if not answers else 1,
        "engineering_smoke": bool(case.get("engineering_smoke")),
    }
    if has_route_id_gold(row) or leak_markers(row["prompt_text"]):
        raise ValueError("route_id_leak")
    return row


def compile_mw_row(case: dict, query: str, *, teacher_model: str = "template") -> dict:
    banned = query_banned(query)
    if banned:
        raise ValueError(banned)
    if not protected_slots_ok(case, query):
        raise ValueError("protected_slot")
    book = codebook_by_reason()
    reason = str(case.get("reason_code") or "")
    if reason not in book:
        raise ValueError(f"unknown_reason:{reason}")
    mapped = book[reason]
    act = mapped["act"]
    cell = mapped["cell"]
    calls = list(case.get("function_calls") or [])
    if act != "execute":
        calls = []
    gaps = list(case.get("gaps") or [])
    if act == "execute" and gaps:
        raise ValueError("execute_with_gaps")
    if act == "execute" and not calls:
        raise ValueError("execute_without_calls")
    if act == "expand" and not gaps:
        raise ValueError("expand_without_gaps")
    row = {
        "sample_id": "MW-" + sha256_text(case["case_id"] + "\n" + query)[:16],
        "case_id": case["case_id"],
        "task": "mw",
        "query": query,
        "reason_code": reason,
        "reason_class_id": int(mapped["class_id"]),
        "head_target": "reason_code",
        "act": act,
        "cell": cell,
        "gaps": gaps,
        "audit_calls": calls,
        "answers": [],
        "function_calls": calls if act == "execute" else [],
        "toolset_id": case.get("toolset_id") or "needle-vrm-agent-v0",
        "family": case.get("family") or act,
        "kind": case.get("kind") or act,
        "split": case.get("split") or split_for_key(str(case.get("cf_group") or case["case_id"])),
        "cf_group": case.get("cf_group"),
        "zh_slots": dict(case.get("zh_slots") or {}),
        "source_role": "schema-program",
        "teacher_model": teacher_model,
        "prompt_version": PROMPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "high_risk": bool(case.get("high_risk")),
    }
    if has_route_id_gold(row):
        raise ValueError("route_id_leak")
    return row


def freeze_family_splits(cases: list[dict]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        split = str(case.get("split") or "")
        cf = str(case.get("cf_group") or "")
        fam = str(case.get("family") or "")
        if cf:
            groups[cf].add(split)
        if fam:
            families[fam].add(split)
    cross = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    return {
        "n": len(cases),
        "n_train": sum(1 for c in cases if c.get("split") == "train"),
        "n_valid": sum(1 for c in cases if c.get("split") == "valid"),
        "cf_cross_split": cross,
        "ok": not cross,
    }


def isolation_queries(rows: list[dict]) -> set[str]:
    return {str(r.get("query") or "").strip() for r in rows if str(r.get("query") or "").strip()}


def query_overlaps(train_rows: list[dict], eval_rows: list[dict]) -> list[str]:
    blocked = isolation_queries(eval_rows)
    hits = []
    for row in train_rows:
        q = str(row.get("query") or "").strip()
        if q and q in blocked:
            hits.append(f"{row.get('sample_id') or row.get('case_id')}:{q[:48]}")
    return hits


def kind_balance(rows: list[dict], key: str = "kind") -> dict:
    return dict(Counter(str(r.get(key)) for r in rows))
