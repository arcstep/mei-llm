#!/usr/bin/env python3
"""Freeze a conservative natural-Chinese augmentation for SFT-v3.

The structural v7 release deliberately obtains complete 147-tool coverage
from a deterministic schema program.  This companion release admits a small,
source-balanced subset of the earlier multi-teacher query candidates.  The
teacher is allowed to vary wording only: tool identity, arguments, schemas,
serializer output, candidates and split isolation are recomputed locally.

Historical train candidates become augmentation rows.  Historical valid
candidates are never trained; they are partitioned by counterfactual group
into an independent cross-generator dev/test bank.  Neither the frozen v7
release nor eval-v6 is rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import contracts.sft_v3_contract_51m as contract


AUGMENTATION_ID = "mei-1.0-51m-tool-sft-natural-aug300m-v1"
SCHEMA_ID = "mei-sft-natural-augmentation-v1"
ADMISSION_ID = "mei-sft-natural-admission-v1"
SOURCE_RELEASE_ID = "mei-1.0-51m-tool-sft-v2-agent300m-v1"
SOURCE_RELEASE_DIR = contract.DEFAULT_RELEASE_ROOT / SOURCE_RELEASE_ID
PARENT_RELEASE_DIR = contract.DEFAULT_RELEASE_ROOT / contract.RELEASE_ID
EVAL_DIR = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID
DEFAULT_OUTPUT_ROOT = contract.DEFAULT_RELEASE_ROOT

ALLOWED_TEACHERS = {
    "deepseek-v4-flash-0731",
    "qwen-plus-2025-12-01",
    "qwen3.7-plus",
}
RETRIEVAL_TRAIN_PER_TOOL = 24
FULLCALL_TRAIN_PER_KIND_TOOL = 10
CROSSGEN_PER_KIND_TOOL_SPLIT = 8
FULLCALL_FIXED_STEPS = 4_000
TRAILING_NUMBER = re.compile(r"(?<!\d)(\d{1,4})\s*$")
NUMBER_FRAGMENT = re.compile(r"\d+(?:\.\d+)?")
CHINESE_NUMBER_RUN = re.compile(r"[零〇一二两三四五六七八九十百千万亿]{3,}")
NUMBER_UNIT = re.compile(
    r"^(?:℃|°C|度|%|分钟|小时|点|元|块|个|档|号|年|月|日|次|米|公里|人|份|张|台|条)"
)
REPEATED_CHARACTER = re.compile(r"(.)\1{2,}")
TIME_PHRASE = re.compile(
    r"(?:时间(?:是|为|在)?|几点|点半|上午|中午|下午|傍晚|晚上|凌晨|今晚|明天|后天|日期)"
)
DEFERRED_ACTION = re.compile(
    r"(?:过会儿|稍后|等会儿|待会儿|等下|一会儿|明天|后天|今晚|下周|以后|之后再)"
)
TIME_SCHEMA_MARKERS = (
    "time",
    "date",
    "start",
    "end",
    "duration",
    "deadline",
    "due",
    "day",
    "hour",
    "minute",
    "时间",
    "日期",
    "时长",
    "截止",
)
WHITESPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()【】\[\]<>《》…·—_-]+")
PHYSICAL_ZERO_ARG_TOOLS = {
    "bow",
    "come_here",
    "nod",
    "shake_head",
    "sit",
    "stand",
    "wave",
}


def source_spec(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(contract.ROOT)),
        "sha256": contract.sha_file(path),
        "bytes": path.stat().st_size,
    }


def normalize_query(value: Any) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def query_signature(value: Any) -> str:
    text = normalize_query(value).casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "<n>", text)
    return PUNCTUATION.sub("", text)


def scalar_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        result: list[Any] = []
        for item in value.values():
            result.extend(scalar_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(scalar_values(item))
        return result
    return [value]


def query_rejection(row: dict[str, Any], arguments: dict[str, Any] | None = None) -> str | None:
    query = normalize_query(row.get("query"))
    if len(query) < 4:
        return "query_too_short"
    if len(query) > 160:
        return "query_too_long"
    if contract.EVAL_MARKER.search(query):
        return "eval_marker"
    if query.startswith(("{", "[")) or re.search(r"[\"']?query[\"']?\s*[:：]", query, re.IGNORECASE):
        return "serialized_wrapper_fragment"
    if any(value in query for value in ("请直接，请", "请为当前对象，请", "，，", "。。", "？？")):
        return "mechanical_punctuation"
    if REPEATED_CHARACTER.search(query):
        return "repeated_character_filler"
    protected_slots = row.get("zh_slots") or (
        (row.get("natural_source") or {}).get("protected_slots") or {}
    )
    protected_values = [
        normalize_query(value)
        for value in scalar_values(protected_slots)
        if value is not None
    ]
    for match in CHINESE_NUMBER_RUN.finditer(query):
        rendered = match.group(0)
        if any(rendered in value for value in protected_values):
            continue
        if NUMBER_UNIT.match(query[match.end() :]):
            continue
        return "unlicensed_chinese_number_run"
    match = TRAILING_NUMBER.search(query)
    if match:
        terminal = match.group(1)
        grounded_numbers = {
            str(value)
            for value in scalar_values(arguments or {})
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if terminal not in grounded_numbers:
            return "synthetic_numeric_suffix"
    if arguments is not None:
        visible_facts = normalize_query(row.get("system_facts") or row.get("scene"))
        grounded_texts = [str(value) for value in scalar_values(arguments)]
        for number in NUMBER_FRAGMENT.finditer(query):
            rendered = number.group(0)
            if rendered in visible_facts or any(rendered in value for value in grounded_texts):
                continue
            if NUMBER_UNIT.match(query[number.end() :]):
                continue
            return "ungrounded_numeric_fragment"
    return None


def schema_supports_time(tool: dict[str, Any]) -> bool:
    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    schema_text = " ".join(
        str(value)
        for name in sorted(properties, key=lambda value: str(value).encode("utf-8"))
        for value in (
            name,
            (properties[name] or {}).get("description", "")
            if isinstance(properties[name], dict)
            else "",
            (properties[name] or {}).get("format", "")
            if isinstance(properties[name], dict)
            else "",
        )
    ).casefold()
    return any(marker in schema_text for marker in TIME_SCHEMA_MARKERS)


def retrieval_query_rejection(
    row: dict[str, Any], tool: dict[str, Any]
) -> str | None:
    """Reject teacher additions that cannot be verified from retrieval metadata.

    Retrieval labels intentionally contain no executable arguments, so a free
    number introduced by a teacher cannot be assumed to be a legitimate slot.
    The only auditable exception is a number already present in a protected
    source slot (for example the ``200`` in ``SKU-B200``).  Time expressions
    are likewise admitted only for tools whose schema actually has a temporal
    field.  The release is an augmentation, so false negatives are safer than
    teaching contaminated wording.
    """

    rejection = query_rejection(row)
    if rejection:
        return rejection
    query = normalize_query(row.get("query"))
    slots = row.get("zh_slots") or (
        (row.get("natural_source") or {}).get("protected_slots") or {}
    )
    if not isinstance(slots, dict):
        slots = {}
    protected = [normalize_query(value) for value in scalar_values(slots) if value is not None]
    for number in NUMBER_FRAGMENT.finditer(query):
        rendered = number.group(0)
        if not any(rendered in value for value in protected):
            return "retrieval_unlicensed_numeric_fragment"

    if TIME_PHRASE.search(query) and not schema_supports_time(tool):
        return "retrieval_schema_irrelevant_time_phrase"
    return None


def variant_for(row: dict[str, Any]) -> int:
    digest = contract.sha_bytes(
        contract.canonical_bytes(
            [row.get("sample_id"), row.get("cf_group"), normalize_query(row.get("query"))]
        )
    )
    return int(digest[:8], 16)


def selected_for(
    tool_name: str,
    tools: Sequence[dict[str, Any]],
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_name = {str(tool["name"]): tool for tool in tools}
    if tool_name not in by_name:
        raise KeyError(tool_name)
    return contract.selected_tools(by_name[tool_name], tools, variant_for(row))


def grounding_ok(row: dict[str, Any], arguments: dict[str, Any]) -> bool:
    # This release adds language-surface variation, not fact-derived action
    # planning.  Every executable argument must therefore be recoverable from
    # the user query itself.  The frozen structural/Agent banks retain the
    # separately audited cases that legitimately derive values from verified
    # facts or ToolResultV2.
    visible = normalize_query(row.get("query"))
    aliases = row.get("zh_slots") or (
        (row.get("natural_source") or {}).get("grounding_aliases") or {}
    )
    if not isinstance(aliases, dict):
        aliases = {}
    for key, value in arguments.items():
        for scalar in scalar_values(value):
            if contract.value_grounded_in_query(scalar, visible):
                continue
            alias = aliases.get(key)
            if alias is not None and normalize_query(alias) in visible:
                continue
            return False
    return True


def accepted_grounding_aliases(
    row: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, str]:
    visible = normalize_query(row.get("query"))
    aliases = row.get("zh_slots") or {}
    if not isinstance(aliases, dict):
        return {}
    output: dict[str, str] = {}
    for key, value in arguments.items():
        if all(contract.value_grounded_in_query(scalar, visible) for scalar in scalar_values(value)):
            continue
        alias = normalize_query(aliases.get(key))
        if alias and alias in visible:
            output[str(key)] = alias
    return output


def teacher_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_release_id": SOURCE_RELEASE_ID,
        "source_sample_id": row.get("sample_id"),
        "source_cf_group": row.get("cf_group") or row.get("counterfactual_group"),
        "source_split": row.get("split"),
        "teacher_model": row.get("teacher_model"),
        "prompt_version": row.get("prompt_version"),
        "request_sha256": row.get("request_sha256"),
        "response_sha256": row.get("response_sha256"),
    }


def convert_retrieval(
    row: dict[str, Any], tools: Sequence[dict[str, Any]], split: str
) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("teacher_model") not in ALLOWED_TEACHERS:
        return None, "teacher_not_admitted"
    if row.get("status") != "accepted" or row.get("compiler_ok") is not True:
        return None, "source_not_accepted"
    gold = str(row.get("gold_tool") or "")
    by_name = {str(tool["name"]): tool for tool in tools}
    names = set(by_name)
    if not gold or gold not in names:
        return None, "gold_not_portable"
    rejection = retrieval_query_rejection(row, by_name[gold])
    if rejection:
        return None, rejection
    selected, negatives = selected_for(gold, tools, row)
    query = normalize_query(row["query"])
    sample_id = contract.stable_id("NRET", split, row.get("sample_id"), query)
    natural_source = teacher_meta(row)
    if isinstance(row.get("zh_slots"), dict) and row["zh_slots"]:
        natural_source["protected_slots"] = dict(row["zh_slots"])
    return {
        "sample_id": sample_id,
        "case_id": sample_id,
        "cf_group": "NRETG-" + str(row.get("cf_group") or row.get("sample_id")),
        "task": "retrieval",
        "split": split,
        "family": str(next(tool for tool in tools if tool["name"] == gold).get("family") or "unknown"),
        "kind": "hard_positive",
        "query": query,
        "gold_tool": gold,
        "hard_negatives": negatives,
        "catalog_tools": selected,
        "retrieval_encoding_id": contract.RETRIEVAL_ENCODING_ID,
        "retrieval_max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        "source_role": "admitted-multiteacher-natural-query",
        "generator_version": ADMISSION_ID,
        "tool_universe_id": "mei-51m-portable-tool-universe-v2",
        "natural_source": natural_source,
        "status": "accepted",
    }, None


REFUSAL_REASON = {
    "fault": "tool_state_fault",
    "illegal_pair": "unknown_slot_value",
    "missing": "missing_slot",
    "no_rsv": "missing_external_fact",
    "noop": "ambiguous_scope",
    "offtopic": "offtopic",
    "scene_conflict": "state_conflict",
}


def lexical_units(value: Any) -> set[str]:
    text = normalize_query(value).casefold()
    ascii_words = set(re.findall(r"[a-z0-9_]+", text))
    han = "".join(re.findall(r"[\u3400-\u9fff]", text))
    return ascii_words | {han[index : index + 2] for index in range(max(0, len(han) - 1))}


def refusal_anchor(row: dict[str, Any], tools: Sequence[dict[str, Any]]) -> str:
    by_name = {str(tool["name"]): tool for tool in tools}
    for key in ("gold_name", "candidate_tool"):
        preferred = str(row.get(key) or "")
        if preferred in by_name:
            return preferred
    query = normalize_query(row.get("query"))
    if str(row.get("toolset_id") or "") == "mei-park-room-v1":
        if "取消" in query and "预约" in query and "cancel_reservation" in by_name:
            return "cancel_reservation"
        if any(value in query for value in ("策略", "舒适温度", "办公时段", "预冷", "宽限")) and "update_room_policy" in by_name:
            return "update_room_policy"
        if any(value in query for value in ("预约", "预订")) and "create_or_update_reservation" in by_name:
            return "create_or_update_reservation"
        if any(value in query for value in ("空调", "冷气", "制冷", "灯", "照明", "温度")) and "control_room_devices" in by_name:
            return "control_room_devices"
    if str(row.get("kind") or "") != "offtopic":
        intent_rules = (
            (("取消订单", "取消外卖", "退餐", "退了"), "cancel_order"),
            (("点餐", "点一个", "点一份", "送一份", "牛肉面", "巨无霸"), "order_food"),
            (("关门", "关上前门", "关上后门", "关闭前门", "关闭后门"), "close_door"),
            (("开门", "打开前门", "打开后门", "开启前门", "开启后门"), "open_door"),
            (("去房间", "去厨房", "去客厅", "去门口", "前往", "走到"), "go_to"),
            (("点头",), "nod"),
            (("摇头",), "shake_head"),
            (("挥手",), "wave"),
            (("鞠躬",), "bow"),
            (("坐下", "坐一下"), "sit"),
            (("站起来", "起立"), "stand"),
            (("停下", "停住", "暂停"), "stop"),
            (("过来", "到我这里"), "come_here"),
            (("指出", "指一下", "指向"), "point"),
            (("倒垃圾", "扔垃圾", "垃圾袋"), "take_out_trash"),
        )
        for phrases, name in intent_rules:
            if name in by_name and any(phrase in query for phrase in phrases):
                return name
    candidates: list[str] = []
    for value in row.get("retrieved_tools") or []:
        name = str(value)
        if name in by_name and name not in candidates:
            candidates.append(name)
    for value in row.get("catalog_tools") or []:
        name = str(value.get("name") or "") if isinstance(value, dict) else str(value)
        if name in by_name and name not in candidates:
            candidates.append(name)
    query_units = lexical_units(row.get("query"))
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda name: (
                -len(query_units & lexical_units(contract.retrieval_tool_text(by_name[name]))),
                name.encode("utf-8"),
            ),
        )
        return ranked[0]
    ordered = sorted(by_name, key=lambda value: value.encode("utf-8"))
    return ordered[variant_for(row) % len(ordered)]


def convert_fullcall(
    row: dict[str, Any], tools: Sequence[dict[str, Any]], split: str
) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("teacher_model") not in ALLOWED_TEACHERS:
        return None, "teacher_not_admitted"
    if row.get("generator_version") != "sft-synth-v2":
        return None, "non_single_step_source"
    if row.get("status") != "accepted" or row.get("compiler_ok") is not True:
        return None, "source_not_accepted"
    if row.get("history") or row.get("prior_tool_results") or row.get("tool_results"):
        return None, "nonempty_history_or_results"
    answers = list(row.get("answers") or [])
    if len(answers) > 1:
        return None, "multiple_calls"
    by_name = {str(tool["name"]): tool for tool in tools}
    if answers:
        answer = answers[0]
        name = str(answer.get("name") or "")
        arguments = answer.get("arguments") or {}
        if name not in by_name:
            return None, "gold_not_portable"
        if not contract.arguments_match_schema(arguments, by_name[name].get("parameters") or {}):
            return None, "schema_invalid"
        query = normalize_query(row.get("query"))
        if DEFERRED_ACTION.search(query) and not schema_supports_time(by_name[name]):
            return None, "deferred_action_without_schedule_slot"
        if name in PHYSICAL_ZERO_ARG_TOOLS and (
            "让我" in query
            or re.search(r"我想(?!让你|请你|让机器人|让它)", query)
        ):
            return None, "ambiguous_action_subject"
        if arguments and not grounding_ok(row, arguments):
            return None, "arguments_not_grounded"
        properties = (by_name[name].get("parameters") or {}).get("properties") or {}
        if not arguments and properties:
            return None, "configurable_empty_call"
        kind = "execute"
        reason = "ready_to_execute"
        anchor = name
        normalized_answers = [{"name": name, "arguments": arguments}]
    else:
        source_kind = str(row.get("kind") or "")
        if source_kind not in REFUSAL_REASON:
            return None, "unsupported_refusal_kind"
        kind = "refuse"
        reason = REFUSAL_REASON[source_kind]
        anchor = refusal_anchor(row, tools)
        normalized_answers = []
        arguments = {}
    rejection = query_rejection(row, arguments)
    if rejection:
        return None, rejection
    selected, negatives = selected_for(anchor, tools, row)
    query = normalize_query(row["query"])
    sample_id = contract.stable_id("NFC", split, row.get("sample_id"), query)
    facts = normalize_query(row.get("system_facts") or row.get("scene"))
    natural_source = teacher_meta(row)
    if normalized_answers:
        natural_source["grounding_aliases"] = accepted_grounding_aliases(
            row, normalized_answers[0]["arguments"]
        )
    return {
        "sample_id": sample_id,
        "case_id": sample_id,
        "cf_group": "NFCG-" + str(row.get("cf_group") or row.get("counterfactual_group") or row.get("sample_id")),
        "task": "fullcall",
        "split": split,
        "family": str(row.get("family") or by_name[anchor].get("family") or "unknown"),
        "kind": kind,
        "query": query,
        "system_facts": facts,
        "context": {"locale": "zh-CN", **({"facts": [facts]} if facts else {})},
        "evidence": [],
        "history": [],
        "permissions": {},
        "state": {},
        "mw": {},
        "prior_calls": [],
        "prior_tool_results": [],
        "tool_results": [],
        "slot_provenance": [],
        "candidate_tool": anchor,
        "gold_name": normalized_answers[0]["name"] if normalized_answers else None,
        "gold_args": normalized_answers[0]["arguments"] if normalized_answers else {},
        "answers": normalized_answers,
        "target_text": contract.serialize_tool_target(normalized_answers),
        "reason_code": reason,
        "retrieved_tools": [str(tool["name"]) for tool in selected],
        "oracle_top5": selected,
        "hard_negatives": negatives,
        "serializer": contract.SERIALIZER_ID,
        "prompt_framing": contract.PROMPT_FRAMING_ID,
        "wire_version": contract.WIRE_ID,
        "source_role": "admitted-multiteacher-natural-query",
        "generator_version": ADMISSION_ID,
        "natural_source": natural_source,
        "status": "accepted",
    }, None


def convert_all(
    rows: Iterable[dict[str, Any]],
    converter: Callable[[dict[str, Any], Sequence[dict[str, Any]], str], tuple[dict[str, Any] | None, str | None]],
    tools: Sequence[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        converted, reason = converter(row, tools, split)
        if converted is None:
            rejected[str(reason or "unknown")] += 1
            continue
        # An identical normalized user request must never carry two different
        # targets or candidate anchors inside one task bank.
        key = query_signature(converted["query"])
        if key in seen:
            rejected["duplicate_signature_target"] += 1
            continue
        seen.add(key)
        output.append(converted)
    return output, rejected


def balanced_select(
    rows: Sequence[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], tuple[str, ...]],
    limit: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: contract.canonical_bytes(value)):
        by_teacher: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in groups[key]:
            teacher = str((row.get("natural_source") or {}).get("teacher_model") or "unknown")
            by_teacher[teacher].append(row)
        for values in by_teacher.values():
            values.sort(key=lambda row: str(row["sample_id"]).encode("utf-8"))
        teachers = sorted(by_teacher, key=lambda value: value.encode("utf-8"))
        offset = 0
        taken = 0
        while taken < limit and any(offset < len(by_teacher[name]) for name in teachers):
            for teacher in teachers:
                values = by_teacher[teacher]
                if offset < len(values) and taken < limit:
                    selected.append(values[offset])
                    taken += 1
            offset += 1
    return sorted(selected, key=lambda row: str(row["sample_id"]).encode("utf-8"))


def reserve_crossgen_groups(
    rows: Sequence[dict[str, Any]], modulus: int = 10
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    crossgen: list[dict[str, Any]] = []
    for row in rows:
        group = str((row.get("natural_source") or {}).get("source_cf_group") or row["cf_group"])
        bucket = int(contract.sha_bytes(group.encode("utf-8"))[:8], 16) % modulus
        (crossgen if bucket == 0 else train).append(row)
    return train, crossgen


def dedupe_query_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda value: str(value["sample_id"]).encode("utf-8")):
        signature = query_signature(row["query"])
        if signature not in seen:
            output.append(row)
            seen.add(signature)
    return output


def partition_crossgen(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {"dev": [], "test": []}
    for row in rows:
        source_group = str((row.get("natural_source") or {}).get("source_cf_group") or row["cf_group"])
        split = "dev" if int(contract.sha_bytes(source_group.encode("utf-8"))[:8], 16) % 2 == 0 else "test"
        item = dict(row)
        item["split"] = split
        item["sample_id"] = contract.stable_id("XGEN", split, row["sample_id"])
        item["case_id"] = item["sample_id"]
        item["cf_group"] = "XGENG-" + source_group
        output[split].append(item)
    return output


def cap_crossgen(
    rows: Sequence[dict[str, Any]], task: str
) -> dict[str, list[dict[str, Any]]]:
    partitioned = partition_crossgen(rows)
    result: dict[str, list[dict[str, Any]]] = {}
    for split, values in partitioned.items():
        if task == "retrieval":
            key = lambda row: (str(row["gold_tool"]),)
        else:
            key = lambda row: (str(row["kind"]), str(row["candidate_tool"]))
        result[split] = balanced_select(values, key, CROSSGEN_PER_KIND_TOOL_SPLIT)
    return result


def global_tool_round_robin(
    rows: Sequence[dict[str, Any]], total: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("candidate_tool") or row.get("gold_tool") or "")].append(row)
    for values in groups.values():
        values.sort(key=lambda row: str(row["sample_id"]).encode("utf-8"))
    names = sorted(groups, key=lambda value: value.encode("utf-8"))
    output: list[dict[str, Any]] = []
    offset = 0
    while len(output) < total and any(offset < len(groups[name]) for name in names):
        for name in names:
            if offset < len(groups[name]) and len(output) < total:
                output.append(groups[name][offset])
        offset += 1
    return sorted(output, key=lambda row: str(row["sample_id"]).encode("utf-8"))


def balance_crossgen_fullcall(
    values: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, rows in values.items():
        by_kind = {
            kind: [row for row in rows if row.get("kind") == kind]
            for kind in ("execute", "refuse")
        }
        target = min(len(by_kind["execute"]), len(by_kind["refuse"]))
        if target <= 0:
            raise RuntimeError(f"cross-generator {split} lacks execute/refuse coverage")
        output[split] = sorted(
            [
                *global_tool_round_robin(by_kind["execute"], target),
                *global_tool_round_robin(by_kind["refuse"], target),
            ],
            key=lambda row: str(row["sample_id"]).encode("utf-8"),
        )
    return output


def distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    teachers = Counter(
        str((row.get("natural_source") or {}).get("teacher_model") or "unknown")
        for row in rows
    )
    tools = {
        str(row.get("gold_tool") or row.get("candidate_tool") or "")
        for row in rows
    }
    signatures = {query_signature(row.get("query")) for row in rows}
    characters = "".join(normalize_query(row.get("query")) for row in rows)
    bigrams = {characters[index : index + 2] for index in range(max(0, len(characters) - 1))}
    return {
        "rows": len(rows),
        "unique_query_signatures": len(signatures),
        "tools": len(tools - {""}),
        "teachers": dict(sorted(teachers.items())),
        "families": len({str(row.get("family") or "") for row in rows}),
        "kinds": dict(sorted(Counter(str(row.get("kind") or "") for row in rows).items())),
        "character_bigrams": len(bigrams),
        "query_chars": len(characters),
    }


def all_eval_signatures() -> set[str]:
    values: set[str] = set()
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        values.update(query_signature(row.get("query")) for row in contract.load_jsonl(path))
    return values


def verify_isolation(
    train_rows: Sequence[dict[str, Any]],
    crossgen: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    signatures = {
        "train": {query_signature(row["query"]) for row in train_rows},
        "dev": {query_signature(row["query"]) for row in crossgen["dev"]},
        "test": {query_signature(row["query"]) for row in crossgen["test"]},
    }
    groups = {
        "train": {
            str((row.get("natural_source") or {}).get("source_cf_group") or row["cf_group"])
            for row in train_rows
        },
        **{
            split: {
                str((row.get("natural_source") or {}).get("source_cf_group") or row["cf_group"])
                for row in rows
            }
            for split, rows in crossgen.items()
        },
    }
    frozen = all_eval_signatures()
    overlap = {
        "train_dev": len(signatures["train"] & signatures["dev"]),
        "train_test": len(signatures["train"] & signatures["test"]),
        "dev_test": len(signatures["dev"] & signatures["test"]),
        "train_frozen_eval": len(signatures["train"] & frozen),
        "dev_frozen_eval": len(signatures["dev"] & frozen),
        "test_frozen_eval": len(signatures["test"] & frozen),
        "source_group_train_dev": len(groups["train"] & groups["dev"]),
        "source_group_train_test": len(groups["train"] & groups["test"]),
        "source_group_dev_test": len(groups["dev"] & groups["test"]),
    }
    return {
        "schema": "mei-sft-natural-isolation-receipt-v1",
        "status": "passed" if not any(overlap.values()) else "failed",
        "signature_normalization": "nfkc-lower-number-placeholder-strip-punctuation-v1",
        "overlap": overlap,
    }


def artifact_spec(payload: bytes, rows: int | None = None) -> dict[str, Any]:
    result = {"sha256": contract.sha_bytes(payload), "bytes": len(payload)}
    if rows is not None:
        result["rows"] = rows
    return result


def build_payloads(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    parent_manifest = contract.load_json(PARENT_RELEASE_DIR / "manifest.json")
    source_manifest = contract.load_json(SOURCE_RELEASE_DIR / "manifest.json")
    if parent_manifest.get("release_id") != contract.RELEASE_ID:
        raise RuntimeError("parent SFT-v3 release identity drifted")
    if source_manifest.get("release_id") != SOURCE_RELEASE_ID:
        raise RuntimeError("historical multi-teacher release identity drifted")
    tools_doc = contract.load_json(PARENT_RELEASE_DIR / "tool-universe.json")
    tools = list(tools_doc["tools"])

    retrieval_train_raw = contract.load_jsonl(SOURCE_RELEASE_DIR / "retrieval.train.jsonl")
    retrieval_valid_raw = contract.load_jsonl(SOURCE_RELEASE_DIR / "retrieval.valid.jsonl")
    fullcall_train_raw = contract.load_jsonl(SOURCE_RELEASE_DIR / "full-call.train.jsonl")
    fullcall_valid_raw = contract.load_jsonl(SOURCE_RELEASE_DIR / "full-call.valid.jsonl")

    retrieval_train_all, retrieval_train_rejected = convert_all(
        retrieval_train_raw, convert_retrieval, tools, "train"
    )
    retrieval_valid_all, retrieval_valid_rejected = convert_all(
        retrieval_valid_raw, convert_retrieval, tools, "crossgen"
    )
    fullcall_train_all, fullcall_train_rejected = convert_all(
        fullcall_train_raw, convert_fullcall, tools, "train"
    )
    fullcall_valid_all, fullcall_valid_rejected = convert_all(
        fullcall_valid_raw, convert_fullcall, tools, "crossgen"
    )

    # Preserve the historical valid side as the stronger cross-generator
    # holdout.  The old release checked exact query identity only; remove any
    # train candidate that collides after NFKC, number abstraction and
    # punctuation stripping before applying per-tool caps.
    heldout_signatures = {
        query_signature(row["query"])
        for row in [*retrieval_valid_all, *fullcall_valid_all]
    }
    retained_retrieval_train = []
    for row in retrieval_train_all:
        if query_signature(row["query"]) in heldout_signatures:
            retrieval_train_rejected["crossgen_signature_overlap"] += 1
        else:
            retained_retrieval_train.append(row)
    retrieval_train_all = retained_retrieval_train
    retained_fullcall_train = []
    for row in fullcall_train_all:
        if query_signature(row["query"]) in heldout_signatures:
            fullcall_train_rejected["crossgen_signature_overlap"] += 1
        else:
            retained_fullcall_train.append(row)
    fullcall_train_all = retained_fullcall_train

    retrieval_train_pool, retrieval_reserved = reserve_crossgen_groups(
        retrieval_train_all
    )
    fullcall_train_pool, fullcall_reserved = reserve_crossgen_groups(
        fullcall_train_all
    )
    retrieval_eval_pool = dedupe_query_rows(
        [*retrieval_valid_all, *retrieval_reserved]
    )
    fullcall_eval_pool = dedupe_query_rows([*fullcall_valid_all, *fullcall_reserved])
    reserved_signatures = {
        query_signature(row["query"])
        for row in [*retrieval_eval_pool, *fullcall_eval_pool]
    }
    retrieval_train_pool = [
        row
        for row in retrieval_train_pool
        if query_signature(row["query"]) not in reserved_signatures
    ]
    fullcall_train_pool = [
        row
        for row in fullcall_train_pool
        if query_signature(row["query"]) not in reserved_signatures
    ]

    retrieval_train = balanced_select(
        retrieval_train_pool,
        lambda row: (str(row["gold_tool"]),),
        RETRIEVAL_TRAIN_PER_TOOL,
    )
    fullcall_train = balanced_select(
        fullcall_train_pool,
        lambda row: (str(row["kind"]), str(row["candidate_tool"])),
        FULLCALL_TRAIN_PER_KIND_TOOL,
    )
    retrieval_crossgen = cap_crossgen(retrieval_eval_pool, "retrieval")
    fullcall_crossgen = balance_crossgen_fullcall(
        cap_crossgen(fullcall_eval_pool, "fullcall")
    )
    train_all = [*retrieval_train, *fullcall_train]
    crossgen_all = {
        split: [*retrieval_crossgen[split], *fullcall_crossgen[split]]
        for split in ("dev", "test")
    }
    isolation = verify_isolation(train_all, crossgen_all)
    if isolation["status"] != "passed":
        raise RuntimeError(f"natural augmentation isolation failed: {isolation['overlap']}")

    payloads: dict[str, bytes] = {}
    row_sets = {
        "natural-retrieval.train.jsonl": retrieval_train,
        "natural-full-call.train.jsonl": fullcall_train,
        "crossgen-retrieval.dev.jsonl": retrieval_crossgen["dev"],
        "crossgen-retrieval.test.jsonl": retrieval_crossgen["test"],
        "crossgen-full-call.dev.jsonl": fullcall_crossgen["dev"],
        "crossgen-full-call.test.jsonl": fullcall_crossgen["test"],
    }
    for name, rows in row_sets.items():
        payloads[name] = contract.jsonl_bytes(rows)
    admission = {
        "schema": "mei-sft-natural-admission-receipt-v1",
        "status": "passed",
        "admission_id": ADMISSION_ID,
        "teachers": sorted(ALLOWED_TEACHERS),
        "rules": {
            "teacher_changes_wording_only": True,
            "schema_and_candidates_recomputed_locally": True,
            "execute_arguments_require_query_grounding": True,
            "retrieval_numbers_require_protected_source_slot": True,
            "retrieval_time_phrases_require_temporal_schema": True,
            "synthetic_numeric_suffix_rejected": True,
            "unlicensed_chinese_number_runs_rejected": True,
            "repeated_character_fillers_rejected": True,
            "deferred_execute_requires_schedule_slot": True,
            "ambiguous_physical_action_subject_rejected": True,
            "model_visible_slot_provenance": False,
            "historical_valid_is_evaluation_only": True,
            "ten_percent_historical_train_groups_reserved_for_crossgen": True,
        },
        "raw": {
            "retrieval_train": len(retrieval_train_raw),
            "retrieval_valid": len(retrieval_valid_raw),
            "fullcall_train": len(fullcall_train_raw),
            "fullcall_valid": len(fullcall_valid_raw),
        },
        "admissible_before_caps": {
            "retrieval_train": len(retrieval_train_all),
            "retrieval_valid": len(retrieval_valid_all),
            "fullcall_train": len(fullcall_train_all),
            "fullcall_valid": len(fullcall_valid_all),
        },
        "reserved_crossgen_before_caps": {
            "retrieval_from_historical_train": len(retrieval_reserved),
            "fullcall_from_historical_train": len(fullcall_reserved),
            "retrieval_combined_pool": len(retrieval_eval_pool),
            "fullcall_combined_pool": len(fullcall_eval_pool),
        },
        "rejections": {
            "retrieval_train": dict(sorted(retrieval_train_rejected.items())),
            "retrieval_valid": dict(sorted(retrieval_valid_rejected.items())),
            "fullcall_train": dict(sorted(fullcall_train_rejected.items())),
            "fullcall_valid": dict(sorted(fullcall_valid_rejected.items())),
        },
        "selected": {name: distribution(rows) for name, rows in row_sets.items()},
    }
    payloads["admission-receipt.json"] = contract.canonical_bytes(admission) + b"\n"
    payloads["isolation-receipt.json"] = contract.canonical_bytes(isolation) + b"\n"

    source_specs = {
        "parent_manifest": source_spec(PARENT_RELEASE_DIR / "manifest.json"),
        "historical_manifest": source_spec(SOURCE_RELEASE_DIR / "manifest.json"),
        "historical_retrieval_train": source_spec(SOURCE_RELEASE_DIR / "retrieval.train.jsonl"),
        "historical_retrieval_valid": source_spec(SOURCE_RELEASE_DIR / "retrieval.valid.jsonl"),
        "historical_fullcall_train": source_spec(SOURCE_RELEASE_DIR / "full-call.train.jsonl"),
        "historical_fullcall_valid": source_spec(SOURCE_RELEASE_DIR / "full-call.valid.jsonl"),
        "frozen_eval_lock": source_spec(EVAL_DIR / "lock.json"),
        "freezer_source": source_spec(Path(__file__)),
    }
    outputs = {
        name: artifact_spec(payload, len(row_sets[name]) if name in row_sets else None)
        for name, payload in payloads.items()
    }
    fingerprint = contract.sha_bytes(
        contract.canonical_bytes(
            {
                "schema": SCHEMA_ID,
                "sources": source_specs,
                "outputs": outputs,
                "caps": {
                    "retrieval_train_per_tool": RETRIEVAL_TRAIN_PER_TOOL,
                    "fullcall_train_per_kind_tool": FULLCALL_TRAIN_PER_KIND_TOOL,
                    "crossgen_per_kind_tool_split": CROSSGEN_PER_KIND_TOOL_SPLIT,
                },
            }
        )
    )
    manifest = {
        "schema": SCHEMA_ID,
        "release_id": args.release_id,
        "status": "frozen",
        "product": contract.PRODUCT_ID,
        "release_fingerprint": fingerprint,
        "parent": {
            "release_id": contract.RELEASE_ID,
            "manifest_sha256": source_specs["parent_manifest"]["sha256"],
            "release_fingerprint": parent_manifest["release_fingerprint"],
        },
        "source_release": {
            "release_id": SOURCE_RELEASE_ID,
            "manifest_sha256": source_specs["historical_manifest"]["sha256"],
        },
        "weight_contract_id": contract.WEIGHT_CONTRACT_ID,
        "tokenizer_id": contract.TOKENIZER_ID,
        "portable_tool_universe_fingerprint": tools_doc["fingerprint"],
        "training_policy": {
            "base_rows": "consume parent SFT-v3 release unchanged",
            "augmentation_rows": [
                "natural-retrieval.train.jsonl",
                "natural-full-call.train.jsonl",
            ],
            "crossgen_rows": "evaluation_only",
            "source_balanced": True,
            "fullcall_fixed_steps": FULLCALL_FIXED_STEPS,
            "main_longitudinal_curve": "record parent-v7 and natural-aug-v1 as distinct data fingerprints",
        },
        "sources": source_specs,
        "outputs": outputs,
        "admission_receipt": {
            "file": "admission-receipt.json",
            "sha256": outputs["admission-receipt.json"]["sha256"],
        },
        "isolation_receipt": {
            "file": "isolation-receipt.json",
            "sha256": outputs["isolation-receipt.json"]["sha256"],
        },
    }
    payloads["manifest.json"] = contract.canonical_bytes(manifest) + b"\n"
    return payloads, manifest


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    payloads, manifest = build_payloads(args)
    target = args.output_root / args.release_id
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "target": str(target),
            "release_fingerprint": manifest["release_fingerprint"],
            "outputs": manifest["outputs"],
        }
    if target.exists():
        actual = {path.name for path in target.iterdir() if path.is_file()}
        if actual != set(payloads):
            raise RuntimeError("existing natural augmentation inventory differs")
        for name, payload in payloads.items():
            if (target / name).read_bytes() != payload:
                raise RuntimeError(f"existing natural augmentation differs: {name}")
        return {"ok": True, "reused": True, "path": str(target)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.release_id}-", dir=args.output_root) as name:
        temporary = Path(name) / args.release_id
        temporary.mkdir()
        for filename, payload in payloads.items():
            (temporary / filename).write_bytes(payload)
        temporary.replace(target)
    return {
        "ok": True,
        "reused": False,
        "path": str(target),
        "release_fingerprint": manifest["release_fingerprint"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-id", default=AUGMENTATION_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(freeze(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
