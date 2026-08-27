"""Program gold + isolated templates for mei-tool-sft-v1 and mei-tool-schema-v1 eval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from repo_paths import EVAL_SHARED_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

import sys

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from schema_render import load_toolset_json, schema_hash  # noqa: E402

SERIALIZER_ID = "mei-schema-serializer-v1"
GENERATOR_VERSION = "mei-tool-schema-v1"
TRAIN_TOOLSETS = ("needle-home-v0", "needle-invoice-v0", "mei-office-v0", "mei-type-v0")
HOLDOUT_TOOLSETS = ("mei-retail-v0",)
EVAL_ONLY_TOOLSETS = ("mei-office-mut-v0", "mei-office-rename-v0")
VRM_SEEN = "needle-vrm-agent-v0"

TRAIN_STEMS = (
    "训练登记：{body}",
    "按办公口径处理：{body}",
    "帮我做这一步：{body}",
    "路由这条：{body}",
)
EVAL_STEMS = (
    "评测执行：{body}",
    "holdout 请路由：{body}",
    "闭集检查：{body}",
    "schema 题：{body}",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ts(toolset_id: str) -> dict[str, Any]:
    return load_toolset_json(toolset_id, root=EVAL_SHARED_ROOT)


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def tool_index(toolset: dict) -> dict[str, dict]:
    return {str(t["name"]): t for t in toolset.get("tools") or []}


def gold_call(name: str, arguments: dict | None = None) -> list[dict]:
    return [{"name": name, "arguments": arguments or {}}]


def row_base(
    *,
    sample_id: str,
    query: str,
    toolset_id: str,
    answers: list[dict],
    split: str,
    family: str,
    kind: str,
    scene: str | None = None,
    extra: dict | None = None,
) -> dict:
    ts = load_ts(toolset_id)
    row = {
        "sample_id": sample_id,
        "item_id": sample_id if sample_id.startswith("EVAL-") else None,
        "split": split,
        "lang": "zh",
        "family": family,
        "kind": kind,
        "toolset_id": toolset_id,
        "schema_hash": schema_hash(ts),
        "schema_conditioned": True,
        "serializer": SERIALIZER_ID,
        "generator_version": GENERATOR_VERSION,
        "query": query,
        "answers": answers,
        "gold": {"function_calls": answers},
        "pass": "exact_match",
        "confidence_label": 0 if not answers else 1,
        "act": None,
        "provenance": "schema-program",
    }
    if scene:
        row["scene"] = scene
    if extra:
        row.update(extra)
    if row["item_id"] is None:
        row.pop("item_id")
        row.pop("gold", None)
        row.pop("pass", None)
    else:
        row.pop("answers", None)
    return row


def _stem(body: str, stems: tuple[str, ...], i: int) -> str:
    return stems[i % len(stems)].format(body=body)


# --- gold programs (canonical slots, never the full query) ---

HOME_CASES = [
    ("天气", "查成都天气", "get_weather", {"city": "成都"}),
    ("天气", "查深圳天气", "get_weather", {"city": "深圳"}),
    ("灯", "厨房灯亮度 40", "set_lights", {"room": "厨房", "brightness": 40}),
    ("灯", "客厅灯亮度 0", "set_lights", {"room": "客厅", "brightness": 0}),
]
INVOICE_CASES = [
    ("发票", "摘录供应商华为金额 88.5", "invoice", {"vendor": "华为", "total": 88.5}),
    ("发票", "摘录供应商联想金额 120", "invoice", {"vendor": "联想", "total": 120.0}),
]
OFFICE_CASES = [
    ("日程", "创建标题 standup", "create_event", {"title": "standup"}),
    ("日程", "创建标题 review 全天", "create_event", {"title": "review", "all_day": True}),
    ("音量", "音量调到 4", "set_volume", {"level": 4}),
    ("音量", "音量调到 0", "set_volume", {"level": 0}),
    ("报价", "查询茶叶数量 2.5", "lookup_price", {"item": "茶叶", "qty": 2.5}),
    ("报价", "查询咖啡数量 1", "lookup_price", {"item": "咖啡", "qty": 1.0}),
]
TYPE_CASES = [
    ("类型", "回显 alpha", "echo_text", {"text": "alpha"}),
    ("类型", "回显 beta", "echo_text", {"text": "beta"}),
    ("类型", "打开标志", "set_flag", {"on": True}),
    ("类型", "关闭标志", "set_flag", {"on": False}),
    ("类型", "计数 3", "set_count", {"n": 3}),
    ("类型", "计数 0", "set_count", {"n": 0}),
    ("类型", "比率 1.5", "set_rate", {"x": 1.5}),
    ("类型", "比率 0.25", "set_rate", {"x": 0.25}),
]
RETAIL_CASES = [
    ("订位", "订 4 人桌", "book_table", {"party_size": 4}),
    ("订位", "订 2 人户外桌", "book_table", {"party_size": 2, "outdoor": True}),
    ("收款", "收款 36.5 人民币", "charge_card", {"amount": 36.5, "currency": "CNY"}),
    ("收款", "收款 9 美元", "charge_card", {"amount": 9.0, "currency": "USD"}),
    ("标签", "打印 sku A100", "print_label", {"sku": "A100"}),
    ("标签", "打印 sku B200", "print_label", {"sku": "B200"}),
]
RENAME_CASES = [
    ("日程", "创建会议主题 standup", "add_meeting", {"topic": "standup"}),
    ("音量", "响度调到 4", "adjust_loudness", {"gain": 4}),
    ("报价", "报价茶叶数量 2.5", "quote_item", {"sku_name": "茶叶", "count": 2.5}),
]
MUT_CASES = [
    ("日程", "创建 standup 时长 30", "create_event", {"title": "standup", "duration_min": 30}),
    ("音量", "音量调到 7", "set_volume", {"level": 7}),
]
OFFTOPIC = ("离题", "今天股市怎样", [])
MISSING = [
    ("缺槽", "查天气但不给城市", "needle-home-v0"),
    ("缺槽", "订位但不给人数", "mei-retail-v0"),
    ("缺槽", "创建日程但不给标题", "mei-office-v0"),
]


def iter_execute_pool(toolset_id: str, cases: list[tuple], stems: tuple[str, ...]) -> Iterator[tuple[str, str, list[dict], str]]:
    for i, (family, body, name, args) in enumerate(cases):
        yield family, _stem(body, stems, i), gold_call(name, args), "execute"


def make_sft_rows(n_train: int, n_valid: int, *, seed_tag: str) -> list[dict]:
    pools: list[tuple[str, list]] = [
        ("needle-home-v0", HOME_CASES),
        ("needle-invoice-v0", INVOICE_CASES),
        ("mei-office-v0", OFFICE_CASES),
        ("mei-type-v0", TYPE_CASES),
    ]
    rows: list[dict] = []
    idx = 0

    def push(split: str, toolset_id: str, family: str, query: str, answers: list, kind: str, extra: dict | None = None):
        nonlocal idx
        idx += 1
        sid = f"SFT-TOOL-{seed_tag}-{idx:05d}"
        rows.append(
            row_base(
                sample_id=sid,
                query=query,
                toolset_id=toolset_id,
                answers=answers,
                split=split,
                family=family,
                kind=kind,
                extra=extra,
            )
        )

    target = n_train + n_valid
    cycle = 0
    while len(rows) < target:
        cycle += 1
        for toolset_id, cases in pools:
            for j, (family, body, name, args) in enumerate(cases):
                q = _stem(f"{body} #{cycle}-{j}", TRAIN_STEMS, cycle + j)
                kind = "execute"
                answers = gold_call(name, args)
                # mix negatives: missing / offtopic / no-match / rename-trap
                mode = (len(rows) + cycle) % 11
                extra = {"template_id": f"tpl-{toolset_id}-{j}", "counterfactual_id": None}
                if mode == 0:
                    family, q, answers, kind = "offtopic", _stem(f"聊聊天气之外的宇宙起源 #{cycle}", TRAIN_STEMS, cycle), [], "refuse"
                    extra["template_id"] = "tpl-offtopic"
                elif mode == 1:
                    family, q, answers, kind = "missing", _stem(f"执行{tool_index(load_ts(toolset_id))[name]['name']}但缺必填 #{cycle}", TRAIN_STEMS, cycle), [], "refuse"
                    extra["template_id"] = "tpl-missing"
                elif mode == 2:
                    family, q, answers, kind = "nomatch", _stem(f"请调用不存在的工具 warp_drive #{cycle}", TRAIN_STEMS, cycle), [], "refuse"
                    extra["template_id"] = "tpl-nomatch"
                elif mode == 3:
                    # same query, competing similar tool: still gold from this schema
                    q = _stem(f"{body}（本 schema 内选择）#{cycle}", TRAIN_STEMS, cycle + 3)
                    extra["template_id"] = "tpl-compete"
                push("train" if len(rows) < n_train else "valid", toolset_id, family, q, answers, kind, extra)
                if len(rows) >= target:
                    break
            if len(rows) >= target:
                break
    for r in rows:
        if r["split"] == "valid" and r["sample_id"].startswith("SFT-TOOL"):
            pass
    # relabel last n_valid as valid (already done via length)
    return rows


def _eval_item(i: int, slice_id: str, toolset_id: str, family: str, query: str, answers: list, kind: str, extra: dict | None = None) -> dict:
    sid = f"EVAL-SCHEMA-{slice_id}-{i:04d}"
    row = row_base(
        sample_id=sid,
        query=query,
        toolset_id=toolset_id,
        answers=answers,
        split="eval",
        family=family,
        kind=kind,
        extra=extra,
    )
    row["slice"] = slice_id
    row["gold"] = {"function_calls": answers}
    return row


def make_eval_rows() -> tuple[list[dict], list[dict]]:
    eval_rows: list[dict] = []
    dev_rows: list[dict] = []
    i = 0

    def add(slice_id, toolset_id, family, query, answers, kind, extra=None, dev=False):
        nonlocal i
        i += 1
        row = _eval_item(i, slice_id, toolset_id, family, query, answers, kind, extra)
        (dev_rows if dev else eval_rows).append(row)

    # S0 seen
    for n, (ts, cases) in enumerate((
        ("needle-home-v0", HOME_CASES),
        ("needle-invoice-v0", INVOICE_CASES),
        ("mei-office-v0", OFFICE_CASES),
        ("mei-type-v0", TYPE_CASES),
    )):
        for j, (family, body, name, args) in enumerate(cases):
            add("S0", ts, family, _stem(body, EVAL_STEMS, j + n), gold_call(name, args), "execute", {"template_id": f"eval-s0-{ts}-{j}"}, dev=j == 0)
            add("S0", ts, family, _stem(body + "再确认", EVAL_STEMS, j + 2), gold_call(name, args), "execute", {"template_id": f"eval-s0b-{ts}-{j}"})
    # S1 unseen retail
    for j, (family, body, name, args) in enumerate(RETAIL_CASES):
        add("S1", "mei-retail-v0", family, _stem(body, EVAL_STEMS, j), gold_call(name, args), "execute", {"template_id": f"eval-s1-{j}"}, dev=j == 0)
        add("S1", "mei-retail-v0", family, _stem(body + "立刻", EVAL_STEMS, j + 1), gold_call(name, args), "execute")
    # S2 mutation: missing duration must refuse; complete duration executes
    add("S2", "mei-office-mut-v0", "mutation", _stem("创建标题 standup", EVAL_STEMS, 0), [], "refuse", {"template_id": "eval-s2-missing-required", "mutation": "required+enum"}, True)
    add("S2", "mei-office-mut-v0", "mutation", _stem("创建 standup 时长 30", EVAL_STEMS, 1), gold_call("create_event", {"title": "standup", "duration_min": 30}), "execute")
    add("S2", "mei-office-mut-v0", "mutation", _stem("创建标题 picnic", EVAL_STEMS, 2), [], "refuse", {"mutation": "enum-out"})
    add("S2", "mei-office-mut-v0", "mutation", _stem("音量调到 7", EVAL_STEMS, 3), gold_call("set_volume", {"level": 7}), "execute")
    # S3 rename
    for j, (family, body, name, args) in enumerate(RENAME_CASES):
        add("S3", "mei-office-rename-v0", family, _stem(body, EVAL_STEMS, j), gold_call(name, args), "execute", {"template_id": f"eval-s3-{j}"}, dev=j == 0)
    add("S3", "mei-office-rename-v0", "rename-leak", _stem("调用 create_event 标题 standup", EVAL_STEMS, 9), [], "refuse", {"rename_trap": "create_event"})
    # S4 crosstalk: home-like intent under retail schema -> []
    add("S4", "mei-retail-v0", "crosstalk", _stem("把厨房灯亮度调到 40", EVAL_STEMS, 0), [], "refuse", {"crosstalk": "home-on-retail"}, True)
    add("S4", "mei-retail-v0", "crosstalk", _stem("查成都天气", EVAL_STEMS, 1), [], "refuse", {"crosstalk": "home-on-retail"})
    add("S4", "mei-office-v0", "crosstalk", _stem("订 4 人桌", EVAL_STEMS, 2), [], "refuse", {"crosstalk": "retail-on-office"})
    add("S4", "mei-type-v0", "crosstalk", _stem("收款 36.5 人民币", EVAL_STEMS, 3), [], "refuse", {"crosstalk": "retail-on-type"})
    # S5 type stress
    add("S5", "mei-type-v0", "type", _stem("关闭标志", EVAL_STEMS, 0), gold_call("set_flag", {"on": False}), "execute", {"scalar": "boolean"}, True)
    add("S5", "mei-type-v0", "type", _stem("打开标志", EVAL_STEMS, 1), gold_call("set_flag", {"on": True}), "execute", {"scalar": "boolean"})
    add("S5", "mei-type-v0", "type", _stem("计数 0", EVAL_STEMS, 2), gold_call("set_count", {"n": 0}), "execute", {"scalar": "integer"})
    add("S5", "mei-type-v0", "type", _stem("计数 12", EVAL_STEMS, 3), gold_call("set_count", {"n": 12}), "execute", {"scalar": "integer"})
    add("S5", "mei-type-v0", "type", _stem("比率 0.25", EVAL_STEMS, 4), gold_call("set_rate", {"x": 0.25}), "execute", {"scalar": "number"})
    add("S5", "mei-type-v0", "type", _stem("回显 gamma", EVAL_STEMS, 5), gold_call("echo_text", {"text": "gamma"}), "execute", {"scalar": "string"})
    add("S5", "mei-type-v0", "type", _stem("把计数设成 true", EVAL_STEMS, 6), [], "refuse", {"scalar": "type-mismatch"})
    # S6 refuse
    add("S6", "mei-office-v0", "offtopic", _stem("讲个笑话", EVAL_STEMS, 0), [], "refuse", {}, True)
    add("S6", "mei-office-v0", "missing", _stem("创建日程但不给标题", EVAL_STEMS, 1), [], "refuse")
    add("S6", "mei-retail-v0", "nomatch", _stem("请调用 warp_drive", EVAL_STEMS, 2), [], "refuse")
    add("S6", "needle-home-v0", "offtopic", _stem("月球几度", EVAL_STEMS, 3), [], "refuse")
    add("S6", "mei-type-v0", "missing", _stem("设置标志但没说开或关", EVAL_STEMS, 4), [], "refuse")
    add("S6", "mei-office-v0", "nomatch", _stem("发射火箭", EVAL_STEMS, 5), [], "refuse")
    return eval_rows, dev_rows


def isolation_names() -> dict[str, set[str]]:
    train_names: set[str] = set()
    for tid in TRAIN_TOOLSETS:
        train_names |= set(tool_index(load_ts(tid)))
    hold = set()
    for tid in HOLDOUT_TOOLSETS + ("mei-office-rename-v0",):
        hold |= set(tool_index(load_ts(tid)))
    return {"train_tool_names": train_names, "holdout_tool_names": hold - train_names}
