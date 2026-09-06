#!/usr/bin/env python3
"""Shared, deterministic SFT-v3 data and evaluation contract for mei-1.0-51m.

This module deliberately contains no MLX imports.  It is safe to use while a
live CPT run owns Metal and is the single source of truth for the synthetic
coverage supplement used by both the immutable training release and the
longitudinal evaluation lock.

The v3 supplement fixes a concrete defect in the historical v2 release: only
26/147 tools were gold retrieval targets and only 31/147 were gold full-call
targets.  Every deployable tool is represented uniformly here, with explicit
hard negatives and balanced execute/refuse targets.  MW disposition,
confidence calibration and narration remain separate capabilities and are not
collapsed into the full-call target.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from common._repo import ROOT


PRODUCT_ID = "mei-1.0-51m"
RELEASE_ID = "mei-1.0-51m-tool-sft-v3-300m-v7"
EVAL_ID = "mei-51m-longitudinal-eval-v6"
CONTRACT_ID = "mei-sft-data-contract-v3"
GENERATOR_ID = "mei-sft-v3-schema-grounded-coverage-v4"
SAMPLER_ID = "mei-sft-v3-tool-uniform-diverse-negative-v4"
SERIALIZER_ID = "mei-tool-call-serializer-v2"
PROMPT_FRAMING_ID = "mei-tool-prompt-framing-v1"
RETRIEVAL_ENCODING_ID = "mei-retrieval-text-encoding-v1"
RETRIEVAL_MAX_TOKENS = 384
STABLE_PREFIX_TOKENS_MAX = 1024
ROLLING_WINDOW_TOKENS = 256
TASK_CONTRACT = (
    "任务：只输出一个 schema 合法的工具 JSON 数组，或 []。最多一次调用。"
    "缺少 required 证据时输出 []。禁止输出解释或 route_id。"
)
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
WIRE_ID = "mei-runtime-wire-v2"
SCHEMA_SUBSET_ID = "mei-json-schema-subset-v2"
GRAMMAR_ID = "mei-byte-grammar-v2"
WEIGHT_CONTRACT_ID = "mei-1.0-51m-weight-contract-v1"
TOKENIZER_ID = "zh-24k-v1"

HISTORICAL_EVAL_DIR = (
    ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/banks/sft-v2-eval-lock-v3-20class"
)
TOOL_UNIVERSE_PATH = HISTORICAL_EVAL_DIR / "tool-universe-v1.json"
PARENT_RELEASE_DIR = (
    ROOT
    / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases"
    / "mei-1.0-51m-tool-sft-v2-agent300m-v1"
)
NARRATION_RELEASE_DIR = (
    ROOT
    / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases"
    / "mei-1.0-51m-narration-sft-agent300m-v3"
)
DEFAULT_RELEASE_ROOT = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases"
DEFAULT_EVAL_ROOT = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/banks"

TRAIN_RETRIEVAL_PER_TOOL = 16
VALID_RETRIEVAL_PER_TOOL = 4
TRAIN_EXECUTE_PER_TOOL = 12
TRAIN_REFUSE_PER_TOOL = 12
VALID_EXECUTE_PER_TOOL = 4
VALID_REFUSE_PER_TOOL = 4
EVAL_RETRIEVAL_PER_TOOL = 4
EVAL_EXECUTE_PER_TOOL = 2
EVAL_REFUSE_PER_TOOL = 2
TRAIN_AGENT_TERMINAL_PER_TOOL = 4
VALID_AGENT_TERMINAL_PER_TOOL = 1
EVAL_AGENT_TERMINAL_PER_TOOL = 1
AGENT_TRAIN_STEPS = 4_000

EVAL_MARKER = re.compile(r"\b(?:EVAL|GOLD|ANSWER)[-_][A-Z0-9_-]+\b", re.IGNORECASE)
PORTABLE_PATTERN_REWRITES = {r"^\d{2}:\d{2}$": r"^[0-9]{2}:[0-9]{2}$"}
SUPPORTED_FORMATS = {"date", "date-time", "time", "email", "uuid", "uri", "ipv4", "ipv6"}


ACTION_ZH = {
    "control_room_devices": "同时设置房间空调或灯光",
    "update_room_policy": "更新房间温控与办公时段策略",
    "get_weather": "查询城市当前天气",
    "set_lights": "设置房间灯光亮度",
    "create_event": "创建日历事件",
    "set_volume": "设置播放音量",
    "lookup_price": "按商品和数量查询价格",
    "add_meeting": "创建日历会议",
    "adjust_loudness": "调整播放响度",
    "quote_item": "按商品和数量报价",
    "book_table": "预订餐桌",
    "charge_card": "从支付卡扣款",
    "echo_text": "回显规范文本",
    "set_flag": "设置布尔标记",
    "set_count": "设置整数计数",
    "set_rate": "设置数值比率",
    "invoice": "从文本提取发票字段",
    "nod": "点头确认",
    "shake_head": "摇头表示拒绝",
    "come_here": "走到用户身边",
    "stop": "停止移动",
    "point": "指向指定目标",
    "wave": "挥手",
    "bow": "鞠躬一次",
    "sit": "坐下",
    "stand": "站起",
    "go_to": "前往指定位置",
    "set_switch": "切换指定灯开关",
    "open_door": "打开指定门",
    "close_door": "关闭指定门",
    "take_out_trash": "把生活垃圾拿出去",
    "order_food": "下单购买餐食",
    "cancel_order": "取消正在处理的餐食订单",
    "set_lamps": "设置房间灯具亮度",
    "create_meeting": "创建会议预约",
    "set_switch_level": "设置开关档位",
}


ARG_ZH = {
    "ac_power": "空调电源",
    "ac_target_c": "空调目标温度",
    "account": "账户",
    "address": "地址",
    "airline": "航空公司",
    "airport": "机场",
    "all_day": "是否全天",
    "amount": "金额",
    "appointment_id": "预约号",
    "bin": "废料箱",
    "brightness": "亮度",
    "cabinet": "柜号",
    "card_id": "卡号",
    "chamber": "培养箱腔室",
    "city": "城市",
    "comfort_c": "舒适温度",
    "confirmation": "确认号",
    "copies": "份数",
    "count": "数量",
    "courier": "快递员",
    "course": "课程",
    "currency": "币种",
    "date": "日期",
    "days": "天数",
    "dept": "科室",
    "dest": "目的位置",
    "dish": "餐品",
    "door": "门",
    "dorm": "宿舍",
    "drug": "药品",
    "due_date": "到期日",
    "duration_min": "持续分钟",
    "duration_minutes": "持续分钟",
    "expense_id": "报销单号",
    "from_account": "转出账户",
    "from_ccy": "原币种",
    "from_city": "出发城市",
    "from_port": "出发港口",
    "gain": "响度",
    "hood": "通风橱",
    "hour": "小时",
    "hours": "小时数",
    "hs_code": "海关编码",
    "id": "设备编号",
    "invoice_id": "发票号",
    "issue": "问题",
    "item": "项目",
    "kg": "重量公斤",
    "lab": "实验室",
    "label": "标签",
    "leave_grace_minutes": "离场宽限分钟",
    "level": "档位",
    "light_on": "灯光开关",
    "limit": "限额",
    "minutes": "分钟",
    "modality": "检查类型",
    "mode": "模式",
    "n": "计数",
    "name": "名称",
    "new": "新名称",
    "nights": "晚数",
    "number": "号码",
    "od": "OD 值",
    "old": "原名称",
    "on": "是否开启",
    "order_id": "检验单号",
    "outdoor": "是否户外",
    "package": "套餐",
    "pair": "货币对",
    "party_size": "用餐人数",
    "passport": "护照号",
    "payment_id": "支付流水号",
    "place": "位置",
    "pnr": "订座编码",
    "precool_minutes": "预冷分钟",
    "probe": "探头",
    "qty": "数量",
    "reason": "原因",
    "room": "房间",
    "route": "线路",
    "rpm": "转速",
    "run_id": "实验记录号",
    "rx_id": "处方号",
    "sample_id": "样本号",
    "seat": "座位",
    "shop": "商家",
    "sku": "SKU",
    "sku_name": "商品名",
    "slot": "时段",
    "start_iso": "开始时间",
    "station": "站点",
    "substance": "过敏物",
    "symptom": "症状",
    "tag": "行李牌",
    "target": "目标",
    "temp_c": "温度",
    "term": "学期",
    "text": "文本",
    "time": "时间",
    "title": "标题",
    "to_account": "收款账户",
    "to_ccy": "目标币种",
    "to_city": "到达城市",
    "to_port": "到达港口",
    "topic": "主题",
    "total": "总额",
    "train_no": "车次",
    "type": "类型",
    "unit": "设备单元",
    "vaccine": "疫苗",
    "value": "数值",
    "vendor": "供应商",
    "visit_id": "就诊号",
    "ward": "病区",
    "waybill": "运单号",
    "when": "执行时间",
    "window": "时间窗",
    "work_end": "下班时间",
    "work_start": "上班时间",
    "x": "比率",
}


STRING_VALUES = {
    "account": ["账户A102", "账户B208", "账户C315", "账户D426"],
    "address": ["上海市虹桥路88号", "北京市学院路12号", "深圳市科苑路6号"],
    "airline": ["东方航空", "南方航空", "中国国航"],
    "airport": ["浦东机场", "首都机场", "宝安机场"],
    "appointment_id": ["APT-24031", "APT-35042", "APT-46053"],
    "bin": ["危废箱A", "生物废料箱B", "锐器箱C"],
    "cabinet": ["试剂柜A3", "试剂柜B7", "试剂柜C2"],
    "card_id": ["CARD-1024", "CARD-2048", "CARD-4096"],
    "chamber": ["培养箱A", "培养箱B", "培养箱C"],
    "city": ["上海", "北京", "深圳", "杭州"],
    "confirmation": ["HTL-48320", "HTL-57310", "HTL-68240"],
    "courier": ["李明", "王静", "陈涛"],
    "course": ["高等数学", "数据结构", "大学物理"],
    "currency": ["CNY", "USD", "EUR"],
    "date": ["2026-10-12", "2026-11-08", "2026-12-16"],
    "dept": ["内科", "眼科", "皮肤科"],
    "dest": ["冷库A区", "样本架B2", "实验台C"],
    "dish": ["牛肉面", "巨无霸"],
    "door": ["front", "back"],
    "dorm": ["东区3号楼", "西区6号楼", "南区2号楼"],
    "drug": ["阿莫西林", "布洛芬", "氯雷他定"],
    "due_date": ["2026-10-20", "2026-11-18", "2026-12-22"],
    "expense_id": ["EXP-1028", "EXP-2049", "EXP-3096"],
    "from_account": ["账户A102", "账户B208", "账户C315"],
    "from_ccy": ["CNY", "USD", "EUR"],
    "from_city": ["北京", "上海", "广州"],
    "from_port": ["上海港", "宁波港", "深圳港"],
    "hood": ["通风橱A", "通风橱B", "通风橱C"],
    "hs_code": ["847130", "300490", "901890"],
    "id": ["kitchen_light", "living_light"],
    "invoice_id": ["INV-2026102", "INV-2026204", "INV-2026308"],
    "issue": ["灯具闪烁", "门锁失灵", "空调漏水"],
    "item": ["打印纸", "离心管", "校准服务"],
    "lab": ["化学实验室", "生物实验室", "材料实验室"],
    "label": ["差旅备用卡", "采购专用卡", "订阅服务卡"],
    "modality": ["CT", "核磁共振", "超声"],
    "mode": ["阅读", "夜间", "会议"],
    "name": ["乙醇", "缓冲液", "培养基"],
    "new": ["会议室新标签", "实验室新标签", "仓库新标签"],
    "old": ["会议室旧标签", "实验室旧标签", "仓库旧标签"],
    "order_id": ["LAB-10082", "LAB-20164", "LAB-30246"],
    "package": ["基础体检", "入职体检", "心血管筛查"],
    "pair": ["CNY/USD", "EUR/CNY", "USD/JPY"],
    "passport": ["E12345678", "G87654321", "P24681357"],
    "payment_id": ["PAY-10824", "PAY-21648", "PAY-32472"],
    "place": ["kitchen", "living", "entry", "trash"],
    "pnr": ["PNR6A2K", "PNR8M4Q", "PNR3T7X"],
    "probe": ["pH探头A", "pH探头B", "pH探头C"],
    "reason": ["科研加班", "航班延误", "临时任务"],
    "room": ["客厅", "会议室A", "实验室B"],
    "route": ["东区一线", "南北环线", "地铁接驳线"],
    "run_id": ["RUN-20261001", "RUN-20261002", "RUN-20261003"],
    "rx_id": ["RX-10293", "RX-20586", "RX-30879"],
    "sample_id": ["SMP-1007", "SMP-2014", "SMP-3021"],
    "seat": ["靠窗", "靠过道", "前排"],
    "shop": ["兰州拉面", "麦当劳"],
    "sku": ["SKU-1024", "SKU-2048", "SKU-4096"],
    "sku_name": ["工业滤芯", "打印纸", "离心管"],
    "slot": ["上午09:00", "下午14:00", "晚间18:30"],
    "start_iso": ["2026-10-12T09:30:00+08:00", "2026-11-08T14:00:00+08:00"],
    "station": ["图书馆一层", "教学楼A座", "实验楼大厅"],
    "substance": ["青霉素", "花生", "乳胶"],
    "symptom": ["咳嗽", "头痛", "皮疹"],
    "tag": ["BAG-102938", "BAG-564738", "BAG-918273"],
    "target": ["left", "right", "user"],
    "term": ["2026秋季", "2027春季", "2027秋季"],
    "text": ["确认收到", "任务完成", "设备正常"],
    "time": ["2026-10-12T18:30:00+08:00", "2026-11-08T09:00:00+08:00"],
    "title": ["采购申请", "差旅会议", "设备维护"],
    "to_account": ["账户D426", "账户E537", "账户F648"],
    "to_ccy": ["USD", "EUR", "JPY"],
    "to_city": ["上海", "深圳", "杭州"],
    "to_port": ["宁波港", "青岛港", "天津港"],
    "topic": ["项目周会", "预算复盘", "实验评审"],
    "train_no": ["G102", "D2288", "K56"],
    "type": ["血清", "组织", "培养液"],
    "unit": ["离心机A", "冰箱B", "摇床C"],
    "vaccine": ["流感疫苗", "乙肝疫苗", "带状疱疹疫苗"],
    "vendor": ["华东实验器材", "北辰办公用品", "南方医疗供应"],
    "visit_id": ["VIS-10203", "VIS-20406", "VIS-30609"],
    "ward": ["内科一病区", "外科二病区", "康复病区"],
    "waybill": ["SF1029384756", "YT5647382910", "JD9182736450"],
    "when": ["2026-10-12T10:00:00+08:00", "2026-11-08T15:30:00+08:00"],
    "window": ["上午9点到11点", "下午2点到4点", "晚间6点到8点"],
    "work_end": ["18:00", "17:30", "19:00"],
    "work_start": ["09:00", "08:30", "10:00"],
}


CONTEXT_PHRASES = {
    "train": [
        "现在",
        "这次",
        "按当前安排",
        "为了今天的任务",
        "在当前会话里",
        "根据刚确认的信息",
        "按我给出的参数",
        "在不改动其他项目的前提下",
        "直接处理这项请求",
        "接下来",
        "此刻",
        "按这份明确请求",
        "针对当前对象",
        "在本次操作中",
        "按现有条件",
        "根据下面的完整信息",
    ],
    "valid": ["核对参数后", "针对这项请求", "依照已确认的信息", "为当前对象"],
    "dev": ["在这项独立任务中", "根据本次明确需求", "请只处理当前事项", "请核对后"],
    "test": ["在当前这个场景里", "按这次给定条件", "请为这项具体需求", "仅针对本次请求"],
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{sha_bytes(canonical_bytes(parts))[:16]}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def write_once(path: Path, payload: bytes) -> None:
    """Atomically create an immutable artifact, permitting byte-identical reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    write_once(path, canonical_bytes(value) + b"\n")


def _project_portable_patterns(value: Any, rewrites: Counter[str]) -> Any:
    if isinstance(value, list):
        return [_project_portable_patterns(item, rewrites) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key == "pattern" and isinstance(item, str) and item in PORTABLE_PATTERN_REWRITES:
            output[key] = PORTABLE_PATTERN_REWRITES[item]
            rewrites[item] += 1
        else:
            output[key] = _project_portable_patterns(item, rewrites)
    return output


def portable_universe_document(
    path: Path = TOOL_UNIVERSE_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the historical catalog into the Python/Rust/ECMAScript subset.

    The raw source is never modified.  The only current rewrite replaces the
    non-portable ``\\d`` escape with an explicit ASCII digit class.  Every
    rewrite is recorded so the frozen release can prove what changed.
    """

    raw = load_json(path)
    rewrites: Counter[str] = Counter()
    projected = _project_portable_patterns(copy.deepcopy(raw), rewrites)
    tools = projected.get("tools") or []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        source_fingerprint = tool.get("fingerprint")
        if source_fingerprint:
            tool["source_fingerprint"] = source_fingerprint
        identity = {
            key: tool.get(key)
            for key in (
                "name",
                "description",
                "parameters",
                "family",
                "similar_to",
                "tool_id",
                "source_toolset",
            )
        }
        tool["fingerprint"] = sha_bytes(canonical_bytes(identity))
    source_fingerprint = raw.get("fingerprint")
    projected["source_universe_id"] = raw.get("universe_id")
    projected["source_fingerprint"] = source_fingerprint
    projected["universe_id"] = "mei-51m-portable-tool-universe-v2"
    projected["n_tools"] = len(tools)
    projected["projection_id"] = "mei-portable-schema-projection-v1"
    projected["fingerprint"] = sha_bytes(
        canonical_bytes([compact_tool(tool) for tool in tools])
    )
    receipt = {
        "schema": "mei-tool-universe-projection-receipt-v1",
        "status": "passed",
        "projection_id": projected["projection_id"],
        "source": {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha_file(path),
            "universe_id": raw.get("universe_id"),
            "fingerprint": source_fingerprint,
        },
        "output": {
            "universe_id": projected["universe_id"],
            "fingerprint": projected["fingerprint"],
            "tool_count": len(tools),
        },
        "pattern_rewrites": [
            {
                "from": source,
                "to": PORTABLE_PATTERN_REWRITES[source],
                "count": count,
            }
            for source, count in sorted(rewrites.items())
        ],
        "other_semantic_changes": 0,
    }
    return projected, receipt


def universe_tools(path: Path = TOOL_UNIVERSE_PATH) -> list[dict[str, Any]]:
    blob, _ = portable_universe_document(path)
    tools = blob.get("tools") or []
    if not isinstance(tools, list) or len(tools) != 147:
        raise RuntimeError("mei-51m v3 requires the frozen 147-tool universe")
    names = [str(tool.get("name") or "") for tool in tools if isinstance(tool, dict)]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise RuntimeError("tool universe contains an empty or duplicate name")
    errors = [
        f"{tool['name']}: {error}"
        for tool in tools
        for error in schema_subset_errors(tool.get("parameters") or {})
    ]
    if errors:
        raise RuntimeError("unsupported tool universe schema: " + errors[0])
    return tools


def compact_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(tool["name"]),
        "description": str(tool.get("description") or ""),
        "parameters": tool.get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def schema_subset_errors(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("type") != "object":
        errors.append("root type must be object")
        return errors
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if not isinstance(properties, dict) or not isinstance(required, list):
        errors.append("properties/required shape invalid")
        return errors
    if not set(required).issubset(properties):
        errors.append("required references unknown property")
    if schema.get("additionalProperties", False) not in {False, None}:
        errors.append("additionalProperties must be false")
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            errors.append(f"{name}: property schema must be object")
            continue
        if spec.get("type") not in {"string", "boolean", "integer", "number"}:
            errors.append(f"{name}: unsupported type {spec.get('type')!r}")
        if "enum" in spec and (
            not isinstance(spec["enum"], list) or not spec["enum"]
        ):
            errors.append(f"{name}: enum must be non-empty")
        if "format" in spec and spec["format"] not in SUPPORTED_FORMATS:
            errors.append(f"{name}: unsupported format {spec['format']!r}")
        pattern = spec.get("pattern")
        if isinstance(pattern, str):
            escaped = False
            for char in pattern:
                if escaped:
                    if char.isascii() and char.isalnum():
                        errors.append(f"{name}: non-portable pattern escape \\{char}")
                        break
                    escaped = False
                elif char == "\\":
                    escaped = True
            if escaped:
                errors.append(f"{name}: pattern ends with an escape")
        unknown = set(spec) - {
            "type",
            "description",
            "enum",
            "const",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "format",
        }
        if unknown:
            errors.append(f"{name}: unsupported keywords {sorted(unknown)}")
    return errors


def value_matches_schema(value: Any, spec: dict[str, Any]) -> bool:
    kind = spec.get("type")
    if kind == "string":
        valid = isinstance(value, str)
    elif kind == "boolean":
        valid = isinstance(value, bool)
    elif kind == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        return False
    if not valid:
        return False
    if "enum" in spec and value not in spec["enum"]:
        return False
    if "const" in spec and value != spec["const"]:
        return False
    if isinstance(value, str):
        if len(value) < int(spec.get("minLength", 0)):
            return False
        if "maxLength" in spec and len(value) > int(spec["maxLength"]):
            return False
        if "pattern" in spec and re.search(str(spec["pattern"]), value) is None:
            return False
        format_name = spec.get("format")
        if format_name == "date" and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
            return False
        if format_name == "time" and re.fullmatch(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?", value) is None:
            return False
        if format_name == "date-time" and "T" not in value:
            return False
        if format_name == "email" and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is None:
            return False
        if format_name == "uuid" and re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            value,
        ) is None:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            return False
        if "maximum" in spec and value > spec["maximum"]:
            return False
        if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in spec and value >= spec["exclusiveMaximum"]:
            return False
        if "multipleOf" in spec:
            quotient = float(value) / float(spec["multipleOf"])
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                return False
    return True


def arguments_match_schema(arguments: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict):
        return False
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not required.issubset(arguments):
        return False
    if schema.get("additionalProperties", False) is False and not set(arguments).issubset(
        properties
    ):
        return False
    return all(
        name in properties and value_matches_schema(value, properties[name])
        for name, value in arguments.items()
    )


def _number_value(spec: dict[str, Any], variant: int, *, integer: bool) -> int | float:
    low = float(spec.get("minimum", spec.get("exclusiveMinimum", 0)))
    high = float(spec.get("maximum", spec.get("exclusiveMaximum", low + 100)))
    multiple = float(spec.get("multipleOf", 1 if integer else 0.01))
    if multiple <= 0:
        raise RuntimeError(f"invalid multipleOf: {spec}")
    if "exclusiveMinimum" in spec:
        low += multiple
    if "exclusiveMaximum" in spec:
        high -= multiple
    low = math.ceil((low - 1e-12) / multiple) * multiple
    high = math.floor((high + 1e-12) / multiple) * multiple
    if high < low:
        raise RuntimeError(f"invalid numeric bounds: {spec}")
    raw_candidates = [
        low,
        high,
        low + (high - low) * 0.5,
        low + (high - low) * 0.25,
        low + (high - low) * 0.75,
    ]
    candidates: list[int | float] = []
    for raw in raw_candidates:
        snapped = round(raw / multiple) * multiple
        value: int | float = int(round(snapped)) if integer else round(snapped, 10)
        if value not in candidates and value_matches_schema(value, spec):
            candidates.append(value)
    if not candidates:
        raise RuntimeError(f"cannot synthesize a valid numeric value: {spec}")
    return candidates[variant % len(candidates)]


def example_value(name: str, spec: dict[str, Any], variant: int, partition: str) -> Any:
    if "const" in spec:
        return spec["const"]
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        return enum[(variant + (1 if partition == "test" else 0)) % len(enum)]
    kind = spec.get("type")
    if kind == "boolean":
        return bool((variant + (1 if partition == "test" else 0)) % 2)
    if kind == "integer":
        return _number_value(spec, variant, integer=True)
    if kind == "number":
        return _number_value(spec, variant, integer=False)
    format_name = spec.get("format")
    if format_name == "date":
        return f"2026-{10 + variant % 3:02d}-{10 + variant % 17:02d}"
    if format_name == "time":
        return f"{8 + variant % 10:02d}:{(variant * 10) % 60:02d}"
    if format_name == "date-time":
        return f"2026-{10 + variant % 3:02d}-{10 + variant % 17:02d}T{8 + variant % 10:02d}:30:00+08:00"
    if format_name == "email":
        return f"user{variant % 17 + 1}@example.cn"
    if format_name == "uuid":
        return f"00000000-0000-4000-8000-{variant % 1000000000000:012d}"
    values = STRING_VALUES.get(name)
    if values:
        offset = {"train": 0, "valid": 1, "dev": 1, "test": 2}.get(partition, 0)
        value = values[(variant + offset) % len(values)]
    elif name == "date":
        value = f"2026-{10 + variant % 3:02d}-{10 + variant % 17:02d}"
    else:
        prefix = {"train": "甲", "valid": "乙", "dev": "丙", "test": "丁"}.get(
            partition, "甲"
        )
        value = f"{prefix}{ARG_ZH.get(name, name.replace('_', ''))}{variant % 17 + 1}"
    pattern = spec.get("pattern")
    if pattern in {r"^\d{2}:\d{2}$", r"^[0-9]{2}:[0-9]{2}$"}:
        value = f"{8 + variant % 10:02d}:{(variant * 10) % 60:02d}"
    return value


def action_zh(tool: dict[str, Any]) -> str:
    name = str(tool["name"])
    if name in ACTION_ZH:
        return ACTION_ZH[name]
    description = str(tool.get("description") or name).strip().rstrip("。.")
    return description or name


def example_arguments(
    tool: dict[str, Any], variant: int, partition: str, *, include_optional: bool = True
) -> dict[str, Any]:
    schema = tool.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    selected = list(required)
    optional = [name for name in properties if name not in required]
    # Configurable actions must not be labelled executable with an empty
    # no-op. Empty arguments remain legitimate only for true zero-argument
    # tools such as ``wave`` and ``stop``.
    if optional and (include_optional or not required):
        selected.append(optional[variant % len(optional)])
    arguments = {
        name: example_value(name, properties[name], variant, partition) for name in selected
    }
    if not arguments_match_schema(arguments, schema):
        raise RuntimeError(f"generated arguments violate schema for {tool['name']}: {arguments}")
    return arguments


def _value_text(value: Any) -> str:
    if value is True:
        return "开启"
    if value is False:
        return "关闭"
    return str(value)


def argument_text(arguments: dict[str, Any], variant: int) -> str:
    style = variant % 4
    parts: list[str] = []
    for name, value in arguments.items():
        label = ARG_ZH.get(name, name.replace("_", ""))
        rendered = _value_text(value)
        if style == 0:
            parts.append(f"{label}为{rendered}")
        elif style == 1:
            parts.append(f"{label}：{rendered}")
        elif style == 2:
            parts.append(f"把{label}设为{rendered}")
        else:
            parts.append(f"{label}用{rendered}")
    return "，".join(parts)


def positive_query(tool: dict[str, Any], arguments: dict[str, Any], variant: int, partition: str) -> str:
    action = action_zh(tool)
    slots = argument_text(arguments, variant)
    templates = {
        "train": [
            "{context}，请{action}{slots}。",
            "{context}，麻烦帮我{action}{slots}。",
            "{context}，我需要{action}{slots}。",
            "{context}，请按这些信息{action}{slots}。",
            "{context}，我要{action}{slots}。",
            "{context}，替我{action}{slots}。",
            "{context}，请直接{action}{slots}。",
            "{context}，帮忙{action}{slots}。",
        ],
        "valid": [
            "{context}，能否替我{action}{slots}？",
            "{context}，请依据以下信息{action}{slots}。",
        ],
        "dev": [
            "{context}，我想办理这件事：{action}{slots}。",
            "{context}，请正确选择工具来{action}{slots}。",
        ],
        "test": [
            "{context}，帮我完成以下操作：{action}{slots}。",
            "{context}，需要你现在{action}{slots}。",
        ],
    }
    candidates = templates[partition]
    joiner = "，" if slots else ""
    context = CONTEXT_PHRASES[partition][variant % len(CONTEXT_PHRASES[partition])]
    return candidates[variant % len(candidates)].format(
        context=context, action=action, slots=joiner + slots
    )


def refusal_query(
    tool: dict[str, Any], variant: int, partition: str
) -> tuple[str, str, dict[str, Any]]:
    action = action_zh(tool)
    schema = tool.get("parameters") or {}
    required = list(schema.get("required") or [])
    properties = list((schema.get("properties") or {}).keys())
    mode = (variant + int(sha_bytes(str(tool["name"]).encode("utf-8"))[:4], 16)) % 6
    context = CONTEXT_PHRASES[partition][variant % len(CONTEXT_PHRASES[partition])]
    if mode == 0:
        return f"{context}，先不要{action}，我只是确认这里有没有这个功能。", "negation_cancels", {}
    if mode == 1:
        if required:
            missing_name = required[variant % len(required)]
            missing = ARG_ZH.get(missing_name, missing_name)
            return f"{context}，请{action}，但{missing}我还没有确定。", "missing_slot", {}
        return f"{context}，我还没决定是否要{action}，请先不要执行。", "ambiguous_scope", {}
    if mode == 2:
        return f"{context}，关于{action}我还没决定具体对象和参数，暂时不要执行。", "ambiguous_scope", {}
    if mode == 3:
        if properties:
            property_name = properties[variant % len(properties)]
            unknown = ARG_ZH.get(property_name, property_name)
            return f"{context}，请{action}，{unknown}就用那个常用的，但我说不清具体值。", "unknown_slot_value", {}
        return f"{context}，如果以后需要{action}会怎样？现在不要执行。", "offtopic", {}
    if mode == 4:
        return f"{context}，既想{action}，又想顺便处理另一件还没说清楚的事，先别执行。", "mixed_intent", {}
    if mode == 5:
        return f"{context}，请解释这个功能能否{action}，不要实际调用任何工具。", "offtopic", {}
    return f"{context}，我还没有明确要求{action}，请先不要执行。", "ambiguous_scope", {}


def _similarity_tokens(tool: dict[str, Any]) -> set[str]:
    name = str(tool.get("name") or "")
    description = str(tool.get("description") or "")
    pieces = set(name.split("_"))
    pieces.update(description[index : index + 2] for index in range(max(0, len(description) - 1)))
    return {piece for piece in pieces if piece}


def ranked_hard_negatives(
    tool: dict[str, Any], tools: Sequence[dict[str, Any]]
) -> list[str]:
    name = str(tool["name"])
    family = str(tool.get("family") or "")
    similar_to = tool.get("similar_to")
    tokens = _similarity_tokens(tool)
    scored: list[tuple[int, str]] = []
    for candidate in tools:
        candidate_name = str(candidate["name"])
        if candidate_name == name:
            continue
        explicit = (
            similar_to == candidate_name
            or candidate.get("similar_to") == name
            or (similar_to is not None and candidate.get("similar_to") == similar_to)
        )
        overlap = len(tokens & _similarity_tokens(candidate))
        same_family = family == str(candidate.get("family") or "")
        score = (1000 if explicit else 0) + (100 if same_family else 0) + overlap
        scored.append((score, candidate_name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate_name for _, candidate_name in scored]


def hard_negatives(
    tool: dict[str, Any], tools: Sequence[dict[str, Any]], variant: int, count: int = 4
) -> list[str]:
    """Keep the closest negative and rotate three within the top-48 window.

    Sixteen training variants therefore expose every tool to exactly 27
    distinct negatives instead of repeating the same four candidates.
    """

    ranked = ranked_hard_negatives(tool, tools)
    if len(ranked) < count:
        raise RuntimeError(f"not enough hard negatives for {tool['name']}")
    if count == 1:
        return ranked[:1]
    window = min(48, len(ranked) - 1)
    selected = [ranked[0]]
    for index in range(count - 1):
        selected.append(ranked[1 + ((variant * 11 + index * 7) % window)])
    if len(set(selected)) != count:
        raise RuntimeError(f"hard-negative sampler produced duplicates for {tool['name']}")
    return selected


def selected_tools(
    tool: dict[str, Any], tools: Sequence[dict[str, Any]], variant: int
) -> tuple[list[dict[str, Any]], list[str]]:
    by_name = {str(item["name"]): item for item in tools}
    negatives = hard_negatives(tool, tools, variant, 4)
    names = list(negatives)
    names.insert(variant % 5, str(tool["name"]))
    return [compact_tool(by_name[name]) for name in names], negatives


def serialize_tool_target(answers: Sequence[dict[str, Any]] | None) -> str:
    if not answers:
        return "[]"
    if len(answers) != 1:
        raise ValueError("mei-tool-call-serializer-v2 permits exactly one call")
    answer = answers[0]
    ordered = {
        "name": str(answer["name"]),
        "arguments": dict(sorted((answer.get("arguments") or {}).items())),
    }
    return json.dumps([ordered], ensure_ascii=False, separators=(",", ":"))


def _query_forms(value: Any) -> set[str]:
    if value is True:
        return {"true", "开启", "打开", "是", "启用"}
    if value is False:
        return {"false", "关闭", "关掉", "否", "停用"}
    if value is None:
        return {"null", "空值", "无"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rendered = str(value)
        forms = {rendered}
        if isinstance(value, float) and value.is_integer():
            forms.add(str(int(value)))
        return forms
    return {str(value)}


def value_grounded_in_query(value: Any, query: str) -> bool:
    normalized = query.casefold()
    return any(form.casefold() in normalized for form in _query_forms(value))


def make_retrieval_row(
    tool: dict[str, Any], tools: Sequence[dict[str, Any]], variant: int, split: str
) -> dict[str, Any]:
    arguments = example_arguments(
        tool, variant, split, include_optional=bool(variant % 2)
    )
    query = positive_query(tool, arguments, variant, split)
    catalog, negatives = selected_tools(tool, tools, variant)
    sample_id = stable_id("RET3", split, tool["name"], variant, query)
    return {
        "sample_id": sample_id,
        "case_id": sample_id,
        "cf_group": stable_id("RET3G", split, tool["name"], variant // 2),
        "task": "retrieval",
        "split": split,
        "family": str(tool.get("family") or "unknown"),
        "kind": "hard_positive",
        "query": query,
        "gold_tool": str(tool["name"]),
        "hard_negatives": negatives,
        "catalog_tools": catalog,
        "seen_schema": True,
        "tool_universe_id": "mei-51m-portable-tool-universe-v2",
        "retrieval_encoding_id": RETRIEVAL_ENCODING_ID,
        "retrieval_max_tokens": RETRIEVAL_MAX_TOKENS,
        "generator_version": GENERATOR_ID,
        "source_role": "deterministic-schema-program",
        "status": "accepted",
    }


def make_fullcall_row(
    tool: dict[str, Any],
    tools: Sequence[dict[str, Any]],
    variant: int,
    split: str,
    *,
    execute: bool,
) -> dict[str, Any]:
    catalog, negatives = selected_tools(tool, tools, variant)
    if execute:
        arguments = example_arguments(
            tool, variant, split, include_optional=bool(variant % 2)
        )
        query = positive_query(tool, arguments, variant, split)
        reason_code = "ready_to_execute"
        answers: list[dict[str, Any]] = [
            {"name": str(tool["name"]), "arguments": arguments}
        ]
        kind = "execute"
    else:
        query, reason_code, arguments = refusal_query(tool, variant, split)
        answers = []
        kind = "refuse"
    sample_id = stable_id("FC3", split, tool["name"], variant, kind, query)
    return {
        "sample_id": sample_id,
        "case_id": sample_id,
        # Execute/refuse rows with the same tool and variant form one explicit
        # counterfactual pair and therefore share an isolation group.
        "cf_group": stable_id("FC3G", split, tool["name"], variant),
        "task": "fullcall",
        "split": split,
        "family": str(tool.get("family") or "unknown"),
        "kind": kind,
        "query": query,
        "answers": answers,
        "gold_name": str(tool["name"]) if execute else None,
        "candidate_tool": str(tool["name"]),
        "gold_args": arguments if execute else {},
        "reason_code": reason_code,
        "retrieved_tools": [str(item["name"]) for item in catalog],
        "oracle_top5": catalog,
        "hard_negatives": negatives,
        # Empty means no extra host restriction.  The historical ``[]`` was
        # converted into ``allowed_tools: []`` by the trainer and denied every
        # call; gold slot provenance also leaked the answer into model input.
        "permissions": {},
        "slot_provenance": [],
        "history": [],
        "tool_results": [],
        "prior_tool_results": [],
        "serializer": SERIALIZER_ID,
        "prompt_framing": PROMPT_FRAMING_ID,
        "target_text": serialize_tool_target(answers),
        "wire_version": WIRE_ID,
        "generator_version": GENERATOR_ID,
        "source_role": "deterministic-schema-program",
        "status": "accepted",
    }


def build_single_step_rows(
    tools: Sequence[dict[str, Any]],
    *,
    split: str,
    retrieval_per_tool: int,
    execute_per_tool: int,
    refuse_per_tool: int,
) -> dict[str, list[dict[str, Any]]]:
    retrieval: list[dict[str, Any]] = []
    fullcall: list[dict[str, Any]] = []
    for tool in tools:
        retrieval.extend(
            make_retrieval_row(tool, tools, variant, split)
            for variant in range(retrieval_per_tool)
        )
        fullcall.extend(
            make_fullcall_row(tool, tools, variant, split, execute=True)
            for variant in range(execute_per_tool)
        )
        fullcall.extend(
            make_fullcall_row(tool, tools, variant, split, execute=False)
            for variant in range(refuse_per_tool)
        )
    return {"retrieval": retrieval, "fullcall": fullcall}


def _agent_call_id(trajectory_id: str, call: dict[str, Any]) -> str:
    nonce = sha_bytes(trajectory_id.encode("utf-8"))[:8]
    digest = sha_bytes(canonical_bytes([trajectory_id, 1, call]))[:12]
    return f"call-s{nonce}-1-{digest}"


def build_agent_terminal_rows(
    tools: Sequence[dict[str, Any]],
    *,
    split: str,
    variants_per_tool: int,
) -> list[dict[str, Any]]:
    """Build call→verified-result→terminal coverage for every tool.

    These two-step trajectories teach the generic closed-loop termination
    rule. They deliberately do not invent semantically unsupported cross-tool
    chains; the reviewed historical Agent bank remains the source for
    result-grounded second calls.
    """

    rows: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool["name"])
        for variant in range(variants_per_tool):
            arguments = example_arguments(
                tool, variant, split, include_optional=bool(variant % 2)
            )
            query = positive_query(tool, arguments, variant, split)
            catalog, negatives = selected_tools(tool, tools, variant)
            catalog_names = [str(item["name"]) for item in catalog]
            trajectory_id = stable_id("AGT3", split, name, variant, query)
            call_without_id = {"name": name, "arguments": arguments}
            call = {
                "call_id": _agent_call_id(trajectory_id, call_without_id),
                **call_without_id,
            }
            result = {
                "call_id": call["call_id"],
                "status": "ok",
                "payload": {
                    "accepted": True,
                    "tool_name": name,
                    "arguments": arguments,
                },
                "provenance": {
                    "source": f"deterministic-host-simulator:mei-agent-host-simulator-v1:{name}",
                    "verified": True,
                },
                "wire_version": WIRE_ID,
            }
            common = {
                "case_id": trajectory_id,
                "cf_group": trajectory_id,
                "trajectory_id": trajectory_id,
                "trajectory_length": 2,
                "task": "fullcall",
                "split": split,
                "family": str(tool.get("family") or "unknown"),
                "query": query,
                "history": [],
                "permissions": {},
                "slot_provenance": [],
                "catalog_tools": catalog_names,
                "retrieved_tools": catalog_names,
                "oracle_top5": catalog,
                "hard_negatives": negatives,
                "serializer": SERIALIZER_ID,
                "prompt_framing": PROMPT_FRAMING_ID,
                "wire_version": WIRE_ID,
                "generator_version": GENERATOR_ID,
                "source_role": "deterministic-schema-agent-simulator",
                "status": "accepted",
            }
            first_id = stable_id("AGT3S", trajectory_id, 1)
            rows.append(
                {
                    **common,
                    "sample_id": first_id,
                    "source_sample_id": first_id,
                    "trajectory_step": 1,
                    "kind": "execute",
                    "answers": [call_without_id],
                    "gold_name": name,
                    "gold_args": arguments,
                    "target_text": serialize_tool_target([call_without_id]),
                    "prior_calls": [],
                    "prior_tool_results": [],
                    "tool_results": [],
                    "agent_trace": {
                        "simulator_id": "mei-agent-host-simulator-v1",
                        "target_kind": "call",
                        "trusted_offline_fixture": True,
                    },
                }
            )
            terminal_id = stable_id("AGT3S", trajectory_id, 2)
            rows.append(
                {
                    **common,
                    "sample_id": terminal_id,
                    "source_sample_id": terminal_id,
                    "trajectory_step": 2,
                    "kind": "respond",
                    "answers": [],
                    "gold_name": None,
                    "gold_args": {},
                    "target_text": "[]",
                    "prior_calls": [call],
                    "prior_tool_results": [result],
                    "tool_results": [result],
                    "agent_trace": {
                        "simulator_id": "mei-agent-host-simulator-v1",
                        "target_kind": "respond",
                        "trusted_offline_fixture": True,
                    },
                }
            )
    return rows


def _scalar_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        output: list[Any] = []
        for item in value.values():
            output.extend(_scalar_values(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_scalar_values(item))
        return output
    return [value]


def audit_agent_rows(
    rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    expected_split: str,
    minimum_unique_call_tools: int,
    minimum_terminal_tools: int,
) -> dict[str, Any]:
    by_name = {str(tool["name"]): tool for tool in tools}
    errors: list[str] = []
    ids: set[str] = set()
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    call_tools: set[str] = set()
    terminal_tools: set[str] = set()
    call_rows = terminal_rows = query_grounded = result_grounded = 0
    nonempty_result_rows = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in ids:
            errors.append(f"empty or duplicate Agent sample ID: {sample_id}")
        ids.add(sample_id)
        if row.get("split") != expected_split:
            errors.append(f"{sample_id}: Agent split mismatch")
        trajectory_id = str(row.get("trajectory_id") or row.get("cf_group") or "")
        if not trajectory_id:
            errors.append(f"{sample_id}: Agent trajectory ID missing")
        trajectories[trajectory_id].append(row)
        if row.get("permissions") != {} or row.get("slot_provenance"):
            errors.append(f"{sample_id}: Agent input leaks permission/provenance gold")
        trace = row.get("agent_trace") or {}
        if not (
            isinstance(trace, dict)
            and trace.get("trusted_offline_fixture") is True
            and trace.get("simulator_id") == "mei-agent-host-simulator-v1"
        ):
            errors.append(f"{sample_id}: Agent fixture is not trusted")
        selected = [str(value) for value in row.get("retrieved_tools") or []]
        if len(selected) != 5 or len(set(selected)) != 5:
            errors.append(f"{sample_id}: Agent row lacks five unique schemas")
        calls = list(row.get("prior_calls") or [])
        results = list(row.get("tool_results") or row.get("prior_tool_results") or [])
        if len(calls) != len(results):
            errors.append(f"{sample_id}: Agent call/result prefix length mismatch")
        if results:
            nonempty_result_rows += 1
        for call, result in zip(calls, results):
            if (
                not isinstance(call, dict)
                or not isinstance(result, dict)
                or result.get("call_id") != call.get("call_id")
                or result.get("status") != "ok"
                or (result.get("provenance") or {}).get("verified") is not True
            ):
                errors.append(f"{sample_id}: invalid trusted ToolResultV2 prefix")
        answers = list(row.get("answers") or [])
        expected_target = serialize_tool_target(answers)
        if row.get("target_text") != expected_target:
            errors.append(f"{sample_id}: Agent target serializer drift")
        if answers:
            call_rows += 1
            if len(answers) != 1:
                errors.append(f"{sample_id}: Agent step emits more than one call")
                continue
            answer = answers[0]
            name = str(answer.get("name") or "")
            call_tools.add(name)
            tool = by_name.get(name)
            if (
                tool is None
                or name not in selected
                or not arguments_match_schema(
                    answer.get("arguments") or {}, tool.get("parameters") or {}
                )
            ):
                errors.append(f"{sample_id}: invalid Agent call target")
                continue
            result_values = _scalar_values(
                [
                    item.get("payload")
                    for item in results
                    if isinstance(item, dict)
                ]
            )
            for value in (answer.get("arguments") or {}).values():
                if value_grounded_in_query(value, str(row.get("query") or "")):
                    query_grounded += 1
                elif value in result_values:
                    result_grounded += 1
                else:
                    errors.append(f"{sample_id}: Agent argument lacks query/result grounding")
        else:
            terminal_rows += 1
            if calls:
                terminal_tools.add(str(calls[-1].get("name") or ""))
    for trajectory_id, values in trajectories.items():
        ordered = sorted(values, key=lambda row: int(row.get("trajectory_step") or 0))
        steps = [int(row.get("trajectory_step") or 0) for row in ordered]
        declared = {int(row.get("trajectory_length") or 0) for row in ordered}
        if steps != list(range(1, len(ordered) + 1)) or declared != {len(ordered)}:
            errors.append(f"{trajectory_id}: Agent trajectory shape mismatch")
        if not ordered or ordered[-1].get("answers"):
            errors.append(f"{trajectory_id}: Agent trajectory lacks terminal respond")
        if any(not row.get("answers") for row in ordered[:-1]):
            errors.append(f"{trajectory_id}: Agent terminal appears before final step")
        for offset, row in enumerate(ordered):
            if len(row.get("prior_calls") or []) != offset:
                errors.append(f"{row.get('sample_id')}: Agent prefix depth mismatch")
    if len(call_tools) < minimum_unique_call_tools:
        errors.append(
            f"Agent call-tool coverage {len(call_tools)} < {minimum_unique_call_tools}"
        )
    if len(terminal_tools) < minimum_terminal_tools:
        errors.append(
            f"Agent terminal-tool coverage {len(terminal_tools)} < {minimum_terminal_tools}"
        )
    return {
        "schema": "mei-agent-continuation-input-audit-v3",
        "status": "passed" if not errors else "failed",
        "errors": errors[:100],
        "split": expected_split,
        "rows": len(rows),
        "trajectories": len(trajectories),
        "call_rows": call_rows,
        "terminal_respond_rows": terminal_rows,
        "nonempty_tool_result_rows": nonempty_result_rows,
        "unique_call_tools": len(call_tools),
        "unique_terminal_tools": len(terminal_tools),
        "query_grounded_argument_bindings": query_grounded,
        "trusted_result_grounded_argument_bindings": result_grounded,
        "gold_slot_provenance_rows": sum(
            1 for row in rows if row.get("slot_provenance")
        ),
        "unrestricted_permission_rows": sum(
            1 for row in rows if row.get("permissions") == {}
        ),
    }


def retrieval_tool_text(tool: dict[str, Any]) -> str:
    compact = compact_tool(tool)
    return (
        f"工具：{compact['name']}\n"
        f"说明：{compact['description']}\n"
        "参数："
        + json.dumps(
            compact["parameters"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def render_request_parts(
    row: dict[str, Any], selected_tools: Sequence[dict[str, Any]]
) -> dict[str, str]:
    """Pure-data mirror of the v2 runtime framing frozen by this release."""

    sink = (
        TASK_CONTRACT
        + "\n<tools>"
        + json.dumps(
            [compact_tool(tool) for tool in selected_tools],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "</tools>"
    )
    ordinary: list[str] = []
    for turn in row.get("history") or []:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            ordinary.append(f"{str(turn.get('role') or 'user')}：{content}")
    for result in row.get("tool_results") or row.get("prior_tool_results") or []:
        ordinary.append("tool：" + json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    for tag, value in (
        ("context", row.get("context") or {"locale": "zh-CN"}),
        ("evidence", row.get("evidence") or []),
        ("permissions", row.get("permissions") or {}),
        ("state", row.get("state") or {}),
        ("mw", row.get("mw") or row.get("mw_disposition") or {}),
    ):
        ordinary.append(
            f"<{tag}>"
            + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + f"</{tag}>"
        )
    ordinary.append("user：" + str(row.get("query") or "").strip())
    ordinary_text = "\n".join(ordinary)
    return {
        "sink": sink,
        "ordinary": ordinary_text,
        "prompt": sink + "\n" + ordinary_text + ASSISTANT_SUFFIX,
    }


def _quantiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]

    return {"min": ordered[0], "p50": at(0.50), "p95": at(0.95), "max": ordered[-1]}


def token_budget_audit(
    rows: dict[str, list[dict[str, Any]]],
    tools: Sequence[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    sink_lengths: list[int] = []
    ordinary_lengths: list[int] = []
    for row in rows.get("fullcall") or []:
        rendered = render_request_parts(row, row.get("oracle_top5") or [])
        sink_lengths.append(len(tokenizer.encode(rendered["sink"], add_bos=True)))
        ordinary_lengths.append(len(tokenizer.encode(rendered["ordinary"])))
    retrieval_lengths = [
        len(tokenizer.encode(retrieval_tool_text(tool))) for tool in tools
    ]
    errors: list[str] = []
    sink_over = sum(length > STABLE_PREFIX_TOKENS_MAX for length in sink_lengths)
    retrieval_over = sum(length > RETRIEVAL_MAX_TOKENS for length in retrieval_lengths)
    if sink_over:
        errors.append(f"{sink_over} selected-schema sinks exceed stable prefix budget")
    if retrieval_over:
        errors.append(f"{retrieval_over} tool encodings exceed retrieval max tokens")
    return {
        "schema": "mei-sft-v3-token-budget-audit-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "prompt_framing_id": PROMPT_FRAMING_ID,
        "retrieval_encoding_id": RETRIEVAL_ENCODING_ID,
        "stable_prefix_tokens_max": STABLE_PREFIX_TOKENS_MAX,
        "rolling_window_tokens": ROLLING_WINDOW_TOKENS,
        "retrieval_max_tokens": RETRIEVAL_MAX_TOKENS,
        "selected_schema_sink_tokens": _quantiles(sink_lengths),
        "ordinary_tokens_before_tail_window": _quantiles(ordinary_lengths),
        "retrieval_tool_tokens": _quantiles(retrieval_lengths),
        "selected_schema_sink_over_budget": sink_over,
        "retrieval_tool_over_budget": retrieval_over,
    }


def _query_hash(row: dict[str, Any]) -> str:
    text = ""
    for key in ("query", "prompt", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if not text:
        text = "sample-id:" + str(row.get("sample_id") or row.get("item_id") or "")
    return sha_bytes(text.encode("utf-8"))


def audit_single_step_rows(
    rows: dict[str, list[dict[str, Any]]],
    tools: Sequence[dict[str, Any]],
    *,
    expected_split: str,
    min_retrieval_per_tool: int,
    min_execute_per_tool: int,
    min_refuse_per_tool: int,
    forbidden_query_hashes: set[str] | None = None,
) -> dict[str, Any]:
    names = {str(tool["name"]) for tool in tools}
    by_name = {str(tool["name"]): tool for tool in tools}
    forbidden = forbidden_query_hashes or set()
    errors: list[str] = []
    ids: set[str] = set()
    query_hashes: set[str] = set()
    task_query_hashes: dict[str, set[str]] = defaultdict(set)
    retrieval_counts: Counter[str] = Counter()
    execute_counts: Counter[str] = Counter()
    refuse_counts: Counter[str] = Counter()
    refusal_reasons: Counter[str] = Counter()
    negative_diversity: dict[str, set[str]] = defaultdict(set)
    optional_present: Counter[str] = Counter()
    required_only: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    counterfactual_kinds: dict[str, set[str]] = defaultdict(set)
    configurable_empty_execute = 0
    inapplicable_refusal_reason = 0
    mechanical_query_defects = 0
    for task, task_rows in rows.items():
        for row in task_rows:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in ids:
                errors.append(f"empty or duplicate sample id: {sample_id}")
            ids.add(sample_id)
            split = str(row.get("split") or "")
            if split != expected_split:
                errors.append(f"{sample_id}: split {split} != {expected_split}")
            group_splits[str(row.get("cf_group") or sample_id)].add(split)
            query = str(row.get("query") or "").strip()
            digest = _query_hash(row)
            if not query or digest in task_query_hashes[task]:
                errors.append(f"empty or duplicate query: {sample_id}")
            task_query_hashes[task].add(digest)
            query_hashes.add(digest)
            if digest in forbidden:
                errors.append(f"query overlaps a forbidden split: {sample_id}")
            if EVAL_MARKER.search(query):
                errors.append(f"marker contamination: {sample_id}")
            if any(
                defect in query
                for defect in ("请直接，请", "请为当前对象，请", "，，", "。。", "？？")
            ):
                mechanical_query_defects += 1
                errors.append(f"mechanical query defect: {sample_id}")
            if task == "retrieval":
                gold = str(row.get("gold_tool") or "")
                retrieval_counts[gold] += 1
                if gold not in names:
                    errors.append(f"unknown retrieval gold: {sample_id}")
                candidates = [str(item.get("name")) for item in row.get("catalog_tools") or []]
                if len(candidates) != 5 or gold not in candidates or len(set(candidates)) != 5:
                    errors.append(f"invalid retrieval candidate set: {sample_id}")
                negatives = [str(value) for value in row.get("hard_negatives") or []]
                if len(negatives) != 4 or gold in negatives:
                    errors.append(f"invalid retrieval negatives: {sample_id}")
                negative_diversity[gold].update(negatives)
            elif task == "fullcall":
                candidate = str(row.get("candidate_tool") or row.get("gold_name") or "")
                counterfactual_kinds[str(row.get("cf_group") or sample_id)].add(
                    str(row.get("kind") or "")
                )
                if candidate not in names:
                    errors.append(f"unknown fullcall candidate: {sample_id}")
                selected = [str(value) for value in row.get("retrieved_tools") or []]
                if len(selected) != 5 or candidate not in selected or len(set(selected)) != 5:
                    errors.append(f"invalid fullcall top5: {sample_id}")
                if row.get("permissions") != {}:
                    errors.append(f"permissions must be an unrestricted object: {sample_id}")
                if row.get("slot_provenance"):
                    errors.append(f"gold slot provenance leaked into input: {sample_id}")
                if row.get("kind") == "execute":
                    execute_counts[candidate] += 1
                    answers = row.get("answers") or []
                    if len(answers) != 1 or answers[0].get("name") != candidate:
                        errors.append(f"invalid execute answer: {sample_id}")
                    elif not arguments_match_schema(
                        answers[0].get("arguments"), by_name[candidate]["parameters"]
                    ):
                        errors.append(f"schema-invalid execute answer: {sample_id}")
                    arguments = answers[0].get("arguments") or {}
                    properties = by_name[candidate]["parameters"].get("properties") or {}
                    if properties and not arguments:
                        configurable_empty_execute += 1
                        errors.append(f"configurable execute action has empty arguments: {sample_id}")
                    if any(
                        not value_grounded_in_query(value, query)
                        for value in arguments.values()
                    ):
                        errors.append(f"execute argument is not query-grounded: {sample_id}")
                    expected_target = serialize_tool_target(answers)
                    if row.get("target_text") != expected_target:
                        errors.append(f"non-canonical target serialization: {sample_id}")
                    schema = by_name[candidate]["parameters"]
                    required = set(schema.get("required") or [])
                    optional = set((schema.get("properties") or {})) - required
                    if optional and set(arguments) & optional:
                        optional_present[candidate] += 1
                    if set(arguments) == required:
                        required_only[candidate] += 1
                else:
                    refuse_counts[candidate] += 1
                    reason_code = str(row.get("reason_code") or "unknown")
                    refusal_reasons[reason_code] += 1
                    schema = by_name[candidate]["parameters"]
                    if reason_code == "missing_slot" and not schema.get("required"):
                        inapplicable_refusal_reason += 1
                        errors.append(f"missing-slot refusal on tool without required slots: {sample_id}")
                    if reason_code == "unknown_slot_value" and not (
                        schema.get("properties") or {}
                    ):
                        inapplicable_refusal_reason += 1
                        errors.append(f"unknown-slot refusal on zero-argument tool: {sample_id}")
                    if row.get("answers") != []:
                        errors.append(f"refusal answer is not []: {sample_id}")
                    if row.get("target_text") != "[]":
                        errors.append(f"refusal target is not canonical []: {sample_id}")
            else:
                errors.append(f"unknown task key: {task}")
    for name in sorted(names):
        if retrieval_counts[name] < min_retrieval_per_tool:
            errors.append(f"retrieval coverage below floor for {name}")
        if execute_counts[name] < min_execute_per_tool:
            errors.append(f"execute coverage below floor for {name}")
        if refuse_counts[name] < min_refuse_per_tool:
            errors.append(f"refuse coverage below floor for {name}")
        optional = set((by_name[name]["parameters"].get("properties") or {})) - set(
            by_name[name]["parameters"].get("required") or []
        )
        if optional and min_execute_per_tool >= 2:
            if not optional_present[name]:
                errors.append(f"optional argument coverage missing for {name}")
            if (by_name[name]["parameters"].get("required") or []) and not required_only[name]:
                errors.append(f"required-only coverage missing for {name}")
    expected_negative_diversity = len(
        {0}
        | {
            1 + ((variant * 11 + index * 7) % 48)
            for variant in range(min_retrieval_per_tool)
            for index in range(3)
        }
    )
    for name in sorted(names):
        if len(negative_diversity[name]) < expected_negative_diversity:
            errors.append(
                f"negative diversity below {expected_negative_diversity} for {name}"
            )
    expected_refusal_reasons = {
        "negation_cancels",
        "missing_slot",
        "ambiguous_scope",
        "unknown_slot_value",
        "mixed_intent",
        "offtopic",
    }
    if min_refuse_per_tool >= 2 and set(refusal_reasons) != expected_refusal_reasons:
        errors.append(
            "refusal reason coverage differs: "
            + repr(sorted(set(refusal_reasons) ^ expected_refusal_reasons))
        )
    if any(len(splits) != 1 for splits in group_splits.values()):
        errors.append("counterfactual group crosses split")
    incomplete_counterfactual_groups = sum(
        kinds != {"execute", "refuse"} for kinds in counterfactual_kinds.values()
    )
    if incomplete_counterfactual_groups:
        errors.append(
            f"{incomplete_counterfactual_groups} full-call counterfactual groups lack both labels"
        )
    return {
        "schema": "mei-sft-v3-coverage-audit-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors[:100],
        "tool_universe_size": len(names),
        "retrieval_rows": len(rows.get("retrieval") or []),
        "fullcall_rows": len(rows.get("fullcall") or []),
        "retrieval_unique_gold_tools": len(retrieval_counts),
        "fullcall_unique_execute_tools": len(execute_counts),
        "fullcall_unique_refusal_candidates": len(refuse_counts),
        "retrieval_min_per_tool": min(retrieval_counts.values(), default=0),
        "execute_min_per_tool": min(execute_counts.values(), default=0),
        "refuse_min_per_tool": min(refuse_counts.values(), default=0),
        "execute_rows": sum(execute_counts.values()),
        "refuse_rows": sum(refuse_counts.values()),
        "refusal_reasons": dict(sorted(refusal_reasons.items())),
        "negative_distinct_min_per_tool": min(
            (len(values) for values in negative_diversity.values()), default=0
        ),
        "negative_distinct_expected_min": expected_negative_diversity,
        "gold_slot_provenance_rows": sum(
            1 for row in rows.get("fullcall") or [] if row.get("slot_provenance")
        ),
        "unrestricted_permission_rows": sum(
            1 for row in rows.get("fullcall") or [] if row.get("permissions") == {}
        ),
        "counterfactual_pair_groups": len(counterfactual_kinds),
        "incomplete_counterfactual_groups": incomplete_counterfactual_groups,
        "configurable_empty_execute_rows": configurable_empty_execute,
        "inapplicable_refusal_reason_rows": inapplicable_refusal_reason,
        "mechanical_query_defect_rows": mechanical_query_defects,
        "query_hashes": sorted(query_hashes),
    }


def query_hashes(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {_query_hash(row) for row in rows}


def file_spec(path: Path) -> dict[str, Any]:
    rows = None
    if path.suffix == ".jsonl":
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    result: dict[str, Any] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    if rows is not None:
        result["rows"] = rows
    return result
