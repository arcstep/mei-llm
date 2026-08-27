#!/usr/bin/env python3
"""Build sft-v2-eval-lock-v2 from the frozen large tool universe.

Does not overwrite lock v1 banks. Gold is compiler-determined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from park_toolcall_lib import char_trigrams, near_dup_holdout
from repo_paths import (
    PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN,
    PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN_V2,
    PACK_MEI_RETRIEVAL_V2_10K_CLEAN,
    PACK_MEI_RETRIEVAL_V2_10K_CLEAN_V2,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN_V2,
    SFT_V2_EVAL_LOCK_DIR_V2,
)
from sft_canonical_lib import compact_tools, dump_jsonl, load_jsonl, query_banned, sha256_text
from sft_v2_baseline_lib import dump_json, sha256_file, wilson_interval
from sft_v2_fair_prompts import dump_prompt_asset
from sft_v2_retrieval_backends import SparseRetriever
from sft_v2_tool_universe import UNIVERSE_ID, build_universe, dump_universe, slice_catalog

LOCK_VERSION = "sft-v2-eval-lock-v2"
GENERATOR = "sft-synth-v2-fair"

# Distinct from lock-v1 EVAL_* and CLEAN_* lexicons.
DEV_CITIES = ["赣州", "揭阳", "宿迁", "咸宁", "荆门", "随州"]
TEST_CITIES = ["黄石", "萍乡", "新余", "鹰潭", "滁州", "宣城"]
DEV_ROOMS = ["消控室", "样本库", "配电间", "茶歇区"]
TEST_ROOMS = ["安检口", "机柜区", "缓冲间", "无菌间"]
DEV_UNSEEN = {"travel", "campus"}
TEST_UNSEEN = {"finance", "medical", "logistics", "lab"}

NO_MATCH = {
    "dev": [
        "碱金属遇水为什么会放氢",
        "请默写一首盛唐五律中的颔联对仗",
        "海王星环带主要成分是什么",
        "把这段吐火罗语残片转写出来",
        "欧拉示性数在亏格为二曲面怎么算",
        "讲讲明代漕运的加耗规则",
        "写一段没有动词的景物短赋",
        "伽罗瓦群可解意味着什么",
        "冥王星的开普勒周期大约多久",
        "用生成函数写卡特兰数通项",
    ],
    "test": [
        "叶绿素荧光淬灭有哪几类",
        "请对仗补全一联边塞颔联",
        "天王星磁场为什么倾斜",
        "把这段西夏文残卷标音",
        "陈类的丁定理在二维怎么叙述",
        "讲讲宋代茶马司的比价",
        "写一首全是名词的咏物绝句",
        "戴德金分割怎样定义实数",
        "谷神星公转周期大概多少",
        "用递推写贝塞尔数的初值",
    ],
}

REASON_CODES = [
    "ambiguous_scope",
    "authority_required",
    "capability_insufficient",
    "correction_incomplete",
    "deixis_unresolved",
    "illegal_pair",
    "missing_external_fact",
    "missing_permission_token",
    "missing_slot",
    "mixed_intent",
    "partial_sequence_blocked",
    "ready_to_execute",
    "safety_judgment",
    "scene_conflict",
    "unknown_slot_value",
    "unsupported_scope",
]


MARKS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", "子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]


def _serial(i: int) -> str:
    a = MARKS[i % len(MARKS)]
    b = MARKS[(i // len(MARKS)) % len(MARKS)]
    c = MARKS[(i // (len(MARKS) ** 2)) % len(MARKS)]
    return f"对照簿{a}{b}{c}"


def _item_id(prefix: str, *parts: str) -> str:
    return prefix + sha256_text("\n".join(parts))[:16]
    return prefix + sha256_text("\n".join(parts))[:16]


def _compact(tools: list[dict]) -> list[dict]:
    return compact_tools(tools)


def _by_family(universe: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for t in universe["tools"]:
        out[str(t["family"])].append(t)
    return out


def _catalog_for(
    universe: dict,
    *,
    gold: dict | None,
    size: int,
    rng: random.Random,
    extra: list[dict] | None = None,
) -> list[dict]:
    pool = list(universe["tools"])
    picked: list[dict] = []
    seen: set[str] = set()

    def add(tool: dict | None) -> None:
        if not tool:
            return
        name = str(tool.get("name"))
        if name in seen:
            return
        picked.append(tool)
        seen.add(name)

    add(gold)
    for t in extra or []:
        add(t)
    rng.shuffle(pool)
    for t in pool:
        if len(picked) >= size:
            break
        add(t)
    if gold and gold["name"] not in seen:
        raise RuntimeError("gold dropped")
    if gold and len(picked) < 16:
        raise RuntimeError("catalog too small for positive")
    return picked[:size]


def _accept(query: str, kept: set[str], kept_tri: list[tuple[str, set[str]]]) -> bool:
    if query_banned(query):
        return False
    if near_dup_holdout(query):
        return False
    if query in kept:
        return False
    tri = char_trigrams(query)
    for other, ot in kept_tri:
        union = tri | ot
        if union and len(tri & ot) / len(union) >= 0.9:
            return False
    return True


def _commit(query: str, kept: set[str], kept_tri: list[tuple[str, set[str]]]) -> None:
    kept.add(query)
    kept_tri.append((query, char_trigrams(query)))


def build_retrieval(universe: dict, split: str, rng: random.Random) -> list[dict]:
    cities = DEV_CITIES if split == "dev" else TEST_CITIES
    rooms = DEV_ROOMS if split == "dev" else TEST_ROOMS
    unseen_fams = DEV_UNSEEN if split == "dev" else TEST_UNSEEN
    by_fam = _by_family(universe)
    kept: set[str] = set()
    kept_tri: list[tuple[str, set[str]]] = []
    rows: list[dict] = []
    sizes = [32, 128, len(universe["tools"])]
    similar = by_fam.get("similar_name") or []
    seen_tools = [t for t in universe["tools"] if t["family"] in {"park", "home", "office", "retail", "vrm", "type"}]
    unseen_tools = [t for t in universe["tools"] if t["family"] in unseen_fams]

    templates = {
        "seen": [
            ("请查询{city}现在的天气", "get_weather", {"city": None}),
            ("把{room}的灯调到{n}", "set_lights", {"room": None, "n": None}),
            ("请把{room}照明打开并关闭空调", "control_room_devices", {}),
            ("请撤销这笔房间占用预约", "cancel_reservation", {}),
            ("把舒适温度改到{n}度", "update_room_policy", {"n": None}),
            ("预约{city}出发的房间时段", "create_or_update_reservation", {"city": None}),
        ],
        "similar_name": [
            ("把{room}的灯具亮度调到{n}", "set_lamps", {"room": None}),
            ("只要{city}未来三天预报", "get_weather_forecast", {"city": None}),
            ("只关暖通不要动灯", "control_room_hvac", {}),
            ("取消那场会议不是房间预约", "cancel_meeting", {}),
        ],
        "unseen": [
            ("帮我订一张{city}出发的机票", "book_flight", {"city": None}),
            ("查一下{city}今晚的火车晚点", "check_train_delay", {"city": None}),
            ("支付发票金额", "pay_invoice", {}),
            ("预约明天的内科门诊", "book_clinic", {}),
            ("追踪这票货物", "track_shipment", {}),
            ("预约一间教室", "book_classroom", {}),
            ("启动培养箱", "start_incubator", {}),
            ("查询账户余额", "get_balance", {}),
        ],
        "hard_negative": [
            ("请查询{city}天气实况不是预警", "get_weather", {"city": None}),
            ("设置{room}灯光亮度不要改模式", "set_lights", {"room": None}),
            ("创建房间预约不要创建会议", "create_or_update_reservation", {}),
        ],
    }

    def emit(family: str, kind: str, query: str, gold_name: str, size: int, hard: list[str], serial: int) -> bool:
        query = f"{_serial(serial)}：{query}"
        if not _accept(query, kept, kept_tri):
            return False
        gold = next((t for t in universe["tools"] if t["name"] == gold_name), None)
        if gold is None:
            return False
        extras = [t for t in universe["tools"] if t["name"] in hard]
        catalog = _catalog_for(universe, gold=gold, size=size, rng=rng, extra=extras)
        names = [t["name"] for t in catalog]
        if gold_name not in names:
            return False
        _commit(query, kept, kept_tri)
        rows.append(
            {
                "item_id": _item_id("RET2-", split, family, query, gold_name),
                "sample_id": _item_id("RET2-", split, family, query, gold_name),
                "task": "retrieval",
                "split": split,
                "query": query,
                "gold_tool": gold_name,
                "family": family,
                "kind": kind,
                "seen_schema": family in {"seen", "hard_negative"},
                "catalog_size": len(catalog),
                "catalog_tool_names": names,
                "hard_negatives": hard,
                "no_match": False,
                "universe_id": UNIVERSE_ID,
                "universe_fp": universe["fingerprint"],
                "generator_version": GENERATOR,
            }
        )
        return True

    quotas = {"seen": 200, "unseen": 200, "similar_name": 200, "hard_negative": 200, "no_match": 200}
    # seen
    n = 0
    i = 0
    while n < quotas["seen"] and i < 8000:
        i += 1
        city, room = rng.choice(cities), rng.choice(rooms)
        tmpl, gold, _ = templates["seen"][i % len(templates["seen"])]
        q = tmpl.format(city=city, room=room, n=20 + (i % 70))
        size = sizes[i % 3]
        if size == 32 and gold in {"book_flight"}:
            size = 128
        if emit("seen", "positive", q, gold, size, [], i):
            n += 1
    n = 0
    i = 0
    while n < quotas["unseen"] and i < 8000:
        i += 1
        city, room = rng.choice(cities), rng.choice(rooms)
        fam_tools = [t for t in unseen_tools]
        if not fam_tools:
            break
        gold_t = fam_tools[i % len(fam_tools)]
        natural = {
            "book_flight": f"帮我订一张从{city}出发去宣城的机票",
            "cancel_flight": f"把{city}那张机票退掉",
            "book_hotel": f"在{city}订两晚酒店",
            "cancel_hotel": f"取消{city}酒店订单",
            "book_train": f"买一张{city}出发的火车票",
            "check_train_delay": f"{city}这趟车晚点了吗",
            "rent_car": f"在{city}租三天车",
            "get_visa_status": f"查一下护照签证办到哪了",
            "book_ferry": f"订{city}出发的轮渡",
            "list_airports": f"{city}有哪些机场",
            "add_frequent_flyer": f"登记航司常旅客号",
            "request_wheelchair": f"机场需要轮椅协助",
            "book_airport_transfer": f"订一趟机场接驳",
            "check_baggage": f"托运行李到哪了",
            "upgrade_seat": f"申请换到靠过道的座位",
            "buy_travel_insurance": f"买一份短期旅行保险",
            "pay_invoice": f"把那张发票的钱付了",
            "refund_payment": f"对上一笔付款发起退款",
            "transfer_funds": f"从常用账户转一笔钱出去",
            "get_balance": f"查一下常用账户还剩多少",
            "create_expense": f"填一张差旅报销单",
            "approve_expense": f"批准那张报销",
            "list_transactions": f"列出最近几笔流水",
            "freeze_card": f"先把那张卡冻住",
            "unfreeze_card": f"把冻住的卡解开",
            "set_payment_limit": f"把支付限额调低",
            "exchange_currency": f"换一些外币",
            "open_virtual_card": f"开通一张虚拟卡",
            "close_virtual_card": f"关掉那张虚拟卡",
            "get_fx_rate": f"查一下当前汇率",
            "schedule_transfer": f"预约明天转一笔账",
            "verify_payee": f"核验收款人是不是本人",
            "book_clinic": f"预约明天内科门诊",
            "cancel_clinic": f"取消那个门诊预约",
            "get_lab_result": f"化验结果出来了吗",
            "refill_prescription": f"把那张处方续上",
            "check_drug_stock": f"药房还有这种药吗",
            "book_imaging": f"预约一次影像检查",
            "report_symptom": f"登记一下发热症状",
            "get_vaccine_slot": f"还有疫苗号源吗",
            "update_allergy": f"过敏史加上青霉素",
            "request_sick_leave_note": f"开三天病假条",
            "book_physical": f"预约个体检套餐",
            "get_queue_number": f"内科取个号",
            "confirm_admission": f"确认住院床位",
            "discharge_summary": f"要出院小结",
            "set_reminder_meds": f"设置服药提醒",
            "check_insurance_cover": f"这个项目医保报不报",
            "create_shipment": f"寄一件货到{city}",
            "track_shipment": f"这票货现在到哪了",
            "cancel_shipment": f"这票货不要了",
            "schedule_pickup": f"预约下午上门揽收",
            "update_address": f"改一下收货地址",
            "declare_customs": f"给这票货报关",
            "get_freight_quote": f"询一下到{city}的运费",
            "book_warehouse_slot": f"预约一个库位",
            "mark_delivered": f"标记这票已签收",
            "report_damage": f"这票货破损了",
            "assign_courier": f"给这票指派快递员",
            "hold_at_depot": f"先把货扣在网点",
            "print_label": f"打印面单",
            "set_cod_amount": f"设置到付金额",
            "list_depots": f"{city}有哪些网点",
            "book_cold_chain": f"订冷链把货送到{city}",
            "book_classroom": f"预约{room}当教室",
            "cancel_classroom": f"取消{room}教室预约",
            "get_course_grade": f"查这门课成绩",
            "enroll_course": f"选修那门课",
            "drop_course": f"退选那门课",
            "book_lab_bench": f"预约实验台",
            "request_transcript": f"申请成绩单两份",
            "pay_tuition": f"把这学期学费交了",
            "reserve_library_room": f"预约研讨室两小时",
            "report_facility": f"{room}的灯坏了要报修",
            "get_shuttle_time": f"班车几点发",
            "apply_leave": f"请三天假",
            "book_printer": f"预约打印机",
            "open_dorm_gate": f"开一下宿舍门禁",
            "set_curfew_exception": f"申请今晚晚归",
            "list_lost_found": f"失物招领有没有工牌",
            "start_incubator": f"把培养箱开到三十七度",
            "stop_centrifuge": f"把离心机停掉",
            "log_sample": f"登记这个样本",
            "move_sample": f"把样本转到冷库",
            "calibrate_ph": f"校准这支 pH 探头",
            "set_fumehood": f"把通风橱开到二档",
            "book_autoclave": f"预约灭菌锅",
            "report_spill": f"{room}有洒漏",
            "order_reagent": f"申购一瓶试剂",
            "get_freezer_alarm": f"超低温有没有报警",
            "unlock_cabinet": f"解锁试剂柜",
            "set_shaker_rpm": f"摇床调到一百二十转",
            "record_od": f"记下这个样本的 OD",
            "schedule_maintenance": f"预约设备维护",
            "export_run_log": f"导出这一批实验记录",
            "seal_waste": f"把废料桶封上",
        }.get(gold_t["name"]) or f"办理与{city}相关的这项业务"
        size = sizes[(i + 1) % 3]
        if emit("unseen", "positive", natural, gold_t["name"], max(size, 32), [], i + 400):
            n += 1
    n = 0
    i = 0
    while n < quotas["similar_name"] and i < 8000:
        i += 1
        city, room = rng.choice(cities), rng.choice(rooms)
        tmpl, gold, _ = templates["similar_name"][i % len(templates["similar_name"])]
        q = tmpl.format(city=city, room=room, n=15 + (i % 80))
        pair = next((t for t in similar if t["name"] == gold), None)
        hard = [pair["similar_to"]] if pair and pair.get("similar_to") else []
        if emit("similar_name", "similar_name", q, gold, 128, hard, i + 800):
            n += 1
    n = 0
    i = 0
    while n < quotas["hard_negative"] and i < 8000:
        i += 1
        city, room = rng.choice(cities), rng.choice(rooms)
        tmpl, gold, _ = templates["hard_negative"][i % len(templates["hard_negative"])]
        q = tmpl.format(city=city, room=room)
        similars = [t["name"] for t in similar if t.get("similar_to") == gold]
        if emit("hard_negative", "hard_negative", q, gold, 128, similars[:3], i + 1200):
            n += 1
    for i, q in enumerate(NO_MATCH[split] * 20):
        if len([r for r in rows if r["family"] == "no_match"]) >= quotas["no_match"]:
            break
        query = f"{_serial(i + 1600)}：{q}"
        if not _accept(query, kept, kept_tri):
            continue
        catalog = _catalog_for(universe, gold=None, size=128, rng=rng)
        _commit(query, kept, kept_tri)
        rows.append(
            {
                "item_id": _item_id("RET2-", split, "no_match", query),
                "sample_id": _item_id("RET2-", split, "no_match", query),
                "task": "retrieval",
                "split": split,
                "query": query,
                "gold_tool": None,
                "family": "no_match",
                "kind": "no_match",
                "seen_schema": False,
                "catalog_size": len(catalog),
                "catalog_tool_names": [t["name"] for t in catalog],
                "hard_negatives": [],
                "no_match": True,
                "universe_id": UNIVERSE_ID,
                "universe_fp": universe["fingerprint"],
                "generator_version": GENERATOR,
            }
        )
    assert len(rows) >= 1000, f"retrieval {split} {len(rows)}"
    return rows[:1000]


FULLCALL_BEHAVIORS = [
    ("control_room_devices", {"light_on": True}, "execute", "把灯打开"),
    ("control_room_devices", {"light_on": False}, "execute", "把灯关掉"),
    ("control_room_devices", {"ac_power": True, "ac_target_c": 24}, "execute", "打开空调并设到二十四度"),
    ("control_room_devices", {"ac_power": True, "ac_target_c": 26}, "execute", "打开空调并设到二十六度"),
    ("control_room_devices", {"ac_power": False}, "execute", "把空调关掉"),
    ("control_room_devices", {"ac_target_c": 25}, "execute", "把空调调到二十五度"),
    ("control_room_devices", {"ac_target_c": 27}, "execute", "把空调调到二十七度"),
    ("control_room_devices", {"light_on": False, "ac_power": False}, "execute", "灯和空调都关掉"),
    ("create_or_update_reservation", {"start_iso": "2026-09-08T09:00:00+08:00", "duration_minutes": 60}, "execute", "预约九点开始用六十分钟"),
    ("create_or_update_reservation", {"start_iso": "2026-09-08T14:00:00+08:00", "duration_minutes": 120, "precool_minutes": 15}, "execute", "预约十四点开始两小时并提前十五分钟预冷"),
    ("cancel_reservation", {}, "execute", "取消当前房间预约"),
    ("update_room_policy", {"comfort_c": 24}, "execute", "把舒适温度改成二十四度"),
    ("update_room_policy", {"comfort_c": 22}, "execute", "把舒适温度改成二十二度"),
    ("get_weather", {"city": "CITY"}, "execute", "查询CITY现在的天气"),
    ("set_lights", {"room": "ROOM", "brightness": 40}, "execute", "把ROOM的灯调到四十"),
    ("set_lights", {"room": "ROOM", "brightness": 80}, "execute", "把ROOM的灯调到八十"),
    ("create_event", {"title": "封存对照会", "duration_min": 30}, "execute", "创建一个三十分钟的封存对照会"),
    ("set_volume", {"level": 7}, "execute", "把音量调到七"),
    ("lookup_price", {"item": "档案夹", "qty": 5}, "execute", "查一下档案夹五份的单价"),
    ("rent_car", {"city": "CITY", "days": 3}, "execute", "在CITY租三天车"),
    ("list_airports", {"city": "CITY"}, "execute", "列出CITY有哪些机场"),
    ("create_shipment", {"sku": "冷链标签", "dest": "CITY"}, "execute", "寄冷链标签到CITY"),
    ("get_course_grade", {"course": "对照实验课"}, "execute", "查对照实验课成绩"),
    ("stop_centrifuge", {"unit": "U3"}, "execute", "把离心机U3停掉"),
    ("lookup_price", {"item": "冷链标签", "qty": 2}, "execute", "查一下冷链标签两份的单价"),
    ("book_flight", {"from_city": "CITY", "to_city": "宣城", "date": "2026-09-12"}, "execute", "订一张CITY去宣城九月十二日的机票"),
    ("pay_invoice", {"invoice_id": "INV-8841", "amount": 320.5}, "execute", "支付发票INV-8841金额三百二十点五"),
    ("track_shipment", {"waybill": "WB77881"}, "execute", "追踪运单WB77881"),
    ("book_clinic", {"dept": "内科", "date": "2026-09-09"}, "execute", "预约九月九日内科门诊"),
    ("get_balance", {"account": "A-19"}, "execute", "查询账户A-19的余额"),
    ("start_incubator", {"chamber": "C2", "temp_c": 37}, "execute", "把培养箱C2开到三十七度"),
    ("book_classroom", {"room": "ROOM", "start_iso": "2026-09-10T08:00:00+08:00"}, "execute", "预约ROOM教室八点开始"),
    ("control_room_devices", {}, "refuse", "把设备调一下"),
    ("set_lights", {}, "refuse", "把灯调亮一点"),
    ("get_weather", {}, "refuse", "查一下天气"),
    ("cancel_reservation", {}, "refuse", "取消预约"),
    ("control_room_devices", {}, "refuse", "灯已经开着请再开一次"),
    ("create_or_update_reservation", {}, "refuse", "随便订个房间"),
    ("update_room_policy", {}, "refuse", "把策略改了"),
    ("book_flight", {}, "refuse", "帮我出门旅行"),
    ("pay_invoice", {}, "refuse", "把钱付了"),
    ("set_lights", {}, "refuse", "光合作用需要开灯吗"),
    ("control_room_devices", {}, "refuse", "空调故障了还是把温度打到十八度"),
    ("cancel_reservation", {}, "refuse", "没有预约也请取消"),
    ("mixed", {}, "refuse", "既要订机票又要把灯关掉"),
    ("unsupported", {}, "refuse", "帮我调度整座城市的电网"),
]


def _fill_slots(args: dict, city: str, room: str) -> dict:
    out = {}
    for k, v in args.items():
        if v == "CITY":
            out[k] = city
        elif v == "ROOM":
            out[k] = room
        else:
            out[k] = v
    return out


def _place_oracle(catalog: list[dict], gold_name: str | None, rank: int, rng: random.Random) -> list[dict]:
    by_name = {t["name"]: t for t in catalog}
    others = [t for t in catalog if t["name"] != gold_name]
    rng.shuffle(others)
    picked = others[:4]
    if gold_name and gold_name in by_name:
        rank = max(0, min(4, rank))
        picked.insert(rank, by_name[gold_name])
    while len(picked) < 5 and others:
        t = others[len(picked)]
        if t["name"] not in {x["name"] for x in picked}:
            picked.append(t)
    return _compact(picked[:5])


def build_fullcall(universe: dict, split: str, rng: random.Random, retriever: SparseRetriever) -> list[dict]:
    cities = DEV_CITIES if split == "dev" else TEST_CITIES
    rooms = DEV_ROOMS if split == "dev" else TEST_ROOMS
    kept: set[str] = set()
    kept_tri: list[tuple[str, set[str]]] = []
    rows: list[dict] = []
    full = universe["tools"]
    retriever.build(_compact(full))
    behavior_ids: set[str] = set()
    i = 0
    while len(rows) < 1200 and i < 20000:
        i += 1
        tmpl_id = i % len(FULLCALL_BEHAVIORS)
        name, args, decision, tmpl = FULLCALL_BEHAVIORS[tmpl_id]
        city, room = cities[i % len(cities)], rooms[i % len(rooms)]
        query = tmpl.replace("CITY", city).replace("ROOM", room)
        query = f"{_serial(i + 3000)}：{query}"
        if decision == "refuse":
            query = query + ("。" if i % 2 == 0 else "。请处理。")
            if i % 5 == 0:
                query = f"{room}侧：{query}" if "灯" in query or "空调" in query else query
        if not _accept(query, kept, kept_tri):
            continue
        gold_args = _fill_slots(args, city, room) if decision == "execute" and name not in {"mixed", "unsupported"} else {}
        gold_name = name if decision == "execute" and name not in {"mixed", "unsupported"} else None
        answers = [{"name": gold_name, "arguments": gold_args}] if gold_name else []
        gold_tool = next((t for t in full if t["name"] == gold_name), None) if gold_name else None
        catalog = _catalog_for(universe, gold=gold_tool, size=128, rng=rng)
        rank = (i // len(FULLCALL_BEHAVIORS)) % 5
        oracle = _place_oracle(catalog, gold_name, rank, rng)
        learned = retriever.search(query, k=5)
        learned_names = [t["name"] for t in learned]
        hit = (gold_name in learned_names) if gold_name else True
        bid = json.dumps([gold_name or decision, gold_args, tmpl_id], ensure_ascii=False, sort_keys=True)
        behavior_ids.add(bid)
        slice_id = "execute" if answers else "refuse"
        _commit(query, kept, kept_tri)
        facts = ""
        if "已经开着" in query:
            facts = "灯光已开启；空调关闭。"
        if "故障" in query:
            facts = "空调健康=fault。"
        if "没有预约" in query:
            facts = "当前无预约。"
        rows.append(
            {
                "item_id": _item_id("FC2-", split, query, str(gold_name), str(rank)),
                "sample_id": _item_id("FC2-", split, query, str(gold_name), str(rank)),
                "task": "fullcall",
                "split": split,
                "query": query,
                "system_facts": facts,
                "answers": answers,
                "gold_name": gold_name,
                "gold_rank": rank if gold_name else None,
                "kind": slice_id,
                "family": (gold_tool or {}).get("family") if gold_tool else "refuse",
                "slice": slice_id,
                "oracle_top5": oracle,
                "learned_top5": _compact(learned),
                "retrieval_hit_learned": hit,
                "catalog_size": 128,
                "catalog_tool_names": [t["name"] for t in catalog],
                "universe_id": UNIVERSE_ID,
                "universe_fp": universe["fingerprint"],
                "serializer": "mei-tool-call-serializer-v2",
                "generator_version": GENERATOR,
                "behavior_id": bid,
                "template_id": tmpl_id,
            }
        )
    n_exec = len({json.dumps(r["answers"], ensure_ascii=False, sort_keys=True) for r in rows if r.get("answers")})
    n_tmpl = len({r.get("template_id") for r in rows})
    assert len(rows) >= 1200, len(rows)
    assert n_exec >= 40, n_exec
    assert n_tmpl >= 30, n_tmpl
    for r in rows:
        assert len(r["oracle_top5"]) == 5, r["item_id"]
        assert len(r["learned_top5"]) == 5, r["item_id"]
    return rows[:1200]


MW_TEMPLATES = {
    "ready_to_execute": ("把{room}的灯调到四十", "set_lights", {"room": "ROOM", "brightness": 40}, ""),
    "missing_slot": ("把灯打开", "set_lights", {}, ""),
    "missing_external_fact": ("按当前室温再降一度", "control_room_devices", {}, ""),
    "missing_permission_token": ("把舒适温度改成二十二度", "update_room_policy", {"comfort_c": 22}, "权限：无"),
    "ambiguous_scope": ("把灯关掉，消控室和样本库都有灯", "control_room_devices", {}, "设备：消控室灯、样本库灯"),
    "mixed_intent": ("订机票并把灯关掉", None, {}, ""),
    "deixis_unresolved": ("把它打开", None, {}, ""),
    "correction_incomplete": ("不是灯，是空调，但别调太冷", "control_room_devices", {}, ""),
    "authority_required": ("全楼紧急断电", None, {}, ""),
    "safety_judgment": ("空调故障仍然打到十八度", "control_room_devices", {}, "空调健康=fault"),
    "capability_insufficient": ("请播放背景音乐", None, {}, ""),
    "unsupported_scope": ("调度全市电网负荷", None, {}, ""),
    "scene_conflict": ("灯已经开着请再开一次", "control_room_devices", {}, "灯光已开启"),
    "illegal_pair": ("预冷九十分钟但只预约十五分钟", "create_or_update_reservation", {}, ""),
    "unknown_slot_value": ("把空调调到八十度", "control_room_devices", {"ac_target_c": 80}, ""),
    "partial_sequence_blocked": ("还没预约先改时长到两小时", "create_or_update_reservation", {}, "当前无预约"),
}


def _mw_learned_reason(code: str, gold_name: str | None, learned: list[dict]) -> str:
    names = {t["name"] for t in learned}
    if code == "ready_to_execute" and gold_name and gold_name not in names:
        return "capability_insufficient"
    return code


def build_mw(universe: dict, split: str, rng: random.Random, retriever: SparseRetriever) -> list[dict]:
    rooms = DEV_ROOMS if split == "dev" else TEST_ROOMS
    cities = DEV_CITIES if split == "dev" else TEST_CITIES
    kept: set[str] = set()
    kept_tri: list[tuple[str, set[str]]] = []
    rows: list[dict] = []
    retriever.build(_compact(universe["tools"]))
    for code in REASON_CODES:
        tmpl, gold_name, _args, facts0 = MW_TEMPLATES[code]
        for i in range(100):
            room = rooms[i % len(rooms)]
            city = cities[i % len(cities)]
            query = tmpl.format(room=room, city=city)
            query = f"{_serial(i + 8000 + REASON_CODES.index(code) * 100)}：{query}"
            if not _accept(query, kept, kept_tri):
                query = f"{city}记录：{query}"
                if not _accept(query, kept, kept_tri):
                    continue
            gold_tool = next((t for t in universe["tools"] if t["name"] == gold_name), None) if gold_name else None
            catalog = _catalog_for(universe, gold=gold_tool, size=128, rng=rng)
            oracle = _place_oracle(catalog, gold_name, i % 5, rng)
            learned = retriever.search(query, k=5)
            learned_code = _mw_learned_reason(code, gold_name, learned)
            facts = facts0
            _commit(query, kept, kept_tri)
            rows.append(
                {
                    "item_id": _item_id("MW2-", split, code, query),
                    "sample_id": _item_id("MW2-", split, code, query),
                    "task": "mw",
                    "split": split,
                    "query": query,
                    "system_facts": facts,
                    "reason_code": code,
                    "learned_reason_code": learned_code,
                    "gold_name": gold_name,
                    "oracle_top5": oracle,
                    "learned_top5": _compact(learned),
                    "retrieval_hit_learned": (gold_name in {t["name"] for t in learned}) if gold_name else True,
                    "family": code,
                    "kind": code,
                    "universe_id": UNIVERSE_ID,
                    "universe_fp": universe["fingerprint"],
                    "generator_version": GENERATOR,
                }
            )
        assert sum(1 for r in rows if r["reason_code"] == code) >= 100, code
    assert len(rows) >= 1600, len(rows)
    return rows


def _assert_banks(retrieval, fullcall, mw, universe) -> None:
    pos = [r for r in retrieval if not r.get("no_match")]
    assert all(r["catalog_size"] >= 16 for r in pos)
    assert all(len(r["oracle_top5"]) == 5 for r in fullcall)
    assert all(len(r["learned_top5"]) == 5 for r in fullcall)
    exec_b = {r["behavior_id"] for r in fullcall if r["kind"] == "execute"}
    assert len(exec_b) >= 16
    names = {t["name"] for t in universe["tools"]}
    assert len(names) >= 128
    qs = [r["query"] for r in retrieval + fullcall + mw]
    assert len(qs) == len(set(qs))


def write_clean_v2(eval_queries: set[str]) -> dict:
    reports = {}
    mapping = [
        (PACK_MEI_RETRIEVAL_V2_10K_CLEAN, PACK_MEI_RETRIEVAL_V2_10K_CLEAN_V2),
        (PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN, PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN_V2),
        (PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN, PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN_V2),
    ]
    eval_tri = [(q, char_trigrams(q)) for q in eval_queries if q]
    for src, dst in mapping:
        rows = load_jsonl(src) if src.is_file() else []
        kept = []
        dropped = 0
        for row in rows:
            q = str(row.get("query") or "").strip()
            if q in eval_queries:
                dropped += 1
                continue
            tq = char_trigrams(q)
            hit = False
            for eq, et in eval_tri:
                union = tq | et
                if union and len(tq & et) / len(union) >= 0.9:
                    hit = True
                    break
            if hit:
                dropped += 1
                continue
            kept.append(row)
        dump_jsonl(dst, kept)
        reports[dst.name] = {"src": src.name, "n": len(kept), "dropped": dropped}
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SFT_V2_EVAL_LOCK_DIR_V2)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    universe = dump_universe(out / "tool-universe-v1.json")
    rng_dev = random.Random(20260827)
    rng_test = random.Random(20260828)
    bm25_dev = SparseRetriever(mode="bm25", k=5)
    bm25_test = SparseRetriever(mode="bm25", k=5)
    ret_dev = build_retrieval(universe, "dev", rng_dev)
    ret_test = build_retrieval(universe, "test", rng_test)
    fc_dev = build_fullcall(universe, "dev", rng_dev, bm25_dev)
    fc_test = build_fullcall(universe, "test", rng_test, bm25_test)
    mw_dev = build_mw(universe, "dev", rng_dev, bm25_dev)
    mw_test = build_mw(universe, "test", rng_test, bm25_test)
    _assert_banks(ret_dev, fc_dev, mw_dev, universe)
    _assert_banks(ret_test, fc_test, mw_test, universe)
    dump_jsonl(out / "eval-retrieval-dev.jsonl", ret_dev)
    dump_jsonl(out / "eval-retrieval-test.jsonl", ret_test)
    dump_jsonl(out / "eval-fullcall-dev.jsonl", fc_dev)
    dump_jsonl(out / "eval-fullcall-test.jsonl", fc_test)
    dump_jsonl(out / "eval-mw-dev.jsonl", mw_dev)
    dump_jsonl(out / "eval-mw-test.jsonl", mw_test)
    prompts = dump_prompt_asset(out / "fair-prompts-v1.json")
    eval_q = {r["query"] for r in ret_dev + ret_test + fc_dev + fc_test + mw_dev + mw_test}
    clean = write_clean_v2(eval_q)
    lock = {
        "id": LOCK_VERSION,
        "universe_id": UNIVERSE_ID,
        "universe_n": universe["n_tools"],
        "universe_sha256": sha256_file(out / "tool-universe-v1.json"),
        "universe_fp": universe["fingerprint"],
        "banks": {
            "retrieval_dev": {"n": len(ret_dev), "sha256": sha256_file(out / "eval-retrieval-dev.jsonl")},
            "retrieval_test": {"n": len(ret_test), "sha256": sha256_file(out / "eval-retrieval-test.jsonl")},
            "fullcall_dev": {"n": len(fc_dev), "sha256": sha256_file(out / "eval-fullcall-dev.jsonl")},
            "fullcall_test": {"n": len(fc_test), "sha256": sha256_file(out / "eval-fullcall-test.jsonl")},
            "mw_dev": {"n": len(mw_dev), "sha256": sha256_file(out / "eval-mw-dev.jsonl")},
            "mw_test": {"n": len(mw_test), "sha256": sha256_file(out / "eval-mw-test.jsonl")},
        },
        "prompt_version": prompts["prompt_version"],
        "fullcall_system_sha256": prompts["fullcall_system_sha256"],
        "mw_system_sha256": prompts["mw_system_sha256"],
        "learned_retriever": "bm25",
        "clean_v2": clean,
        "v1_preserved": True,
    }
    dump_json(out / "lock.json", lock)
    print(json.dumps({k: lock[k] for k in ("id", "universe_n", "banks", "clean_v2")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
