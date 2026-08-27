"""Frozen large tool universe for sft-v2 fair baselines.

Merges canonical toolsets then expands same-name neighbors, same-action/different-object,
and unseen schema families. Each tool locks name/description/parameters/family/fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sft_canonical_lib import compact_tools, dumps_canonical, load_toolset

UNIVERSE_ID = "sft-v2-tool-universe-v1"
SOURCE_TOOLSETS = [
    "mei-park-room-v1",
    "needle-home-v0",
    "mei-office-v0",
    "mei-office-mut-v0",
    "mei-office-rename-v0",
    "mei-retail-v0",
    "mei-type-v0",
    "needle-invoice-v0",
    "needle-vrm-agent-v0",
]

SEEN_FAMILIES = {"park", "home", "office", "retail", "vrm", "type"}
UNSEEN_FAMILIES = {"travel", "finance", "medical", "logistics", "campus", "lab"}
SIMILAR_FAMILIES = {"similar_name"}

TOOLSET_FAMILY = {
    "mei-park-room-v1": "park",
    "needle-home-v0": "home",
    "mei-office-v0": "office",
    "mei-office-mut-v0": "office",
    "mei-office-rename-v0": "office",
    "mei-retail-v0": "retail",
    "mei-type-v0": "type",
    "needle-invoice-v0": "invoice",
    "needle-vrm-agent-v0": "vrm",
}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    family: str,
    *,
    similar_to: str | None = None,
) -> dict[str, Any]:
    schema = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }
    fp = hashlib.sha256(dumps_canonical(compact_tools([schema])).encode("utf-8")).hexdigest()
    return {
        **schema,
        "family": family,
        "fingerprint": fp,
        "similar_to": similar_to,
        "tool_id": name,
    }


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _int_prop(desc: str, lo: int | None = None, hi: int | None = None) -> dict:
    out: dict[str, Any] = {"type": "integer", "description": desc}
    if lo is not None:
        out["minimum"] = lo
    if hi is not None:
        out["maximum"] = hi
    return out


def _num_prop(desc: str, lo: float | None = None, hi: float | None = None) -> dict:
    out: dict[str, Any] = {"type": "number", "description": desc}
    if lo is not None:
        out["minimum"] = lo
    if hi is not None:
        out["maximum"] = hi
    return out


def _bool_prop(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


def _load_canonical() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tid in SOURCE_TOOLSETS:
        family = TOOLSET_FAMILY[tid]
        for tool in load_toolset(tid).get("tools") or []:
            name = str(tool.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            compact = compact_tools([tool])[0]
            fp = hashlib.sha256(dumps_canonical([compact]).encode("utf-8")).hexdigest()
            out.append(
                {
                    **compact,
                    "family": family,
                    "fingerprint": fp,
                    "similar_to": None,
                    "tool_id": name,
                    "source_toolset": tid,
                }
            )
    return out


def _synthetic() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    # similar-name neighbors of seen actions
    tools += [
        _tool("set_lamps", "设置房间灯具亮度（与 set_lights 近邻）。", {"room": _str_prop("房间"), "brightness": _int_prop("亮度", 0, 100)}, ["room", "brightness"], "similar_name", similar_to="set_lights"),
        _tool("set_light_mode", "只切换灯光模式，不改亮度。", {"room": _str_prop("房间"), "mode": {"type": "string", "enum": ["auto", "manual"]}}, ["room", "mode"], "similar_name", similar_to="set_lights"),
        _tool("get_weather_forecast", "查询未来天气预报，不是当前实况。", {"city": _str_prop("城市"), "days": _int_prop("天数", 1, 7)}, ["city"], "similar_name", similar_to="get_weather"),
        _tool("get_weather_alerts", "查询气象预警。", {"city": _str_prop("城市")}, ["city"], "similar_name", similar_to="get_weather"),
        _tool("control_room_hvac", "只控制暖通，不控制灯光。", {"ac_power": _bool_prop("空调电源"), "ac_target_c": _num_prop("目标温度", 16, 32)}, ["ac_power"], "similar_name", similar_to="control_room_devices"),
        _tool("control_room_lighting", "只控制房间灯光。", {"light_on": _bool_prop("灯光开关")}, ["light_on"], "similar_name", similar_to="control_room_devices"),
        _tool("create_meeting", "创建会议预约（相对房间预约的近邻）。", {"start_iso": _str_prop("开始时间"), "title": _str_prop("标题")}, ["start_iso"], "similar_name", similar_to="create_or_update_reservation"),
        _tool("cancel_meeting", "取消会议，不是取消房间预约。", {"title": _str_prop("标题")}, ["title"], "similar_name", similar_to="cancel_reservation"),
        _tool("update_hvac_policy", "只更新暖通策略。", {"comfort_c": _num_prop("舒适温度", 16, 32)}, ["comfort_c"], "similar_name", similar_to="update_room_policy"),
        _tool("check_stock", "按 SKU 查库存余量。", {"item": _str_prop("SKU")}, ["item"], "similar_name", similar_to="check_inventory"),
        _tool("create_bill", "创建账单文档。", {"title": _str_prop("标题")}, ["title"], "similar_name", similar_to="create_invoice"),
        _tool("open_ticket", "打开运维工单。", {"title": _str_prop("标题")}, ["title"], "similar_name", similar_to="create_ticket"),
        _tool("set_switch_level", "设置开关档位而不是通断。", {"id": _str_prop("开关 id"), "level": _int_prop("档位", 0, 3)}, ["id", "level"], "similar_name", similar_to="set_switch"),
        _tool("set_thermostat", "设置恒温器，不是瞬时温度。", {"value": _num_prop("温度", 10, 40)}, ["value"], "similar_name", similar_to="set_temp"),
        _tool("lookup_sku_price", "按 SKU 查价。", {"sku": _str_prop("SKU")}, ["sku"], "similar_name", similar_to="lookup_price"),
        _tool("rename_room_tag", "给房间标签改名。", {"old": _str_prop("旧名"), "new": _str_prop("新名")}, ["old", "new"], "similar_name", similar_to="rename_item"),
    ]
    # unseen families
    travel = [
        ("book_flight", "预订航班。", {"from_city": _str_prop("出发城市"), "to_city": _str_prop("到达城市"), "date": _str_prop("日期")}, ["from_city", "to_city", "date"]),
        ("cancel_flight", "取消已订航班。", {"pnr": _str_prop("订座编码")}, ["pnr"]),
        ("book_hotel", "预订酒店房间。", {"city": _str_prop("城市"), "nights": _int_prop("晚数", 1, 30)}, ["city", "nights"]),
        ("cancel_hotel", "取消酒店预订。", {"confirmation": _str_prop("确认号")}, ["confirmation"]),
        ("book_train", "预订火车票。", {"from_city": _str_prop("出发"), "to_city": _str_prop("到达"), "date": _str_prop("日期")}, ["from_city", "to_city", "date"]),
        ("check_train_delay", "查询列车晚点。", {"train_no": _str_prop("车次")}, ["train_no"]),
        ("rent_car", "租车。", {"city": _str_prop("城市"), "days": _int_prop("天数", 1, 14)}, ["city", "days"]),
        ("get_visa_status", "查询签证进度。", {"passport": _str_prop("护照号")}, ["passport"]),
        ("book_ferry", "预订轮渡。", {"from_port": _str_prop("出发港"), "to_port": _str_prop("到达港")}, ["from_port", "to_port"]),
        ("list_airports", "列出城市机场。", {"city": _str_prop("城市")}, ["city"]),
        ("add_frequent_flyer", "登记常旅客号。", {"airline": _str_prop("航司"), "number": _str_prop("会员号")}, ["airline", "number"]),
        ("request_wheelchair", "申请机场轮椅协助。", {"pnr": _str_prop("订座编码")}, ["pnr"]),
        ("book_airport_transfer", "预订机场接驳。", {"airport": _str_prop("机场"), "time": _str_prop("时间")}, ["airport", "time"]),
        ("check_baggage", "查询托运行李。", {"tag": _str_prop("行李牌")}, ["tag"]),
        ("upgrade_seat", "申请升舱或换座。", {"pnr": _str_prop("订座编码"), "seat": _str_prop("座位")}, ["pnr"]),
        ("buy_travel_insurance", "购买旅行保险。", {"days": _int_prop("天数", 1, 90)}, ["days"]),
    ]
    finance = [
        ("pay_invoice", "支付指定发票。", {"invoice_id": _str_prop("发票号"), "amount": _num_prop("金额")}, ["invoice_id", "amount"]),
        ("refund_payment", "发起退款。", {"payment_id": _str_prop("支付单号")}, ["payment_id"]),
        ("transfer_funds", "内部转账。", {"from_account": _str_prop("转出"), "to_account": _str_prop("转入"), "amount": _num_prop("金额")}, ["from_account", "to_account", "amount"]),
        ("get_balance", "查询账户余额。", {"account": _str_prop("账户")}, ["account"]),
        ("create_expense", "创建报销单。", {"title": _str_prop("标题"), "amount": _num_prop("金额")}, ["title", "amount"]),
        ("approve_expense", "批准报销。", {"expense_id": _str_prop("报销单号")}, ["expense_id"]),
        ("list_transactions", "列出近期流水。", {"account": _str_prop("账户"), "limit": _int_prop("条数", 1, 50)}, ["account"]),
        ("freeze_card", "冻结银行卡。", {"card_id": _str_prop("卡号后四位")}, ["card_id"]),
        ("unfreeze_card", "解冻银行卡。", {"card_id": _str_prop("卡号后四位")}, ["card_id"]),
        ("set_payment_limit", "设置支付限额。", {"account": _str_prop("账户"), "limit": _num_prop("限额")}, ["account", "limit"]),
        ("exchange_currency", "兑换外币。", {"from_ccy": _str_prop("源币种"), "to_ccy": _str_prop("目标币种"), "amount": _num_prop("金额")}, ["from_ccy", "to_ccy", "amount"]),
        ("open_virtual_card", "开通虚拟卡。", {"label": _str_prop("标签")}, ["label"]),
        ("close_virtual_card", "关闭虚拟卡。", {"card_id": _str_prop("卡 id")}, ["card_id"]),
        ("get_fx_rate", "查询汇率。", {"pair": _str_prop("货币对")}, ["pair"]),
        ("schedule_transfer", "预约转账。", {"to_account": _str_prop("转入"), "amount": _num_prop("金额"), "when": _str_prop("时间")}, ["to_account", "amount", "when"]),
        ("verify_payee", "核验收款人。", {"name": _str_prop("姓名"), "account": _str_prop("账户")}, ["name", "account"]),
    ]
    medical = [
        ("book_clinic", "预约门诊。", {"dept": _str_prop("科室"), "date": _str_prop("日期")}, ["dept", "date"]),
        ("cancel_clinic", "取消门诊预约。", {"appointment_id": _str_prop("预约号")}, ["appointment_id"]),
        ("get_lab_result", "查询化验结果。", {"order_id": _str_prop("单号")}, ["order_id"]),
        ("refill_prescription", "续方。", {"rx_id": _str_prop("处方号")}, ["rx_id"]),
        ("check_drug_stock", "查询药房库存。", {"drug": _str_prop("药品")}, ["drug"]),
        ("book_imaging", "预约影像检查。", {"modality": _str_prop("检查类型"), "date": _str_prop("日期")}, ["modality", "date"]),
        ("report_symptom", "登记症状（非诊断）。", {"symptom": _str_prop("症状")}, ["symptom"]),
        ("get_vaccine_slot", "查询疫苗号源。", {"vaccine": _str_prop("疫苗名")}, ["vaccine"]),
        ("update_allergy", "更新过敏史。", {"substance": _str_prop("过敏原")}, ["substance"]),
        ("request_sick_leave_note", "申请病假条。", {"days": _int_prop("天数", 1, 14)}, ["days"]),
        ("book_physical", "预约体检套餐。", {"package": _str_prop("套餐"), "date": _str_prop("日期")}, ["package", "date"]),
        ("get_queue_number", "取就诊排队号。", {"dept": _str_prop("科室")}, ["dept"]),
        ("confirm_admission", "确认住院。", {"ward": _str_prop("病区")}, ["ward"]),
        ("discharge_summary", "获取出院小结。", {"visit_id": _str_prop("就诊号")}, ["visit_id"]),
        ("set_reminder_meds", "设置服药提醒。", {"drug": _str_prop("药品"), "hour": _int_prop("小时", 0, 23)}, ["drug", "hour"]),
        ("check_insurance_cover", "查询医保覆盖。", {"item": _str_prop("项目")}, ["item"]),
    ]
    logistics = [
        ("create_shipment", "创建运单。", {"sku": _str_prop("货物"), "dest": _str_prop("目的地")}, ["sku", "dest"]),
        ("track_shipment", "追踪运单。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("cancel_shipment", "取消运单。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("schedule_pickup", "预约揽收。", {"address": _str_prop("地址"), "window": _str_prop("时段")}, ["address", "window"]),
        ("update_address", "更新收货地址。", {"waybill": _str_prop("运单号"), "address": _str_prop("地址")}, ["waybill", "address"]),
        ("declare_customs", "申报海关。", {"waybill": _str_prop("运单号"), "hs_code": _str_prop("税号")}, ["waybill", "hs_code"]),
        ("get_freight_quote", "询运费。", {"from_city": _str_prop("出发"), "to_city": _str_prop("到达"), "kg": _num_prop("公斤")}, ["from_city", "to_city", "kg"]),
        ("book_warehouse_slot", "预约库位。", {"sku": _str_prop("货物"), "hours": _int_prop("小时", 1, 72)}, ["sku", "hours"]),
        ("mark_delivered", "标记签收。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("report_damage", "报告货损。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("assign_courier", "指派快递员。", {"waybill": _str_prop("运单号"), "courier": _str_prop("姓名")}, ["waybill", "courier"]),
        ("hold_at_depot", "滞留网点。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("print_label", "打印面单。", {"waybill": _str_prop("运单号")}, ["waybill"]),
        ("set_cod_amount", "设置到付金额。", {"waybill": _str_prop("运单号"), "amount": _num_prop("金额")}, ["waybill", "amount"]),
        ("list_depots", "列出城市网点。", {"city": _str_prop("城市")}, ["city"]),
        ("book_cold_chain", "预订冷链。", {"sku": _str_prop("货物"), "temp_c": _num_prop("温度", -30, 10)}, ["sku", "temp_c"]),
    ]
    campus = [
        ("book_classroom", "预约教室。", {"room": _str_prop("教室"), "start_iso": _str_prop("开始")}, ["room", "start_iso"]),
        ("cancel_classroom", "取消教室预约。", {"room": _str_prop("教室")}, ["room"]),
        ("get_course_grade", "查询课程成绩。", {"course": _str_prop("课程")}, ["course"]),
        ("enroll_course", "选修课程。", {"course": _str_prop("课程")}, ["course"]),
        ("drop_course", "退选课程。", {"course": _str_prop("课程")}, ["course"]),
        ("book_lab_bench", "预约实验台。", {"lab": _str_prop("实验室"), "slot": _str_prop("时段")}, ["lab", "slot"]),
        ("request_transcript", "申请成绩单。", {"copies": _int_prop("份数", 1, 5)}, ["copies"]),
        ("pay_tuition", "缴纳学费。", {"term": _str_prop("学期"), "amount": _num_prop("金额")}, ["term", "amount"]),
        ("reserve_library_room", "预约研讨室。", {"room": _str_prop("房间"), "hours": _int_prop("小时", 1, 4)}, ["room", "hours"]),
        ("report_facility", "报修校园设施。", {"place": _str_prop("地点"), "issue": _str_prop("问题")}, ["place", "issue"]),
        ("get_shuttle_time", "查询班车时刻。", {"route": _str_prop("线路")}, ["route"]),
        ("apply_leave", "提交请假。", {"days": _int_prop("天数", 1, 30), "reason": _str_prop("原因")}, ["days"]),
        ("book_printer", "预约打印机。", {"station": _str_prop("站点")}, ["station"]),
        ("open_dorm_gate", "开启宿舍门禁。", {"dorm": _str_prop("宿舍")}, ["dorm"]),
        ("set_curfew_exception", "申请晚归。", {"date": _str_prop("日期")}, ["date"]),
        ("list_lost_found", "查询失物招领。", {"item": _str_prop("物品")}, ["item"]),
    ]
    lab = [
        ("start_incubator", "启动培养箱。", {"chamber": _str_prop("箱号"), "temp_c": _num_prop("温度", 20, 40)}, ["chamber", "temp_c"]),
        ("stop_centrifuge", "停止离心机。", {"unit": _str_prop("设备号")}, ["unit"]),
        ("log_sample", "登记样本。", {"sample_id": _str_prop("样本号"), "type": _str_prop("类型")}, ["sample_id"]),
        ("move_sample", "转移样本。", {"sample_id": _str_prop("样本号"), "dest": _str_prop("位置")}, ["sample_id", "dest"]),
        ("calibrate_ph", "校准 pH 计。", {"probe": _str_prop("探头")}, ["probe"]),
        ("set_fumehood", "设置通风橱风速。", {"hood": _str_prop("橱号"), "level": _int_prop("档位", 1, 3)}, ["hood", "level"]),
        ("book_autoclave", "预约灭菌锅。", {"slot": _str_prop("时段")}, ["slot"]),
        ("report_spill", "报告洒漏。", {"place": _str_prop("地点")}, ["place"]),
        ("order_reagent", "申购试剂。", {"name": _str_prop("试剂"), "qty": _int_prop("数量", 1, 100)}, ["name", "qty"]),
        ("get_freezer_alarm", "查询超低温报警。", {"unit": _str_prop("设备号")}, ["unit"]),
        ("unlock_cabinet", "解锁试剂柜。", {"cabinet": _str_prop("柜号")}, ["cabinet"]),
        ("set_shaker_rpm", "设置摇床转速。", {"unit": _str_prop("设备号"), "rpm": _int_prop("转速", 50, 300)}, ["unit", "rpm"]),
        ("record_od", "记录 OD 值。", {"sample_id": _str_prop("样本号"), "od": _num_prop("OD")}, ["sample_id", "od"]),
        ("schedule_maintenance", "预约设备维护。", {"unit": _str_prop("设备号"), "date": _str_prop("日期")}, ["unit", "date"]),
        ("export_run_log", "导出实验记录。", {"run_id": _str_prop("批次号")}, ["run_id"]),
        ("seal_waste", "封装实验废料。", {"bin": _str_prop("桶号")}, ["bin"]),
    ]
    for fam, rows in (
        ("travel", travel),
        ("finance", finance),
        ("medical", medical),
        ("logistics", logistics),
        ("campus", campus),
        ("lab", lab),
    ):
        for name, desc, props, req in rows:
            tools.append(_tool(name, desc, props, req, fam))
    return tools


def build_universe() -> dict[str, Any]:
    tools = _load_canonical() + _synthetic()
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        by_name[str(tool["name"])] = tool
    ordered = list(by_name.values())
    if len(ordered) < 128:
        raise RuntimeError(f"universe too small: {len(ordered)}")
    families: dict[str, int] = {}
    for t in ordered:
        families[str(t["family"])] = families.get(str(t["family"]), 0) + 1
    payload = {
        "universe_id": UNIVERSE_ID,
        "n_tools": len(ordered),
        "families": families,
        "seen_families": sorted(SEEN_FAMILIES),
        "unseen_families": sorted(UNSEEN_FAMILIES),
        "similar_families": sorted(SIMILAR_FAMILIES),
        "tools": ordered,
        "slices": {
            "32": [t["name"] for t in ordered[:32]],
            "128": [t["name"] for t in ordered[:128]],
            "full": [t["name"] for t in ordered],
        },
    }
    payload["fingerprint"] = hashlib.sha256(
        dumps_canonical({"names": [t["name"] for t in ordered], "fps": [t["fingerprint"] for t in ordered]}).encode("utf-8")
    ).hexdigest()
    return payload


def slice_catalog(universe: dict[str, Any], size: str | int) -> list[dict[str, Any]]:
    key = str(size)
    names = universe["slices"].get(key) or universe["slices"]["full"]
    by_name = {t["name"]: t for t in universe["tools"]}
    return [by_name[n] for n in names if n in by_name]


def dump_universe(path: Path) -> dict[str, Any]:
    payload = build_universe()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    uni = build_universe()
    print(json.dumps({"n": uni["n_tools"], "families": uni["families"], "fp": uni["fingerprint"]}, ensure_ascii=False))
