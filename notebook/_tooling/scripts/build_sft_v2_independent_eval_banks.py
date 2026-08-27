#!/usr/bin/env python3
"""Independently generate and lock sft-v2 Retrieval / Full-call / MW eval banks.

Gold comes from schema compilers only. Queries use eval-only lexicons so they
are not a split of the teacher/train distribution.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from repo_paths import (
    BANK_MEI_MW_DISPOSITION_V2_DEV,
    BANK_MEI_MW_DISPOSITION_V2_TEST,
    BANK_MEI_RETRIEVAL_V2_DEV,
    BANK_MEI_RETRIEVAL_V2_TEST,
    BANK_MEI_TOOLCALL_V2_DEV,
    BANK_MEI_TOOLCALL_V2_TEST,
    EVAL_BANKS_ROOT,
    ROOT,
    SFT_V2_EVAL_LOCK_DIR,
)
from sft_canonical_lib import (
    RETRIEVAL_EXTRA_TOOLS,
    catalog_tools,
    compact_tools,
    compile_fullcall_row,
    compile_mw_row,
    compile_retrieval_row,
    dump_jsonl,
    sha256_text,
)
from sft_v2_baseline_lib import (
    EVAL_CITIES,
    EVAL_DOORS,
    EVAL_ITEMS,
    EVAL_NO_MATCH,
    EVAL_PLACES,
    EVAL_ROOMS,
    EVAL_TIMES,
    EVAL_TITLES,
    LOCK_VERSION,
    REASON_CODES_16,
    blocked_queries,
    dump_json,
    lock_payload,
    query_rejected,
    rel,
)
from park_toolcall_lib import PERM, default_facts, facts_text, park_tools

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20260827
WAVE1_RET_PER_FAMILY = 16
WAVE1_FC_PER_SLICE = 12
WAVE1_MW_PER_CLASS = 5


def _cid(prefix: str, query: str, extra: str = "") -> str:
    return prefix + sha256_text(LOCK_VERSION + "\n" + query + "\n" + extra)[:16]


def _accept(query: str, *, blocked: set[str], seen: set[str], kept_tri: list[set[str]]) -> bool:
    if query_rejected(query, blocked=blocked):
        return False
    if query in seen:
        return False
    return True


def _commit(query: str, *, blocked: set[str], seen: set[str], kept_tri: list[set[str]]) -> None:
    from park_toolcall_lib import char_trigrams

    seen.add(query)
    kept_tri.append(char_trigrams(query))
    blocked.add(query)


def split_even(rows: list[dict], *, per_split: int, key: str) -> tuple[list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "na")].append(row)
    dev: list[dict] = []
    test: list[dict] = []
    for label, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: str(r.get("item_id") or r.get("sample_id") or r.get("query")))
        if len(bucket) < per_split * 2:
            raise RuntimeError(f"{key}={label} only {len(bucket)}, need {per_split * 2}")
        for row in bucket[:per_split]:
            row["split"] = "dev"
            dev.append(row)
        for row in bucket[per_split : per_split * 2]:
            row["split"] = "test"
            test.append(row)
    return dev, test


def build_retrieval(blocked: set[str]) -> tuple[list[dict], list[dict]]:
    seen_home = compact_tools(catalog_tools(["needle-home-v0"], extra=RETRIEVAL_EXTRA_TOOLS))
    office = compact_tools(catalog_tools(["mei-office-v0"]))
    vrm = compact_tools(catalog_tools(["needle-vrm-agent-v0"]))
    retail = compact_tools(catalog_tools(["mei-retail-v0"]))
    invoice = compact_tools(catalog_tools(["needle-invoice-v0"]))
    merged = compact_tools(
        catalog_tools(["needle-home-v0", "mei-office-v0", "needle-vrm-agent-v0"], extra=RETRIEVAL_EXTRA_TOOLS)
    )
    specs = [
        ("seen_schema", "get_weather", seen_home, "needle-home-v0", ["get_weather_status", "set_lights"], "positive"),
        ("seen_schema", "set_lights", seen_home, "needle-home-v0", ["set_switch", "get_weather"], "positive"),
        ("seen_schema", "create_event", office, "mei-office-v0", ["lookup_price", "set_volume"], "positive"),
        ("seen_schema", "open_door", vrm, "needle-vrm-agent-v0", ["close_door", "go_to"], "positive"),
        ("unseen_schema", "book_table", retail, "mei-retail-v0", ["charge_card", "print_label"], "positive"),
        ("unseen_schema", "charge_card", retail, "mei-retail-v0", ["book_table", "print_label"], "positive"),
        ("unseen_schema", "print_label", retail, "mei-retail-v0", ["book_table", "charge_card"], "positive"),
        ("unseen_schema", "invoice", invoice, "needle-invoice-v0", [], "positive"),
        ("similar_name", "get_weather_status", merged, "needle-home-v0", ["get_weather", "lookup_price"], "hard"),
        ("hard_negative", "set_switch", merged, "needle-vrm-agent-v0", ["set_lights", "open_door"], "hard"),
        ("hard_negative", "set_lights", merged, "needle-home-v0", ["set_switch", "get_weather"], "hard"),
        ("no_match", None, seen_home, "needle-home-v0", ["get_weather", "set_lights"], "no_match"),
    ]
    frames = {
        "get_weather": [
            "劳驾确认一下{city}封存日后天闷不闷热",
            "{city}展陈区后天会不会闷热成那样帮忙看气象",
            "盘点间隙查下{city}有没有降水预警",
        ],
        "set_lights": [
            "把{room}展陈灯调到四成亮",
            "请把{room}照明压暗两档",
            "{room}的灯再亮一点但别刺眼",
        ],
        "create_event": [
            "在日历里建一个叫{title}的档期",
            "给{title}加一条不公开的日程",
        ],
        "open_door": [
            "请开启{door}",
            "把{door}从内侧打开",
        ],
        "book_table": [
            "订一张{city}展厅的四人桌",
            "帮展厅订室内四人位",
        ],
        "charge_card": [
            "刷卡收{city}巡楼费二百",
            "把档案押金用卡扣二百",
        ],
        "print_label": [
            "打印 {sku} 的冷链条码",
            "出一张 {sku} 封箱标签",
        ],
        "invoice": [
            "从盘点纪要里抽出供应商和金额",
            "把封存清单做成进项识别",
        ],
        "get_weather_status": [
            "请只检查{city}气象接口通不通，不要报气温",
            "{city}{room}的天气服务健康检查，不是查会不会下雨",
            "盘点间隙 ping 一下{city}天气网关，别给预报",
        ],
        "set_switch": [
            "把{room}灯开关扳到通电",
            "请拨{room}照明开关，不是调亮度",
        ],
        None: [f"{{city}}{{room}}里，{stem}" for stem in EVAL_NO_MATCH],
    }
    by_family: dict[str, list[tuple]] = defaultdict(list)
    for spec in specs:
        by_family[spec[0]].append(spec)
    seen: set[str] = set()
    kept_tri: list[set[str]] = []
    compiled: list[dict] = []
    want = {
        "seen_schema": 420,
        "unseen_schema": 420,
        "similar_name": 420,
        "hard_negative": 420,
        "no_match": 420,
    }
    for family, need in want.items():
        got = 0
        n = 0
        family_specs = by_family[family]
        while got < need and n < need * 80:
            gold_spec = family_specs[n % len(family_specs)]
            _fam, gold, catalog, toolset_id, negs, kind = gold_spec
            city = EVAL_CITIES[n % len(EVAL_CITIES)]
            room = EVAL_ROOMS[(n // len(EVAL_CITIES)) % len(EVAL_ROOMS)]
            door = EVAL_DOORS[(n // (len(EVAL_CITIES) * len(EVAL_ROOMS))) % len(EVAL_DOORS)]
            title = EVAL_TITLES[(n // 5) % len(EVAL_TITLES)]
            sku = EVAL_ITEMS[n % len(EVAL_ITEMS)] + f"-{city}"
            time = EVAL_TIMES[(n // 7) % len(EVAL_TIMES)]
            cands = frames.get(gold, frames[None])
            raw = cands[n % len(cands)]
            query = raw.format(city=city, room=room, door=door, title=title, sku=sku)
            query = f"{time}于{city}{room}靠近{door}，{query}"
            n += 1
            if not _accept(query, blocked=blocked, seen=seen, kept_tri=kept_tri):
                continue
            slots = {}
            if "{city}" in raw:
                slots["city"] = city
            if "{room}" in raw:
                slots["room"] = room
            if "{door}" in raw:
                slots["door"] = door
            if "{title}" in raw:
                slots["title"] = title
            case = {
                "case_id": _cid("EVLRET-", query, str(gold)),
                "query": query,
                "gold_tool": gold,
                "hard_negatives": list(negs),
                "catalog_tools": catalog,
                "toolset_id": toolset_id,
                "family": family,
                "kind": kind,
                "zh_slots": slots,
                "cf_group": "EVALCFG-" + sha256_text(query)[:12],
                "seen_schema": family not in {"unseen_schema", "no_match"},
                "source_role": "schema-program",
            }
            try:
                row = compile_retrieval_row(case, query, teacher_model="eval-lock")
            except ValueError:
                continue
            _commit(query, blocked=blocked, seen=seen, kept_tri=kept_tri)
            row["item_id"] = "mei-retrieval-v2-lock-" + row["sample_id"]
            row["lock_version"] = LOCK_VERSION
            row["gold_origin"] = "schema-compiler"
            compiled.append(row)
            got += 1
        if got < 400:
            raise RuntimeError(f"retrieval family {family} only {got}")
    return split_even(compiled, per_split=200, key="family")


def _fullcall_item(slice_id: str, place: str, city: str, time: str) -> tuple[str, str | None, dict, dict, str]:
    if slice_id == "execute_l0_copy":
        return (
            f"把{place}展陈灯点亮",
            "control_room_devices",
            {"light_on": True},
            default_facts(light_on=False),
            "identity",
        )
    if slice_id == "execute_l1_normalize":
        return (
            f"把{place}制冷调到廿六度",
            "control_room_devices",
            {"ac_power": True, "ac_target_c": 26},
            default_facts(ac_power=False),
            "zh_number",
        )
    if slice_id == "execute_l3_derived":
        return (
            f"让{place}再冷一度",
            "control_room_devices",
            {"ac_power": True, "ac_target_c": 25},
            default_facts(ac_power=True, target_c=26.0),
            "relative_-1+clamp",
        )
    if slice_id == "execute_l2_env":
        return (
            f"当前环境有变化：{place}到点无人，把照明和制冷收掉",
            "control_room_devices",
            {"light_on": False, "ac_power": False},
            default_facts(occupied=False, light_on=True, ac_power=True),
            "identity",
        )
    if slice_id == "refuse_missing":
        return (f"{place}把灯打开，具体哪一盏还没指定", None, {}, default_facts(), "identity")
    if slice_id == "refuse_fault":
        return (
            f"{place}空调报故障了还是先制冷到二十六",
            None,
            {},
            default_facts(ac_health="fault"),
            "identity",
        )
    if slice_id == "refuse_noop":
        return (f"{place}灯已经亮着，再点亮一次", None, {}, default_facts(light_on=True), "identity")
    return (
        f"{place}明明空着却说已经坐满，按满员把灯关掉",
        None,
        {},
        default_facts(occupied=False, light_on=True),
        "identity",
    )


def build_fullcall(blocked: set[str]) -> tuple[list[dict], list[dict]]:
    catalog = park_tools()
    rng = random.Random(SEED)
    slices = [
        ("execute_l0_copy", "execute", "L0"),
        ("execute_l1_normalize", "execute", "L1"),
        ("execute_l3_derived", "execute", "L3"),
        ("execute_l2_env", "execute", "L2"),
        ("refuse_missing", "missing", "L3"),
        ("refuse_fault", "fault", "L3"),
        ("refuse_noop", "noop", "L0"),
        ("refuse_conflict", "scene_conflict", "L3"),
    ]
    seen: set[str] = set()
    kept_tri: list[set[str]] = []
    compiled: list[dict] = []
    n_place, n_city, n_time = len(EVAL_PLACES), len(EVAL_CITIES), len(EVAL_TIMES)
    for slice_id, kind, layer in slices:
        got = 0
        n = 0
        while got < 320 and n < 8000:
            place = EVAL_PLACES[n % n_place]
            city = EVAL_CITIES[(n // n_place) % n_city]
            time = EVAL_TIMES[(n // (n_place * n_city)) % n_time]
            door = EVAL_DOORS[n % len(EVAL_DOORS)]
            n += 1
            core, gold_name, gold_args, facts, transform = _fullcall_item(slice_id, place, city, time)
            query = f"{time}在{city}侧靠近{door}，{core}"
            if not _accept(query, blocked=blocked, seen=seen, kept_tri=kept_tri):
                continue
            answers = [{"name": gold_name, "arguments": dict(gold_args)}] if gold_name else []
            case = {
                "case_id": _cid("EVLFC-", query, slice_id),
                "query": query,
                "toolset_id": "mei-park-room-v1",
                "catalog_tools": catalog,
                "answers": answers,
                "gold_name": gold_name,
                "gold_args": dict(gold_args or {}),
                "kind": kind,
                "family": "park_eval_lock",
                "zh_slots": {"place": place},
                "hard_negatives": ["create_or_update_reservation", "update_room_policy"],
                "cf_group": "EVALCFG-" + sha256_text(query)[:12],
                "system_facts_text": facts_text(facts),
                "permissions": list(PERM),
                "provenance_transform": transform,
                "park_layer": layer,
                "template_id": f"eval-lock-{slice_id}",
                "intent_id": slice_id,
            }
            try:
                row = compile_fullcall_row(case, query, rng=rng, teacher_model="eval-lock")
            except ValueError:
                continue
            _commit(query, blocked=blocked, seen=seen, kept_tri=kept_tri)
            row["item_id"] = "mei-toolcall-v2-lock-" + row["sample_id"]
            row["split"] = "eval"
            row["slice"] = slice_id
            row["lock_version"] = LOCK_VERSION
            row["gold_origin"] = "schema-compiler"
            compiled.append(row)
            got += 1
        if got < 300:
            raise RuntimeError(f"fullcall slice {slice_id} only {got}")
    return split_even(compiled, per_split=150, key="slice")


def _mw_core(reason: str, *, place: str, city: str, time: str, door: str, room: str) -> tuple[str, list[dict], list[str]]:
    calls: list[dict] = []
    gaps: list[str] = []
    if reason == "ready_to_execute":
        query = f"按点选把{place}照明打开"
        calls = [{"name": "set_switch", "arguments": {"place": place}}]
    elif reason == "missing_slot":
        query = f"把灯打开，没说是{place}还是{room}"
        gaps = ["room"]
    elif reason == "missing_external_fact":
        query = f"按{city}室外温度自动调{place}，但我这边没有室外读数"
        gaps = ["outdoor_temp"]
    elif reason == "missing_permission_token":
        query = f"改{place}策略，权限令牌还没下发"
        gaps = ["permission_token"]
    elif reason == "ambiguous_scope":
        query = f"{place}那边弄一下，没说灯还是制冷"
    elif reason == "mixed_intent":
        query = f"把{door}和{place}灯光当成同一个开关一起扳"
    elif reason == "deixis_unresolved":
        query = f"把刚才那个也关上，没指是{place}哪台设备"
    elif reason == "correction_incomplete":
        query = f"不是{place}，另外那个，具体哪个还没说完"
    elif reason == "authority_required":
        query = f"{place}这事得值班长批完才能动设备"
    elif reason == "safety_judgment":
        query = f"{place}有人被困在设备里，先别自动关制冷"
    elif reason == "capability_insufficient":
        query = f"把{city}整栋暖通都按{place}重规划一遍"
    elif reason == "unsupported_scope":
        query = f"帮{place}改消防主机联动逻辑"
    elif reason == "scene_conflict":
        query = f"{place}明明空着却说已经坐满，按满员执行"
    elif reason == "illegal_pair":
        query = f"用开灯接口去取消{place}预约"
    elif reason == "unknown_slot_value":
        query = f"把{place}灯光调成星云紫，这档枚举没有"
    else:
        query = f"先开{place}灯再开制冷，但第一步还没完成不许跳"
    query = f"{time}在{city}的{place}靠近{door}，{query}"
    return query, calls, gaps


def build_mw(blocked: set[str]) -> tuple[list[dict], list[dict]]:
    seen: set[str] = set()
    kept_tri: list[set[str]] = []
    compiled: list[dict] = []
    n_place, n_city, n_time = len(EVAL_PLACES), len(EVAL_CITIES), len(EVAL_TIMES)
    for reason in REASON_CODES_16:
        got = 0
        n = 0
        while got < 210 and n < 8000:
            place = EVAL_PLACES[n % n_place]
            city = EVAL_CITIES[(n // n_place) % n_city]
            time = EVAL_TIMES[(n // (n_place * n_city)) % n_time]
            door = EVAL_DOORS[n % len(EVAL_DOORS)]
            room = EVAL_ROOMS[(n // 3) % len(EVAL_ROOMS)]
            n += 1
            query, calls, gaps = _mw_core(
                reason, place=place, city=city, time=time, door=door, room=room
            )
            if not _accept(query, blocked=blocked, seen=seen, kept_tri=kept_tri):
                continue
            case = {
                "case_id": _cid("EVLMW-", query, reason),
                "query": query,
                "reason_code": reason,
                "gaps": gaps,
                "function_calls": calls,
                "family": reason,
                "kind": reason,
                "cf_group": "EVALCFG-" + sha256_text(query)[:12],
                "toolset_id": "needle-vrm-agent-v0",
                "high_risk": reason in {"safety_judgment", "illegal_pair", "scene_conflict"},
            }
            try:
                row = compile_mw_row(case, query, teacher_model="eval-lock")
            except ValueError:
                continue
            _commit(query, blocked=blocked, seen=seen, kept_tri=kept_tri)
            row["item_id"] = "mei-mw-disposition-v2-lock-" + row["sample_id"]
            row["lock_version"] = LOCK_VERSION
            row["gold_origin"] = "schema-compiler"
            compiled.append(row)
            got += 1
        if got < 200:
            raise RuntimeError(f"mw class {reason} only {got}")
    return split_even(compiled, per_split=100, key="reason_code")


def write_bank(path: Path, rows: list[dict]) -> dict:
    dump_jsonl(path, rows)
    return lock_payload(path, rows)


def wave1_ids(rows: list[dict], *, key: str, per: int) -> list[str]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "na")].append(row)
    ids: list[str] = []
    for _, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: str(r.get("item_id")))
        ids.extend(str(r["item_id"]) for r in bucket[:per])
    return ids


def write_readme(dir_path: Path, title: str, body: str) -> None:
    path = dir_path / "README.md"
    prev = path.read_text(encoding="utf-8") if path.is_file() else ""
    if LOCK_VERSION in prev:
        return
    note = f"\n\n## {LOCK_VERSION}\n\n{body}\n"
    path.write_text((prev.rstrip() + note if prev else f"# {title}\n{note}"), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if BANK_MEI_RETRIEVAL_V2_TEST.is_file() and not args.force:
        print(json.dumps({"ok": True, "skipped": "already_locked", "test": rel(BANK_MEI_RETRIEVAL_V2_TEST)}))
        return 0
    blocked = blocked_queries()
    ret_dev, ret_test = build_retrieval(set(blocked))
    blocked |= {r["query"] for r in ret_dev + ret_test}
    fc_dev, fc_test = build_fullcall(set(blocked))
    blocked |= {r["query"] for r in fc_dev + fc_test}
    mw_dev, mw_test = build_mw(set(blocked))

    banks = [
        (BANK_MEI_RETRIEVAL_V2_DEV, ret_dev),
        (BANK_MEI_RETRIEVAL_V2_TEST, ret_test),
        (BANK_MEI_TOOLCALL_V2_DEV, fc_dev),
        (BANK_MEI_TOOLCALL_V2_TEST, fc_test),
        (BANK_MEI_MW_DISPOSITION_V2_DEV, mw_dev),
        (BANK_MEI_MW_DISPOSITION_V2_TEST, mw_test),
    ]
    locks = [write_bank(path, rows) for path, rows in banks]
    wave1 = {
        "lock_version": LOCK_VERSION,
        "note": "Frozen stratified subsample of official TEST for scorecard wave1. Floors still use full TEST.",
        "retrieval": wave1_ids(ret_test, key="family", per=WAVE1_RET_PER_FAMILY),
        "fullcall": wave1_ids(fc_test, key="slice", per=WAVE1_FC_PER_SLICE),
        "mw": wave1_ids(mw_test, key="reason_code", per=WAVE1_MW_PER_CLASS),
    }
    SFT_V2_EVAL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(SFT_V2_EVAL_LOCK_DIR / "holdout-v1.lock.json", {"banks": locks, "wave1": wave1})
    dump_json(SFT_V2_EVAL_LOCK_DIR / "scorecard-wave1.ids.json", wave1)
    write_readme(
        EVAL_BANKS_ROOT / "mei-retrieval-v2",
        "mei-retrieval-v2",
        "Official independent dev/test lock. Smoke remains isolation wiring only. Park blind is regression-only.",
    )
    write_readme(
        EVAL_BANKS_ROOT / "mei-toolcall-v2",
        "mei-toolcall-v2",
        "Official independent oracle-top5 full-call lock. Gold from compiler+validator. Not a split of the 10k paid pool.",
    )
    write_readme(
        EVAL_BANKS_ROOT / "mei-mw-disposition-v2",
        "mei-mw-disposition-v2",
        "Official independent MW lock: 16 frozen reason_codes × ≥100 per split. Head target is reason_code only.",
    )
    report = {
        "ok": all(lock["n"] >= 1000 for lock in locks),
        "lock_version": LOCK_VERSION,
        "banks": locks,
        "wave1_n": {k: len(v) for k, v in wave1.items() if isinstance(v, list)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
