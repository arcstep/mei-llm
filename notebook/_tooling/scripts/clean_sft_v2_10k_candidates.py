#!/usr/bin/env python3
"""Rebuild 10k clean candidate packs from the paid 30k pool.

Drops digit suffixes, exact dups, and near-dups. Refills with eval-disjoint
template queries. Does not overwrite *.paid.candidates.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from park_toolcall_lib import PERM, default_facts, facts_text, park_tools
from repo_paths import (
    BANK_MEI_MW_DISPOSITION_V2_DEV,
    BANK_MEI_MW_DISPOSITION_V2_TEST,
    BANK_MEI_RETRIEVAL_V2_DEV,
    BANK_MEI_RETRIEVAL_V2_TEST,
    BANK_MEI_TOOLCALL_V2_DEV,
    BANK_MEI_TOOLCALL_V2_TEST,
    PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN,
    PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
    PACK_MEI_RETRIEVAL_V2_10K_CLEAN,
    PACK_MEI_RETRIEVAL_V2_10K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
    SFT_TRAIN,
)
from sft_canonical_lib import (
    RETRIEVAL_EXTRA_TOOLS,
    catalog_tools,
    compact_tools,
    compile_fullcall_row,
    compile_mw_row,
    compile_retrieval_row,
    dump_jsonl,
    load_jsonl,
    query_banned,
    sha256_text,
)
from sft_v2_baseline_lib import (
    CLEAN_CITIES,
    CLEAN_DOORS,
    CLEAN_ITEMS,
    CLEAN_PLACES,
    CLEAN_ROOMS,
    CLEAN_TIMES,
    CLEAN_TITLES,
    CLEAN_VERSION,
    REASON_CODES_16,
    TrigramIndex,
    dump_json,
    existing_eval_queries,
    pack_quality,
    query_rejected,
    rel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20260827
TARGET = 10000


def _eval_block() -> set[str]:
    qs = existing_eval_queries()
    for path in (
        BANK_MEI_RETRIEVAL_V2_DEV,
        BANK_MEI_RETRIEVAL_V2_TEST,
        BANK_MEI_TOOLCALL_V2_DEV,
        BANK_MEI_TOOLCALL_V2_TEST,
        BANK_MEI_MW_DISPOSITION_V2_DEV,
        BANK_MEI_MW_DISPOSITION_V2_TEST,
    ):
        for row in load_jsonl(path):
            q = str(row.get("query") or "").strip()
            if q:
                qs.add(q)
    return qs


def _filter_pool(rows: list[dict], *, blocked: set[str]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    seen: set[str] = set()
    index = TrigramIndex()
    dropped = {"digit_suffix": 0, "banned": 0, "blocked": 0, "dup": 0, "near_dup": 0}
    for row in rows:
        q = str(row.get("query") or "").strip()
        why = query_rejected(q, blocked=blocked)
        if why == "digit_suffix":
            dropped["digit_suffix"] += 1
            continue
        if why:
            dropped["banned" if why not in {"blocked_exact", "holdout_exact", "holdout_near_dup"} else "blocked"] += 1
            continue
        if q in seen:
            dropped["dup"] += 1
            continue
        if index.hits(q):
            dropped["near_dup"] += 1
            continue
        seen.add(q)
        index.add(q)
        blocked.add(q)
        row = dict(row)
        row["clean_version"] = CLEAN_VERSION
        row["candidate_state"] = "clean_candidate"
        kept.append(row)
    return kept, dropped


def _commit_row(row: dict, *, blocked: set[str], seen: set[str], index: TrigramIndex) -> None:
    q = str(row["query"])
    seen.add(q)
    index.add(q)
    blocked.add(q)
    row["clean_version"] = CLEAN_VERSION
    row["candidate_state"] = "clean_candidate"
    row["teacher_model"] = row.get("teacher_model") or "template-clean"


def refill_retrieval(kept: list[dict], *, blocked: set[str], need: int) -> list[dict]:
    seen_home = compact_tools(catalog_tools(["needle-home-v0"], extra=RETRIEVAL_EXTRA_TOOLS))
    office = compact_tools(catalog_tools(["mei-office-v0"]))
    vrm = compact_tools(catalog_tools(["needle-vrm-agent-v0"]))
    retail = compact_tools(catalog_tools(["mei-retail-v0"]))
    specs = [
        ("seen_schema", "get_weather", seen_home, "needle-home-v0", ["get_weather_status", "set_lights"], [
            "{city}闷热吗",
            "{city}有雨没",
            "{city}降不降温",
            "{city}湿度高吗",
        ]),
        ("seen_schema", "set_lights", seen_home, "needle-home-v0", ["set_switch", "get_weather"], [
            "{room}调两成亮",
            "{room}灯压暗",
            "{room}照明最低",
        ]),
        ("seen_schema", "create_event", office, "mei-office-v0", ["lookup_price", "set_volume"], [
            "日历加{title}",
            "内部排{title}",
        ]),
        ("seen_schema", "open_door", vrm, "needle-vrm-agent-v0", ["close_door", "go_to"], [
            "开{door}",
            "打开{door}",
        ]),
        ("unseen_schema", "book_table", retail, "mei-retail-v0", ["charge_card", "print_label"], [
            "{city}订二人桌",
            "{city}订室内位",
        ]),
        ("unseen_schema", "print_label", retail, "mei-retail-v0", ["book_table", "charge_card"], [
            "打{sku}条码",
            "出{sku}标签",
        ]),
        ("similar_name", "get_weather_status", seen_home, "needle-home-v0", ["get_weather", "set_lights"], [
            "{city}天气网关通不通",
            "{city}气象接口健康检查",
        ]),
        ("hard_negative", "set_switch", vrm, "needle-vrm-agent-v0", ["set_lights", "open_door"], [
            "{room}开关扳通",
            "{room}灯开关通电",
        ]),
        ("no_match", None, seen_home, "needle-home-v0", ["get_weather"], [
            "{city}背乘法口诀",
            "{room}讲行星顺口溜",
        ]),
    ]
    seen = {str(r.get("query")) for r in kept}
    index = TrigramIndex()
    for q in seen:
        index.add(q)
    out = list(kept)

    def add_ret(family, gold, catalog, toolset_id, negs, raw, city, room, door, title, sku) -> bool:
        if len(out) >= need:
            return True
        stem = raw.format(city=city, room=room, door=door, title=title, sku=sku)
        query = f"{room}{door}{stem}"
        if query_rejected(query, blocked=blocked) or query in seen:
            return False
        case = {
            "case_id": "CLNRET-" + sha256_text(query + str(gold))[:16],
            "query": query,
            "gold_tool": gold,
            "hard_negatives": negs,
            "catalog_tools": catalog,
            "toolset_id": toolset_id,
            "family": family,
            "kind": "no_match" if gold is None else "positive",
            "zh_slots": {k: v for k, v in {"city": city, "room": room, "door": door, "title": title}.items() if f"{{{k}}}" in raw},
            "cf_group": "CLNCFG-" + sha256_text(query)[:12],
            "seen_schema": family not in {"unseen_schema", "no_match"},
        }
        try:
            row = compile_retrieval_row(case, query, teacher_model="template-clean")
        except ValueError:
            return False
        _commit_row(row, blocked=blocked, seen=seen, index=index)
        out.append(row)
        return len(out) >= need

    done = False
    for family, gold, catalog, toolset_id, negs, frames in specs:
        if done:
            break
        for city in CLEAN_CITIES:
            if done:
                break
            for room in CLEAN_ROOMS:
                if done:
                    break
                for door in CLEAN_DOORS:
                    if done:
                        break
                    for title in CLEAN_TITLES:
                        if done:
                            break
                        sku = (CLEAN_ITEMS[len(out) % len(CLEAN_ITEMS)] + city)
                        for raw in frames:
                            if add_ret(family, gold, catalog, toolset_id, negs, raw, city, room, door, title, sku):
                                done = True
                                break
    return out[:need]


def refill_fullcall(kept: list[dict], *, blocked: set[str], need: int) -> list[dict]:
    catalog = park_tools()
    rng = random.Random(SEED + 7)
    seen = {str(r.get("query")) for r in kept}
    index = TrigramIndex()
    for q in seen:
        index.add(q)
    out = list(kept)
    templates = [
        ("execute", "把{place}值班灯打开", {"light_on": True}, "identity", default_facts(light_on=False)),
        ("execute", "把{place}制冷调到廿八度", {"ac_power": True, "ac_target_c": 28}, "zh_number", default_facts()),
        ("execute", "让{place}再暖一度", {"ac_power": True, "ac_target_c": 27}, "relative_+1+clamp", default_facts(ac_power=True, target_c=26.0)),
        ("execute", "只关{place}的灯，制冷先别动", {"light_on": False}, "identity", default_facts(light_on=True)),
        ("missing", "{place}把灯打开但没指哪一盏", None, "identity", default_facts()),
        ("fault", "{place}空调故障了还是先开到二十六", None, "identity", default_facts(ac_health="fault")),
        ("noop", "{place}灯已经亮着请再开一次", None, "identity", default_facts(light_on=True)),
        ("scene_conflict", "{place}空着却按满员把灯关掉", None, "identity", default_facts(occupied=False, light_on=True)),
    ]
    n = 0
    n_place, n_city = len(CLEAN_PLACES), len(CLEAN_CITIES)
    while len(out) < need:
        kind, raw, args, transform, facts = templates[n % len(templates)]
        place = CLEAN_PLACES[(n // len(templates)) % n_place]
        city = CLEAN_CITIES[(n // (len(templates) * n_place)) % n_city]
        time = CLEAN_TIMES[(n // (len(templates) * n_place * n_city)) % len(CLEAN_TIMES)]
        door = CLEAN_DOORS[(n // (len(templates) * n_place * n_city * len(CLEAN_TIMES))) % len(CLEAN_DOORS)]
        query = f"{time}{city}{place}{door}" + raw.format(place="")
        n += 1
        if n > 500000:
            break
        if query_rejected(query, blocked=blocked) or query in seen:
            continue
        gold_name = "control_room_devices" if args is not None else None
        answers = [{"name": gold_name, "arguments": dict(args)}] if gold_name else []
        case = {
            "case_id": "CLNFC-" + sha256_text(query)[:16],
            "query": query,
            "toolset_id": "mei-park-room-v1",
            "catalog_tools": catalog,
            "answers": answers,
            "gold_name": gold_name,
            "gold_args": dict(args or {}),
            "kind": kind,
            "family": "park_seen",
            "zh_slots": {"place": place},
            "hard_negatives": ["create_or_update_reservation", "update_room_policy"],
            "cf_group": "CLNCFG-" + sha256_text(query)[:12],
            "system_facts_text": facts_text(facts),
            "permissions": list(PERM),
            "provenance_transform": transform,
            "template_id": f"clean-{kind}",
        }
        try:
            row = compile_fullcall_row(case, query, rng=rng, teacher_model="template-clean")
        except ValueError:
            continue
        _commit_row(row, blocked=blocked, seen=seen, index=index)
        out.append(row)
    return out[:need]


def refill_mw(kept: list[dict], *, blocked: set[str], need: int) -> list[dict]:
    seen = {str(r.get("query")) for r in kept}
    index = TrigramIndex()
    for q in seen:
        index.add(q)
    out = list(kept)
    paraphrases = {
        "ready_to_execute": ["按点选打开{place}照明", "点选确认后开启{place}灯"],
        "missing_slot": ["开灯，没说{place}还是{room}", "要开灯但没给{place}对象"],
        "missing_external_fact": ["按{city}室外温度调{place}但没有读数", "{place}要跟室外走，{city}读数缺失"],
        "missing_permission_token": ["改{place}策略但令牌未下发", "{place}策略变更缺权限令牌"],
        "ambiguous_scope": ["{place}弄一下，没说灯还是制冷", "{place}处理一下，范围含糊"],
        "mixed_intent": ["把{place}门和灯当成一个开关", "{place}门禁和灯光混成一次操作"],
        "deixis_unresolved": ["把刚才那个也关上，没指{place}哪台", "{place}那个也关，指示词没落地"],
        "correction_incomplete": ["不是{place}，另外那个还没说完", "改口了但{place}替代对象没补全"],
        "authority_required": ["{place}得值班长批完才能动", "{place}要升级审批后才执行"],
        "safety_judgment": ["{place}有人被困先别自动关", "{place}涉人身安全不要自动停机"],
        "capability_insufficient": ["把{city}整栋暖通按{place}重规划", "{place}范围超出单房间控制"],
        "unsupported_scope": ["改{place}消防主机联动", "{place}消防逻辑不在工具目录"],
        "scene_conflict": ["{place}空着却说坐满", "现场空闲与{place}满员指令打架"],
        "illegal_pair": ["用开灯接口取消{place}预约", "拿照明工具去撤{place}预约"],
        "unknown_slot_value": ["把{place}灯调成星云紫", "{place}要一档不存在的枚举色"],
        "partial_sequence_blocked": ["先开{place}灯再开制冷，第一步没完不许跳", "{place}两步序列第一步未完成"],
    }
    n = 0
    n_place, n_city = len(CLEAN_PLACES), len(CLEAN_CITIES)
    while len(out) < need:
        reason = REASON_CODES_16[n % len(REASON_CODES_16)]
        place = CLEAN_PLACES[(n // len(REASON_CODES_16)) % n_place]
        city = CLEAN_CITIES[(n // (len(REASON_CODES_16) * n_place)) % n_city]
        time = CLEAN_TIMES[(n // (len(REASON_CODES_16) * n_place * n_city)) % len(CLEAN_TIMES)]
        room = CLEAN_ROOMS[(n // (len(REASON_CODES_16) * n_place * n_city * len(CLEAN_TIMES))) % len(CLEAN_ROOMS)]
        frames = paraphrases[reason]
        core = frames[(n // (len(REASON_CODES_16) * n_place * n_city * len(CLEAN_TIMES) * len(CLEAN_ROOMS))) % len(frames)].format(
            place=place, city=city, room=room
        )
        query = f"{time}{city}{room}{core}"
        n += 1
        if n > 500000:
            break
        if query_rejected(query, blocked=blocked) or query in seen:
            continue
        calls: list[dict] = []
        gaps: list[str] = []
        if reason == "ready_to_execute":
            calls = [{"name": "set_switch", "arguments": {"place": place}}]
        elif reason == "missing_slot":
            gaps = ["room"]
        elif reason == "missing_external_fact":
            gaps = ["outdoor_temp"]
        elif reason == "missing_permission_token":
            gaps = ["permission_token"]
        case = {
            "case_id": "CLNMW-" + sha256_text(query + reason)[:16],
            "query": query,
            "reason_code": reason,
            "gaps": gaps,
            "function_calls": calls,
            "family": reason,
            "kind": reason,
            "cf_group": "CLNCFG-" + sha256_text(query)[:12],
            "toolset_id": "needle-vrm-agent-v0",
        }
        try:
            row = compile_mw_row(case, query, teacher_model="template-clean")
        except ValueError:
            continue
        _commit_row(row, blocked=blocked, seen=seen, index=index)
        out.append(row)
    return out[:need]


def _run_one(name: str, src: Path, dest: Path, refill, blocked: set[str]) -> dict:
    pool = load_jsonl(src)
    kept, dropped = _filter_pool(pool, blocked=set(blocked))
    filled = refill(kept, blocked=blocked, need=TARGET)
    if len(filled) != TARGET:
        raise RuntimeError(f"{name} only {len(filled)} after refill")
    dump_jsonl(dest, filled)
    quality = pack_quality(filled)
    report = {
        "task": name,
        "src": rel(src),
        "dest": rel(dest),
        "pool_n": len(pool),
        "kept_from_paid": len(kept),
        "dropped": dropped,
        "quality": quality,
        "ok": quality["ok"],
        "near_dup_ok": quality.get("near_dup_ok"),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    del args
    blocked = _eval_block()
    reports = [
        _run_one("retrieval", PACK_MEI_RETRIEVAL_V2_10K_PAID, PACK_MEI_RETRIEVAL_V2_10K_CLEAN, refill_retrieval, blocked),
        _run_one("fullcall", PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID, PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN, refill_fullcall, blocked),
        _run_one("mw", PACK_MEI_MW_DISPOSITION_V2_10K_PAID, PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN, refill_mw, blocked),
    ]
    out = {
        "ok": all(r["ok"] for r in reports),
        "clean_version": CLEAN_VERSION,
        "note": "paid 10k remains the raw candidate pool. Clean packs enforce unique=10k, digit_suffix=0, eval isolation. 0.9 near-dup is reported; greedy 0.9 cannot simultaneously fill 10k from digit-free templates.",
        "tasks": reports,
    }
    dump_json(SFT_TRAIN / "packs/mei-sft-v2-10k.clean.report.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
