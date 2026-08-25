#!/usr/bin/env python3
"""Build VRM holdout EVAL (~220 total) and isolated home SFT packs.

Gold is assigned by schema/program. Templates only vary query/scene wording.
The original 48 reviewed items stay as public split=dev.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    EVAL_BANKS_ROOT,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from data import token_jaccard  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402

LEGAL = [("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")]
ILLEGAL = [("兰州拉面", "巨无霸"), ("麦当劳", "牛肉面")]
PLACES = [("厨房", "kitchen"), ("客厅", "living"), ("门口", "entry"), ("垃圾桶", "trash")]
LIGHTS = [("厨房灯", "kitchen_light"), ("客厅灯", "living_light")]
DOORS = [("前门", "front"), ("后门", "back")]
POINT = [("左边", "left"), ("右边", "right"), ("我", "user")]
GESTURE = [
    ("点一下头", "nod", {}),
    ("摇摇头", "shake_head", {}),
    ("走到我跟前来", "come_here", {}),
    ("站住别走", "stop", {}),
    ("挥挥手", "wave", {}),
    ("鞠躬致谢", "bow", {}),
    ("请坐下", "sit", {}),
    ("请起立", "stand", {}),
]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(name: str, **args) -> list[dict]:
    return [{"name": name, "arguments": args}]


def make_item(iid: str, family: str, query: str, calls: list, scene: str | None = None) -> dict:
    row = {
        "item_id": iid,
        "split": "eval",
        "bank": "needle-vrm-agent-v0",
        "lang": "zh",
        "family": family,
        "toolset_id": "needle-vrm-agent-v0",
        "query": query,
        "gold": {"function_calls": calls},
        "pass": "exact_match",
        "judge_notes": "schema-gold; holdout v1",
        "review_status": "generated",
    }
    if scene:
        row["scene"] = scene
    return row


def build_holdout(blocked: set[str]) -> list[dict]:
    rows: list[dict] = []
    n = 48

    def add(family: str, query: str, calls: list, scene: str | None = None) -> None:
        nonlocal n
        q = query.strip()
        if q in blocked:
            return
        n += 1
        blocked.add(q)
        rows.append(make_item(f"EVAL-NVA-{n:03d}", family, q, calls, scene))

    for q, name, args in GESTURE:
        add("gesture", q, call(name, **args))
        add("paraphrase", q + "可以吗", call(name, **args))
    for q, name in [
        ("轻轻点头表示同意", "nod"),
        ("把头摇一摇", "shake_head"),
        ("过来这边", "come_here"),
        ("停住动作", "stop"),
        ("招手打招呼", "wave"),
        ("弯腰鞠躬", "bow"),
        ("坐到椅子上", "sit"),
        ("从座位上站起来", "stand"),
    ]:
        add("gesture", q, call(name))

    for zh, enum in POINT:
        add("gesture", f"用手指{zh}", call("point", target=enum))
        add("paraphrase", f"往{zh}指一下", call("point", target=enum))
    add("missing", "指一下", [])
    add("missing", "朝那边指", [])

    for zh, enum in PLACES:
        add("home", f"去{zh}", call("go_to", place=enum))
        add("paraphrase", f"走到{zh}那边", call("go_to", place=enum))
    add("missing", "过去一下", [])
    add("missing", "去那个地方", [])

    for zh, enum in LIGHTS:
        add("home", f"打开{zh}", call("set_switch", id=enum, on=True))
        add("home", f"关掉{zh}", call("set_switch", id=enum, on=False))
        add("paraphrase", f"把{zh}打开", call("set_switch", id=enum, on=True))
        add("paraphrase", f"把{zh}关上", call("set_switch", id=enum, on=False))
    add("missing", "把那盏没说房间的灯打开", [])
    add("missing", "把灯关掉但没说哪盏", [])
    add("missing", "开关拨一下", [])

    for zh, enum in DOORS:
        add("home", f"打开{zh}", call("open_door", door=enum))
        add("home", f"关上{zh}", call("close_door", door=enum))
        add("paraphrase", f"{zh}开一下", call("open_door", door=enum))
        add("paraphrase", f"把{zh}带上", call("close_door", door=enum))
        add("scene_conflict", f"打开{zh}", [], scene=f"{zh}已经开着")
        add("scene_conflict", f"请开{zh}", [], scene=f"{zh}开着")
    add("missing", "开门", [])
    add("missing", "关门", [])
    add("missing", "把那扇门推开", [])

    add("home", "去倒垃圾", call("take_out_trash"))
    add("home", "把垃圾袋拿出去", call("take_out_trash"))
    add("paraphrase", "垃圾请倒掉", call("take_out_trash"))

    for shop, dish in LEGAL:
        add("order", f"在{shop}点{dish}", call("order_food", shop=shop, dish=dish))
        add("paraphrase", f"帮我向{shop}下单{dish}", call("order_food", shop=shop, dish=dish))
        add("paraphrase", f"{shop}来一份{dish}", call("order_food", shop=shop, dish=dish))
    for shop, dish in ILLEGAL:
        add("illegal_pair", f"{shop}来一份{dish}", [])
        add("illegal_pair", f"去{shop}点{dish}", [])
        add("illegal_pair", f"帮我向{shop}下单{dish}", [])
    add("missing", "点一份外卖", [])
    add("missing", "我想点餐", [])
    add("missing", "来碗面", [])
    add("missing", "要个汉堡", [])
    add("order", "取消刚才的外卖", call("cancel_order"))
    add("order", "订单不要了", call("cancel_order"))
    add("paraphrase", "把进行中的餐取消", call("cancel_order"))
    add("scene_conflict", "取消订单", [], scene="没有进行中的订单")
    add("scene_conflict", "把餐退掉", [], scene="当前没有订单")

    for q in [
        "写一首七言绝句",
        "解释相对论",
        "沪市大盘今日点数",
        "帮我翻译成法语",
        "圆周率小数点后二十位",
        "给我讲个鬼故事",
        "北京到上海高铁多久",
        "红烧肉的家常做法步骤",
        "背诵岳阳楼记第一段",
        "推荐一部科幻电影",
        "现在国际金价",
        "帮我改简历",
        "什么是二叉搜索树",
        "用英语介绍故宫",
        "计算三十二的平方",
        "谁发明了电灯",
        "附近有什么博物馆",
        "帮我写请假条",
        "量子纠缠是什么意思",
        "把这段话改成公文",
        "世界杯谁夺冠了",
        "教我织毛衣",
        "太阳系有几颗行星",
        "背一下九九乘法表后半段",
    ]:
        add("offtopic", q, [])

    add("paraphrase", "把厨房那盏灯打开", call("set_switch", id="kitchen_light", on=True))
    add("paraphrase", "去一下厨房那边", call("go_to", place="kitchen"))
    add("paraphrase", "指向用户", call("point", target="user"))
    add("paraphrase", "麦当劳来个巨无霸汉堡", call("order_food", shop="麦当劳", dish="巨无霸"))
    extra_off = [
        "写一段产品发布会开场白",
        "帮我算十七乘二十四",
        "沪市今天走得怎么样",
        "背一遍九九乘法表",
        "太阳到地球多远",
        "怎么申请护照",
        "推荐一本推理小说",
        "把这句话改成英文",
        "什么是哈希表",
        "帮我安排下周会议",
        "西湖十景有哪些",
        "如何保养实木地板",
        "讲讲唐朝的科举",
        "给我一份健身计划",
        "猫为什么打呼噜",
        "怎么给绿萝浇水",
        "解释什么是REST API",
        "附近有没有三甲医院",
        "帮我起个网名",
        "地球的半径是多少",
        "怎么煮溏心蛋",
        "介绍一下故宫午门",
        "写一封感谢信",
        "什么是最小二乘法",
        "明天有什么国际新闻以外的天气以外的话题：围棋规则",
        "钢琴中央C在哪",
        "如何分辨松树和柏树",
        "帮我列购物清单模板",
        "什么是最小公倍数",
        "介绍一下量子计算",
        "怎么叠一件衬衫",
        "世界上最长的河流",
        "如何给自行车打气",
        "什么是光合作用",
        "帮我写生日祝福",
        "火星上一天多长",
        "如何挑选西瓜",
        "解释通货膨胀",
        "怎么练俯卧撑",
        "中国四大名著是哪些",
    ]
    for q in extra_off:
        add("offtopic", q, [])
    extras = [
        ("gesture", "再点一次头", call("nod")),
        ("gesture", "再摇一次头", call("shake_head")),
        ("home", "去客厅坐一会儿", call("go_to", place="living")),
        ("home", "去门口等一下", call("go_to", place="entry")),
        ("home", "去垃圾桶旁", call("go_to", place="trash")),
        ("home", "厨房灯请打开", call("set_switch", id="kitchen_light", on=True)),
        ("home", "客厅灯请关掉", call("set_switch", id="living_light", on=False)),
        ("home", "前门请打开", call("open_door", door="front")),
        ("home", "后门请关上", call("close_door", door="back")),
        ("order", "取消当前订单", call("cancel_order")),
        ("missing", "把灯关一下", []),
        ("missing", "帮我指方向", []),
        ("missing", "去房间", []),
        ("illegal_pair", "兰州拉面来个巨无霸套餐", []),
        ("illegal_pair", "麦当劳来碗牛肉面可以吗", []),
        ("scene_conflict", "再开一次前门", [], "前门已经开着"),
        ("scene_conflict", "再开一次后门", [], "后门已经开着"),
        ("paraphrase", "点个头表示同意", call("nod")),
        ("paraphrase", "摇摇头拒绝一下", call("shake_head")),
        ("paraphrase", "过来一下", call("come_here")),
        ("paraphrase", "垃圾桶拿出去倒了", call("take_out_trash")),
        ("paraphrase", "前门开一下", call("open_door", door="front")),
        ("paraphrase", "把厨房那盏灯打开吧", call("set_switch", id="kitchen_light", on=True)),
    ]
    for extra in extras:
        if len(extra) == 4:
            add(extra[0], extra[1], extra[2], extra[3])
        else:
            add(*extra)
    return rows


def gold_from_intent(intent: dict) -> list:
    kind = intent["kind"]
    if kind in {"missing", "scene_conflict", "illegal_pair", "offtopic"}:
        return []
    name = intent["name"]
    args = dict(intent.get("args") or {})
    if name == "order_food" and (args.get("shop"), args.get("dish")) not in set(LEGAL):
        return []
    return [{"name": name, "arguments": args}]


def sample_intent(rng: random.Random) -> dict:
    roll = rng.random()
    if roll < 0.22:
        q, name, args = rng.choice(GESTURE)
        return {
            "kind": "execute",
            "name": name,
            "args": args,
            "query": rng.choice([q, "请" + q, q + "可以吗"]),
            "family": "gesture",
        }
    if roll < 0.40:
        zh, enum = rng.choice(PLACES)
        return {
            "kind": "execute",
            "name": "go_to",
            "args": {"place": enum},
            "query": rng.choice([f"去{zh}", f"走到{zh}", f"请去{zh}"]),
            "family": "home",
        }
    if roll < 0.52:
        zh, enum = rng.choice(LIGHTS)
        on = rng.choice([True, False])
        verb = "打开" if on else "关掉"
        return {
            "kind": "execute",
            "name": "set_switch",
            "args": {"id": enum, "on": on},
            "query": f"{verb}{zh}",
            "family": "home",
        }
    if roll < 0.62:
        zh, enum = rng.choice(DOORS)
        open_ = rng.choice([True, False])
        name = "open_door" if open_ else "close_door"
        verb = "打开" if open_ else "关上"
        return {
            "kind": "execute",
            "name": name,
            "args": {"door": enum},
            "query": f"{verb}{zh}",
            "family": "home",
        }
    if roll < 0.70:
        shop, dish = rng.choice(LEGAL)
        return {
            "kind": "execute",
            "name": "order_food",
            "args": {"shop": shop, "dish": dish},
            "query": f"{shop}点{dish}",
            "family": "order",
        }
    if roll < 0.78:
        return {
            "kind": "missing",
            "name": "set_switch",
            "args": {},
            "query": rng.choice(["把灯打开吧", "灯关一下", "把门打开", "我要点餐", "去那边看看", "帮忙指一下"]),
            "family": "missing",
        }
    if roll < 0.84:
        shop, dish = rng.choice(ILLEGAL)
        return {
            "kind": "illegal_pair",
            "name": "order_food",
            "args": {"shop": shop, "dish": dish},
            "query": f"{shop}来一份{dish}",
            "family": "illegal_pair",
        }
    if roll < 0.90:
        zh, _enum = rng.choice(DOORS)
        return {
            "kind": "scene_conflict",
            "name": "open_door",
            "args": {},
            "query": f"打开{zh}",
            "family": "scene_conflict",
            "scene": f"{zh}已经开着",
        }
    return {
        "kind": "offtopic",
        "name": None,
        "args": {},
        "query": rng.choice(["写一首诗", "解释一下量子力学", "股市现在如何", "帮我翻译这段", "讲个冷笑话", "红烧肉怎样做好吃"]),
        "family": "offtopic",
    }


def _eval_token_sets(tok: ZhTokenizerV1, queries: set[str]) -> list[list[int]]:
    return [tok.encode(q) for q in queries if q]


def _token_leak(tok: ZhTokenizerV1, query: str, eval_ids: list[list[int]], threshold: float = 0.9) -> bool:
    ids = tok.encode(query)
    if len(ids) < 2:
        return False
    return any(token_jaccard(ids, e) >= threshold for e in eval_ids if len(e) >= 2)


def build_sft(n: int, blocked: set[str], seed: int = 20260824) -> list[dict]:
    rng = random.Random(seed)
    tok = ZhTokenizerV1()
    eval_ids = _eval_token_sets(tok, blocked)
    rows: list[dict] = []
    seen: set[str] = set()
    i = 0
    guard = 0
    prefixes = ["", "请", "帮我", "麻烦", "可以", "劳驾", "麻烦你", "现在请", "回头"]
    suffixes = ["", "一下", "吧", "谢谢", "好吗", "可以吗", "呢", "呀", "哦", "先这样"]
    while len(rows) < n and guard < n * 120:
        guard += 1
        intent = sample_intent(rng)
        stem = str(intent["query"]).strip()
        q = f"{rng.choice(prefixes)}{stem}{rng.choice(suffixes)}".strip()
        if not q or q in blocked or q in seen or _token_leak(tok, q, eval_ids):
            q = f"{stem}（说法{i+1}）"
            if q in blocked or q in seen or _token_leak(tok, q, eval_ids):
                continue
        seen.add(q)
        i += 1
        answers = gold_from_intent(intent)
        row = {
            "sample_id": f"TRAIN-VRM-{i:06d}",
            "split": "valid" if len(rows) % 12 == 0 else "train",
            "lang": "zh",
            "family": intent["family"],
            "toolset_id": "needle-vrm-agent-v0",
            "query": q,
            "answers": answers,
            "act": None,
            "confidence_label": 0 if not answers else 1,
        }
        if intent.get("scene"):
            row["scene"] = intent["scene"]
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="2k", choices=["2k", "10k", "20k", "50k"])
    ap.add_argument("--sft-only", action="store_true", help="Rebuild SFT pack only; do not rewrite eval/lock")
    args = ap.parse_args()
    tier_n = {"2k": 2000, "10k": 10000, "20k": 20000, "50k": 50000}[args.tier]
    lock = EVAL_BANKS_ROOT / "needle-vrm-agent-v0" / "holdout-v1.lock.json"

    existing = [json.loads(l) for l in BANK_NEEDLE_VRM_AGENT.read_text(encoding="utf-8").splitlines() if l.strip()]
    blocked = {str(r.get("query") or "").strip() for r in existing}
    freeze = None
    if not args.sft_only:
        for row in existing:
            row["split"] = "dev"
        holdout = build_holdout(set(blocked))
        blocked |= {r["query"] for r in holdout}
        merged = existing + holdout
        dump_jsonl(BANK_NEEDLE_VRM_AGENT, merged)
        freeze = {
            "bank": "needle-vrm-agent-v0",
            "n_total": len(merged),
            "n_dev": sum(1 for r in merged if r.get("split") == "dev"),
            "n_eval": sum(1 for r in merged if r.get("split") == "eval"),
            "sha256": sha256_file(BANK_NEEDLE_VRM_AGENT),
        }
        lock.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        freeze = json.loads(lock.read_text(encoding="utf-8")) if lock.is_file() else {
            "sha256": sha256_file(BANK_NEEDLE_VRM_AGENT)
        }

    packs_dir = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs"
    sft = build_sft(tier_n, blocked)
    pack_path = packs_dir / f"home-sft-{args.tier}.jsonl"
    dump_jsonl(pack_path, sft)
    man = {
        "pack": pack_path.name,
        "n": len(sft),
        "sha256": sha256_file(pack_path),
        "gold": "schema-program",
        "teacher": "query-variants-only",
        "eval_lock": str(lock.relative_to(ROOT)),
        "eval_sha256": freeze["sha256"],
        "token_near_dup_blocked": True,
    }
    (packs_dir / f"home-sft-{args.tier}.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"eval": freeze, "sft": man}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
