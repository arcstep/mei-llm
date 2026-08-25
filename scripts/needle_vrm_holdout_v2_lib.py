#!/usr/bin/env python3
"""Canonical intents and schema-program gold for Needle VRM holdout v2.

Teachers may only rewrite query/scene. Gold is always gold_from_intent_v2.
Never writes v1 eval-bank or lock files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from needle_home_sft_lib import (
    ASK_RE,
    DOUBLE_QING_RE,
    EVAL_RE,
    FAKE_VARIANT_RE,
    PII_RE,
    TOOL_NAME_RE,
    TOOLSET_ID,
    load_jsonl,
    load_toolset,
    sha256_text,
    tools_of,
)
from repo_paths import (
    BANK_NEEDLE_VRM_AGENT,
    BANK_NEEDLE_VRM_AGENT_V2_RECIPE,
    SEED_NEEDLE_ZH,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

GENERATOR_VERSION = "holdout-v2-pack-v1"
PROMPT_VERSION = "holdout-v2-teacher-v1"
BANK_ID = "needle-vrm-agent-v0"

GESTURE_STEMS = {
    "nod": [
        "请轻轻颔首表示收到",
        "用颔首动作回应一声",
        "朝这边做一个同意的颔首",
        "把头微微低下再抬起表示明白",
        "给一个短促的颔首示意",
        "当场做一次表示认可的颔首",
    ],
    "shake_head": [
        "请左右摆头表示拒绝",
        "用摆头动作说不",
        "把头左右晃一晃表示不同意",
        "给一个明确的拒绝摆头",
        "当场左右摇动头部表示否定",
        "做一个不愿意的摆头动作",
    ],
    "come_here": [
        "请朝我站的位置走近",
        "往说话人这边靠过来",
        "走到发出口令的人面前",
        "向我站立处靠近几步",
        "请移动到我身旁",
        "朝发令者的方向走近来",
    ],
    "stop": [
        "请立刻止步保持原地",
        "动作全部停住进入待机",
        "不要再移动就地立定",
        "马上刹车停在当前位置",
        "停下所有走动保持静止",
        "原地立定不要继续走",
    ],
    "wave": [
        "请抬手做一次招呼手势",
        "朝外面挥动手掌致意",
        "举手晃一晃打个招呼",
        "做一个见面时的挥手动作",
        "抬起手臂左右摆动致意",
        "用手掌朝前方挥动一下",
    ],
    "bow": [
        "请弯腰做一次礼节性鞠躬",
        "上身前倾完成鞠躬礼",
        "行一个短鞠躬表示敬意",
        "低头弯腰完成一次鞠躬",
        "做一个正式的鞠躬动作",
        "当场鞠躬致意即可",
    ],
    "sit": [
        "请就近坐到座位上",
        "弯膝坐下休息片刻",
        "找位子坐稳不要站着",
        "请落座保持坐姿",
        "坐到旁边的椅子上",
        "请坐下把重心放低",
    ],
    "stand": [
        "请从坐姿改成站直",
        "离开座位站起来",
        "把身体直立起来",
        "请起身保持站立",
        "站直身体不要再坐",
        "从座位上起身站好",
    ],
}

POINT_STEMS = {
    "left": [
        "请朝左手边那个方向伸出手指",
        "把手指向左侧区域",
        "对着左边做出指向动作",
        "用食指标明左手一侧",
        "指向左侧那一块",
    ],
    "right": [
        "请朝右手边那个方向伸出手指",
        "把手指向右侧区域",
        "对着右边做出指向动作",
        "用食指标明右手一侧",
        "指向右侧那一块",
    ],
    "user": [
        "请把手指指向正在说话的我",
        "对着发口令的人做出指向",
        "用食指标明我这个人",
        "指向站在你面前的我",
        "把手指向我本人",
    ],
}

PLACE_STEMS = {
    "kitchen": [
        "请移动到做饭的厨房区域",
        "走到厨房那一侧待命",
        "去厨房所在位置站好",
        "把位置换到厨房",
        "前往厨房方向并停下",
    ],
    "living": [
        "请移动到会客的客厅区域",
        "走到客厅那一侧待命",
        "去客厅所在位置站好",
        "把位置换到客厅",
        "前往客厅方向并停下",
    ],
    "entry": [
        "请移动到进门的门口区域",
        "走到门口那一侧待命",
        "去门口所在位置站好",
        "把位置换到门口",
        "前往门口方向并停下",
    ],
    "trash": [
        "请移动到垃圾桶旁边",
        "走到垃圾桶那一侧待命",
        "去垃圾桶所在位置站好",
        "把位置换到垃圾桶旁",
        "前往垃圾桶方向并停下",
    ],
}

LIGHT_STEMS = {
    ("kitchen_light", True): [
        "请把厨房灯切换成亮着",
        "让厨房灯处于开启状态",
        "把厨房那盏灯点亮",
        "厨房灯需要变成开",
        "给厨房灯通电点亮",
    ],
    ("kitchen_light", False): [
        "请把厨房灯切换成熄灭",
        "让厨房灯处于关闭状态",
        "把厨房那盏灯熄掉",
        "厨房灯需要变成关",
        "给厨房灯断电熄灭",
    ],
    ("living_light", True): [
        "请把客厅灯切换成亮着",
        "让客厅灯处于开启状态",
        "把客厅那盏灯点亮",
        "客厅灯需要变成开",
        "给客厅灯通电点亮",
    ],
    ("living_light", False): [
        "请把客厅灯切换成熄灭",
        "让客厅灯处于关闭状态",
        "把客厅那盏灯熄掉",
        "客厅灯需要变成关",
        "给客厅灯断电熄灭",
    ],
}

DOOR_OPEN_STEMS = {
    "front": [
        "请把前门从关闭推到打开",
        "让前门处于敞开状态",
        "把前头那扇门打开",
        "前门需要变成开着",
        "推开前侧的那扇门",
    ],
    "back": [
        "请把后门从关闭推到打开",
        "让后门处于敞开状态",
        "把后头那扇门打开",
        "后门需要变成开着",
        "推开后侧的那扇门",
    ],
}

DOOR_CLOSE_STEMS = {
    "front": [
        "请把前门从打开收到关闭",
        "让前门处于关严状态",
        "把前头那扇门关上",
        "前门需要变成关着",
        "合上前侧的那扇门",
    ],
    "back": [
        "请把后门从打开收到关闭",
        "让后门处于关严状态",
        "把后头那扇门关上",
        "后门需要变成关着",
        "合上后侧的那扇门",
    ],
}

TRASH_STEMS = [
    "请把家里的垃圾带出去丢掉",
    "将垃圾袋拿到室外处理",
    "完成一次把垃圾清出家门",
    "把积着的垃圾送出门外",
    "出门把垃圾处理掉",
    "把垃圾桶里的废物带离室内",
]

CANCEL_STEMS = [
    "请撤销正在进行的那一单外卖",
    "把当前餐饮订单作废",
    "中止已经下出去的餐单",
    "不要继续那笔进行中的点餐",
    "把进行中的外卖单取消掉",
    "终止眼下这张餐饮订单",
]

MISSING_STEMS = [
    "把灯调一下但没说哪盏",
    "门帮我处理一下没说哪扇",
    "过去一趟没说去哪",
    "指一下方向但没说指向谁",
    "来一份外卖没说店和菜",
    "开关动一动没说灯名",
    "把那扇没指名的门推开",
    "想点餐可是没给店名",
    "去房间看看可没说哪个房间",
    "帮我指个方向没给左右",
    "把灯关一关却漏了房间",
    "点个汉堡没说哪家店",
    "来碗面没说哪家馆子",
    "把门带上但没说前门还是后门",
    "去那边站着没给地点名称",
    "把开关拨一下没说厨房还是客厅",
    "指一下没说左边右边还是我",
    "下单吧店名菜名都空着",
    "处理一下那扇门但没编号",
    "把灯打开到合适亮度却没灯名",
]

SCENE_SPECS = [
    ("open_door", "front", "前门此刻已经敞开", "请把前门从关闭推到打开", ["scene_flips_execute"]),
    ("open_door", "front", "前门现在是开着的", "让前门处于敞开状态", ["scene_flips_execute"]),
    ("open_door", "back", "后门此刻已经敞开", "请把后门从关闭推到打开", ["scene_flips_execute"]),
    ("open_door", "back", "后门现在是开着的", "让后门处于敞开状态", ["scene_flips_execute"]),
    ("close_door", "front", "前门此刻已经关严", "请把前门从打开收到关闭", ["scene_flips_execute"]),
    ("close_door", "front", "前门现在是关着的", "让前门处于关严状态", ["scene_flips_execute"]),
    ("close_door", "back", "后门此刻已经关严", "请把后门从打开收到关闭", ["scene_flips_execute"]),
    ("close_door", "back", "后门现在是关着的", "让后门处于关严状态", ["scene_flips_execute"]),
    ("set_switch", "kitchen_light", "厨房灯此刻已经亮着", "请把厨房灯切换成亮着", ["scene_flips_execute"]),
    ("set_switch", "living_light", "客厅灯此刻已经熄灭", "请把客厅灯切换成熄灭", ["scene_flips_execute"]),
    ("cancel_order", None, "当前没有任何进行中的餐饮订单", "请撤销正在进行的那一单外卖", ["scene_flips_execute"]),
    ("cancel_order", None, "系统里没有可撤销的餐单", "把当前餐饮订单作废", ["scene_flips_execute"]),
]

OFFTOPIC_TOPICS = [
    "景德镇青花瓷的釉色层次", "黄山迎客松的树龄争议", "二十四节气里芒种的农事",
    "苏州评弹的伴奏乐器", "敦煌壁画的矿物颜料", "茶马古道的主要驿站",
    "甲骨文里常见的祭祀字", "宋体和楷体在印刷史上的分工", "京杭大运河的水位管理",
    "围棋死活题里的金鸡独立", "京剧青衣的发声位置", "景德镇与龙泉窑的区别",
    "《九章算术》里的盈不足术", "赵州桥的敞肩拱结构", "端午节粽叶的植物种类",
    "南极磷虾的生态位置", "青藏高原冻土的季节变化", "长江江豚的栖息河段",
    "水稻旱育秧的温度窗口", "小麦赤霉病的田间识别", "柑橘溃疡病的叶片症状",
    "太极拳起势的重心移动", "八段锦第三式的呼吸节奏", "少林棍术的基本步型",
    "古琴减字谱怎么记左手", "琵琶轮指的发力位置", "笛子叠音的指法",
    "篆刻阴文和阳文的分别", "工笔花鸟的分染顺序", "青绿山水的矿物色叠加",
    "活字印刷的检字盘布局", "简牍编联的韦编怎么穿", "碑帖拓片的上墨次数",
    "二十四史里《史记》的体例", "《文心雕龙》神思篇的大意", "律诗颔联为什么要对仗",
    "北京中轴线从永定门数起", "西安城墙马面的间距规律", "平遥票号的汇兑手续",
    "泉州刺桐港的宋元贸易", "哈尼梯田的灌溉沟渠", "都江堰鱼嘴如何分流",
    "赵州桥与卢沟桥的结构差异", "应县木塔的明层暗层", "布达拉宫红宫的功能分区",
    "五台山显通寺的铜殿", "龙门石窟奉先寺的卢舍那", "云冈昙曜五窟的形制",
    "良渚玉琮的节数含义", "三星堆青铜纵目面具", "殷墟妇好墓的铜钺",
    "马王堆帛画的T形构图", "曾侯乙编钟的律制", "越王勾践剑的表面硫化",
    "司母戊鼎的铸造分范", "四羊方尊的肩部结构", "何尊里中国二字的写法",
    "贾湖骨笛的测音孔数", "曾侯乙墓冰鉴的双层设计", "秦陵铜车马的缰绳连接",
    "唐三彩骆驼俑的釉色流动", "元青花萧何月下追韩信罐", "明成化斗彩鸡缸杯",
    "清乾隆珐琅彩的轧道工艺", "顾绣的劈丝技法", "云锦妆花的金线盘织",
    "蜀锦经线起花的特点", "宋锦的几何填花", "壮锦的菱形纹样",
    "蓝印花布的刮浆防染", "蜡染冰纹怎么形成", "扎染绞缬的捆扎密度",
    "宣纸棉料和净皮的纤维差", "徽墨的烟料配比", "湖笔羊毫的锋颖",
    "端砚鱼脑冻的识别", "歙砚罗纹的成因", "澄泥砚的烧制温度",
    "紫砂朱泥的收缩率", "钧瓷窑变的铜红", "汝瓷香灰色的烧成气氛",
    "官窑开片的铁线冰裂", "哥窑金丝铁线的成因", "定窑覆烧的芒口",
    "磁州窑白地黑花的画法", "耀州窑刻花的一刀推", "龙泉梅子青的铁含量",
    "建盏兔毫的析晶", "吉州窑木叶盏的贴花", "德化白瓷的猪油白",
    "珐琅彩轧道的锦地", "粉彩洗染的玻璃白", "浅绛彩文人画入瓷",
    "五彩矾红的油料", "素三彩的低温铅釉", "法华器的立粉工艺",
]


def load_recipe(path: Path | None = None) -> dict:
    p = path or BANK_NEEDLE_VRM_AGENT_V2_RECIPE
    return json.loads(p.read_text(encoding="utf-8"))


def pair_tuples(rows: list) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in rows]


def gold_from_intent_v2(intent: dict, recipe: dict | None = None) -> list[dict]:
    rec = recipe or load_recipe()
    kind = str(intent.get("kind") or "")
    if kind in {"missing", "scene_conflict", "illegal_pair", "offtopic"}:
        return []
    if kind == "sequence":
        return [dict(c) for c in (intent.get("calls") or [])]
    name = intent.get("name")
    args = dict(intent.get("args") or {})
    if not name:
        return []
    legal = set(pair_tuples(rec["legal_food"]))
    if name == "order_food" and (args.get("shop"), args.get("dish")) not in legal:
        return []
    return [{"name": str(name), "arguments": args}]


def intent_key(intent: dict) -> str:
    payload = {
        "kind": intent.get("kind"),
        "name": intent.get("name"),
        "args": intent.get("args") or {},
        "calls": intent.get("calls") or [],
        "scene": intent.get("scene") or "",
        "family": intent.get("family"),
        "stem": intent.get("stem") or "",
        "style_tags": intent.get("style_tags") or [],
        "uid": intent.get("uid") or "",
    }
    return "INT2-" + sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))[:16]


def query_banned(text: str) -> str | None:
    if not text or not text.strip():
        return "empty"
    if FAKE_VARIANT_RE.search(text):
        return "fake_variant"
    if DOUBLE_QING_RE.search(text):
        return "double_qing"
    if ASK_RE.search(text):
        return "ask_fill"
    if EVAL_RE.search(text):
        return "eval_id"
    if PII_RE.search(text):
        return "pii"
    if TOOL_NAME_RE.search(text):
        return "tool_name_leak"
    if "�" in text:
        return "garbled"
    return None


EVAL_PREFIX = ["此刻", "当场", "这回", "本条口令：", "按现场指令", "现在这条"]
EVAL_SUFFIX = ["即可", "完成", "照做", "按此执行", "别走样"]


def _affix(rng: random.Random, stem: str, tag: str) -> tuple[str, str]:
    stem = stem.strip()
    prefix, suffix = "", ""
    if tag == "formal":
        prefix, suffix = "劳烦", "，谢谢配合"
    elif tag == "colloquial":
        suffix = "就行"
    elif tag == "command":
        prefix = rng.choice(["", "马上", "立刻"])
        suffix = rng.choice(["", "，执行"])
    elif tag == "statement":
        prefix = "我需要你"
    elif tag == "negation":
        prefix = "别理解成聊天，"
        suffix = "，只要动作"
    elif tag == "ellipsis":
        prefix, suffix = "那个", "……"
    elif tag == "inversion":
        suffix = "，先做这个"
    elif tag == "particle":
        suffix = "哦"
    elif tag == "deixis":
        prefix = "就眼前这事，"
    elif tag == "correction":
        prefix, suffix = "说清楚：", "，不是别的"
    elif tag == "redundant":
        prefix, suffix = "现在立刻当场", "一遍"
    extra_p = rng.choice(EVAL_PREFIX)
    extra_s = rng.choice(EVAL_SUFFIX)
    if rng.random() < 0.55:
        prefix = f"{extra_p}{prefix}"
    if rng.random() < 0.45:
        suffix = f"{suffix}{extra_s}"
    q = f"{prefix}{stem}{suffix}".strip()
    q = q.replace("请请", "请").replace("劳烦请", "劳烦")
    tid = f"ev2-{tag}-{sha256_text(prefix + '|' + stem[:12] + '|' + suffix)[:10]}"
    return q, tid


def slot_present(kind: str, zh: str, query: str) -> bool:
    zh = str(zh or "")
    if not zh:
        return True
    if kind == "target":
        if zh == "我":
            return any(tok in query for tok in ("我", "发口令", "说话", "本人", "这个人", "发令"))
        if zh == "左边":
            return any(tok in query for tok in ("左边", "左手", "左侧", "左面"))
        if zh == "右边":
            return any(tok in query for tok in ("右边", "右手", "右侧", "右面"))
        return zh in query
    if kind == "light":
        if zh == "厨房灯":
            return "厨房灯" in query or ("厨房" in query and "灯" in query)
        if zh == "客厅灯":
            return "客厅灯" in query or ("客厅" in query and "灯" in query)
        return zh in query
    if kind == "door":
        if zh == "前门":
            return "前门" in query or ("前" in query and "门" in query)
        if zh == "后门":
            return "后门" in query or ("后" in query and "门" in query)
        return zh in query
    return zh in query


def protected_slots_ok(intent: dict, query: str) -> bool:
    kind = intent.get("kind")
    slots = intent.get("zh_slots") or {}
    if kind in {"missing", "offtopic"}:
        return True
    if kind == "illegal_pair":
        return str(slots.get("shop") or "") in query and str(slots.get("dish") or "") in query
    if kind == "sequence":
        return all(slot_present(k, v, query) for k, v in slots.items() if v)
    name = intent.get("name")
    if name == "order_food":
        return str(slots.get("shop") or "") in query and str(slots.get("dish") or "") in query
    if name == "set_switch":
        return slot_present("light", slots.get("light"), query)
    if name in {"open_door", "close_door"}:
        return slot_present("door", slots.get("door"), query)
    if name == "go_to":
        return str(slots.get("place") or "") in query
    if name == "point":
        return slot_present("target", slots.get("target"), query)
    return True


# Manner clauses must not name protected slots (rooms/doors/lights/food/point).
_MANNER_WHERE = [
    "原地", "当前站位", "发令者视野内", "口令范围内", "不挪额外位置",
    "面向发令者处", "动作可被看见处", "脚下位置", "短距离内", "视线可及处",
    "不离场的位置", "礼貌距离内", "现场站位", "不增加位移处", "一次收束处",
    "不闲聊的现场", "口令原意内", "节拍范围内", "不添表情戏处", "安静执行区",
]
_MANNER_HOW = [
    "干脆地", "利落地", "安静地", "清楚地", "平稳地",
    "短促地", "明确地", "一次做完地", "不拖沓地", "干净地",
    "有节制地", "按口令地", "不解释地", "直接地", "克制地",
    "连贯地", "准确地", "不添戏地", "照原意地", "不走样地",
]
_MANNER_TAIL = [
    "即可", "就结束", "不要加戏", "保持礼貌", "算完成",
    "收住", "停在结果上", "不要复述", "作为本条动作", "到此为止",
]


def manner_clause(salt: int) -> str:
    w = _MANNER_WHERE[salt % len(_MANNER_WHERE)]
    h = _MANNER_HOW[(salt // len(_MANNER_WHERE)) % len(_MANNER_HOW)]
    t = _MANNER_TAIL[(salt // (len(_MANNER_WHERE) * len(_MANNER_HOW))) % len(_MANNER_TAIL)]
    return f"{h}在{w}执行，{t}"


def render_eval_query(intent: dict, rng: random.Random, *, salt: int = 0) -> tuple[str, str]:
    tags = list(intent.get("style_tags") or ["command"])
    tag = tags[0]
    q, tid = _affix(rng, str(intent.get("stem") or ""), tag)
    q = f"{q}，{manner_clause(salt)}"
    tid = f"{tid}-{salt % 997}"
    return q, tid


def intent_from_row(row: dict) -> dict:
    return {
        "kind": row.get("kind"),
        "name": row.get("gold_name"),
        "args": row.get("gold_args") or {},
        "calls": row.get("gold_calls") or [],
        "family": row.get("family"),
        "scene": row.get("scene"),
        "zh_slots": row.get("zh_slots") or {},
        "stem": row.get("stem") or "",
    }


def _cycle(items: list, n: int) -> list:
    if not items:
        return []
    return [items[i % len(items)] for i in range(n)]


def _style(i: int, recipe: dict) -> list[str]:
    tags = list(recipe["style_tags"])
    return [tags[i % len(tags)]]


def enumerate_canonical_intents(recipe: dict) -> list[dict]:
    intents: list[dict] = []
    n = recipe["family_n"]

    gi = 0
    for name, stems in GESTURE_STEMS.items():
        count = 32
        for i, stem in enumerate(_cycle(stems, count)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "gesture",
                    "name": name,
                    "args": {},
                    "stem": stem,
                    "style_tags": _style(gi + i, recipe),
                    "case_tags": [],
                    "zh_slots": {},
                }
            )
        gi += count
    point_n = {"left": 15, "right": 15, "user": 14}
    zh_map = {"left": "左边", "right": "右边", "user": "我"}
    for enum, count in point_n.items():
        stems = POINT_STEMS[enum]
        for i, stem in enumerate(_cycle(stems, count)):
            tags = ["same_word_diff_slot"] if i % 5 == 0 else []
            intents.append(
                {
                    "kind": "execute",
                    "family": "gesture",
                    "name": "point",
                    "args": {"target": enum},
                    "stem": stem,
                    "style_tags": _style(gi + i, recipe),
                    "case_tags": tags,
                    "zh_slots": {"target": zh_map[enum]},
                }
            )
            gi += 1

    place_enum = {"kitchen": "厨房", "living": "客厅", "entry": "门口", "trash": "垃圾桶"}
    for enum, zh in place_enum.items():
        for i, stem in enumerate(_cycle(PLACE_STEMS[enum], 35)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "home",
                    "name": "go_to",
                    "args": {"place": enum},
                    "stem": stem,
                    "style_tags": _style(i, recipe),
                    "case_tags": [],
                    "zh_slots": {"place": zh},
                }
            )
    light_keys = [
        ("kitchen_light", True, "厨房灯", 35),
        ("kitchen_light", False, "厨房灯", 35),
        ("living_light", True, "客厅灯", 35),
        ("living_light", False, "客厅灯", 35),
    ]
    for lid, on, zh, count in light_keys:
        for i, stem in enumerate(_cycle(LIGHT_STEMS[(lid, on)], count)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "home",
                    "name": "set_switch",
                    "args": {"id": lid, "on": on},
                    "stem": stem,
                    "style_tags": _style(i, recipe),
                    "case_tags": ["negation_scope"] if not on and i % 7 == 0 else [],
                    "zh_slots": {"light": zh},
                }
            )
    for enum, zh in (("front", "前门"), ("back", "后门")):
        for i, stem in enumerate(_cycle(DOOR_OPEN_STEMS[enum], 25)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "home",
                    "name": "open_door",
                    "args": {"door": enum},
                    "stem": stem,
                    "style_tags": _style(i, recipe),
                    "case_tags": [],
                    "zh_slots": {"door": zh},
                }
            )
        for i, stem in enumerate(_cycle(DOOR_CLOSE_STEMS[enum], 25)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "home",
                    "name": "close_door",
                    "args": {"door": enum},
                    "stem": stem,
                    "style_tags": _style(i, recipe),
                    "case_tags": [],
                    "zh_slots": {"door": zh},
                }
            )
    for i, stem in enumerate(_cycle(TRASH_STEMS, 100)):
        intents.append(
            {
                "kind": "execute",
                "family": "home",
                "name": "take_out_trash",
                "args": {},
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": [],
                "zh_slots": {},
            }
        )

    legal = pair_tuples(recipe["legal_food"])
    for shop, dish in legal:
        count = 45
        stems = [
            f"请向{shop}下单{dish}",
            f"在{shop}把{dish}加入订单",
            f"给{shop}点一份{dish}",
            f"{shop}的{dish}请下一单",
            f"向{shop}提交{dish}的订单",
        ]
        for i, stem in enumerate(_cycle(stems, count)):
            intents.append(
                {
                    "kind": "execute",
                    "family": "order",
                    "name": "order_food",
                    "args": {"shop": shop, "dish": dish},
                    "stem": stem,
                    "style_tags": _style(i, recipe),
                    "case_tags": [],
                    "zh_slots": {"shop": shop, "dish": dish},
                }
            )
    for i, stem in enumerate(_cycle(CANCEL_STEMS, 70)):
        intents.append(
            {
                "kind": "execute",
                "family": "order",
                "name": "cancel_order",
                "args": {},
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": [],
                "zh_slots": {},
            }
        )

    seq_specs = [
        (
            [{"name": "nod", "arguments": {}}, {"name": "wave", "arguments": {}}],
            "请先轻轻颔首表示收到，再抬手做一次招呼手势",
            {},
        ),
        (
            [{"name": "wave", "arguments": {}}, {"name": "bow", "arguments": {}}],
            "请先朝外面挥动手掌致意，再弯腰做一次礼节性鞠躬",
            {},
        ),
        (
            [{"name": "sit", "arguments": {}}, {"name": "stand", "arguments": {}}],
            "请先就近坐到座位上，随后从坐姿改成站直",
            {},
        ),
        (
            [{"name": "come_here", "arguments": {}}, {"name": "stop", "arguments": {}}],
            "请先朝我站的位置走近，到达后立刻止步保持原地",
            {},
        ),
        (
            [{"name": "shake_head", "arguments": {}}, {"name": "bow", "arguments": {}}],
            "请先左右摆头表示拒绝，再做一个正式的鞠躬动作",
            {},
        ),
        (
            [
                {"name": "go_to", "arguments": {"place": "kitchen"}},
                {"name": "set_switch", "arguments": {"id": "kitchen_light", "on": True}},
            ],
            "请先移动到做饭的厨房区域，再把厨房灯切换成亮着",
            {"place": "厨房", "light": "厨房灯"},
        ),
        (
            [
                {"name": "go_to", "arguments": {"place": "living"}},
                {"name": "set_switch", "arguments": {"id": "living_light", "on": False}},
            ],
            "请先移动到会客的客厅区域，再把客厅灯切换成熄灭",
            {"place": "客厅", "light": "客厅灯"},
        ),
        (
            [
                {"name": "go_to", "arguments": {"place": "entry"}},
                {"name": "open_door", "arguments": {"door": "front"}},
            ],
            "请先移动到进门的门口区域，再把前门从关闭推到打开",
            {"place": "门口", "door": "前门"},
        ),
        (
            [
                {"name": "go_to", "arguments": {"place": "trash"}},
                {"name": "take_out_trash", "arguments": {}},
            ],
            "请先移动到垃圾桶旁边，再把家里的垃圾带出去丢掉",
            {"place": "垃圾桶"},
        ),
        (
            [
                {"name": "open_door", "arguments": {"door": "back"}},
                {"name": "go_to", "arguments": {"place": "entry"}},
            ],
            "请先把后门从关闭推到打开，再走到门口那一侧待命",
            {"door": "后门", "place": "门口"},
        ),
        (
            [
                {"name": "close_door", "arguments": {"door": "front"}},
                {"name": "take_out_trash", "arguments": {}},
            ],
            "请先把前门从打开收到关闭，再将垃圾袋拿到室外处理",
            {"door": "前门"},
        ),
        (
            [
                {"name": "point", "arguments": {"target": "left"}},
                {"name": "nod", "arguments": {}},
            ],
            "请先朝左手边那个方向伸出手指，再轻轻颔首表示收到",
            {"target": "左边"},
        ),
        (
            [
                {"name": "order_food", "arguments": {"shop": "兰州拉面", "dish": "牛肉面"}},
                {"name": "nod", "arguments": {}},
            ],
            "请先向兰州拉面下单牛肉面，再给一个短促的颔首示意",
            {"shop": "兰州拉面", "dish": "牛肉面"},
        ),
        (
            [
                {"name": "order_food", "arguments": {"shop": "麦当劳", "dish": "巨无霸"}},
                {"name": "wave", "arguments": {}},
            ],
            "请先在麦当劳把巨无霸加入订单，再举手晃一晃打个招呼",
            {"shop": "麦当劳", "dish": "巨无霸"},
        ),
        (
            [
                {"name": "go_to", "arguments": {"place": "kitchen"}},
                {"name": "take_out_trash", "arguments": {}},
                {"name": "wave", "arguments": {}},
            ],
            "请先移动到做饭的厨房区域，再把家里的垃圾带出去丢掉，最后抬手做一次招呼手势",
            {"place": "厨房"},
        ),
        (
            [
                {"name": "sit", "arguments": {}},
                {"name": "point", "arguments": {"target": "right"}},
                {"name": "stand", "arguments": {}},
            ],
            "请先就近坐到座位上，再朝右手边那个方向伸出手指，然后从坐姿改成站直",
            {"target": "右边"},
        ),
    ]
    seq_need = int(n["sequence"])
    for i in range(seq_need):
        calls, stem, slots = seq_specs[i % len(seq_specs)]
        intents.append(
            {
                "kind": "sequence",
                "family": "sequence",
                "name": None,
                "args": {},
                "calls": [dict(c) for c in calls],
                "stem": stem if i < len(seq_specs) else f"{stem}；本条按独立顺序执行",
                "style_tags": _style(i, recipe),
                "case_tags": ["sequence_order"],
                "zh_slots": dict(slots),
            }
        )

    for i, stem in enumerate(_cycle(MISSING_STEMS, int(n["missing"]))):
        tags = ["short_ellipsis"] if i % 4 == 0 else []
        intents.append(
            {
                "kind": "missing",
                "family": "missing",
                "name": None,
                "args": {},
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": tags,
                "zh_slots": {},
            }
        )

    sc_need = int(n["scene_conflict"])
    for i in range(sc_need):
        tool, slot, scene, stem, tags = SCENE_SPECS[i % len(SCENE_SPECS)]
        zh_slots = {}
        args: dict = {}
        if tool == "open_door":
            args = {"door": slot}
            zh_slots = {"door": "前门" if slot == "front" else "后门"}
        elif tool == "close_door":
            args = {"door": slot}
            zh_slots = {"door": "前门" if slot == "front" else "后门"}
        elif tool == "set_switch":
            args = {"id": slot, "on": slot == "kitchen_light"}
            zh_slots = {"light": "厨房灯" if "kitchen" in str(slot) else "客厅灯"}
        intents.append(
            {
                "kind": "scene_conflict",
                "family": "scene_conflict",
                "name": tool,
                "args": args,
                "scene": scene,
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": list(tags),
                "zh_slots": zh_slots,
            }
        )

    illegal = pair_tuples(recipe["illegal_food"])
    ill_need = int(n["illegal_pair"])
    ill_stems = []
    for shop, dish in illegal:
        ill_stems.extend(
            [
                f"{shop}来一份{dish}",
                f"去{shop}点{dish}",
                f"帮我向{shop}下单{dish}",
                f"{shop}的{dish}套餐来一份",
                f"给{shop}点{dish}外卖",
            ]
        )
    for i, stem in enumerate(_cycle(ill_stems, ill_need)):
        shop = "兰州拉面" if "兰州拉面" in stem else "麦当劳"
        dish = "巨无霸" if "巨无霸" in stem else "牛肉面"
        intents.append(
            {
                "kind": "illegal_pair",
                "family": "illegal_pair",
                "name": "order_food",
                "args": {"shop": shop, "dish": dish},
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": ["illegal_food"],
                "zh_slots": {"shop": shop, "dish": dish},
            }
        )

    wraps = [
        "帮我备一份关于{t}的口述笔记",
        "把{t}讲成三段课堂提纲",
        "出一道考查{t}的简答题",
        "用对比方式说明{t}",
        "给新手解释{t}的入门要点",
        "写三条关于{t}的复习提示",
        "把{t}缩成一句口诀",
        "从材料学角度聊聊{t}",
    ]
    off_stems = [w.format(t=t) for t in OFFTOPIC_TOPICS for w in wraps]
    for i, stem in enumerate(_cycle(off_stems, int(n["offtopic"]))):
        intents.append(
            {
                "kind": "offtopic",
                "family": "offtopic",
                "name": None,
                "args": {},
                "stem": stem,
                "style_tags": _style(i, recipe),
                "case_tags": [],
                "zh_slots": {},
            }
        )

    by = {}
    for it in intents:
        by.setdefault(it["family"], []).append(it)
    out: list[dict] = []
    for fam, want in n.items():
        got = by.get(fam) or []
        if len(got) < want:
            raise RuntimeError(f"family {fam} generated {len(got)} < {want}")
        out.extend(got[:want])
    if len(out) != recipe["n_total"]:
        raise RuntimeError(f"intent n={len(out)} want={recipe['n_total']}")
    for i, it in enumerate(out):
        it["uid"] = f"{it['family']}-{i:04d}"
    return out


def collect_blocked_queries() -> set[str]:
    blocked: set[str] = set()
    for path in [
        BANK_NEEDLE_VRM_AGENT,
        SEED_NEEDLE_ZH,
        TASKS_ROOT / TASK_NEEDLE_ZH / "train/seed/sft-phase1-v0.jsonl",
        TASKS_ROOT / TASK_NEEDLE_ZH / "train/seed/sft-smoke-v0.jsonl",
        TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs/home-sft-2k.jsonl",
        TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs/home-sft-10k.jsonl",
    ]:
        for row in load_jsonl(path):
            q = str(row.get("query") or "").strip()
            if q:
                blocked.add(q)
    return blocked


def assign_dev_split(rows: list[dict], recipe: dict, rng: random.Random) -> None:
    want = dict(recipe["dev_family_n"])
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(str(row["family"]), []).append(row)
    chosen: set[str] = set()
    for fam, n in want.items():
        pool = list(by.get(fam) or [])
        rng.shuffle(pool)
        # keep intent groups together: one row per intent in v2
        for row in pool[:n]:
            chosen.add(row["item_id"])
    for row in rows:
        row["split"] = "dev" if row["item_id"] in chosen else "eval"


def make_eval_item(intent: dict, query: str, *, recipe: dict, index: int, template_id: str, teacher_model: str) -> dict:
    gold = gold_from_intent_v2(intent, recipe)
    iid = f"{recipe['item_id_prefix']}{index:04d}"
    row = {
        "item_id": iid,
        "split": "eval",
        "bank": BANK_ID,
        "holdout_version": "v2",
        "lang": "zh",
        "family": intent["family"],
        "toolset_id": TOOLSET_ID,
        "query": query,
        "gold": {"function_calls": gold},
        "pass": "exact_match",
        "judge_notes": "schema-gold; holdout v2",
        "review_status": "generated",
        "intent_id": intent_key(intent),
        "template_id": template_id,
        "style_tags": list(intent.get("style_tags") or []),
        "case_tags": list(intent.get("case_tags") or []),
        "source_role": "schema-intent",
        "teacher_model": teacher_model,
        "prompt_version": recipe.get("prompt_version") or PROMPT_VERSION,
        "generator_version": recipe.get("generator_version") or GENERATOR_VERSION,
        "kind": intent.get("kind"),
        "gold_name": intent.get("name"),
        "gold_args": intent.get("args") or {},
        "gold_calls": intent.get("calls") or gold,
        "zh_slots": dict(intent.get("zh_slots") or {}),
        "stem": intent.get("stem") or "",
    }
    if intent.get("scene"):
        row["scene"] = intent["scene"]
    return row


def catalog_ok(calls: list[dict]) -> bool:
    cat = tools_of(load_toolset())
    for call in calls:
        name = str(call.get("name") or "")
        if name not in cat:
            return False
        spec = cat[name]
        params = spec.get("parameters") or {}
        required = list(params.get("required") or [])
        props = params.get("properties") or {}
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            return False
        if any(k not in args for k in required):
            return False
        for key, val in args.items():
            prop = props.get(key) or {}
            enum = prop.get("enum")
            if enum is not None and val not in enum:
                return False
    return True
