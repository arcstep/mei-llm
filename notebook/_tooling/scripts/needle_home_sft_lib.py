#!/usr/bin/env python3
"""Shared intent, gold, wording, and scene helpers for needle-zh home SFT.

Teachers (template or qwen-max) may only rewrite query/scene. Gold is always
recomputed by gold_from_intent. Holdout/lock files are never written here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from repo_paths import EVAL_SHARED_ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

MIXTURE_PATH = TASKS_ROOT / TASK_NEEDLE_ZH / "recipes" / "sft-mixture-v1.json"
GENERATOR_VERSION = "sft-pack-v1"
PROMPT_VERSION = "sft-teacher-v1"
TOOLSET_ID = "needle-vrm-agent-v0"
FAKE_VARIANT_RE = re.compile(r"（说法\s*\d+）|\(说法\s*\d+\)")
DOUBLE_QING_RE = re.compile(r"请请")
ASK_RE = re.compile(r"请补充|请提供城市|请告诉我房间|请告诉我哪盏")
EVAL_RE = re.compile(r"\bEVAL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+\b")
PII_RE = re.compile(
    r"(?:\+?86[-\s]?)?1[3-9]\d{9}"
    r"|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}"
    r"|worker[_\-]?id\s*[:=]\s*\S+",
    re.I,
)
TOOL_NAME_RE = re.compile(
    r"\b(?:nod|shake_head|come_here|stop|point|wave|bow|sit|stand|go_to|"
    r"set_switch|open_door|close_door|take_out_trash|order_food|cancel_order)\b"
)
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


def load_mixture(path: Path | None = None) -> dict:
    p = path or MIXTURE_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def format_sft_user_text(row: dict, *, lang: str | None = None) -> str:
    """Match eval/Qwen runner prefixes so scene_conflict is visible at train time."""
    query = str(row.get("query") or "").strip()
    scene = row.get("scene")
    use_en = str(lang or row.get("lang") or "zh").lower().startswith("en")
    if use_en:
        if isinstance(scene, str) and scene.strip():
            return f"Scene: {scene.strip()}. User: {query}"
        return f"User: {query}"
    if isinstance(scene, str) and scene.strip():
        return f"场景：{scene.strip()}。用户：{query}"
    return f"用户：{query}"


def load_toolset(toolset_id: str = TOOLSET_ID) -> dict:
    path = EVAL_SHARED_ROOT / "toolsets" / f"{toolset_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tools_of(toolset: dict) -> dict[str, dict]:
    return {str(t.get("name")): t for t in (toolset.get("tools") or []) if t.get("name")}


def pair_tuples(rows: list) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in rows]


def gold_from_intent(intent: dict, mixture: dict | None = None) -> list[dict]:
    mix = mixture or load_mixture()
    kind = str(intent.get("kind") or "")
    if kind in {"missing", "scene_conflict", "illegal_pair", "offtopic"}:
        return []
    name = intent.get("name")
    args = dict(intent.get("args") or {})
    if not name:
        return []
    legal = set(pair_tuples(mix["legal_food"]))
    if name == "order_food" and (args.get("shop"), args.get("dish")) not in legal:
        return []
    return [{"name": str(name), "arguments": args}]


def intent_key(intent: dict) -> str:
    payload = {
        "kind": intent.get("kind"),
        "name": intent.get("name"),
        "args": intent.get("args") or {},
        "scene": intent.get("scene") or "",
        "family": intent.get("family"),
        "stem": intent.get("stem") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "INT-" + sha256_text(raw)[:16]


def split_for_key(key: str, *, valid_frac: float) -> str:
    cut = max(1, int(valid_frac * 10_000))
    bucket = int(sha256_text(key)[:8], 16) % 10_000
    return "valid" if bucket < cut else "train"


def _starts_with_qing(text: str) -> bool:
    return text.startswith("请")


def _join_affix(prefix: str, stem: str, suffix: str) -> str:
    stem = stem.strip()
    prefix = prefix.strip()
    suffix = suffix.strip()
    if prefix and _starts_with_qing(stem) and prefix.endswith("请"):
        prefix = prefix[: -len("请")].rstrip()
    if prefix and stem.startswith(prefix):
        prefix = ""
    if suffix and stem.endswith(suffix):
        suffix = ""
    q = f"{prefix}{stem}{suffix}".strip()
    q = re.sub(r"请请+", "请", q)
    return q


STYLE_AFFIX = {
    "formal": ("麻烦", "，谢谢"),
    "colloquial": ("", "呗"),
    "command": ("", ""),
    "statement": ("我想", ""),
    "negation": ("先别搞错，", ""),
    "ellipsis": ("那个", "…"),
    "inversion": ("", "，现在"),
    "particle": ("", "呀"),
    "deixis": ("这边", ""),
    "correction": ("我说的是", "，不是别的"),
    "redundant": ("现在马上", "一下"),
}


def offtopic_stems() -> list[str]:
    topics = [
        "相对论", "量子力学", "光合作用", "通货膨胀", "二叉搜索树", "哈希表",
        "最小二乘法", "圆周率", "九九乘法表", "岳阳楼记", "七言绝句", "故宫午门",
        "西湖十景", "唐朝科举", "四大名著", "红烧肉做法", "溏心蛋", "绿萝浇水",
        "实木地板保养", "自行车打气", "衬衫折叠", "俯卧撑", "挑选西瓜", "织毛衣",
        "围棋规则", "钢琴中央C", "松树和柏树", "猫打呼噜", "火星一天", "地球半径",
        "太阳到地球", "世界上最长的河流", "沪市大盘", "国际金价", "北京到上海高铁",
        "护照申请", "请假条", "感谢信", "生日祝福", "产品发布会开场白", "网名",
        "购物清单模板", "健身计划", "下周会议", "法语翻译", "英文介绍故宫",
        "科幻电影", "推理小说", "鬼故事", "冷笑话", "REST API", "量子计算",
        "光合作用以外的呼吸作用", "三十二的平方", "十七乘二十四", "电灯发明者",
        "三甲医院", "博物馆", "量子纠缠", "最小公倍数",
    ]
    wraps = [
        "解释一下{t}",
        "写一段关于{t}的介绍",
        "帮我科普{t}",
        "{t}是什么意思",
        "推荐资料学习{t}",
        "把{t}讲给小孩听",
        "用三句话说明{t}",
        "{t}有什么常见误区",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for t in topics:
        for w in wraps:
            q = w.format(t=t)
            if q not in seen:
                seen.add(q)
                out.append(q)
    extras = [
        "帮我改简历", "背一遍九九乘法表后半段", "怎么申请护照", "如何给绿萝浇水",
        "太阳系有几颗行星", "谁发明了电灯", "附近有什么博物馆", "红烧肉的家常做法步骤",
        "帮我翻译成法语", "圆周率小数点后二十位", "沪市大盘今日点数", "写一首七言绝句",
        "解释相对论", "给我讲个鬼故事", "北京到上海高铁多久", "推荐一部科幻电影",
        "现在国际金价", "什么是二叉搜索树", "用英语介绍故宫", "计算三十二的平方",
        "帮我写请假条", "量子纠缠是什么意思", "把这段话改成公文", "世界杯谁夺冠了",
        "教我织毛衣", "背诵岳阳楼记第一段",
    ]
    for q in extras:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


OFFTOPIC_STEMS = offtopic_stems()

MISSING_STEMS = [
    "把灯打开吧",
    "灯关一下",
    "把门打开",
    "我要点餐",
    "去那边看看",
    "帮忙指一下",
    "开关拨一下",
    "去房间",
    "来碗面",
    "要个汉堡",
    "点一份外卖",
    "把那盏没说房间的灯打开",
]


def sample_intent(rng: random.Random, mixture: dict) -> dict:
    weights = mixture["family_weights"]
    families = list(weights)
    probs = [float(weights[f]) for f in families]
    total = sum(probs)
    roll = rng.random() * total
    acc = 0.0
    family = families[-1]
    for f, p in zip(families, probs):
        acc += p
        if roll <= acc:
            family = f
            break
    style = rng.choice(list(mixture["style_tags"]))
    if family == "gesture":
        if rng.random() < 0.22:
            zh, enum = rng.choice(pair_tuples(mixture["point_targets"]))
            return {
                "kind": "execute",
                "name": "point",
                "args": {"target": enum},
                "family": "gesture",
                "stem": f"用手指{zh}",
                "style_tags": [style],
                "zh_slots": {"target": zh},
            }
        q, name, args = rng.choice(GESTURE)
        return {
            "kind": "execute",
            "name": name,
            "args": dict(args),
            "family": "gesture",
            "stem": q,
            "style_tags": [style],
            "zh_slots": {},
        }
    if family == "home":
        hw = mixture["home_tool_weights"]
        tools = list(hw)
        troll = rng.random() * sum(float(hw[t]) for t in tools)
        tacc = 0.0
        tool = tools[-1]
        for t, p in zip(tools, [float(hw[x]) for x in tools]):
            tacc += p
            if troll <= tacc:
                tool = t
                break
        if tool == "go_to":
            zh, enum = rng.choice(pair_tuples(mixture["places"]))
            return {
                "kind": "execute",
                "name": "go_to",
                "args": {"place": enum},
                "family": "home",
                "stem": f"去{zh}",
                "style_tags": [style],
                "zh_slots": {"place": zh},
            }
        if tool == "set_switch":
            zh, enum = rng.choice(pair_tuples(mixture["lights"]))
            on = rng.choice([True, False])
            verb = "打开" if on else "关掉"
            return {
                "kind": "execute",
                "name": "set_switch",
                "args": {"id": enum, "on": on},
                "family": "home",
                "stem": f"{verb}{zh}",
                "style_tags": [style],
                "zh_slots": {"light": zh},
            }
        if tool in {"open_door", "close_door"}:
            zh, enum = rng.choice(pair_tuples(mixture["doors"]))
            open_ = tool == "open_door"
            verb = "打开" if open_ else "关上"
            return {
                "kind": "execute",
                "name": tool,
                "args": {"door": enum},
                "family": "home",
                "stem": f"{verb}{zh}",
                "style_tags": [style],
                "zh_slots": {"door": zh},
            }
        return {
            "kind": "execute",
            "name": "take_out_trash",
            "args": {},
            "family": "home",
            "stem": "去倒垃圾",
            "style_tags": [style],
            "zh_slots": {},
        }
    if family == "order":
        ow = mixture["order_tool_weights"]
        if rng.random() < float(ow["order_food"]):
            shop, dish = rng.choice(pair_tuples(mixture["legal_food"]))
            return {
                "kind": "execute",
                "name": "order_food",
                "args": {"shop": shop, "dish": dish},
                "family": "order",
                "stem": f"在{shop}点{dish}",
                "style_tags": [style],
                "zh_slots": {"shop": shop, "dish": dish},
            }
        return {
            "kind": "execute",
            "name": "cancel_order",
            "args": {},
            "family": "order",
            "stem": "取消刚才的外卖",
            "style_tags": [style],
            "zh_slots": {},
        }
    if family == "missing":
        return {
            "kind": "missing",
            "name": None,
            "args": {},
            "family": "missing",
            "stem": rng.choice(MISSING_STEMS),
            "style_tags": [style],
            "zh_slots": {},
        }
    if family == "illegal_pair":
        shop, dish = rng.choice(pair_tuples(mixture["illegal_food"]))
        return {
            "kind": "illegal_pair",
            "name": "order_food",
            "args": {"shop": shop, "dish": dish},
            "family": "illegal_pair",
            "stem": f"{shop}来一份{dish}",
            "style_tags": [style],
            "zh_slots": {"shop": shop, "dish": dish},
        }
    if family == "scene_conflict":
        if rng.random() < 0.7:
            zh, _enum = rng.choice(pair_tuples(mixture["doors"]))
            return {
                "kind": "scene_conflict",
                "name": "open_door",
                "args": {},
                "family": "scene_conflict",
                "stem": f"打开{zh}",
                "scene": f"{zh}已经开着",
                "style_tags": [style],
                "zh_slots": {"door": zh},
            }
        return {
            "kind": "scene_conflict",
            "name": "cancel_order",
            "args": {},
            "family": "scene_conflict",
            "stem": "取消订单",
            "scene": "没有进行中的订单",
            "style_tags": [style],
            "zh_slots": {},
        }
    return {
        "kind": "offtopic",
        "name": None,
        "args": {},
        "family": "offtopic",
        "stem": rng.choice(OFFTOPIC_STEMS),
        "style_tags": [style],
        "zh_slots": {},
    }


def render_template_query(intent: dict, rng: random.Random) -> tuple[str, str]:
    """Return (query, template_id). Never emits （说法N） or stacked 请请."""
    stem = str(intent.get("stem") or "").strip()
    tags = list(intent.get("style_tags") or ["command"])
    tag = tags[0] if tags else "command"
    prefix, suffix = STYLE_AFFIX.get(tag, ("", ""))
    extra_prefixes = ["", "帮我", "麻烦", "可以", "劳驾", "回头", "请你", "能不能", "拜托", "赶紧"]
    extra_suffixes = ["", "一下", "吧", "好吗", "可以吗", "呢", "呀", "哦", "哈", "行不行"]
    if tag == "command" or rng.random() < 0.5:
        extra_p = rng.choice(extra_prefixes)
        extra_s = rng.choice(extra_suffixes)
        prefix = extra_p or prefix
        suffix = extra_s or suffix
    if _starts_with_qing(stem) and prefix in {"请", "现在请"}:
        prefix = "帮我" if prefix else ""
    q = _join_affix(prefix, stem, suffix)
    tid = f"tpl-{intent.get('name') or intent.get('kind')}-{tag}-{sha256_text(prefix + '|' + suffix)[:8]}"
    slots = intent.get("zh_slots") or {}
    if intent.get("name") == "point" and slots.get("target"):
        zh = slots["target"]
        alts = [f"用手指{zh}", f"往{zh}指一下", f"指向{zh}"]
        if zh == "我":
            alts = ["指向我", "用手指着我", "指我这边"]
        q = _join_affix(prefix, rng.choice(alts), suffix)
    if intent.get("name") == "take_out_trash":
        alts = ["去倒垃圾", "把垃圾袋拿出去", "垃圾请倒掉", "把垃圾扔掉", "出门倒垃圾"]
        stem2 = rng.choice(alts)
        q = _join_affix(prefix if not _starts_with_qing(stem2) else "", stem2, suffix)
    if intent.get("name") == "cancel_order" and intent.get("kind") == "execute":
        alts = ["取消刚才的外卖", "订单不要了", "把进行中的餐取消", "取消当前订单", "这单取消掉"]
        q = _join_affix(prefix, rng.choice(alts), suffix)
    if intent.get("name") == "nod":
        q = _join_affix(prefix, rng.choice(["点一下头", "点头", "点个头"]), suffix)
    if intent.get("name") == "shake_head":
        q = _join_affix(prefix, rng.choice(["摇摇头", "摇头", "摇一下头"]), suffix)
    if intent.get("name") == "wave":
        q = _join_affix(prefix, rng.choice(["挥挥手", "挥手", "招一下手"]), suffix)
    if intent.get("name") == "bow":
        q = _join_affix(prefix, rng.choice(["鞠躬致谢", "鞠个躬", "鞠躬"]), suffix)
    if intent.get("name") == "sit":
        q = _join_affix(prefix, rng.choice(["请坐下", "坐下", "坐一会儿"]), suffix)
    if intent.get("name") == "stand":
        q = _join_affix(prefix, rng.choice(["请起立", "站起来", "起立"]), suffix)
    if intent.get("name") == "come_here":
        q = _join_affix(prefix, rng.choice(["走到我跟前来", "过来一下", "到我这边来"]), suffix)
    if intent.get("name") == "stop":
        q = _join_affix(prefix, rng.choice(["站住别走", "停住", "先停下"]), suffix)
    return q, tid


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


def protected_slots_ok(intent: dict, query: str) -> bool:
    """Teacher must keep visible zh slot nouns for execute/illegal; missing may omit them."""
    kind = intent.get("kind")
    slots = intent.get("zh_slots") or {}
    if kind in {"missing", "offtopic"}:
        return True
    if kind == "illegal_pair":
        return str(slots.get("shop") or "") in query and str(slots.get("dish") or "") in query
    name = intent.get("name")
    if name == "order_food":
        return str(slots.get("shop") or "") in query and str(slots.get("dish") or "") in query
    if name == "set_switch":
        return str(slots.get("light") or "") in query
    if name in {"open_door", "close_door"}:
        return str(slots.get("door") or "") in query
    if name == "go_to":
        return str(slots.get("place") or "") in query
    if name == "point":
        return str(slots.get("target") or "") in query
    return True


def row_from_intent(
    intent: dict,
    query: str,
    *,
    mixture: dict,
    index: int,
    teacher_model: str,
    source_role: str,
    template_id: str,
) -> dict:
    answers = gold_from_intent(intent, mixture)
    key = intent_key(intent)
    row = {
        "sample_id": f"TRAIN-VRM-{index:06d}",
        "split": split_for_key(key, valid_frac=float(mixture["valid_frac"])),
        "lang": "zh",
        "family": intent["family"],
        "toolset_id": TOOLSET_ID,
        "query": query,
        "answers": answers,
        "act": None,
        "confidence_label": 0 if not answers else 1,
        "intent_id": key,
        "template_id": template_id,
        "style_tags": list(intent.get("style_tags") or []),
        "source_role": source_role,
        "teacher_model": teacher_model,
        "prompt_version": mixture.get("prompt_version") or PROMPT_VERSION,
        "generator_version": mixture.get("generator_version") or GENERATOR_VERSION,
        "kind": intent.get("kind"),
        "gold_name": intent.get("name"),
        "gold_args": intent.get("args") or {},
    }
    if intent.get("scene"):
        row["scene"] = intent["scene"]
    return row


def seen_key(row: dict) -> str:
    scene = str(row.get("scene") or "").strip()
    return sha256_text(str(row.get("query") or "").strip() + "\n" + scene)


def assistant_truncated(tok: Any, row: dict, seq_len: int) -> bool:
    from data import encode_sft_row

    packed = encode_sft_row(tok, row, seq_len)
    answers = row.get("answers") or []
    if not answers:
        return False
    return int(packed.get("n_unmasked") or 0) <= 0
