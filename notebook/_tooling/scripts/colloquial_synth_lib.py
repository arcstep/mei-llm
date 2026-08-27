#!/usr/bin/env python3
"""Frozen colloquial-synth contract, frames, filters, and generators.

Production language realization is a pinned qwen-plus snapshot.
The offline renderer exists for smoke/CI and engineering pilots; it cannot
open formal CPT.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from repo_paths import ROOT, SPEC_NEEDLE_ZH

CONTRACT_PATH = SPEC_NEEDLE_ZH / "colloquial-synth-v1.json"
FALLBACK_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TOOLISH_RE = re.compile(r'(\{"route_id"|<\s*routes\s*>|EVAL-[A-Z0-9]+|function_calls)', re.I)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
TURN_LINE_RE = re.compile(r"^(甲|乙|A|B)[：:](.+)$")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
UNK_CHAR_REWRITE = str.maketrans(
    {
        "诶": "哎",
        "欸": "哎",
        "呃": "额",
        "嗯": "恩",
        "噢": "哦",
        "啧": "切",
        "嗐": "咳",
        "喏": "那",
        "嘞": "了",
        "呗": "吧",
        "喽": "了",
        "仨": "三",
        "冇": "没",
        "唔": "不",
        "咗": "了",
        "嘅": "的",
        "嚟": "来",
        "佢": "他",
        "㗎": "的",
        "瞓": "睡",
        "晡": "晚",
        "～": "。",
    }
)

SCENE_BEATS = {
    "home": ["灯", "空调", "晚饭", "垃圾袋", "门锁", "洗衣机", "遥控器"],
    "commute": ["地铁", "公交", "堵车", "换乘", "迟到", "共享单车"],
    "restaurant": ["位子", "菜单", "辣", "打包", "账单", "等位"],
    "shopping": ["尺码", "打折", "退货", "购物袋", "试衣间"],
    "school": ["作业", "家长会", "校服", "迟到", "考试"],
    "clinic": ["挂号", "排队", "药", "复查", "发烧"],
    "workplace": ["会议", "周报", "加班", "工位", "截止日期"],
    "phone_call": ["信号", "回电", "占线", "微信", "语音"],
    "delivery": ["快递", "取件码", "放门口", "驿站", "破损"],
    "weather": ["下雨", "降温", "晒", "台风", "雾霾"],
    "sports": ["球场", "跑步", "拉伸", "报名", "请假"],
    "travel": ["车票", "酒店", "行李", "改签", "景点"],
    "family": ["爸妈", "孩子", "过年", "买菜", "看病"],
    "neighbors": ["楼道", "噪音", "快递柜", "物业", "钥匙"],
    "repair": ["师傅", "水管", "预约", "零件", "保修"],
    "pet": ["遛狗", "猫粮", "疫苗", "洗澡", "叫"],
    "cooking": ["下锅", "盐", "剩菜", "烤箱", "买菜"],
    "gaming": ["排位", "掉线", "组队", "皮肤", "更新"],
}

ASK = [
    "这个{beat}你弄了没？",
    "{beat}咋样了，靠谱吗？",
    "要不咱先把{beat}搞定？",
    "你觉得{beat}还行不？",
    "诶，{beat}现在方便弄吗？",
]
ANSWER = [
    "弄了弄了，刚才顺手的。",
    "还行吧，比我想的省事。",
    "没呢，我正忙着呢。",
    "可以，那我一会儿弄。",
    "嗯，我看着还行。",
]
REQUEST = [
    "帮我看一眼{beat}呗。",
    "你顺手把{beat}处理下。",
    "麻烦你盯一下{beat}。",
    "能不能先把{beat}排上？",
]
REFUSE = [
    "这会儿真不行，过会儿再说。",
    "别，我这边腾不出手。",
    "先不用，放着就行。",
    "不是我不想，是现在没空。",
]
CONFIRM = [
    "行，那就这么着。",
    "嗯嗯，我记下了。",
    "好，我按这个来。",
    "成，回头我跟你说。",
]
CORRECT = [
    "不对，我是说{beat}，不是那个。",
    "等下，我讲错了，应该是{beat}。",
    "不是那个意思，我是想先看{beat}。",
]
COMPLAIN = [
    "这{beat}也太磨叽了吧。",
    "真的无语，{beat}又出状况。",
    "我都说了两遍了，{beat}还没好。",
]
COMFORT = [
    "没事没事，不急。",
    "别皱着了，咱们慢慢弄。",
    "我懂，先歇一会儿。",
]
TEASE = [
    "你还挺会拖的啊。",
    "得得得，又来了。",
    "行吧你赢了。",
]
PLAN = [
    "那咱们明天先把{beat}排上。",
    "晚上回来再说{beat}。",
    "我先去，你随后。",
]
REMIND = [
    "提醒你一声，{beat}别忘了。",
    "回头看手机，我把{beat}发你了。",
    "出门前再确认一眼{beat}。",
]

ACT_LINES = {
    "ask": ASK,
    "answer": ANSWER,
    "request": REQUEST,
    "refuse": REFUSE,
    "confirm": CONFIRM,
    "correct": CORRECT,
    "complain": COMPLAIN,
    "comfort": COMFORT,
    "tease": TEASE,
    "plan": PLAN,
    "remind": REMIND,
}

BACKCHANNELS = ["嗯。", "行。", "好。", "哦。", "成。", "得。", "啊？", "啥？", "真的？", "还行吧。"]
FILLERS = ["那个", "就是", "然后", "反正", "其实", "你看", "咱们"]
NUM_DATE = ["后天", "三点半", "二十块", "两斤", "三百米", "周五", "半小时", "三回"]
CODE_MIX = ["OK", "wait", "app", "busy", "done"]
EMOTION = {
    "happy": ["太好了", "可以可以", "成啊"],
    "annoyed": ["烦死了", "真的假的", "无语"],
    "worried": ["有点慌", "怕来不及", "我有点不踏实"],
    "tired": ["好累", "不想动了", "先躺会儿"],
    "hurried": ["快点", "来不及了", "赶紧"],
    "neutral": ["行", "嗯", "那行"],
}
REGION = {
    "none": [],
    "north": ["咋", "得劲", "整"],
    "south": ["冇", "噻", "咯"],
    "wu_light": ["阿拉", "伐", "老"],
}


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def terms_hash(contract: dict | None = None) -> str:
    spec = contract or load_contract()
    text = str((spec.get("terms") or {}).get("text") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_sha256() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def axes_of(contract: dict | None = None) -> dict:
    spec = contract or load_contract()
    return dict(spec.get("axes") or {})


def sanitize_spoken_text(text: str) -> str:
    out = EMOJI_RE.sub("", str(text or ""))
    out = out.translate(UNK_CHAR_REWRITE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_dotenv_quiet() -> None:
    root = ROOT.parent
    for path in (root / ".env", ROOT / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def resolve_qwen_endpoint() -> dict:
    load_dotenv_quiet()
    key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or FALLBACK_BASE
    ).rstrip("/")
    return {"api_key": key, "base_url": base}


def frame_id(index: int, *, prefix: str = "csynth-v1") -> str:
    return f"{prefix}-{int(index):08d}"


def frame_salt(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{int(index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_000_000


def iter_frames(
    n: int,
    *,
    seed: int = 0,
    axes: dict | None = None,
    start: int = 0,
    id_prefix: str = "csynth-v1",
) -> Iterator[dict]:
    ax = axes or axes_of()
    scenes = list(ax["scenes"])
    relations = list(ax["relations"])
    acts = list(ax["speech_acts"])
    styles = list(ax["styles"])
    moods = list(ax["moods"])
    regions = list(ax["region_light"])
    turns = list(ax["turns"])
    for i in range(start, start + n):
        scene = scenes[i % len(scenes)]
        beats = SCENE_BEATS[scene]
        beat = beats[(i // len(scenes)) % len(beats)]
        style = styles[i % len(styles)]
        extra = styles[(i * 3) % len(styles)]
        if extra == style:
            extra = styles[(i * 3 + 1) % len(styles)]
        yield {
            "frame_id": frame_id(i, prefix=id_prefix),
            "index": i,
            "scene": scene,
            "beat": beat,
            "relation": relations[(i * 5) % len(relations)],
            "speech_act": acts[(i * 7) % len(acts)],
            "styles": [style, extra],
            "mood": moods[(i * 11) % len(moods)],
            "region_light": regions[(i * 13) % len(regions)],
            "n_turns": turns[(i * 17) % len(turns)],
            "seed": seed,
            "salt": frame_salt(seed, i),
        }


def _fill(template: str, beat: str) -> str:
    return template.replace("{beat}", beat)


def _apply_style(text: str, styles: list[str], mood: str, region: str, rng: random.Random) -> str:
    out = text
    if "filler" in styles and rng.random() < 0.8:
        out = rng.choice(FILLERS) + "，" + out
    if "ellipsis" in styles:
        out = out.replace("。", "…") if rng.random() < 0.5 else out.rstrip("。") + "…"
    if "repair" in styles and rng.random() < 0.45:
        out = "不对，" + out
    if "interrupt" in styles and rng.random() < 0.4:
        out = out[: max(2, len(out) // 2)] + "—"
    if "negation" in styles and rng.random() < 0.35:
        out = "先别急，" + out
    if "deixis" in styles and rng.random() < 0.5:
        out = "那个，" + out
    if "number_date_unit" in styles and rng.random() < 0.6:
        out = out.rstrip("。…") + "，" + rng.choice(NUM_DATE) + "。"
    if "code_mix" in styles and rng.random() < 0.35:
        out = out.rstrip("。") + " " + rng.choice(CODE_MIX) + "。"
    emo = EMOTION.get(mood) or EMOTION["neutral"]
    if "emotion" in styles or mood != "neutral":
        if rng.random() < 0.55:
            out = rng.choice(emo) + "，" + out
    marks = REGION.get(region) or []
    if marks and rng.random() < 0.35:
        out = out.rstrip("。") + rng.choice(marks) + "。"
    return out


def render_offline(frame: dict) -> dict:
    rng = random.Random((frame.get("index") or 0) * 1009 + int(frame.get("salt") or 0))
    beat = str(frame.get("beat") or "那个")
    act = str(frame.get("speech_act") or "ask")
    styles = list(frame.get("styles") or ["filler"])
    n_turns = int(frame.get("n_turns") or 4)
    mood = str(frame.get("mood") or "neutral")
    region = str(frame.get("region_light") or "none")
    lead = _fill(rng.choice(ACT_LINES.get(act) or ASK), beat)
    turns = [{"speaker": "甲", "text": _apply_style(lead, styles, mood, region, rng)}]
    for t in range(1, n_turns):
        speaker = "乙" if t % 2 else "甲"
        if "short_reply" in styles and t % 2 == 1:
            text = rng.choice(BACKCHANNELS)
        else:
            pool = ACT_LINES[list(ACT_LINES.keys())[(t + hash(act)) % len(ACT_LINES)]]
            text = _fill(rng.choice(pool), beat)
        if t == n_turns - 1 and rng.random() < 0.5:
            text = rng.choice(CONFIRM)
        turns.append({"speaker": speaker, "text": _apply_style(text, styles, mood, region, rng)})
    if not any("吗" in x["text"] or "呢" in x["text"] or "啥" in x["text"] for x in turns):
        turns[0]["text"] = turns[0]["text"].rstrip("。…") + "，咋样了呢？"
    text = "\n".join(f"{row['speaker']}：{row['text']}" for row in turns)
    return {"turns": turns, "text": text}


def system_prompt(prompt_version: str) -> str:
    return (
        "写自然中文日常口语对话，按语义框推进，像真人聊天。"
        "必须体现 styles（省略/改口/打断/短答/口头禅/数字日期单位/中英夹杂/情绪/否定/指代）。"
        "只用大陆通行简体常用字，用哎/啊/哦/呀/恩代替诶嗯呃噢，不要emoji和生僻方言字。"
        f"pv={prompt_version}。"
        '只输出JSON：{"turns":[{"speaker":"甲或乙","text":"..."}]}。'
    )


def user_prompt_for_frame(frame: dict, *, min_turns: int = 0) -> str:
    n_turns = max(int(frame.get("n_turns") or 2), int(min_turns or 0))
    payload = {
        "scene": frame.get("scene"),
        "beat": frame.get("beat"),
        "relation": frame.get("relation"),
        "speech_act": frame.get("speech_act"),
        "styles": frame.get("styles"),
        "mood": frame.get("mood"),
        "region_light": frame.get("region_light"),
        "n_turns": n_turns,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_turns_payload(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    turns = obj.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    cleaned = []
    for row in turns:
        if not isinstance(row, dict):
            continue
        speaker = str(row.get("speaker") or "").strip()
        utt = str(row.get("text") or "").strip()
        if speaker in {"A", "B"}:
            speaker = "甲" if speaker == "A" else "乙"
        if speaker not in {"甲", "乙"} or not utt:
            continue
        cleaned.append({"speaker": speaker, "text": utt})
    if len(cleaned) < 2:
        return None
    rendered = "\n".join(f"{r['speaker']}：{r['text']}" for r in cleaned)
    return {"turns": cleaned, "text": rendered}


class QwenHttpError(RuntimeError):
    def __init__(self, code: int, retry_after: float | None = None, body: str = ""):
        self.code = int(code)
        self.retry_after = retry_after
        self.body = body
        super().__init__(f"qwen_http_{self.code}")


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    raw = ""
    try:
        raw = str((exc.headers or {}).get("Retry-After") or "")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _usage_tokens(usage: dict) -> tuple[int, int]:
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


def qwen_complete(frame: dict, *, contract: dict, timeout_s: int = 60) -> dict:
    spec = contract
    gen = spec["generator"]
    sampling = spec["sampling"]
    endpoint = resolve_qwen_endpoint()
    if not endpoint["api_key"]:
        raise RuntimeError("missing QWEN_API_KEY / DASHSCOPE_API_KEY")
    want_snap = str(gen["frozen_snapshot"])
    body = {
        "model": want_snap,
        "messages": [
            {"role": "system", "content": system_prompt(spec["prompt_version"])},
            {"role": "user", "content": user_prompt_for_frame(frame, min_turns=6)},
        ],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "max_tokens": sampling["max_tokens"],
        "enable_thinking": False,
        "extra_body": {"enable_thinking": False},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint["base_url"] + "/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + endpoint["api_key"],
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise QwenHttpError(exc.code, _retry_after_seconds(exc), raw) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"qwen_timeout_or_network:{exc}") from exc
    payload = json.loads(raw)
    got_model = str(payload.get("model") or "")
    if got_model and want_snap not in got_model and got_model != want_snap:
        raise RuntimeError(f"snapshot_mismatch:{got_model}")
    content = (
        (((payload.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    )
    usage = payload.get("usage") or {}
    prompt_tokens, completion_tokens = _usage_tokens(usage)
    usage = {
        **usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    parsed = parse_turns_payload(str(content))
    if parsed is None:
        raise RuntimeError("qwen_invalid_json")
    return {
        "text": parsed["text"],
        "turns": parsed["turns"],
        "raw_content": str(content),
        "usage": usage,
        "http_status": status,
        "model_snapshot": got_model or want_snap,
        "request_sha256": sha256_obj(
            {k: v for k, v in body.items() if k != "messages"} | {"frame_id": frame["frame_id"]}
        ),
        "response_sha256": sha256_text(str(content)),
    }


def generate_one(
    frame: dict,
    *,
    generator: str,
    contract: dict,
    retries: int = 5,
) -> dict:
    if generator == "offline-frame-renderer":
        rendered = render_offline(frame)
        text = sanitize_spoken_text(rendered["text"])
        turns = [{"speaker": r["speaker"], "text": sanitize_spoken_text(r["text"])} for r in rendered["turns"]]
        return {
            "ok": True,
            "text": text,
            "turns": turns,
            "request_sha256": sha256_obj(frame),
            "response_sha256": sha256_text(text),
            "usage": {"completion_tokens": 0, "prompt_tokens": 0},
            "model_snapshot": "offline-v1",
            "retries": 0,
        }
    last = "qwen_error"
    delay = 0.5
    for attempt in range(retries):
        try:
            got = qwen_complete(frame, contract=contract)
            got["text"] = sanitize_spoken_text(str(got.get("text") or ""))
            got["turns"] = [
                {"speaker": r.get("speaker"), "text": sanitize_spoken_text(str(r.get("text") or ""))}
                for r in (got.get("turns") or [])
            ]
            got["text"] = "\n".join(f"{r['speaker']}：{r['text']}" for r in got["turns"] if r.get("speaker") and r.get("text"))
            got["ok"] = True
            got["retries"] = attempt
            return got
        except QwenHttpError as exc:
            last = str(exc)
            if exc.code in {401, 403}:
                return {"ok": False, "error": last, "retries": attempt + 1}
            wait = exc.retry_after if exc.code == 429 and exc.retry_after is not None else delay
            time.sleep(min(45.0, float(wait)))
            delay = min(30.0, delay * 2)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(min(30.0, delay))
            delay = min(30.0, delay * 2)
    return {"ok": False, "error": last, "retries": retries}


def filter_reasons(text: str, *, leaks: list[str], pii_fn) -> list[str]:
    reasons: list[str] = []
    if not text or len(text) < 24:
        reasons.append("too_short")
    if text.count("甲：") + text.count("乙：") < 2:
        reasons.append("illegal_structure")
    if TOOLISH_RE.search(text):
        reasons.append("tool_or_eval_marker")
    hit = pii_fn(text)
    if hit == "pii":
        reasons.append("pii")
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    if cjk < 12:
        reasons.append("low_cjk")
    blob = text
    for leak in leaks:
        if leak and len(leak) >= 12 and leak in blob:
            reasons.append("eval_leak")
            break
    return reasons


def spend_cny(*, prompt_tokens: int, completion_tokens: int, contract: dict) -> float:
    budget = contract.get("budget") or {}
    inp = float(budget.get("input_cny_per_million") or 0.8)
    out = float(budget.get("output_cny_per_million_non_thinking") or 2.0)
    return (int(prompt_tokens) / 1_000_000.0) * inp + (int(completion_tokens) / 1_000_000.0) * out


def estimate_cny(completion_tokens: int, contract: dict) -> float:
    return spend_cny(prompt_tokens=0, completion_tokens=completion_tokens, contract=contract)


def unique_by_first_frame(rows: list[dict]) -> dict:
    seen: set[str] = set()
    n_dup = 0
    unique_train = 0
    unique_valid = 0
    exposure_train = 0
    exposure_valid = 0
    n_unique_ids = 0
    for row in rows:
        frame = row.get("frame") or {}
        fid = str(row.get("doc_id") or frame.get("frame_id") or "")
        n_tok = int(row.get("n_tokens") or 0)
        split = str(row.get("split") or "train")
        if split == "train":
            exposure_train += n_tok
        else:
            exposure_valid += n_tok
        if not fid or fid in seen:
            if fid:
                n_dup += 1
            continue
        seen.add(fid)
        n_unique_ids += 1
        if split == "train":
            unique_train += n_tok
        else:
            unique_valid += n_tok
    return {
        "n_docs": len(rows),
        "n_unique_frame_ids": n_unique_ids,
        "n_duplicate_frame_ids": n_dup,
        "unique_train_tokens": unique_train,
        "unique_valid_tokens": unique_valid,
        "exposure_train_tokens": exposure_train,
        "exposure_valid_tokens": exposure_valid,
    }


def provenance_row(
    *,
    frame: dict,
    text: str,
    generator: str,
    contract: dict,
    request_sha256: str,
    response_sha256: str,
    usage: dict,
    filter_ok: bool,
    reasons: list[str],
    n_tokens: int,
    split: str,
) -> dict:
    return {
        "doc_id": frame["frame_id"],
        "frame": frame,
        "text": text,
        "prompt_version": contract["prompt_version"],
        "generator": generator,
        "model_snapshot": (
            "offline-v1"
            if generator == "offline-frame-renderer"
            else str(contract["generator"].get("frozen_snapshot") or generator)
        ),
        "sampling": {"deterministic": True} if generator == "offline-frame-renderer" else contract["sampling"],
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "usage": usage,
        "filter": {"ok": filter_ok, "reasons": reasons},
        "n_tokens": n_tokens,
        "split": split,
        "terms_hash": terms_hash(contract),
        "contract_sha256": contract_sha256(),
    }
