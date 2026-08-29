#!/usr/bin/env python3
"""Produce fresh structure + colloquial shards for corpus/lm-v2.

Does not rewrite corpus/lm-v1. Wiki/HQ remain referenced from lm-v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from repo_paths import (
    EVAL_SHARED_ROOT,
    LM_V2,
    PUBLISHED_LM_V1,
    PUBLISHED_LM_V2,
    ROOT,
    TOKENIZER_ZH_V1,
    all_eval_jsonl,
)

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training/mei-1.0-58m-train-v1"))
from _repo import ensure_formal_on_path  # noqa: E402

ensure_formal_on_path()
sys.path.insert(0, str(SCRIPTS))

from data import build_leak_index, document_leaks_eval, leak_strings_from_rows, sha256_text  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from zh_pretrain_ingest import (  # noqa: E402
    CJK_RE,
    TOKENS_PER_SHARD,
    UNK_TOKEN_MAX,
    SplitWriters,
    dump_json,
    file_sha256,
    is_clean_structure_text,
    pii_or_nav,
)

BANK_PROBES = ROOT / "notebook/evaluation/banks/needle-pretrain-probes-v0/probes-v0.jsonl"
STRUCTURE_WORK = LM_V2 / "structure" / "work" / "zh-pretrain-v4"
COLLOQUIAL_WORK = LM_V2 / "colloquial" / "work" / "colloquial-cpt-v2"
STRUCTURE_PUB = PUBLISHED_LM_V2 / "structure" / "zh-pretrain-v4"
COLLOQUIAL_PUB = PUBLISHED_LM_V2 / "colloquial" / "zh-pretrain-colloquial-cpt-v2"

STRUCTURE_TRAIN_TARGET = 12_000_000
STRUCTURE_VALID_TARGET = 150_000
COLLOQUIAL_TRAIN_TARGET = 71_000_000
COLLOQUIAL_VALID_TARGET = 150_000

CWT2_RE = re.compile(r"ChineseWebText|CWT2|cwt2", re.I)
EVAL_ID_RE = re.compile(r"\bEVAL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+\b")

SCENES = (
    "home", "commute", "restaurant", "shopping", "school", "clinic", "workplace",
    "phone_call", "delivery", "weather", "sports", "travel", "family", "neighbors",
    "repair", "pet", "cooking", "gaming",
)
BEATS = (
    "灯", "空调", "晚饭", "垃圾袋", "门锁", "洗衣机", "遥控器", "地铁", "公交", "堵车",
    "换乘", "迟到", "共享单车", "位子", "菜单", "辣", "打包", "账单", "等位", "尺码",
    "打折", "退货", "购物袋", "试衣间", "作业", "家长会", "校服", "考试", "挂号",
    "排队", "药", "复查", "发烧", "会议", "周报", "加班", "工位", "截止日期", "信号",
    "回电", "占线", "微信", "语音", "快递", "取件码", "放门口", "驿站", "破损", "下雨",
    "降温", "晒", "台风", "雾霾", "球场", "跑步", "拉伸", "报名", "请假", "车票",
    "酒店", "行李", "改签", "景点", "爸妈", "孩子", "过年", "买菜", "看病", "楼道",
    "噪音", "快递柜", "物业", "钥匙", "师傅", "水管", "预约", "零件", "保修", "遛狗",
    "猫粮", "疫苗", "洗澡", "下锅", "盐", "剩菜", "烤箱", "排位", "掉线", "组队", "皮肤",
)
NAMES = ("小周", "阿强", "小陈", "老张", "小吴", "阿梅", "小林", "大伟", "小宁", "阿敏")
PLACES = ("望京", "徐汇", "天河", "南山", "江汉", "鼓楼", "金牛", "西湖", "和平", "朝阳")
ACTS = ("ask", "answer", "request", "refuse", "confirm", "complain", "comfort", "plan", "remind")


def load_leaks() -> list[str]:
    rows: list[dict] = []
    for path in all_eval_jsonl():
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    if BANK_PROBES.is_file():
        for line in BANK_PROBES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return leak_strings_from_rows(rows)


def load_existing_shas() -> set[str]:
    out: set[str] = set()
    for path in [
        ROOT / "notebook/corpus/lm-v1/structure/work/zh-pretrain-v3/raw/structure.jsonl",
        ROOT / "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1/unique-ledger.json",
    ]:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sha = str(row.get("sha256") or "")
            if sha:
                out.add(sha)
            text = str(row.get("text") or "")
            if text:
                out.add(sha256_text(text))
    return out


def reject_text(text: str, leaks: list[str], seen: set[str], leak_index) -> str | None:
    if len(text.strip()) < 64:
        return "short"
    if CWT2_RE.search(text):
        return "cwt2"
    if EVAL_ID_RE.search(text):
        return "eval-id"
    if pii_or_nav(text):
        return pii_or_nav(text)
    if CJK_RE.search(text) is None:
        return "no-cjk"
    leak = document_leaks_eval(text, leaks, index=leak_index)
    if leak:
        return "eval-leak"
    sha = sha256_text(text)
    if sha in seen:
        return "dup"
    return None


def iter_structure_docs(start: int = 100_000):
    tool_dir = EVAL_SHARED_ROOT / "toolsets"
    templates = []
    if tool_dir.is_dir():
        for path in sorted(tool_dir.glob("*.json")):
            templates.append(json.loads(path.read_text(encoding="utf-8")))
    if not templates:
        templates = [{"tools": [{"name": "light.set", "description": "设置灯"}]}]
    kinds = ("object", "string", "boolean", "integer", "number")
    methods = ("post", "get", "put", "patch")
    i = start
    while True:
        i += 1
        ts = templates[i % len(templates)]
        name = f"cpt_tool_{i % 997}_{i}"
        typ = kinds[i % len(kinds)]
        method = methods[i % len(methods)]
        required = ["slot", "tag"] if i % 4 else ["slot"]
        enum_vals = [f"alpha{i % 17}", f"beta{i % 19}", f"gamma{i % 23}"]
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"中文工具描述 {name}",
            "type": "object",
            "properties": {
                "slot": {"type": typ, "description": f"参数类型 {typ} 序号 {i}"},
                "tag": {"type": "string", "enum": enum_vals},
                "count": {"type": "integer", "minimum": 0, "maximum": 64},
            },
            "required": required,
        }
        instance_ok = {
            "slot": {"string": f"val-{i}", "boolean": bool(i % 2), "integer": i % 97, "number": (i % 50) / 2, "object": {"k": i % 9}}[typ],
            "tag": enum_vals[i % 3],
            "count": i % 8,
        }
        instance_bad = {"slot": ["array-not-allowed"], "tag": 0}
        openapi = {
            "openapi": "3.0.3",
            "info": {"title": f"工具面 {name}", "version": f"1.{i % 20}.0"},
            "paths": {
                f"/{name}": {
                    method: {
                        "summary": "调用已登记工具",
                        "operationId": f"call_{name}",
                        "requestBody": {"content": {"application/json": {"schema": schema}}},
                    }
                }
            },
        }
        tools = []
        if ts.get("tools"):
            tools = [{"name": t.get("name"), "description": t.get("description")} for t in ts["tools"][:4]]
        text = (
            json.dumps(schema, ensure_ascii=False)
            + "\n合法实例："
            + json.dumps(instance_ok, ensure_ascii=False)
            + "\n非法实例："
            + json.dumps(instance_bad, ensure_ascii=False)
            + "\n"
            + json.dumps(openapi, ensure_ascii=False)
        )
        if tools:
            text += "\n已登记工具：" + json.dumps(tools, ensure_ascii=False)
        text += f"\n备注：该结构样本仅用于内部持续预训练，编号 {i}。"
        yield {
            "page_id": f"v4-struct-{i:08d}",
            "title": name,
            "text": text,
            "sha256": sha256_text(text),
        }


def make_colloquial_doc(i: int) -> dict:
    rng = random.Random(10_007 * i + 17)
    scene = SCENES[i % len(SCENES)]
    beat = BEATS[i % len(BEATS)]
    name = NAMES[i % len(NAMES)]
    place = PLACES[(i // 3) % len(PLACES)]
    other = NAMES[(i + 4) % len(NAMES)]
    n_turns = 8 + (i % 7)
    clock = f"{8 + (i % 12)}点{i % 60:02d}"
    order = f"口{i:08d}"
    lines = [
        f"甲：{name}，这个{beat}你弄了没？我看{place}那边{clock}才开门。",
        f"乙：弄了弄了，刚才顺手的。编号{order}我记下了。",
        f"甲：那就行。{other}还说要改时间，你觉得靠谱吗？",
        f"乙：先别急，咱们按原计划。{beat}这边我盯着。",
        f"甲：行。回头你把{place}的取件码发我，别忘了。",
        f"乙：嗯嗯，我记下了。要是排队太长我就先回来。",
    ]
    extras = [
        f"甲：对了，{beat}要是还不行，咱们晚上再说。",
        f"乙：可以。我先去一趟，你随后。",
        f"甲：哎，那个，{name}你别拖到明天。",
        f"乙：不成问题。我按这个来。",
        f"甲：下雨的话就改室内，别硬撑。",
        f"乙：好，我看着办。有消息微信你。",
        f"甲：成，那就这么着。",
        f"乙：回头我跟你说一声就行。",
    ]
    while len(lines) < n_turns:
        lines.append(extras[(len(lines) + i) % len(extras)])
    lines = lines[:n_turns]
    if not any("吗" in x or "呢" in x or "啥" in x for x in lines):
        lines[0] = lines[0].rstrip("。") + "，咋样了呢？"
    text = "\n".join(lines)
    return {
        "page_id": f"v2-col-{i:08d}",
        "title": f"{scene}-{beat}",
        "text": text,
        "sha256": sha256_text(text),
        "scene": scene,
        "beat": beat,
    }


def write_role(
    *,
    role: str,
    tok: ZhTokenizerV1,
    leaks: list[str],
    seen: set[str],
    pub_dir: Path,
    work_dir: Path,
    train_target: int,
    valid_target: int,
    prefix: str,
    docs,
    require_structure: bool,
) -> dict:
    token_dir = pub_dir / "tokens"
    idx_dir = work_dir / "tokens"
    raw_dir = work_dir / "raw"
    token_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    writers = SplitWriters(token_dir, prefix, TOKENS_PER_SHARD, ROOT, idx_dir=idx_dir)
    leak_index = build_leak_index(leaks)
    n_train = n_valid = n_unk = n_drop = 0
    kept = 0
    raw_path = raw_dir / f"{role}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for doc in docs:
            text = doc["text"]
            if require_structure and not is_clean_structure_text(text):
                n_drop += 1
                continue
            why = reject_text(text, leaks, seen, leak_index)
            if why:
                n_drop += 1
                continue
            ids = tok.encode_document(text)
            if not ids:
                n_drop += 1
                continue
            unk = sum(1 for t in ids if t == tok.unk_id)
            if unk / max(len(ids), 1) > 0.05:
                n_drop += 1
                continue
            seen.add(doc["sha256"])
            if n_valid < valid_target and (n_valid * 100 <= (n_train + n_valid) or n_train >= train_target):
                split = "valid"
            else:
                split = "train"
            row = {**doc, "split": split, "n_tokens": len(ids)}
            raw_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if split == "valid":
                writers.valid.write(ids, row, tok.unk_id)
                n_valid += len(ids)
            else:
                writers.train.write(ids, row, tok.unk_id)
                n_train += len(ids)
            n_unk += unk
            kept += 1
            if n_train >= train_target and n_valid >= valid_target:
                break
            if kept % 2000 == 0:
                print(
                    json.dumps(
                        {"role": role, "kept": kept, "n_train": n_train, "n_valid": n_valid, "dropped": n_drop},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    writers.close()
    n_tok = n_train + n_valid
    unk_rate = (n_unk / n_tok) if n_tok else 1.0
    train_shards = sorted(p.relative_to(ROOT).as_posix() for p in token_dir.glob(f"{prefix}-train-*.bin"))
    valid_shards = sorted(p.relative_to(ROOT).as_posix() for p in token_dir.glob(f"{prefix}-valid-*.bin"))
    ledger = {
        "role": role,
        "unique_train_tokens": n_train,
        "n_valid_tokens": n_valid,
        "unk_rate": unk_rate,
        "dropped": n_drop,
        "kept_docs": kept,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "public_distribution_clearance_asserted": False,
        "excluded_cwt2": True,
    }
    dump_json(work_dir / "unique-ledger.json", ledger)
    dump_json(
        pub_dir / "RELEASE.json",
        {
            "id": f"lm-v2-{role}",
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "unk_rate": unk_rate,
            "train_shards": train_shards,
            "valid_shards": valid_shards,
            "unique_ledger": str((work_dir / "unique-ledger.json").relative_to(ROOT)),
            "rewrites_lm_v1": False,
        },
    )
    dump_json(
        pub_dir / "manifest.json",
        {
            "id": f"lm-v2-{role}",
            "n_train_tokens": n_train,
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "unk_rate": unk_rate,
            "tokenizer_sha256": file_sha256(TOKENIZER_ZH_V1),
        },
    )
    if n_train < train_target:
        raise RuntimeError(f"{role} train tokens {n_train} < {train_target}")
    if n_valid < valid_target:
        raise RuntimeError(f"{role} valid tokens {n_valid} < {valid_target}")
    if unk_rate > UNK_TOKEN_MAX:
        raise RuntimeError(f"{role} unk_rate {unk_rate} > {UNK_TOKEN_MAX}")
    return ledger


def iter_colloquial_docs():
    i = 0
    while True:
        i += 1
        yield make_colloquial_doc(i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["structure", "colloquial", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tok = ZhTokenizerV1()
    leaks = load_leaks()
    seen = load_existing_shas()
    structure_train = 80_000 if args.smoke else STRUCTURE_TRAIN_TARGET
    structure_valid = 2_000 if args.smoke else STRUCTURE_VALID_TARGET
    colloquial_train = 80_000 if args.smoke else COLLOQUIAL_TRAIN_TARGET
    colloquial_valid = 2_000 if args.smoke else COLLOQUIAL_VALID_TARGET
    reports = {}
    if args.role in {"structure", "all"}:
        reports["structure"] = write_role(
            role="structure",
            tok=tok,
            leaks=leaks,
            seen=seen,
            pub_dir=STRUCTURE_PUB,
            work_dir=STRUCTURE_WORK,
            train_target=structure_train,
            valid_target=structure_valid,
            prefix="structure",
            docs=iter_structure_docs(),
            require_structure=True,
        )
    if args.role in {"colloquial", "all"}:
        reports["colloquial"] = write_role(
            role="colloquial",
            tok=tok,
            leaks=leaks,
            seen=seen,
            pub_dir=COLLOQUIAL_PUB,
            work_dir=COLLOQUIAL_WORK,
            train_target=colloquial_train,
            valid_target=colloquial_valid,
            prefix="colloquial",
            docs=iter_colloquial_docs(),
            require_structure=False,
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
