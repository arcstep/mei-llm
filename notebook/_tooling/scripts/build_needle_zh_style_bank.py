#!/usr/bin/env python3
"""Build a redacted style bank from CrossWOZ/KdConv *train* only.

Cards never carry product gold. Original utterances are hashed for near-dup
filters; teachers see tags + a short redacted snippet, not dialog acts or KG.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from repo_paths import CORPORA_ROOT, ROOT

from needle_home_sft_lib import PII_RE, dump_jsonl, load_jsonl, sha256_text

BANK_ROOT = CORPORA_ROOT / "sft-style-v0"
PROBE_CROSSWOZ = ROOT / "notebook/archive/corpus/_probe/samples/crosswoz-train/train.json"
PROBE_KDCONV = ROOT / "notebook/archive/corpus/_probe/samples/kdconv-train.jsonl"
CROSSWOZ_ZIP = (
    "https://raw.githubusercontent.com/thu-coai/CrossWOZ/master/data/train.json.zip"
)
PHONE_RE = re.compile(r"(?:\+?86[-\s]?)?1[3-9]\d{9}|\d{3,4}-\d{7,8}")
PAREN_RE = re.compile(r"（[^）]{1,40}）|\([^)]{1,40}\)")


def redact(text: str) -> str:
    t = PII_RE.sub("[PII]", text)
    t = PHONE_RE.sub("[PHONE]", t)
    t = PAREN_RE.sub("", t)
    t = re.sub(r"worker[_\-]?id\s*[:=]\s*\S+", "[WORKER]", t, flags=re.I)
    t = re.sub(r"\d{5,}", "[N]", t)
    return " ".join(t.split()).strip()


def tag_text(text: str) -> list[str]:
    tags: list[str] = []
    if any(x in text for x in ("请", "帮我", "麻烦")):
        tags.append("command")
    if any(x in text for x in ("不要", "不是", "别", "不用", "取消")):
        tags.append("negation")
    if "…" in text or text.endswith("呢") or text.endswith("吧") or "那个" in text:
        tags.append("ellipsis")
    if any(x in text for x in ("它", "那个", "刚才", "上面")):
        tags.append("deixis")
    if any(x in text for x in ("改成", "换成", "还是")):
        tags.append("correction")
    if len(text) < 12:
        tags.append("ellipsis")
    if "谢谢" in text or "麻烦" in text:
        tags.append("formal")
    if not tags:
        tags.append("colloquial")
    out: list[str] = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return out[:4]


def iter_crosswoz_user(obj) -> list[str]:
    out: list[str] = []
    dialogs = list(obj.values()) if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    for dlg in dialogs:
        if not isinstance(dlg, dict):
            continue
        for msg in dlg.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role") or "").lower() != "usr":
                continue
            text = str(msg.get("content") or "").strip()
            if text:
                out.append(text)
    return out


def iter_kdconv_turns(path: Path) -> list[str]:
    out: list[str] = []
    for row in load_jsonl(path):
        domain = str(row.get("domain") or "")
        if domain in {"travel", "attraction", "景点"}:
            continue
        text = str(row.get("text") or "")
        turns = [t.strip() for t in text.split("\n") if t.strip()]
        for i, turn in enumerate(turns):
            if i % 2 != 0:
                continue
            if not (4 <= len(turn) <= 40):
                continue
            if PHONE_RE.search(turn):
                continue
            out.append(turn)
    return out


def locate_crosswoz(raw_dir: Path) -> Path:
    dest = raw_dir / "crosswoz-train.json"
    if dest.is_file():
        return dest
    if PROBE_CROSSWOZ.is_file():
        return PROBE_CROSSWOZ
    zip_path = raw_dir / "train.json.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"download {CROSSWOZ_ZIP}", flush=True)
    urllib.request.urlretrieve(CROSSWOZ_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        dest.write_bytes(zf.read(name))
    return dest


def locate_kdconv(raw_dir: Path) -> Path:
    dest = raw_dir / "kdconv-train.jsonl"
    if dest.is_file():
        return dest
    if PROBE_KDCONV.is_file():
        return PROBE_KDCONV
    raise FileNotFoundError("KdConv train sample missing; expected probe cache or raw copy")


def to_cards(source: str, texts: list[str], rng: random.Random, cap: int) -> tuple[list[dict], list[str]]:
    hashes: list[str] = []
    cards: list[dict] = []
    seen: set[str] = set()
    rng.shuffle(texts)
    for i, raw in enumerate(texts):
        red = redact(raw)
        if not red or len(red) < 4:
            continue
        h = sha256_text(red)
        if h in seen:
            continue
        seen.add(h)
        hashes.append(h)
        cards.append(
            {
                "style_id": f"{source}-{i:06d}",
                "source": source,
                "source_role": "style-only",
                "split": "train",
                "tags": tag_text(red),
                "abstract": "口吻参考，不含任务 gold、POI 或 KG",
                "example_redacted": red[:32],
                "n_chars": len(red),
                "sha256": h,
            }
        )
        if len(cards) >= cap:
            break
    return cards, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    raw_dir = BANK_ROOT / "raw"
    cards_dir = BANK_ROOT / "cards"
    hash_dir = BANK_ROOT / "hashes"
    cards_dir.mkdir(parents=True, exist_ok=True)
    hash_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cw_path = locate_crosswoz(raw_dir)
    kd_path = locate_kdconv(raw_dir)
    cw_obj = json.loads(cw_path.read_text(encoding="utf-8"))
    cw_texts = iter_crosswoz_user(cw_obj)
    kd_texts = iter_kdconv_turns(kd_path)
    cw_cards, cw_hashes = to_cards("crosswoz-train", cw_texts, rng, args.cap)
    kd_cards, kd_hashes = to_cards("kdconv-train", kd_texts, rng, max(1, args.cap // 2))
    dump_jsonl(cards_dir / "crosswoz-style.jsonl", cw_cards)
    dump_jsonl(cards_dir / "kdconv-style.jsonl", kd_cards)
    (hash_dir / "utterance.sha256").write_text(
        "\n".join(cw_hashes + kd_hashes) + "\n", encoding="utf-8"
    )
    man = {
        "id": "sft-style-v0",
        "role": "style-only",
        "gold": None,
        "product_rows": 0,
        "crosswoz_n": len(cw_cards),
        "kdconv_n": len(kd_cards),
        "n_hashes": len(cw_hashes) + len(kd_hashes),
        "license": {
            "crosswoz": "Apache-2.0; train split only; dev/test never read",
            "kdconv": "repo Apache-2.0; drop travel/phone; style only; no KG gold",
        },
        "sources": {
            "crosswoz": str(cw_path.relative_to(ROOT)) if str(cw_path).startswith(str(ROOT)) else str(cw_path),
            "kdconv": str(kd_path.relative_to(ROOT)) if str(kd_path).startswith(str(ROOT)) else str(kd_path),
        },
    }
    (BANK_ROOT / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(man, ensure_ascii=False, indent=2))
    if not cw_cards:
        print("no CrossWOZ style cards", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
