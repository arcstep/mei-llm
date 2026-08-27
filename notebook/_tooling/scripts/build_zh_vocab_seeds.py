#!/usr/bin/env python3
"""Build committable zh-vocab seed lists (hanzi, cities, rooms, reserved)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

from repo_paths import CORPUS_ZH_VOCAB

HANZI_L1 = (
    "https://raw.githubusercontent.com/shengdoushi/"
    "common-standard-chinese-characters-table/master/level-1.txt"
)
HANZI_L2 = (
    "https://raw.githubusercontent.com/shengdoushi/"
    "common-standard-chinese-characters-table/master/level-2.txt"
)
THUOCL_DIMING = (
    "https://raw.githubusercontent.com/thunlp/THUOCL/master/data/THUOCL_diming.txt"
)
CITIES_JSON = (
    "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/"
    "master/dist/cities.json"
)
PROVINCES_JSON = (
    "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/"
    "master/dist/provinces.json"
)

ROOMS = [
    "客厅",
    "卧室",
    "书房",
    "厨房",
    "卫生间",
    "阳台",
    "餐厅",
    "主卧",
    "次卧",
    "玄关",
    "儿童房",
    "办公室",
    "会议室",
    "走廊",
    "浴室",
    "厕所",
]

RESERVED = [
    "get_weather",
    "set_lights",
    "invoice",
    "city",
    "room",
    "vendor",
    "total",
    "due_date",
    "brightness",
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
]


def fetch(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "mei-llm-zh-vocab/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8-sig")


def write_lines(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq: list[str] = []
    seen: set[str] = set()
    for raw in items:
        s = raw.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    path.write_text("\n".join(uniq) + "\n", encoding="utf-8")
    print(f"wrote {len(uniq):>6} → {path}")


def hanzi_chars(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) == 1 and "\u4e00" <= line <= "\u9fff":
            out.append(line)
    return out


def city_aliases(name: str) -> list[str]:
    name = name.strip()
    if not name or name in {"市辖区", "县", "省直辖县级行政区划"}:
        return []
    aliases = [name]
    for suf in ("市", "地区", "盟", "自治州"):
        if name.endswith(suf) and len(name) > len(suf):
            aliases.append(name[: -len(suf)])
            break
    if name.endswith("特别行政区") and len(name) > 5:
        aliases.append(name[: -len("特别行政区")])
    return aliases


def cities_from_admin() -> list[str]:
    names: list[str] = []
    provinces = json.loads(fetch(PROVINCES_JSON))
    cities = json.loads(fetch(CITIES_JSON))
    for row in provinces:
        names.extend(city_aliases(str(row.get("name") or "")))
    for row in cities:
        names.extend(city_aliases(str(row.get("name") or "")))
    return names


def cities_from_thuocl(limit: int = 800) -> list[str]:
    text = fetch(THUOCL_DIMING)
    scored: list[tuple[int, str]] = []
    for line in text.splitlines():
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 2:
            continue
        word, freq_s = parts[0], parts[-1]
        if not word.endswith("市") or len(word) < 2:
            continue
        try:
            freq = int(freq_s)
        except ValueError:
            continue
        scored.append((freq, word))
        scored.append((freq, word[:-1]))
    scored.sort(key=lambda x: -x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, w in scored:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_ZH_VOCAB / "seeds")
    args = ap.parse_args()
    seeds = args.out

    try:
        l1 = hanzi_chars(fetch(HANZI_L1))
        l2 = hanzi_chars(fetch(HANZI_L2))
    except Exception as exc:  # noqa: BLE001
        print(f"hanzi fetch failed: {exc}", file=sys.stderr)
        return 1
    if len(l1) < 3000 or len(l2) < 2500:
        print(f"unexpected hanzi sizes l1={len(l1)} l2={len(l2)}", file=sys.stderr)
        return 1

    cities: list[str] = []
    try:
        cities.extend(cities_from_admin())
    except Exception as exc:  # noqa: BLE001
        print(f"admin cities fetch failed: {exc}", file=sys.stderr)
    try:
        cities.extend(cities_from_thuocl())
    except Exception as exc:  # noqa: BLE001
        print(f"THUOCL fetch failed: {exc}", file=sys.stderr)
    if len(cities) < 50:
        print("too few cities", file=sys.stderr)
        return 1

    write_lines(seeds / "hanzi-level1.txt", l1)
    write_lines(seeds / "hanzi-level2.txt", l2)
    write_lines(seeds / "cities-zh.txt", cities)
    write_lines(seeds / "rooms-zh.txt", ROOMS)
    write_lines(seeds / "reserved-tokens.txt", RESERVED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
