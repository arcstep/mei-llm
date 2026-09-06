"""Versioned deterministic normalizers. No LM, no guesswork."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

NORMALIZER_VERSION = "mei-normalizer-v1"
CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNITS = [("亿", 100_000_000), ("万", 10_000), ("千", 1000), ("百", 100), ("十", 10)]
CN_NUM_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
ARABIC_FLOAT_RE = re.compile(r"(?<![0-9])-?[0-9]+\.[0-9]+(?![0-9])")
ARABIC_INT_RE = re.compile(r"(?<![0-9.])-?[0-9]+(?![0-9.])")
BOOL_TRUE = ("打开", "开启", "开着", "设为开", "设为true", "true", "开灯", "打开开关")
BOOL_FALSE = ("关闭", "关掉", "关上", "设为关", "设为false", "false", "关灯", "关闭开关")
CURRENCY = (
    ("人民币", "CNY"),
    ("美金", "USD"),
    ("美元", "USD"),
    ("CNY", "CNY"),
    ("USD", "USD"),
)


@dataclass(frozen=True)
class NormHit:
    raw: str
    value: Any
    start: int
    end: int
    normalizer_id: str
    source_text: str


def parse_chinese_int(text: str) -> int | None:
    raw = (text or "").strip()
    if not raw or any(ch not in CN_DIGITS and ch not in "十百千万亿" for ch in raw):
        return None
    if raw in CN_DIGITS:
        return CN_DIGITS[raw]
    total = 0
    rest = raw
    for unit, mul in CN_UNITS:
        if unit not in rest:
            continue
        left, _, right = rest.partition(unit)
        if left == "":
            head = 1
        else:
            head = parse_chinese_int(left)
            if head is None:
                return None
        total += head * mul
        rest = right
    if rest:
        tail = parse_chinese_int(rest)
        if tail is None:
            return None
        total += tail
    return total


def find_numbers(text: str) -> list[NormHit]:
    hits: list[NormHit] = []
    seen: set[tuple[int, int, str]] = set()
    src = text or ""
    for m in ARABIC_FLOAT_RE.finditer(src):
        val = float(m.group())
        key = (m.start(), m.end(), "float")
        seen.add(key)
        hits.append(NormHit(m.group(), val, m.start(), m.end(), "arabic_float", src))
    for m in ARABIC_INT_RE.finditer(src):
        key = (m.start(), m.end(), "int")
        if any(a <= m.start() and m.end() <= b and k == "float" for a, b, k in seen):
            continue
        hits.append(NormHit(m.group(), int(m.group()), m.start(), m.end(), "arabic_int", src))
    for m in CN_NUM_RE.finditer(src):
        parsed = parse_chinese_int(m.group())
        if parsed is None:
            continue
        hits.append(NormHit(m.group(), parsed, m.start(), m.end(), "zh_int", src))
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    return hits


def find_booleans(text: str) -> list[NormHit]:
    src = text or ""
    hits: list[NormHit] = []
    for phrase in sorted(BOOL_TRUE, key=len, reverse=True):
        start = 0
        while True:
            i = src.find(phrase, start)
            if i < 0:
                break
            hits.append(NormHit(phrase, True, i, i + len(phrase), "zh_bool", src))
            start = i + len(phrase)
    for phrase in sorted(BOOL_FALSE, key=len, reverse=True):
        start = 0
        while True:
            i = src.find(phrase, start)
            if i < 0:
                break
            hits.append(NormHit(phrase, False, i, i + len(phrase), "zh_bool", src))
            start = i + len(phrase)
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    return _dedupe_spans(hits)


def find_currency(text: str) -> list[NormHit]:
    src = text or ""
    hits: list[NormHit] = []
    for phrase, canon in CURRENCY:
        start = 0
        while True:
            i = src.find(phrase, start)
            if i < 0:
                break
            hits.append(NormHit(phrase, canon, i, i + len(phrase), "currency", src))
            start = i + len(phrase)
    return _dedupe_spans(hits)


def _dedupe_spans(hits: list[NormHit]) -> list[NormHit]:
    kept: list[NormHit] = []
    occupied: list[tuple[int, int]] = []
    for hit in sorted(hits, key=lambda h: (h.start, -(h.end - h.start))):
        if any(not (hit.end <= a or hit.start >= b) for a, b in occupied):
            continue
        occupied.append((hit.start, hit.end))
        kept.append(hit)
    return kept


def coerce_schema_value(value: Any, schema_type: str) -> Any:
    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        raise ValueError("not boolean")
    if schema_type == "integer":
        if isinstance(value, bool):
            raise ValueError("bool is not int")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError("not integer")
    if schema_type == "number":
        if isinstance(value, bool):
            raise ValueError("bool is not number")
        if isinstance(value, int):
            return float(value)
        if isinstance(value, float):
            return value
        raise ValueError("not number")
    return value
