#!/usr/bin/env python3
"""Deterministic, offline Chinese natural-language surface-variant utilities.

No paid teacher/provider is called anywhere in this module. All variation is
rule-based and seeded so regeneration is byte-identical. This is intentionally
richer than factory_51m.py's fixed per-class template strings (see
corpus-factory/generators/rebuild_zh_v1/common.py docstring): each helper
composes several independent axes (register, ellipsis, coreference, typo,
synonym, code-mixing) so the cross product of a small template bank still
yields many surface-distinct natural queries grounded in the real tool
description text, not filler placeholders.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

# Common near-synonym substitutions for everyday Chinese request verbs/nouns.
# Used to paraphrase without changing meaning.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "打开": ("开启", "启动", "开一下", "给我开"),
    "关闭": ("关掉", "停掉", "关一下", "给我关"),
    "查询": ("查一下", "帮我查", "看看", "查询一下"),
    "预订": ("订一下", "帮我订", "预约", "订购"),
    "取消": ("撤销", "退掉", "不要了，取消"),
    "调高": ("调大", "往上调", "提高"),
    "调低": ("调小", "往下调", "降低"),
    "设置": ("设为", "调成", "改成"),
    "查看": ("看一下", "查一下", "帮我看"),
    "申请": ("办理", "提交申请", "帮我申请"),
    "帮我": ("麻烦", "请", "能不能帮我"),
    "现在": ("目前", "此刻", ""),
    "一下": ("", "一下下", "下"),
}

_TYPO_PAIRS: tuple[tuple[str, str], ...] = (
    ("的", "地"), ("在", "再"), ("以", "已"), ("account", "acount"),
    ("查询", "查寻"), ("预订", "预定"), ("温度", "湿度"), ("空调", "空掉"),
    ("设置", "设至"), ("申请", "申清"),
)

_COLLOQUIAL_PREFIX: tuple[str, ...] = (
    "", "喂，", "那个，", "嗯，", "对了，", "麻烦一下，", "不好意思，", "急，",
)

_COLLOQUIAL_SUFFIX: tuple[str, ...] = (
    "", "谢谢。", "可以吗？", "行不行？", "麻烦了。", "急用。", "越快越好。",
)

_COREF_OPENERS: tuple[str, ...] = (
    "刚才说的那个，", "还是那件事，", "接着上次的，", "那个东西，",
)

_MIXED_EN_TEMPLATES: tuple[str, ...] = (
    "帮我 {en} 一下 {obj}",
    "用 {en} 处理一下{obj}",
    "{obj}的 {en} 麻烦搞一下",
)


def _seed_int(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def _pick(options: Sequence[str], seed: int) -> str:
    if not options:
        return ""
    return options[seed % len(options)]


def apply_synonym(text: str, seed: str) -> str:
    s = _seed_int("syn", seed)
    out = text
    i = 0
    for key, alts in _SYNONYMS.items():
        if key in out:
            replacement = _pick(alts, s + i)
            out = out.replace(key, replacement, 1)
            i += 1
            if i >= 2:
                break
    return out


def apply_typo(text: str, seed: str) -> str:
    s = _seed_int("typo", seed)
    for i, (correct, typo) in enumerate(_TYPO_PAIRS):
        if correct in text and (s + i) % 5 == 0:
            return text.replace(correct, typo, 1)
    return text


def apply_colloquial(text: str, seed: str) -> str:
    s = _seed_int("collo", seed)
    prefix = _pick(_COLLOQUIAL_PREFIX, s)
    suffix = _pick(_COLLOQUIAL_SUFFIX, s // 7)
    return f"{prefix}{text}{suffix}"


def apply_coreference(text_with_placeholder: str, referent_desc: str, seed: str) -> str:
    """Replace an explicit object mention with a coreference opener, relying
    on `history`/`context` to carry the real referent (the row builder is
    responsible for populating that side channel)."""

    s = _seed_int("coref", seed)
    opener = _pick(_COREF_OPENERS, s)
    return f"{opener}{text_with_placeholder}"


def apply_ellipsis(text: str) -> str:
    """Drop a leading '请'/'帮我' politeness marker and trailing particle to
    simulate a terse colloquial request."""

    out = text
    for lead in ("请帮我", "请", "帮我", "麻烦"):
        if out.startswith(lead):
            out = out[len(lead) :]
            break
    return out.strip("，, ")


def mixed_zh_en(en_term: str, obj: str, seed: str) -> str:
    s = _seed_int("mix", seed)
    template = _pick(_MIXED_EN_TEMPLATES, s)
    return template.format(en=en_term, obj=obj)


VARIANT_KINDS = (
    "formal",
    "colloquial",
    "ellipsis",
    "typo",
    "synonym",
    "coreference",
    "mixed_zh_en",
)


def build_variant(base_sentence: str, kind: str, *, seed: str, referent_desc: str = "", en_term: str = "", obj: str = "") -> str:
    if kind == "formal":
        return base_sentence
    if kind == "colloquial":
        return apply_colloquial(base_sentence, seed)
    if kind == "ellipsis":
        return apply_ellipsis(base_sentence)
    if kind == "typo":
        return apply_typo(base_sentence, seed)
    if kind == "synonym":
        return apply_synonym(base_sentence, seed)
    if kind == "coreference":
        return apply_coreference(base_sentence, referent_desc, seed)
    if kind == "mixed_zh_en" and en_term:
        return mixed_zh_en(en_term, obj or base_sentence, seed)
    return base_sentence
