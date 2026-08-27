#!/usr/bin/env python3
"""Shared helpers for sft-v2 independent eval lock, clean packs, and scorecard."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from park_toolcall_lib import char_trigrams, near_dup_holdout
from repo_paths import (
    BANK_MEI_MW_DISPOSITION_V2,
    BANK_MEI_PARK_TOOLCALL_V1,
    BANK_MEI_RETRIEVAL_V2,
    BANK_NEEDLE_VRM_MW,
    EVAL_BANKS_ROOT,
    PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
    PACK_MEI_MW_DISPOSITION_V2_2K,
    PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
    PACK_MEI_RETRIEVAL_V2_10K_PAID,
    PACK_MEI_RETRIEVAL_V2_2K,
    PACK_MEI_RETRIEVAL_V2_2K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K,
    PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
    ROOT,
    SFT_TRAIN,
    SFT_V2_BASELINE_CONTRACT,
)
from sft_canonical_lib import load_jsonl, query_banned

LOCK_VERSION = "sft-v2-eval-lock-v1"
CLEAN_VERSION = "sft-v2-clean-10k-v1"
REASON_CODES_16 = json.loads(SFT_V2_BASELINE_CONTRACT.read_text(encoding="utf-8"))["tasks"]["mw"][
    "reason_codes"
]

EVAL_CITIES = ["合肥", "厦门", "宁波", "长沙", "郑州", "福州", "无锡", "大连", "哈尔滨", "昆明", "南宁", "兰州"]
EVAL_ROOMS = ["客房", "影音室", "储藏室", "衣帽间", "洗衣房", "阁楼", "地下室", "阳光房"]
EVAL_PLACES = ["放映厅", "档案室", "会客室", "值班台", "冷冻库", "天台", "展厅", "暗房"]
EVAL_TIMES = ["交接班时", "闭馆前", "巡楼后", "清场时", "盘点间隙", "封存前"]
EVAL_TITLES = ["封存会", "巡楼会", "交接班", "盘点会"]
EVAL_ITEMS = ["封箱胶带", "冷链标签", "档案夹"]
EVAL_DOORS = ["消防通道门", "档案室门", "天台门"]

CLEAN_CITIES = ["济南", "长春", "沈阳", "珠海", "佛山", "东莞", "中山", "惠州", "烟台", "威海", "洛阳", "保定", "徐州", "扬州", "嘉兴", "金华", "临沂", "淄博"]
CLEAN_ROOMS = ["打印室", "母婴室", "值班室", "更衣室", "工具间", "天桥", "收发室", "值班走廊", "茶水房", "南侧廊", "北侧廊", "档案柜"]
CLEAN_PLACES = ["打印室", "母婴室", "值班室", "更衣室", "工具间", "收发室"]
CLEAN_TIMES = ["换班后", "清点时", "巡检完", "封箱前"]
CLEAN_TITLES = ["换班会", "巡检会", "封箱会"]
CLEAN_ITEMS = ["标签纸", "封箱钉", "巡检表"]
CLEAN_DOORS = ["值班室门", "工具间门", "收发室门"]

EVAL_NO_MATCH = [
    "光合作用暗反应分哪几步",
    "请默写一首边塞七绝",
    "土星有没有固态表面",
    "帮我把这段拉丁文译成上古汉语",
    "圆周率小数点后第一千位是多少",
    "讲讲宋代市舶司怎么收税",
    "写一段没有人物的风景俳句",
    "解释一下黎曼猜想的现代叙述",
    "金星一天有多长",
    "把斐波那契通项用矩阵写出来",
]

STUDENT_SYSTEM = (
    "任务：只输出一个 schema 合法的工具 JSON 数组，或 []。"
    "最多一次调用。缺少 required 证据时输出 []。禁止输出解释或 route_id。"
)
RETRIEVAL_E2E_SYSTEM = "只输出一个工具名，没有匹配则输出 NONE。禁止解释。"
MW_CLOSED_SYSTEM = "只输出一个 reason_code。不要输出 HD/SF 文本，不要输出 JSON 工具调用。"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"n": 0, "k": 0, "rate": None, "ci95_lo": None, "ci95_hi": None}
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return {
        "n": n,
        "k": k,
        "rate": round(p, 6),
        "ci95_lo": round(max(0.0, center - margin), 6),
        "ci95_hi": round(min(1.0, center + margin), 6),
    }


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def existing_eval_queries() -> set[str]:
    rows: list[dict] = []
    for path in (
        BANK_MEI_RETRIEVAL_V2,
        BANK_MEI_MW_DISPOSITION_V2,
        BANK_MEI_PARK_TOOLCALL_V1,
        BANK_NEEDLE_VRM_MW,
        EVAL_BANKS_ROOT / "mei-toolcall-v2/eval-bank-smoke.jsonl",
    ):
        rows.extend(load_jsonl(path))
    return {str(r.get("query") or "").strip() for r in rows if str(r.get("query") or "").strip()}


def train_pack_paths() -> list[Path]:
    extra = sorted((SFT_TRAIN / "packs").glob("*.jsonl")) if (SFT_TRAIN / "packs").is_dir() else []
    named = [
        PACK_MEI_RETRIEVAL_V2_10K_PAID,
        PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
        PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
        PACK_MEI_RETRIEVAL_V2_2K_PAID,
        PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID,
        PACK_MEI_MW_DISPOSITION_V2_2K_PAID,
        PACK_MEI_RETRIEVAL_V2_2K,
        PACK_MEI_TOOLCALL_V2_ORACLE_2K,
        PACK_MEI_MW_DISPOSITION_V2_2K,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in named + extra:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        out.append(p)
    return out


def train_queries() -> set[str]:
    qs: set[str] = set()
    for path in train_pack_paths():
        for row in load_jsonl(path):
            q = str(row.get("query") or "").strip()
            if q:
                qs.add(q)
    return qs


def blocked_queries(*, extra: Iterable[str] = ()) -> set[str]:
    blocked = existing_eval_queries() | train_queries()
    blocked.update(str(x).strip() for x in extra if str(x).strip())
    return blocked


def query_rejected(query: str, *, blocked: set[str]) -> str | None:
    q = (query or "").strip()
    if not q:
        return "empty"
    banned = query_banned(q)
    if banned:
        return banned
    if q in blocked:
        return "blocked_exact"
    hold = near_dup_holdout(q)
    if hold:
        return hold
    return None


def near_dup_against(query: str, kept_trigrams: list[set[str]], *, threshold: float = 0.9) -> bool:
    tq = char_trigrams(query)
    if not tq:
        return False
    for tk in kept_trigrams:
        union = tq | tk
        if not union:
            continue
        if len(tq & tk) / len(union) >= threshold:
            return True
    return False


class TrigramIndex:
    def __init__(self) -> None:
        self.sets: list[set[str]] = []
        self.inv: dict[str, list[int]] = defaultdict(list)

    def add(self, query: str) -> None:
        ts = char_trigrams(query)
        idx = len(self.sets)
        self.sets.append(ts)
        for tri in ts:
            self.inv[tri].append(idx)

    def hits(self, query: str, *, threshold: float = 0.9) -> bool:
        tq = char_trigrams(query)
        if not tq:
            return False
        cands: set[int] = set()
        for tri in tq:
            cands.update(self.inv.get(tri) or [])
        for j in cands:
            union = tq | self.sets[j]
            if union and len(tq & self.sets[j]) / len(union) >= threshold:
                return True
        return False


def pack_quality(rows: list[dict], *, threshold: float = 0.9) -> dict[str, Any]:
    qs = [str(r.get("query") or "").strip() for r in rows]
    n = len(qs)
    unique = len(set(qs))
    digit = sum(1 for q in qs if query_banned(q) == "digit_suffix")
    banned = sum(1 for q in qs if query_banned(q))
    trigrams = [char_trigrams(q) for q in qs]
    index: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(trigrams):
        for tri in ts:
            index[tri].append(i)
    flagged = [False] * n
    for i, ts in enumerate(trigrams):
        cands: set[int] = set()
        for tri in ts:
            cands.update(index[tri])
        for j in cands:
            if j <= i:
                continue
            union = ts | trigrams[j]
            if union and len(ts & trigrams[j]) / len(union) >= threshold:
                flagged[i] = True
                flagged[j] = True
    near = sum(flagged)
    return {
        "n": n,
        "unique": unique,
        "exact_duplicate": n - unique,
        "digit_suffix": digit,
        "query_banned": banned,
        "near_dup_rows": near,
        "near_dup_rate": round(near / n, 6) if n else None,
        "near_dup_ok": (near / n if n else 0) <= 0.05,
        "ok": n == 10000 and unique == 10000 and (n - unique) == 0 and digit == 0 and banned == 0,
    }


def lock_payload(path: Path, rows: list[dict], extra: dict | None = None) -> dict[str, Any]:
    families: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    kinds: dict[str, int] = defaultdict(int)
    for row in rows:
        families[str(row.get("family") or "na")] += 1
        if row.get("reason_code"):
            reasons[str(row.get("reason_code"))] += 1
        kinds[str(row.get("kind") or row.get("slice") or "na")] += 1
    payload = {
        "lock_version": LOCK_VERSION,
        "path": rel(path),
        "n": len(rows),
        "sha256": sha256_file(path) if path.is_file() else None,
        "families": dict(sorted(families.items())),
        "kinds": dict(sorted(kinds.items())),
        "reason_codes": dict(sorted(reasons.items())) if reasons else {},
        "gold_origin": "schema-compiler",
    }
    if extra:
        payload.update(extra)
    return payload


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None
