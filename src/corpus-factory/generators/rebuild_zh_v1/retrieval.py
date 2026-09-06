#!/usr/bin/env python3
"""Retrieval / no-match / rank-depth / fixed-five cross-batch scan generator.

Produces training rows for the retrieval head + the scanning-boundary MW
classes (0/10 plus the scan-stop classes), keeping retrieval_relevance, MW
disposition and execution-confidence strictly separate fields per
AGENTS.md's tool-context invariants. Catalog sizes are 10/20/50; gold tools
are placed at controlled rank buckets 1-5/6-10/11-20 so downstream fixed-five
batching exercises batch index 0/1/2/3 explicitly.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from . import common as C
from . import mw_scenarios as MS
from . import text_variants as tv

FAMILY = "retrieval"
BATCH = C.BATCH_SIZE

OFFTOPIC_BANK = (
    "今天出门要不要带伞？",
    "帮我写一首关于秋天的小诗。",
    "世界上最高的山是哪座？",
    "帮我把这句话翻译成日语：我很高兴认识你。",
    "推荐一部好看的悬疑电影。",
    "我家猫一直打喷嚏，是不是感冒了？",
    "给我讲讲黑洞是怎么形成的。",
    "帮我想一个宝宝的名字，姓陈。",
    "红烧肉怎么做才不腻？",
    "给我出一道脑筋急转弯。",
    "帮我算一下从北京到东京大概有多少公里。",
    "唱一句你喜欢的歌词给我听。",
    "帮我看看这段代码为什么会死循环（我贴给你）。",
    "股票明天会不会涨，你猜一下呗。",
    "帮我瘦身，制定一个减肥计划。",
    "今天历史上发生过什么大事？",
    "帮我起草一封给朋友的生日祝福。",
    "月亮为什么会有阴晴圆缺？",
    "帮我查一下这个梦是什么意思。",
    "给我讲个鬼故事。",
)

NEAR_MISS_EXTENSIONS = (
    "顺便把过去十年所有相关记录永久删除并寄送纸质存档给我",
    "同时把这件事同步转发到公司全员群和外部社交媒体",
    "另外把后台系统的源代码和密钥一起发给我",
    "顺便绕开所有审批流程直接强制执行",
    "同时把其他所有人的相同记录也一起打包发给我",
    "另外帮我把这个系统整体换成另一套完全不同的产品",
)


def _pick(seq: Sequence[Any], seed: str) -> Any:
    idx = int(C.sha256_text(seed)[:8], 16)
    return seq[idx % len(seq)]


def similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    if a.get("name") == b.get("name"):
        return 1.0
    score = 0.0
    if a.get("family") == b.get("family"):
        score += 0.5
    ta = C.char_trigrams(str(a.get("description") or ""))
    tb = C.char_trigrams(str(b.get("description") or ""))
    score += 0.5 * C.jaccard(ta, tb)
    return score


def _candidates_sorted_by_similarity(gold: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    scored = [(similarity(gold, t), t.get("name"), t) for t in pool if t.get("name") != gold.get("name")]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in scored]


def build_catalog(
    gold: Mapping[str, Any],
    all_tools: Sequence[Mapping[str, Any]],
    *,
    size: int,
    gold_rank: int,
    seed: str,
) -> list[Mapping[str, Any]]:
    """Deterministically construct a catalog of `size` tools with `gold` at
    1-indexed position `gold_rank`. Tools immediately around gold are the
    most lexically/family-similar (hard negatives); tools far from gold are
    lower-similarity filler, mimicking a real cosine-descending ranking."""

    pool = [t for t in all_tools if t.get("name") != gold.get("name")]
    ranked_pool = _candidates_sorted_by_similarity(gold, pool)
    rnd = random.Random(seed)
    rnd.shuffle(ranked_pool[: min(len(ranked_pool), size * 3)])  # local jitter, keep deterministic via seeded Random
    before_n = gold_rank - 1
    after_n = size - gold_rank
    before = ranked_pool[:before_n]
    after = ranked_pool[before_n : before_n + after_n]
    catalog = before + [gold] + after
    if len(catalog) < size:
        filler = [t for t in all_tools if t not in catalog and t.get("name") != gold.get("name")]
        catalog += filler[: size - len(catalog)]
    return catalog[:size]


def build_batches(catalog: Sequence[Mapping[str, Any]], gold_name: str | None, *, max_batches: int) -> tuple[list[dict[str, Any]], str, int | None]:
    names = [t.get("name") for t in catalog]
    batches: list[dict[str, Any]] = []
    found_idx: int | None = None
    scanned = 0
    for i in range(0, len(names), BATCH):
        if scanned >= max_batches:
            break
        batch_names = names[i : i + BATCH]
        batch_idx = i // BATCH
        if gold_name and gold_name in batch_names:
            batches.append({"batch_index": batch_idx, "tool_names": batch_names, "disposition": "found"})
            found_idx = batch_idx
            scanned += 1
            break
        batches.append({"batch_index": batch_idx, "tool_names": batch_names, "disposition": "capability_insufficient"})
        scanned += 1
    if found_idx is None:
        remaining = len(names) - scanned * BATCH
        if gold_name is None:
            terminal = "retrieval_no_match"
        elif remaining <= 0:
            terminal = "candidate_exhausted"
        else:
            terminal = "candidate_scan_limit_reached"
    else:
        terminal = "found"
    return batches, terminal, found_idx


def _budget_for_batch(deploy: C.ToolRegistry, batch_names: Sequence[str], query: str, context, evidence, history, tool_results) -> C.BudgetResult:
    tools_batch = [deploy.by_name[n] for n in batch_names if n in deploy.by_name]
    _, receipt = C.fit_batch_to_budget(
        tools_batch=tools_batch, query=query, context=context, evidence=evidence,
        history=history, tool_results=tool_results, profile="standard",
    )
    return receipt


def _row_scaffold(
    *, scenario: str, variant_kind: str, catalog: list[Mapping[str, Any]], gold: Mapping[str, Any] | None,
    gold_rank: int | None, query: str, context: dict, evidence: list, history: list,
    permissions: dict, state: dict, deploy: C.ToolRegistry, max_batches: int,
    mw_class: dict[str, Any] | None, seed: str,
) -> dict[str, Any]:
    gold_name = gold.get("name") if gold else None
    batches, terminal, found_idx = build_batches(catalog, gold_name, max_batches=max_batches)
    batch0_names = batches[0]["tool_names"] if batches else []
    receipt = _budget_for_batch(deploy, batch0_names, query, context, evidence, history, [])
    hard_neg = [t.get("name") for t in catalog[: min(4, len(catalog))] if t.get("name") != gold_name]
    row = {
        "case_id": C.case_id(FAMILY, scenario, seed),
        "cf_group": C.cf_group(FAMILY, scenario, gold_name or "none", str(gold_rank)),
        "family": FAMILY,
        "task": "retrieval",
        "generator_version": C.GENERATOR_VERSION,
        "scenario": scenario,
        "variant_kind": variant_kind,
        "catalog_size": len(catalog),
        "query": query,
        "context": context,
        "evidence": evidence,
        "history": history,
        "catalog_tool_names": [t.get("name") for t in catalog],
        "catalog_registry": {"path": "tool-universe.json", "sha256": deploy.sha256},
        "gold_name": gold_name,
        "gold_rank": gold_rank,
        "hard_negative_names": hard_neg,
        "batches": batches,
        "retrieval_terminal_reason": terminal,
        "found_at_batch_index": found_idx,
        "max_candidate_batches": max_batches,
        "mw": mw_class,
        "permissions": permissions,
        "state": state,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
    }
    return row


def _query_for_gold(gold: Mapping[str, Any], variant_kind: str, seed: str) -> str:
    scenario = MS.class_0_ready_to_execute(gold, seed)
    base = scenario["query"]
    if variant_kind in ("formal",):
        return base
    return tv.build_variant(base, variant_kind, seed=seed, en_term=gold.get("family", ""), obj=gold.get("description", ""))


def gen_rank_bucket_cases(deploy: C.ToolRegistry, *, rank_lo: int, rank_hi: int, catalog_sizes: Sequence[int], count: int, scenario_tag: str) -> list[dict[str, Any]]:
    tools = list(deploy.tools)
    rows = []
    for i in range(count):
        gold = tools[i % len(tools)]
        size = catalog_sizes[i % len(catalog_sizes)]
        rank_span = max(1, min(rank_hi, size) - rank_lo + 1)
        gold_rank = rank_lo + (i % rank_span)
        if gold_rank > size:
            gold_rank = size
        seed = f"{scenario_tag}:{gold['name']}:{i}"
        variant_kind = tv.VARIANT_KINDS[i % len(tv.VARIANT_KINDS)]
        query = _query_for_gold(gold, variant_kind, seed)
        catalog = build_catalog(gold, tools, size=size, gold_rank=gold_rank, seed=seed)
        max_batches = max(1, -(-size // BATCH))
        row = _row_scaffold(
            scenario=scenario_tag, variant_kind=variant_kind, catalog=catalog, gold=gold, gold_rank=gold_rank,
            query=query, context={"locale": "zh-CN"}, evidence=[], history=[],
            permissions={"principal": "user", "grants": ["standard"]}, state={}, deploy=deploy,
            max_batches=max_batches, mw_class=None, seed=seed,
        )
        rows.append(row)
    return rows


def gen_no_match_cases(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    tools = list(deploy.tools)
    rows = []
    half = count // 2
    for i in range(count):
        size = (10, 20, 50)[i % 3]
        seed = f"no_match:{i}"
        variant_kind = tv.VARIANT_KINDS[i % len(tv.VARIANT_KINDS)]
        if i < half:
            base = _pick(OFFTOPIC_BANK, seed)
            query = tv.build_variant(base, variant_kind, seed=seed)
            near_miss = False
        else:
            anchor = tools[i % len(tools)]
            ext = _pick(NEAR_MISS_EXTENSIONS, seed)
            base = f"帮我{anchor.get('description')}，{ext}。"
            query = tv.build_variant(base, variant_kind, seed=seed)
            near_miss = True
        catalog_seed_tool = tools[(i * 7) % len(tools)]
        ranked = _candidates_sorted_by_similarity(catalog_seed_tool, [t for t in tools if t.get("name") != catalog_seed_tool.get("name")])
        rnd = random.Random(seed)
        pool = ranked[: size * 2] or ranked
        rnd.shuffle(pool)
        catalog = pool[:size]
        max_batches = max(1, -(-size // BATCH))
        row = _row_scaffold(
            scenario="no_match_offtopic" if not near_miss else "no_match_near_miss",
            variant_kind=variant_kind, catalog=catalog, gold=None, gold_rank=None,
            query=query, context={"locale": "zh-CN"}, evidence=[], history=[],
            permissions={"principal": "user", "grants": ["standard"]}, state={}, deploy=deploy,
            max_batches=max_batches, mw_class={"reason_class_id": 11, "reason_code": "unsupported_scope"}, seed=seed,
        )
        rows.append(row)
    return rows


def gen_cross_batch_exhausted_cases(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    """Gold tool exists in the full universe but is entirely excluded from
    the eligible candidate set for this query (a discard-threshold-style
    recall failure), so the row's `gold_name` records ground truth for
    audit/gate purposes while `catalog_tool_names` never contains it. Scanning
    the whole eligible set yields `candidate_exhausted`; capping
    `max_candidate_batches` below full coverage yields
    `candidate_scan_limit_reached`."""

    tools = list(deploy.tools)
    rows = []
    for i in range(count):
        gold = tools[i % len(tools)]
        size = 20
        seed = f"cross_batch_exhausted:{gold['name']}:{i}"
        variant_kind = tv.VARIANT_KINDS[i % len(tv.VARIANT_KINDS)]
        query = _query_for_gold(gold, variant_kind, seed)
        pool = [t for t in tools if t.get("name") != gold.get("name")]
        ranked = _candidates_sorted_by_similarity(gold, pool)
        catalog = ranked[:size]
        full_batches = max(1, -(-size // BATCH))
        max_batches = full_batches if i % 2 == 0 else max(1, full_batches - 1)
        row = _row_scaffold(
            scenario="cross_batch_exhausted", variant_kind=variant_kind, catalog=catalog, gold=gold, gold_rank=None,
            query=query, context={"locale": "zh-CN"}, evidence=[], history=[],
            permissions={"principal": "user", "grants": ["standard"]}, state={}, deploy=deploy,
            max_batches=max_batches, mw_class=None, seed=seed,
        )
        rows.append(row)
    return rows


def gen_stop_before_scan_cases(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    tools = list(deploy.tools)
    rows = []
    codes = list(MS.SCAN_STOP_CLASSES)
    codebook = {c["reason_code"]: c for c in C.mw_classes()}
    for i in range(count):
        code = codes[i % len(codes)]
        gold = tools[i % len(tools)]
        seed = f"stop_before_scan:{code}:{i}"
        builder = MS.BUILDERS[code]
        if code in MS.NEEDS_SIBLING:
            other = tools[(i + 3) % len(tools)]
            scenario = builder(gold, seed, other)
        else:
            scenario = builder(gold, seed)
        size = (10, 20, 50)[i % 3]
        gold_rank = 1 + (i % min(size, 20))
        catalog = build_catalog(gold, tools, size=size, gold_rank=gold_rank, seed=seed)
        max_batches = max(1, -(-size // BATCH))
        cls = codebook[code]
        row = _row_scaffold(
            scenario=f"stop_{code}", variant_kind="grounded", catalog=catalog, gold=gold, gold_rank=gold_rank,
            query=scenario["query"], context=scenario["context"], evidence=scenario["evidence"],
            history=scenario["history"], permissions=scenario["permissions"], state=scenario["state"],
            deploy=deploy, max_batches=max_batches,
            mw_class={"reason_class_id": cls["class_id"], "reason_code": code}, seed=seed,
        )
        # Stop scenarios terminate at batch 0 regardless of gold position.
        row["batches"] = [{"batch_index": 0, "tool_names": row["batches"][0]["tool_names"] if row["batches"] else [], "disposition": f"stop_{code}"}]
        row["retrieval_terminal_reason"] = f"stop_{code}"
        row["found_at_batch_index"] = None
        rows.append(row)
    return rows


def gen_hard_negative_discrimination_cases(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    """Gold at rank 1 but ranks 2-5 are deliberately the most confusable
    same-family / similar-schema tools, stressing top-1 precision."""

    tools = list(deploy.tools)
    rows = []
    for i in range(count):
        gold = tools[i % len(tools)]
        size = (10, 20)[i % 2]
        seed = f"hard_negative:{gold['name']}:{i}"
        variant_kind = tv.VARIANT_KINDS[i % len(tv.VARIANT_KINDS)]
        query = _query_for_gold(gold, variant_kind, seed)
        catalog = build_catalog(gold, tools, size=size, gold_rank=1, seed=seed)
        max_batches = max(1, -(-size // BATCH))
        row = _row_scaffold(
            scenario="hard_negative_discrimination", variant_kind=variant_kind, catalog=catalog, gold=gold, gold_rank=1,
            query=query, context={"locale": "zh-CN"}, evidence=[], history=[],
            permissions={"principal": "user", "grants": ["standard"]}, state={}, deploy=deploy,
            max_batches=max_batches, mw_class={"reason_class_id": 0, "reason_code": "ready_to_execute"}, seed=seed,
        )
        rows.append(row)
    return rows


def generate(deploy: C.ToolRegistry, targets: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += gen_rank_bucket_cases(deploy, rank_lo=1, rank_hi=5, catalog_sizes=(10, 20, 50), count=targets["rank_1_5"], scenario_tag="rank_1_5")
    rows += gen_rank_bucket_cases(deploy, rank_lo=6, rank_hi=10, catalog_sizes=(10, 20, 50), count=targets["rank_6_10"], scenario_tag="rank_6_10")
    rows += gen_rank_bucket_cases(deploy, rank_lo=11, rank_hi=15, catalog_sizes=(20, 50), count=targets["rank_11_15"], scenario_tag="rank_11_15")
    rows += gen_rank_bucket_cases(deploy, rank_lo=16, rank_hi=20, catalog_sizes=(20, 50), count=targets["rank_16_20"], scenario_tag="rank_16_20")
    rows += gen_no_match_cases(deploy, count=targets["no_match"])
    rows += gen_cross_batch_exhausted_cases(deploy, count=targets["cross_batch_exhausted"])
    rows += gen_stop_before_scan_cases(deploy, count=targets["stop_before_scan"])
    rows += gen_hard_negative_discrimination_cases(deploy, count=targets["hard_negative_discrimination"])

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_scenario": _count_by(rows, "scenario"),
        "by_terminal_reason": _count_by(rows, "retrieval_terminal_reason"),
        "by_found_batch_index": _count_by(rows, "found_at_batch_index"),
        "by_split": _count_by(rows, "split"),
        "by_catalog_size": _count_by(rows, "catalog_size"),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
