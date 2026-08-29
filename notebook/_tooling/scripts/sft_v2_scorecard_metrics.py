"""Accuracy + rate blocks for sft-v2 fair baseline scorecards.

Every scored column must expose both. Accuracy splits only when the trace/bank
supports them. Rate is sequential throughput from per-item wall_ms
(n / sum(wall_s)), not batch QPS.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval_sft_v2_layered import confusion, content_match, score_mw
from sft_v2_baseline_lib import wilson_interval


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return round(ys[idx], 1)


def rate_block(traces: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(t.get("wall_ms") or 0.0) for t in traces]
    n = len(traces)
    total_ms = float(sum(walls))
    total_s = total_ms / 1000.0
    if total_s <= 0:
        items_per_s = None
        zero_wall = True
    else:
        items_per_s = n / total_s
        zero_wall = False
    out_tok = [int(t["output_tokens"]) for t in traces if t.get("output_tokens") is not None]
    prompt_tok = [int(t["prompt_tokens"]) for t in traces if t.get("prompt_tokens") is not None]
    tok_s = [float(t["output_tok_s"]) for t in traces if t.get("output_tok_s") is not None]
    tok_total_s = None
    if out_tok and total_s > 0:
        tok_total_s = round(sum(out_tok) / total_s, 2)
    return {
        "n": n,
        "items_per_s": None if items_per_s is None else round(items_per_s, 4),
        "items_per_min": None if items_per_s is None else round(items_per_s * 60.0, 2),
        "mean_ms": None if n == 0 else round(total_ms / n, 1),
        "p50_ms": _pctl(walls, 0.5),
        "p95_ms": _pctl(walls, 0.95),
        "max_ms": None if not walls else round(max(walls), 1),
        "total_wall_s": round(total_s, 3),
        "prompt_tokens_sum": sum(prompt_tok) if prompt_tok else None,
        "output_tokens_sum": sum(out_tok) if out_tok else None,
        "output_tok_s_mean": None if not tok_s else round(sum(tok_s) / len(tok_s), 2),
        "output_tok_s_from_wall": tok_total_s,
        "token_stats_coverage": round(len(out_tok) / n, 4) if n else 0.0,
        "note": (
            "all wall_ms=0 (deterministic/not timed); items_per_s omitted"
            if zero_wall
            else "items_per_s = n / sum(wall_ms); sequential eval, not batch QPS."
        ),
    }


def _split_wilson(traces: list[dict[str, Any]], pred) -> dict[str, Any]:
    n = len(traces)
    k = sum(1 for t in traces if pred(t))
    return wilson_interval(k, n)


def _content_from_trace(t: dict[str, Any]) -> dict[str, bool]:
    gold = list(t.get("gold") or [])
    pred = list(t.get("extracted") or [])
    if gold or pred:
        return content_match(pred, gold)
    return {
        "execute_refuse": True,
        "tool_name": bool(t.get("content_exact")),
        "arguments": bool(t.get("content_exact")),
        "content_exact": bool(t.get("content_exact")),
    }


def accuracy_fullcall(traces: list[dict[str, Any]], *, bank_by_id: dict[str, dict] | None = None) -> dict[str, Any]:
    n = len(traces)
    bank_by_id = bank_by_id or {}
    contents = [_content_from_trace(t) for t in traces]
    exe = [t for t in traces if t.get("gold_execute")]
    ref = [t for t in traces if not t.get("gold_execute")]
    learned = traces and traces[0].get("top5_mode") == "learned_top5"
    hit = [t for t in traces if t.get("retrieval_hit")]
    miss = [t for t in traces if learned and not t.get("retrieval_hit")]

    by_family: dict[str, list[dict]] = defaultdict(list)
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        meta = bank_by_id.get(str(t.get("item_id") or ""), {})
        fam = str(t.get("family") or meta.get("family") or "na")
        sl = str(t.get("slice") or meta.get("slice") or "na")
        by_family[fam].append(t)
        by_slice[sl].append(t)

    def pack(rows: list[dict]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "strict_e2e": _split_wilson(rows, lambda t: t.get("strict_e2e")),
            "content": _split_wilson(rows, lambda t: t.get("content_exact")),
            "format": _split_wilson(rows, lambda t: t.get("format_ok")),
        }

    primary = wilson_interval(sum(1 for t in traces if t.get("strict_e2e")), n)
    return {
        "primary_metric": "strict_e2e",
        "primary": primary,
        "metrics": {
            "strict_e2e": primary,
            "content": wilson_interval(sum(1 for t in traces if t.get("content_exact")), n),
            "format": wilson_interval(sum(1 for t in traces if t.get("format_ok")), n),
            "tool_name": wilson_interval(sum(1 for c in contents if c.get("tool_name")), n),
            "arguments": wilson_interval(sum(1 for c in contents if c.get("arguments")), n),
            "legal": wilson_interval(sum(1 for t in traces if t.get("legal")), n),
        },
        "splits": {
            "execute": pack(exe),
            "refuse": pack(ref),
            "by_family": {k: pack(v) for k, v in sorted(by_family.items()) if k != "na" or len(by_family) == 1},
            "by_slice": {k: pack(v) for k, v in sorted(by_slice.items()) if k != "na" or len(by_slice) == 1},
            "cascade": {
                "retrieval_hit@5": wilson_interval(len(hit), n) if learned else None,
                "retrieval_miss": len(miss) if learned else 0,
                "pipeline_strict": wilson_interval(sum(1 for t in traces if t.get("pipeline_strict")), n),
                "generator_content_fail": sum(1 for t in traces if t.get("generator_content_fail")),
                "generator_format_fail": sum(1 for t in traces if t.get("generator_format_fail")),
            },
        },
    }


def accuracy_mw(traces: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [score_mw({"reason_code": t["gold"]}, raw_text=t.get("raw") or "") for t in traces]
    n = len(scored)
    conf = confusion(scored)
    primary = wilson_interval(sum(1 for s in scored if s["strict_e2e"]), n)
    return {
        "primary_metric": "strict_e2e",
        "primary": primary,
        "metrics": {
            "strict_e2e": primary,
            "content": wilson_interval(sum(1 for s in scored if s["content_ok"]), n),
            "format": wilson_interval(sum(1 for s in scored if s["format_ok"]), n),
            "macro_f1": conf["macro_f1"],
            "unsafe_execute": sum(1 for s in scored if s["unsafe_execute"]),
        },
        "splits": {"per_class": conf["per_class"]},
        "matrix": conf["matrix"],
    }


def _agg_ranks(ranks: list[int]) -> dict[str, Any]:
    n = len(ranks)
    r1 = sum(1 for r in ranks if r == 0)
    r5 = sum(1 for r in ranks if 0 <= r < 5)
    mrr = sum((1.0 / (r + 1) if r >= 0 else 0.0) for r in ranks)
    return {
        "n": n,
        "recall_at_1": wilson_interval(r1, n),
        "recall_at_5": wilson_interval(r5, n),
        "mrr": round(mrr / max(1, n), 6),
    }


def accuracy_retrieval(traces: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [t for t in traces if t.get("family") != "no_match"]
    ranks = [int(t.get("rank")) for t in pos if t.get("rank") is not None]
    by_fam: dict[str, list[int]] = defaultdict(list)
    by_size: dict[str, list[int]] = defaultdict(list)
    for t in pos:
        rank = int(t.get("rank")) if t.get("rank") is not None else -1
        by_fam[str(t.get("family") or "na")].append(rank)
        if t.get("catalog_size") is not None:
            by_size[str(t.get("catalog_size"))].append(rank)
    overall = _agg_ranks(ranks)
    nm = [t for t in traces if t.get("family") == "no_match"]
    return {
        "primary_metric": "recall_at_5",
        "primary": overall["recall_at_5"],
        "metrics": {
            "recall_at_1": overall["recall_at_1"],
            "recall_at_5": overall["recall_at_5"],
            "mrr": overall["mrr"],
        },
        "splits": {
            "by_family": {k: _agg_ranks(v) for k, v in sorted(by_fam.items())},
            "by_catalog_size": {k: _agg_ranks(v) for k, v in sorted(by_size.items())},
            "no_match": {
                "n": len(nm),
                "note": "no-match is not in the positive Recall denominator; sparse/dense always return k=5",
            },
        },
        "n_positive": overall["n"],
    }


def summarize_column(traces: list[dict[str, Any]], *, task: str, bank_by_id: dict[str, dict] | None = None) -> dict[str, Any]:
    if task == "retrieval":
        acc = accuracy_retrieval(traces)
    elif task == "mw":
        acc = accuracy_mw(traces)
    else:
        acc = accuracy_fullcall(traces, bank_by_id=bank_by_id)
    rate = rate_block(traces)
    return {
        "n": len(traces),
        "accuracy": acc,
        "rate": rate,
        "headline": {
            "accuracy": (acc.get("primary") or {}).get("rate") if isinstance(acc.get("primary"), dict) else acc.get("primary"),
            "accuracy_metric": acc.get("primary_metric"),
            "items_per_s": rate.get("items_per_s"),
            "p50_ms": rate.get("p50_ms"),
            "p95_ms": rate.get("p95_ms"),
        },
        "errors": sum(1 for t in traces if t.get("error")),
    }
