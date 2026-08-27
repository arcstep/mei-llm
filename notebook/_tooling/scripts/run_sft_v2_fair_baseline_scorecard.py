#!/usr/bin/env python3
"""Fair sft-v2 baseline scorecard (lock v2). Does not train or change CURRENT.json.

Tune retrieval/prompt knobs on DEV only, then run TEST once.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval_sft_v2_layered import cascade_score, confusion, score_fullcall, score_mw
from repo_paths import (
    NOTEBOOK_JOBS,
    ROOT,
    SFT_V2_BASELINE_CONTRACT_V2,
    SFT_V2_BASELINE_MODELS_V2,
    SFT_V2_EVAL_LOCK_DIR_V2,
)
from sft_canonical_lib import load_jsonl
from sft_v2_baseline_adapters import (
    load_mei58m,
    mei58m_status,
    minimind_status,
    ollama_available,
    qwen_ollama_factory,
)
from sft_v2_baseline_lib import dump_json, rel, sha256_file, wilson_interval
from sft_v2_fair_prompts import (
    fullcall_json_schema,
    fullcall_system,
    mw_codebook_block,
    prompt_asset,
    render_fullcall_user,
    render_mw_user,
    tools_to_ollama,
)
from sft_v2_retrieval_backends import (
    DenseExactRetriever,
    HnswRetriever,
    SparseRetriever,
    mei58m_encode_fn,
    try_sentence_transformer,
)

OUT_DEFAULT = NOTEBOOK_JOBS / "toolcall-sft/outbox/draft/sft-v2-baseline-scorecard-v2.json"
TRACE_DIR = NOTEBOOK_JOBS / "toolcall-sft/outbox/draft/sft-v2-baseline-traces-v2"
QWEN_MODELS = [
    ("qwen3.5-0.8b", "qwen3.5:0.8b-mlx", 800_000_000),
    ("qwen3.5-4b", "qwen3.5:4b-mlx", 4_000_000_000),
    ("qwen3.5-9b", "qwen3.5:9b-mlx", 9_000_000_000),
    ("qwen3.6-35b", "qwen3.6:35b-mlx", 35_000_000_000),
]
GEN_MODES = ("prompt_adapted_raw", "native_tools", "format_constrained")


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return round(ys[idx], 1)


def _peak_mem() -> dict:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS ru_maxrss is bytes; Linux is kilobytes
        mb = rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024
        return {"peak_rss_mb": round(mb, 1), "status": "rusage"}
    except Exception:
        return {"peak_rss_mb": None, "status": "unavailable"}


def load_universe(lock_dir: Path) -> dict:
    return json.loads((lock_dir / "tool-universe-v1.json").read_text(encoding="utf-8"))


def hydrate_catalog(row: dict, universe: dict) -> list[dict]:
    by_name = {t["name"]: t for t in universe["tools"]}
    names = row.get("catalog_tool_names") or []
    return [by_name[n] for n in names if n in by_name]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def retrieval_report(rows: list[dict], universe: dict, search_rank: Callable[[str, list[dict]], list[str]], *, name: str) -> dict:
    pos_ranks: list[int] = []
    by_fam: dict[str, list[int]] = defaultdict(list)
    by_size: dict[str, list[int]] = defaultdict(list)
    pos_top1: list[float] = []
    nm_top1: list[float] = []
    traces = []
    for row in rows:
        catalog = hydrate_catalog(row, universe)
        t0 = time.perf_counter()
        order = search_rank(str(row.get("query") or ""), catalog)
        wall = (time.perf_counter() - t0) * 1000
        gold = row.get("gold_tool")
        if row.get("no_match") or not gold:
            # score proxy: 1 if top-1 name looks weakly related is not used; keep rank=-1
            nm_top1.append(1.0 if order else 0.0)
            traces.append({"item_id": row["item_id"], "family": "no_match", "rank": -1, "pred_top5": order[:5], "wall_ms": wall})
            continue
        rank = order.index(gold) if gold in order else -1
        pos_ranks.append(rank)
        by_fam[str(row.get("family") or "na")].append(rank)
        by_size[str(row.get("catalog_size") or "na")].append(rank)
        pos_top1.append(1.0 if rank == 0 else 0.0)
        traces.append({"item_id": row["item_id"], "family": row.get("family"), "gold": gold, "rank": rank, "pred_top5": order[:5], "wall_ms": wall})
    def agg(ranks: list[int]) -> dict:
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
    nm_n = sum(1 for r in rows if r.get("no_match"))
    # FPR: retrieved a "confident" hit; without calibrated score, use top-1 always-returns.
    # Operational: no_match items never have gold in top-5, so miss is correct;
    # FPR = fraction whose top-1 is a similar-name/seen tool (always 1 if catalog nonempty).
    # Report accuracy as 1.0 for "did not retrieve the nonexistent gold" and FPR via DEV-frozen
    # rule: if top-1 exists, count as retrieval attempt (FPR=1). Explicitly labeled.
    fpr = 1.0 if nm_n else 0.0
    walls = [t["wall_ms"] for t in traces]
    return {
        "retriever": name,
        "overall": agg(pos_ranks),
        "by_family": {k: agg(v) for k, v in sorted(by_fam.items())},
        "by_catalog_size": {k: agg(v) for k, v in sorted(by_size.items())},
        "no_match": {
            "n": nm_n,
            "accuracy_no_gold_in_top5": wilson_interval(nm_n, nm_n) if nm_n else wilson_interval(0, 0),
            "fpr_always_returns_top5": {"rate": fpr, "note": "sparse/dense retrievers always return k=5; no-match is not in positive Recall denominator"},
        },
        "latency": {"p50": _pctl(walls, 0.5), "p95": _pctl(walls, 0.95), "n": len(walls)},
        "traces": traces,
    }


def run_sparse(rows, universe, mode: str) -> dict:
    retriever = SparseRetriever(mode=mode, k=5)
    retriever.build(universe["tools"])

    def rank(query: str, catalog: list[dict]) -> list[str]:
        names = [t["name"] for t in catalog]
        return [n for n, _ in retriever.rank_subset(query, names)]

    return retrieval_report(rows, universe, rank, name=mode)


def run_dense(rows, universe, encode_fn, name: str) -> dict:
    retriever = DenseExactRetriever(encode_fn, name=name, k=5)
    retriever.build(universe["tools"])

    def rank(query: str, catalog: list[dict]) -> list[str]:
        names = set(t["name"] for t in catalog)
        return [n for n, _ in retriever.rank_all(query) if n in names]

    return retrieval_report(rows, universe, rank, name=name)


def summarize_from_traces(traces: list[dict], *, task: str) -> dict:
    if task == "fullcall":
        n = len(traces)
        content = sum(1 for t in traces if t.get("content_exact"))
        fmt = sum(1 for t in traces if t.get("format_ok"))
        strict = sum(1 for t in traces if t.get("strict_e2e"))
        exe = [t for t in traces if t.get("gold_execute")]
        ref = [t for t in traces if not t.get("gold_execute")]
        hit = [t for t in traces if t.get("retrieval_hit")]
        miss = [t for t in traces if t.get("top5_mode") == "learned_top5" and not t.get("retrieval_hit")]
        walls = [t.get("wall_ms") or 0 for t in traces]
        return {
            "n": n,
            "content": wilson_interval(content, n),
            "format": wilson_interval(fmt, n),
            "strict_e2e": wilson_interval(strict, n),
            "execute_content": wilson_interval(sum(1 for t in exe if t.get("content_exact")), len(exe)),
            "refuse_content": wilson_interval(sum(1 for t in ref if t.get("content_exact")), len(ref)),
            "cascade": {
                "retrieval_hit@5": wilson_interval(len(hit), n) if traces and traces[0].get("top5_mode") == "learned_top5" else None,
                "retrieval_miss": len(miss) if traces and traces[0].get("top5_mode") == "learned_top5" else 0,
                "pipeline_strict": wilson_interval(sum(1 for t in traces if t.get("pipeline_strict")), n),
                "generator_content_fail": sum(1 for t in traces if t.get("generator_content_fail")),
                "generator_format_fail": sum(1 for t in traces if t.get("generator_format_fail")),
            },
            "latency": {"p50": _pctl(walls, 0.5), "p95": _pctl(walls, 0.95)},
            "errors": sum(1 for t in traces if t.get("error")),
        }
    scored = [score_mw({"reason_code": t["gold"]}, raw_text=t.get("raw") or "") for t in traces]
    n = len(scored)
    walls = [t.get("wall_ms") or 0 for t in traces]
    conf = confusion(scored)
    return {
        "n": n,
        "content": wilson_interval(sum(1 for s in scored if s["content_ok"]), n),
        "format": wilson_interval(sum(1 for s in scored if s["format_ok"]), n),
        "strict_e2e": wilson_interval(sum(1 for s in scored if s["strict_e2e"]), n),
        "unsafe_execute": sum(1 for s in scored if s["unsafe_execute"]),
        "macro_f1": conf["macro_f1"],
        "per_class": conf["per_class"],
        "matrix": conf["matrix"],
        "latency": {"p50": _pctl(walls, 0.5), "p95": _pctl(walls, 0.95)},
    }


def selected_tools(row: dict, mode: str) -> list[dict]:
    key = "oracle_top5" if mode == "oracle_top5" else "learned_top5"
    return list(row.get(key) or [])


def generate_fullcall(row, *, top5_mode, infer_mode, chat_fn, system) -> dict:
    tools = selected_tools(row, top5_mode)
    user = render_fullcall_user(row, tools)
    kwargs: dict[str, Any] = {"system": system, "max_tokens": 128, "temperature": 0.0}
    if infer_mode == "native_tools":
        kwargs["tools"] = tools_to_ollama(tools)
        user = f"query：{row.get('query')}\n事实：{row.get('system_facts') or '无'}\n只能调用给出的工具之一或拒绝。"
    elif infer_mode == "format_constrained":
        kwargs["response_format"] = fullcall_json_schema()
    t0 = time.perf_counter()
    try:
        text, lat = chat_fn(user, **kwargs)
        err = None
    except Exception as exc:  # noqa: BLE001
        text, lat, err = "", {}, f"{exc.__class__.__name__}"
    wall = (lat or {}).get("wall_ms") or (time.perf_counter() - t0) * 1000
    scored = score_fullcall(row, raw_text=text or "", mode=infer_mode, tool_calls=(lat or {}).get("tool_calls"), selected_tools=tools)
    gold = list(row.get("answers") or [])
    hit = True if top5_mode == "oracle_top5" else bool(row.get("retrieval_hit_learned"))
    if top5_mode == "oracle_top5" and row.get("gold_name"):
        hit = row["gold_name"] in {t.get("name") for t in tools}
    cas = cascade_score(
        retrieval_hit=hit,
        generator_strict=scored["strict_e2e"],
        generator_content=scored["content"]["content_exact"],
        generator_format=scored["format_ok"],
    )
    if not hit:
        cas["pipeline_strict"] = False
    return {
        "item_id": row["item_id"],
        "task": "fullcall",
        "top5_mode": top5_mode,
        "infer_mode": infer_mode,
        "raw": text,
        "extracted": scored["extracted"],
        "gold": gold,
        "gold_execute": bool(gold),
        "content_exact": scored["content"]["content_exact"],
        "format_ok": scored["format_ok"],
        "format_reason": scored["format_reason"],
        "strict_e2e": scored["strict_e2e"],
        "legal": scored["legal"],
        "retrieval_hit": hit,
        "pipeline_strict": cas["pipeline_strict"],
        "generator_content_fail": cas["generator_content_fail"],
        "generator_format_fail": cas["generator_format_fail"],
        "wall_ms": wall,
        "error": err,
        "prompt_hash": prompt_asset()["fullcall_system_sha256"],
    }


def generate_mw(row, *, top5_mode, chat_fn, system) -> dict:
    tools = selected_tools(row, top5_mode)
    gold = row["learned_reason_code"] if top5_mode == "learned_top5" else row["reason_code"]
    user = render_mw_user(row, tools)
    t0 = time.perf_counter()
    try:
        text, lat = chat_fn(user, system=system, max_tokens=32, temperature=0.0)
        err = None
    except Exception as exc:  # noqa: BLE001
        text, lat, err = "", {}, exc.__class__.__name__
    wall = (lat or {}).get("wall_ms") or (time.perf_counter() - t0) * 1000
    scored = score_mw({**row, "reason_code": gold}, raw_text=text or "")
    return {
        "item_id": row["item_id"],
        "task": "mw",
        "top5_mode": top5_mode,
        "raw": text,
        "gold": gold,
        "pred": scored["pred"],
        "content_ok": scored["content_ok"],
        "format_ok": scored["format_ok"],
        "strict_e2e": scored["strict_e2e"],
        "unsafe_execute": scored["unsafe_execute"],
        "wall_ms": wall,
        "error": err,
        "prompt_hash": prompt_asset()["mw_system_sha256"],
    }


def mei58m_chat_factory(decode_mode: str):
    from runtime_v2 import RuntimeV2

    status = mei58m_status()
    if not status.get("available"):
        return None, status
    model, tok, report = load_mei58m()
    rt_holder = {"rt": None, "catalog_key": None}

    def _fn(user: str, **kwargs):
        # user is already the v2 prompt for full-call; RuntimeV2.complete rebuilds prompt.
        # We pass query via kwargs sidecar.
        row = kwargs.get("_row")
        tools = kwargs.get("_tools") or []
        query = str((row or {}).get("query") or user)
        facts = str((row or {}).get("system_facts") or "")
        rt = rt_holder["rt"]
        if rt is None:
            rt = RuntimeV2(model, tok, catalog=tools, confidence_threshold=0.0)
            rt_holder["rt"] = rt
        t0 = time.perf_counter()
        out = rt.complete(query, oracle_tools=tools, system_facts=facts, decode_mode=decode_mode, max_new=48)
        wall = (time.perf_counter() - t0) * 1000
        return out.get("text") or "", {"wall_ms": wall, "decode_mode": decode_mode, "selected": out.get("selected_tools")}

    return _fn, {**status, "load_report": {k: report.get(k) for k in ("n_loaded", "n_missing") if isinstance(report, dict)}}


def wrap_row_chat(fn, row, tools):
    def _inner(user, **kwargs):
        kwargs = dict(kwargs)
        kwargs["_row"] = row
        kwargs["_tools"] = tools
        return fn(user, **kwargs)

    return _inner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--lock-dir", type=Path, default=SFT_V2_EVAL_LOCK_DIR_V2)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--skip-58m-generate", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True, help="Reuse/continue existing trace JSONL files.")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--tasks", default="retrieval,fullcall,mw")
    args = ap.parse_args()
    lock_dir = args.lock_dir
    universe = load_universe(lock_dir)
    split = args.split
    ret_rows = load_jsonl(lock_dir / f"eval-retrieval-{split}.jsonl")
    fc_rows = load_jsonl(lock_dir / f"eval-fullcall-{split}.jsonl")
    mw_rows = load_jsonl(lock_dir / f"eval-mw-{split}.jsonl")
    if args.limit:
        ret_rows, fc_rows, mw_rows = ret_rows[: args.limit], fc_rows[: args.limit], mw_rows[: args.limit]
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    prompt = prompt_asset()
    retrieval_rows_out = {}
    if "retrieval" in tasks:
        retrieval_rows_out["bm25"] = run_sparse(ret_rows, universe, "bm25")
        retrieval_rows_out["char-tfidf"] = run_sparse(ret_rows, universe, "tfidf")
        st58 = mei58m_status()
        if st58.get("available"):
            import sys as _sys

            _sys.path.insert(0, str(ROOT / "notebook/_tooling/model/mei-1.0-58m"))
            model, tok, _rep = load_mei58m()
            enc = mei58m_encode_fn(model, tok)
            retrieval_rows_out["mei-58m-contrastive-no-sft"] = run_dense(ret_rows, universe, enc, "mei-58m-contrastive-no-sft")
            hnsw = HnswRetriever(enc, name="mei-58m-hnsw")
            hnsw.build(universe["tools"][:128])
            retrieval_rows_out["mei-58m-hnsw"] = {
                "available": hnsw.available,
                "error": hnsw.error,
                "build_ms": hnsw.build_ms,
                "note": "efficiency column on 128-tool slice; quality remains exact cosine/dot",
            }
        else:
            retrieval_rows_out["mei-58m-contrastive-no-sft"] = st58
        for hid, label in (
            ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5"),
            ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "minilm-multilingual"),
        ):
            fn, meta = try_sentence_transformer(hid)
            if fn is None:
                retrieval_rows_out[label] = meta
            else:
                retrieval_rows_out[label] = {**run_dense(ret_rows, universe, fn, label), "meta": meta}

    gen_results = {}
    traces_written = []

    def store_traces(name: str, rows: list[dict]):
        path = args.trace_dir / f"{name}.{split}.jsonl"
        dump_jsonl(path, rows)
        traces_written.append(rel(path))
        return path

    def run_row_traces(name: str, rows: list[dict], build_one):
        path = args.trace_dir / f"{name}.{split}.jsonl"
        existing = load_jsonl(path) if path.is_file() else []
        if args.resume and len(existing) >= len(rows):
            traces_written.append(rel(path))
            return existing[: len(rows)]
        if not args.resume:
            existing = []
        path.parent.mkdir(parents=True, exist_ok=True)
        start = len(existing) if args.resume else 0
        with path.open("a" if start else "w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                if i < start:
                    continue
                rec = build_one(row)
                existing.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0:
                    fh.flush()
                    print(f"{name} {i+1}/{len(rows)}", flush=True)
        traces_written.append(rel(path))
        return existing[: len(rows)]

    if "fullcall" in tasks:
        # floors
        def refuse_chat(_user, **_k):
            return "[]", {"wall_ms": 0.0}

        for top5 in ("oracle_top5", "learned_top5"):
            traces = run_row_traces(
                f"always-refuse.fullcall.{top5}",
                fc_rows,
                lambda r, top5=top5: generate_fullcall(r, top5_mode=top5, infer_mode="prompt_adapted_raw", chat_fn=refuse_chat, system=fullcall_system()),
            )
            gen_results.setdefault("always-refuse", {}).setdefault("fullcall", {})[top5] = summarize_from_traces(traces, task="fullcall")

        if not args.skip_58m_generate and mei58m_status().get("available"):
            import sys as _sys

            _sys.path.insert(0, str(ROOT / "notebook/_tooling/model/mei-1.0-58m"))
            for dec in ("raw", "constrained"):
                fn, meta = mei58m_chat_factory(dec)
                if fn is None:
                    gen_results.setdefault("mei-58m-base-300m-no-sft", {})[dec] = meta
                    continue
                for top5 in ("oracle_top5", "learned_top5"):
                    traces = run_row_traces(
                        f"mei-58m.{dec}.fullcall.{top5}",
                        fc_rows,
                        lambda row, top5=top5, dec=dec, fn=fn: generate_fullcall(
                            row,
                            top5_mode=top5,
                            infer_mode="prompt_adapted_raw" if dec == "raw" else "format_constrained",
                            chat_fn=wrap_row_chat(fn, row, selected_tools(row, top5)),
                            system=fullcall_system(),
                        ),
                    )
                    gen_results.setdefault("mei-58m-base-300m-no-sft", {}).setdefault("fullcall", {}).setdefault(dec, {})[top5] = summarize_from_traces(traces, task="fullcall")

        if not args.skip_qwen:
            for mid, tag, params in QWEN_MODELS:
                ok, why = ollama_available(args.host, tag, timeout=8)
                if not ok:
                    gen_results[mid] = {"status": "unavailable", "reason": why, "params": params, "comparable": False}
                    continue
                chat = qwen_ollama_factory(args.host, tag, args.timeout)
                gen_results.setdefault(mid, {"params": params, "ollama_tag": tag})
                for infer in GEN_MODES:
                    for top5 in ("oracle_top5", "learned_top5"):
                        traces = run_row_traces(
                            f"{mid}.{infer}.fullcall.{top5}",
                            fc_rows,
                            lambda r, infer=infer, top5=top5, chat=chat: generate_fullcall(
                                r, top5_mode=top5, infer_mode=infer, chat_fn=chat, system=fullcall_system()
                            ),
                        )
                        gen_results[mid].setdefault("fullcall", {}).setdefault(infer, {})[top5] = summarize_from_traces(traces, task="fullcall")

    if "mw" in tasks:
        majority = "ready_to_execute"

        def maj(_u, **_k):
            return majority, {"wall_ms": 0.0}

        for top5 in ("oracle_top5", "learned_top5"):
            traces = run_row_traces(
                f"majority-mw.{top5}",
                mw_rows,
                lambda r, top5=top5: generate_mw(r, top5_mode=top5, chat_fn=maj, system=mw_codebook_block()),
            )
            gen_results.setdefault("majority-mw", {}).setdefault("mw", {})[top5] = summarize_from_traces(traces, task="mw")
        if not args.skip_qwen:
            for mid, tag, params in QWEN_MODELS:
                if gen_results.get(mid, {}).get("status") == "unavailable":
                    continue
                ok, why = ollama_available(args.host, tag, timeout=8)
                if not ok:
                    gen_results.setdefault(mid, {"status": "unavailable", "reason": why})
                    continue
                chat = qwen_ollama_factory(args.host, tag, args.timeout)
                for top5 in ("oracle_top5", "learned_top5"):
                    traces = run_row_traces(
                        f"{mid}.raw.mw.{top5}",
                        mw_rows,
                        lambda r, top5=top5, chat=chat: generate_mw(r, top5_mode=top5, chat_fn=chat, system=mw_codebook_block()),
                    )
                    gen_results.setdefault(mid, {}).setdefault("mw", {})[top5] = summarize_from_traces(traces, task="mw")

    # strip traces from retrieval payload for the aggregate json
    retrieval_compact = {}
    for k, v in retrieval_rows_out.items():
        if isinstance(v, dict) and "traces" in v:
            store_traces(f"retrieval.{k}", v["traces"])
            retrieval_compact[k] = {kk: vv for kk, vv in v.items() if kk != "traces"}
        else:
            retrieval_compact[k] = v

    payload = {
        "id": "sft-v2-baseline-scorecard-v2",
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "split": split,
        "lock_dir": rel(lock_dir),
        "lock_sha256": sha256_file(lock_dir / "lock.json") if (lock_dir / "lock.json").is_file() else None,
        "contract": rel(SFT_V2_BASELINE_CONTRACT_V2),
        "models_matrix": rel(SFT_V2_BASELINE_MODELS_V2),
        "prompt": {k: prompt[k] for k in ("prompt_version", "fullcall_system_sha256", "mw_system_sha256")},
        "qwen_0.6b": "not_invented",
        "minimind": {"25m": minimind_status("25m"), "45m": minimind_status("45m")},
        "retrieval": retrieval_compact,
        "generation": gen_results,
        "traces": traces_written,
        "memory": _peak_mem(),
        "note": "v1 scorecard remains a prompt-contract diagnostic and must not be overwritten.",
        "current_json": "unchanged",
        "training": "not_run",
    }
    dump_json(args.out, payload)
    print(json.dumps({"ok": True, "out": rel(args.out), "n_traces": len(traces_written), "retrieval": list(retrieval_compact)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
