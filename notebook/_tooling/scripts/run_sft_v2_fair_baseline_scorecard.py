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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

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
    MEI51M_SDK_BACKEND,
    load_mei51m_sdk_engine,
    mei51m_chat_factory,
    mei51m_status,
    minimind_status,
    ollama_available,
    qwen_ollama_factory,
    resolve_promoted_51m_base,
    sdk_backend_revision,
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
from sft_v2_scorecard_metrics import summarize_column
from sft_v2_retrieval_backends import (
    DenseExactRetriever,
    HnswRetriever,
    SparseRetriever,
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
    traces = []
    for row in rows:
        catalog = hydrate_catalog(row, universe)
        t0 = time.perf_counter()
        order = search_rank(str(row.get("query") or ""), catalog)
        wall = (time.perf_counter() - t0) * 1000
        gold = row.get("gold_tool")
        if row.get("no_match") or not gold:
            traces.append(
                {
                    "item_id": row["item_id"],
                    "family": "no_match",
                    "rank": -1,
                    "pred_top5": order[:5],
                    "wall_ms": wall,
                    "catalog_size": row.get("catalog_size"),
                    "no_match": True,
                }
            )
            continue
        rank = order.index(gold) if gold in order else -1
        traces.append(
            {
                "item_id": row["item_id"],
                "family": row.get("family"),
                "gold": gold,
                "rank": rank,
                "pred_top5": order[:5],
                "wall_ms": wall,
                "catalog_size": row.get("catalog_size"),
                "seen_schema": row.get("seen_schema"),
                "no_match": False,
            }
        )
    summary = summarize_column(traces, task="retrieval")
    return {"retriever": name, **summary, "traces": traces}


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


def summarize_from_traces(traces: list[dict], *, task: str, bank_by_id: dict[str, dict] | None = None) -> dict:
    return summarize_column(traces, task=task, bank_by_id=bank_by_id)


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
        "prompt_tokens": (lat or {}).get("prompt_tokens"),
        "output_tokens": (lat or {}).get("output_tokens"),
        "output_tok_s": (lat or {}).get("output_tok_s"),
        "family": row.get("family"),
        "slice": row.get("slice"),
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
        "prompt_tokens": (lat or {}).get("prompt_tokens"),
        "output_tokens": (lat or {}).get("output_tokens"),
        "output_tok_s": (lat or {}).get("output_tok_s"),
        "family": row.get("family"),
        "error": err,
        "prompt_hash": prompt_asset()["mw_system_sha256"],
    }


def wrap_row_chat(fn, row, tools):
    def _inner(user, **kwargs):
        kwargs = dict(kwargs)
        kwargs["_row"] = row
        kwargs["_tools"] = tools
        return fn(user, **kwargs)

    return _inner


INFER_MODES = GEN_MODES


def parse_trace_stem(stem: str) -> dict[str, str] | None:
    if "." not in stem:
        return None
    body, split = stem.rsplit(".", 1)
    if split not in {"dev", "test"}:
        return None
    if body.startswith("retrieval."):
        return {"task": "retrieval", "name": body[len("retrieval.") :], "split": split}
    if ".fullcall." in body:
        left, top5 = body.split(".fullcall.", 1)
        return {"task": "fullcall", "name": left, "top5": top5, "split": split}
    if body.startswith("majority-mw."):
        return {"task": "mw", "name": "majority-mw", "top5": body.split(".", 1)[1], "split": split}
    if ".mw." in body:
        left, top5 = body.split(".mw.", 1)
        if left.endswith(".raw"):
            left = left[: -len(".raw")]
        return {"task": "mw", "name": left, "top5": top5, "split": split}
    return None


def nest_summary(generation: dict, retrieval: dict, parsed: dict, summary: dict) -> None:
    task = parsed["task"]
    name = parsed["name"]
    if task == "retrieval":
        retrieval[name] = summary
        return
    if task == "fullcall":
        top5 = parsed["top5"]
        if name.startswith("mei-51m."):
            dec = name.split(".", 1)[1]
            generation.setdefault("mei-51m-base-300m-no-sft", {}).setdefault("fullcall", {}).setdefault(dec, {})[top5] = summary
            return
        if name == "always-refuse":
            generation.setdefault("always-refuse", {}).setdefault("fullcall", {})[top5] = summary
            return
        for infer in INFER_MODES:
            suffix = "." + infer
            if name.endswith(suffix):
                mid = name[: -len(suffix)]
                generation.setdefault(mid, {}).setdefault("fullcall", {}).setdefault(infer, {})[top5] = summary
                return
        generation.setdefault(name, {}).setdefault("fullcall", {})[top5] = summary
        return
    generation.setdefault(name, {}).setdefault("mw", {})[parsed["top5"]] = summary


def aggregate_from_traces(args) -> int:
    split = args.split
    lock_dir = args.lock_dir
    fc_by_id = {str(r["item_id"]): r for r in load_jsonl(lock_dir / f"eval-fullcall-{split}.jsonl")}
    mw_by_id = {str(r["item_id"]): r for r in load_jsonl(lock_dir / f"eval-mw-{split}.jsonl")}
    generation: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}
    traces_written = []
    for path in sorted(args.trace_dir.glob(f"*.{split}.jsonl")):
        parsed = parse_trace_stem(path.stem)
        if not parsed:
            continue
        rows = load_jsonl(path)
        if not rows:
            continue
        traces_written.append(rel(path))
        if parsed["task"] == "retrieval":
            summary = summarize_column(rows, task="retrieval")
        elif parsed["task"] == "mw":
            summary = summarize_column(rows, task="mw", bank_by_id=mw_by_id)
        else:
            summary = summarize_column(rows, task="fullcall", bank_by_id=fc_by_id)
        summary["partial"] = True
        nest_summary(generation, retrieval, parsed, summary)
    prompt = prompt_asset()
    payload = {
        "id": "sft-v2-baseline-scorecard-v2",
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "split": split,
        "source": "aggregate_from_traces",
        "lock_dir": rel(lock_dir),
        "lock_sha256": sha256_file(lock_dir / "lock.json") if (lock_dir / "lock.json").is_file() else None,
        "contract": rel(SFT_V2_BASELINE_CONTRACT_V2),
        "models_matrix": rel(SFT_V2_BASELINE_MODELS_V2),
        "prompt": {k: prompt[k] for k in ("prompt_version", "fullcall_system_sha256", "mw_system_sha256")},
        "qwen_0.6b": "not_invented",
        "minimind": {"25m": minimind_status("25m"), "45m": minimind_status("45m")},
        "retrieval": retrieval,
        "generation": generation,
        "traces": traces_written,
        "mei51m_checkpoint": resolve_promoted_51m_base(),
        "scorecard_metrics": {
            "required": ["accuracy", "rate"],
            "accuracy_primary": {"retrieval": "recall_at_5", "fullcall": "strict_e2e", "mw": "strict_e2e"},
            "rate": ["items_per_s", "items_per_min", "mean_ms", "p50_ms", "p95_ms", "total_wall_s"],
        },
        "note": "Aggregated from traces. Partial columns are allowed. Each column has accuracy + sequential rate.",
        "current_json": "sft_and_runtime_unchanged",
        "training": "not_run",
    }
    dump_json(args.out, payload)
    print(
        json.dumps(
            {"ok": True, "out": rel(args.out), "n_traces": len(traces_written), "retrieval": list(retrieval), "generation": list(generation)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


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
    ap.add_argument("--skip-51m-generate", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True, help="Reuse/continue existing trace JSONL files.")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--tasks", default="retrieval,fullcall,mw")
    ap.add_argument(
        "--aggregate-from-traces",
        action="store_true",
        help="Rebuild scorecard accuracy+rate from existing JSONL traces; do not call models.",
    )
    args = ap.parse_args()
    if args.aggregate_from_traces:
        return aggregate_from_traces(args)
    lock_dir = args.lock_dir
    print(
        json.dumps(
            {
                "mei51m_checkpoint": resolve_promoted_51m_base(),
                "split": args.split,
                "tasks": args.tasks,
                "skip_qwen": args.skip_qwen,
                "skip_51m_generate": args.skip_51m_generate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    universe = load_universe(lock_dir)
    split = args.split
    ret_rows = load_jsonl(lock_dir / f"eval-retrieval-{split}.jsonl")
    fc_rows = load_jsonl(lock_dir / f"eval-fullcall-{split}.jsonl")
    mw_rows = load_jsonl(lock_dir / f"eval-mw-{split}.jsonl")
    if args.limit:
        ret_rows, fc_rows, mw_rows = ret_rows[: args.limit], fc_rows[: args.limit], mw_rows[: args.limit]
    fc_by_id = {str(r["item_id"]): r for r in fc_rows}
    mw_by_id = {str(r["item_id"]): r for r in mw_rows}
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    prompt = prompt_asset()
    retrieval_rows_out = {}
    if "retrieval" in tasks:
        retrieval_rows_out["bm25"] = run_sparse(ret_rows, universe, "bm25")
        retrieval_rows_out["char-tfidf"] = run_sparse(ret_rows, universe, "tfidf")
        st58 = mei51m_status()
        if st58.get("available"):
            engine = load_mei51m_sdk_engine()
            session = engine.create_session()

            def enc(text: str):
                return session.embed(text)

            dense = run_dense(ret_rows, universe, enc, "mei-51m-contrastive-no-sft")
            for t in dense.get("traces") or []:
                t["checkpoint"] = st58.get("path")
                t["checkpoint_sha256"] = st58.get("weights_sha256")
                t["via_sdk"] = True
                t["eval_surface"] = "mei_sdk.embed"
                t["sdk_backend"] = MEI51M_SDK_BACKEND
                t["sdk_backend_revision"] = sdk_backend_revision()
                t["performance_profile"] = "clean"
            retrieval_rows_out["mei-51m-contrastive-no-sft"] = {
                **dense,
                "checkpoint": {k: st58.get(k) for k in ("path", "weights_sha256", "model_id", "abandoned_archive")},
            }
            hnsw = HnswRetriever(enc, name="mei-51m-hnsw")
            hnsw.build(universe["tools"][:128])
            retrieval_rows_out["mei-51m-hnsw"] = {
                "available": hnsw.available,
                "error": hnsw.error,
                "build_ms": hnsw.build_ms,
                "note": "efficiency column on 128-tool slice; quality remains exact cosine/dot",
            }
        else:
            retrieval_rows_out["mei-51m-contrastive-no-sft"] = st58
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
    retrieval_compact: dict[str, Any] = {}

    def store_traces(name: str, rows: list[dict]):
        path = args.trace_dir / f"{name}.{split}.jsonl"
        dump_jsonl(path, rows)
        traces_written.append(rel(path))
        return path

    def run_row_traces(name: str, rows: list[dict], build_one):
        path = args.trace_dir / f"{name}.{split}.jsonl"
        existing = load_jsonl(path) if path.is_file() else []
        if name.startswith("mei-51m"):
            want_sha = resolve_promoted_51m_base().get("weights_sha256")
            got_sha = (existing[0] or {}).get("checkpoint_sha256") if existing else None
            via_sdk = bool(existing and existing[0].get("via_sdk"))
            got_rev = (existing[0] or {}).get("sdk_backend_revision") if existing else None
            want_rev = sdk_backend_revision()
            profile = (existing[0] or {}).get("performance_profile") if existing else None
            if existing and got_sha != want_sha:
                print(
                    f"discard {name}: trace checkpoint {got_sha} != promoted {want_sha}",
                    flush=True,
                )
                existing = []
            elif existing and not via_sdk:
                print(f"discard {name}: traces did not go through mei_sdk.complete", flush=True)
                existing = []
            elif existing and got_rev != want_rev:
                print(
                    f"discard {name}: sdk_backend_revision {got_rev} != {want_rev}",
                    flush=True,
                )
                existing = []
            elif existing and profile in {"preopt/contended", "contended"}:
                print(f"discard {name}: performance_profile={profile} is not a clean baseline", flush=True)
                existing = []
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

    for k, v in retrieval_rows_out.items():
        if isinstance(v, dict) and "traces" in v:
            store_traces(f"retrieval.{k}", v["traces"])
            retrieval_compact[k] = {kk: vv for kk, vv in v.items() if kk != "traces"}
        else:
            retrieval_compact[k] = v

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
            gen_results.setdefault("always-refuse", {}).setdefault("fullcall", {})[top5] = summarize_from_traces(
                traces, task="fullcall", bank_by_id=fc_by_id
            )

        if not args.skip_51m_generate and mei51m_status().get("available"):
            for dec in ("raw", "constrained"):
                fn, meta = mei51m_chat_factory(dec)
                if fn is None:
                    gen_results.setdefault("mei-51m-base-300m-no-sft", {})[dec] = meta
                    continue
                ckpt_meta = resolve_promoted_51m_base()

                def _stamp_51m(rec: dict, lat: dict | None = None) -> dict:
                    rec["checkpoint"] = ckpt_meta.get("path")
                    rec["checkpoint_sha256"] = ckpt_meta.get("weights_sha256")
                    rec["base_model_id"] = ckpt_meta.get("model_id")
                    rec["via_sdk"] = True
                    rec["eval_surface"] = "mei_sdk.complete"
                    rec["sdk_backend"] = MEI51M_SDK_BACKEND
                    rec["sdk_backend_revision"] = sdk_backend_revision()
                    rec["performance_profile"] = "clean"
                    rec["max_new"] = 128
                    rec["prompt_hash"] = "mei-tool-call-serializer-v2"
                    return rec

                for top5 in ("oracle_top5", "learned_top5"):
                    traces = run_row_traces(
                        f"mei-51m.{dec}.fullcall.{top5}",
                        fc_rows,
                        lambda row, top5=top5, dec=dec, fn=fn: _stamp_51m(
                            {
                                **generate_fullcall(
                                    row,
                                    top5_mode=top5,
                                    infer_mode="prompt_adapted_raw",
                                    chat_fn=wrap_row_chat(fn, row, selected_tools(row, top5)),
                                    system="",
                                ),
                                "decode_mode": dec,
                            }
                        ),
                    )
                    gen_results.setdefault("mei-51m-base-300m-no-sft", {}).setdefault("fullcall", {}).setdefault(dec, {})[top5] = summarize_from_traces(
                        traces, task="fullcall", bank_by_id=fc_by_id
                    )

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
                        gen_results[mid].setdefault("fullcall", {}).setdefault(infer, {})[top5] = summarize_from_traces(
                            traces, task="fullcall", bank_by_id=fc_by_id
                        )

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
            gen_results.setdefault("majority-mw", {}).setdefault("mw", {})[top5] = summarize_from_traces(traces, task="mw", bank_by_id=mw_by_id)
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
                    gen_results.setdefault(mid, {}).setdefault("mw", {})[top5] = summarize_from_traces(traces, task="mw", bank_by_id=mw_by_id)

    # retrieval traces already persisted before generation
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
        "mei51m_checkpoint": resolve_promoted_51m_base(),
        "scorecard_metrics": {
            "required": ["accuracy", "rate"],
            "accuracy_primary": {"retrieval": "recall_at_5", "fullcall": "strict_e2e", "mw": "strict_e2e"},
            "rate": ["items_per_s", "items_per_min", "mean_ms", "p50_ms", "p95_ms", "total_wall_s"],
        },
        "note": "v1 scorecard remains a prompt-contract diagnostic and must not be overwritten. 51M columns use CURRENT.json promoted base only; archive 300M parents are not eval targets. Each column reports accuracy (with splits) and sequential rate.",
        "current_json": "sft_and_runtime_unchanged",
        "training": "not_run",
    }
    dump_json(args.out, payload)
    print(json.dumps({"ok": True, "out": rel(args.out), "n_traces": len(traces_written), "retrieval": list(retrieval_compact)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
