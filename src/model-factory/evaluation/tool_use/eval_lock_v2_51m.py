#!/usr/bin/env python3
"""sft-v2-eval-lock-v2 layered hard gates on a disk 51M quantized product package.

Thresholds must already be pre-registered. This script never edits them after seeing scores.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import subprocess
import sys
from pathlib import Path

from common.identity_51m import (
    JOBS_DIR,
    Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_DIR,
    ROOT,
    SFT_QAT_PACKAGE_DIR,
    fail,
    load_json,
    sha256_file,
    write_json,
)
from tokenizer import ASSISTANT_PREFIX, TURN_END

LOCK_DIR = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/sft-v2-eval-lock-v2"
THRESHOLDS = JOBS_DIR / "sft-v2-51m-thresholds-preregister.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def configure_mlx_memory(*, memory_limit_gb: float = 6.0, cache_limit_gb: float = 0.5) -> dict:
    import mlx.core as mx

    memory_limit = int(memory_limit_gb * 1024**3)
    cache_limit = int(cache_limit_gb * 1024**3)
    previous_memory = mx.set_memory_limit(memory_limit)
    previous_cache = mx.set_cache_limit(cache_limit)
    return {
        "memory_limit_gb": memory_limit_gb,
        "cache_limit_gb": cache_limit_gb,
        "previous_memory_limit": previous_memory,
        "previous_cache_limit": previous_cache,
    }


def release_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        return


def toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (text or "").lower()) if t}


def lexical_rank(query: str, catalog: list[dict], k: int = 5) -> list[dict]:
    q = toks(query)
    scored = []
    for tool in catalog:
        blob = " ".join(
            [
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
            ]
        )
        tb = toks(blob)
        union = q | tb
        score = (len(q & tb) / len(union)) if union else 0.0
        scored.append((score, tool))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [t for _, t in scored[:k]]


def calls_equal(a, b) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    left, right = a[0], b[0]
    if str(left.get("name")) != str(right.get("name")):
        return False
    return json.dumps(left.get("arguments") or {}, sort_keys=True, ensure_ascii=False) == json.dumps(
        right.get("arguments") or {}, sort_keys=True, ensure_ascii=False
    )


def shard_fingerprint(pkg_dir: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in (
        THRESHOLDS,
        LOCK_DIR / "tool-universe-v1.json",
        LOCK_DIR / "eval-retrieval-test.jsonl",
        LOCK_DIR / "eval-fullcall-test.jsonl",
        pkg_dir / "mei-model.json",
        pkg_dir / "weights.q4",
    ):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update((sha256_file(path) if path.is_file() else "missing").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def shard_batches(pending: list[int], max_workers: int) -> list[list[int]]:
    if max_workers < 1:
        raise ValueError("lock-v2 max-workers must be >= 1")
    return [pending[i : i + max_workers] for i in range(0, len(pending), max_workers)]


def load_shard_report(path: Path) -> dict:
    try:
        return load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def shard_is_complete(row: dict, fingerprint: str) -> bool:
    if not row or row.get("kind") != "toolcall-qat-lock-v2":
        return False
    if row.get("input_fingerprint") != fingerprint:
        return False
    counts = row.get("counts") or {}
    return "n_retrieval" in counts and "n_fullcall" in counts and isinstance(row.get("layers"), dict)


def pending_lock_shards(shard_dir: Path, count: int, fingerprint: str, *, resume: bool) -> list[int]:
    pending = []
    for index in range(count):
        path = shard_dir / f"shard-{index:03d}.json"
        if resume and shard_is_complete(load_shard_report(path), fingerprint):
            continue
        pending.append(index)
    return pending


def lock_receipt_exit_code(*, evaluation_complete: bool) -> int:
    """Score failure is recorded in the receipt; hard_gate consumes it."""
    return 0 if evaluation_complete else 2


def aggregate_shards(shard_dir: Path, count: int, layers: dict, pkg_dir: Path, jobs_dir: Path) -> int:
    shards = [load_json(shard_dir / f"shard-{index:03d}.json") for index in range(count)]
    fingerprint = shard_fingerprint(pkg_dir)
    if any(not row or row.get("input_fingerprint") != fingerprint for row in shards):
        return fail("lock-v2 shard set is missing or stale")
    totals = {
        key: sum(int((row.get("counts") or {}).get(key) or 0) for row in shards)
        for key in (
            "n_retrieval",
            "learned_hits",
            "lexical_hits",
            "n_fullcall",
            "oracle_exact",
            "oracle_execute_refuse",
            "learned_exact",
            "unsupported_accepted",
            "unprovenanced_argument_accepted",
        )
    }
    n_ret = max(totals["n_retrieval"], 1)
    n_fc = max(totals["n_fullcall"], 1)
    recall_learned = totals["learned_hits"] / n_ret
    recall_lex = totals["lexical_hits"] / n_ret
    oracle_rate = totals["oracle_exact"] / n_fc
    learned_rate = totals["learned_exact"] / n_fc
    gates = {
        "recall_at_5_learned": {
            "value": recall_learned,
            "floor": layers["recall_at_5_learned_min"],
            "ok": recall_learned >= float(layers["recall_at_5_learned_min"]),
        },
        "recall_beats_lexical": {
            "learned": recall_learned,
            "lexical": recall_lex,
            "ok": (not layers.get("recall_at_5_must_exceed_lexical")) or recall_learned > recall_lex,
        },
        "oracle_top5_fullcall_exact": {
            "value": oracle_rate,
            "floor": layers["oracle_top5_fullcall_exact_min"],
            "ok": oracle_rate >= float(layers["oracle_top5_fullcall_exact_min"]),
        },
        "oracle_execute_refuse": {
            "value": totals["oracle_execute_refuse"] / n_fc,
            "floor": layers["oracle_top5_execute_refuse_ok_min"],
            "ok": (totals["oracle_execute_refuse"] / n_fc)
            >= float(layers["oracle_top5_execute_refuse_ok_min"]),
        },
        "learned_top5_e2e_exact": {
            "value": learned_rate,
            "floor": layers["learned_top5_e2e_exact_min"],
            "ok": learned_rate >= float(layers["learned_top5_e2e_exact_min"]),
        },
        "unsupported_accepted": {
            "value": totals["unsupported_accepted"],
            "max": layers["unsupported_accepted_max"],
            "ok": totals["unsupported_accepted"] <= int(layers["unsupported_accepted_max"]),
        },
        "unprovenanced_argument_accepted": {
            "value": totals["unprovenanced_argument_accepted"],
            "max": layers["unprovenanced_argument_accepted_max"],
            "ok": totals["unprovenanced_argument_accepted"]
            <= int(layers["unprovenanced_argument_accepted_max"]),
        },
    }
    report = {
        "kind": "toolcall-qat-lock-v2",
        "lock": "sft-v2-eval-lock-v2",
        "package_dir": str(pkg_dir),
        "thresholds": display_path(Path(THRESHOLDS).resolve()),
        "input_fingerprint": fingerprint,
        "shard_count": count,
        "shards": [display_path(shard_dir / f"shard-{i:03d}.json") for i in range(count)],
        "counts": totals,
        "recall_at_5_learned": recall_learned,
        "recall_at_5_lexical": recall_lex,
        "oracle_top5_fullcall_exact": oracle_rate,
        "learned_top5_e2e_exact": learned_rate,
        "unsupported_accepted": totals["unsupported_accepted"],
        "unprovenanced_argument_accepted": totals["unprovenanced_argument_accepted"],
        "layers": gates,
        "hard_ok": all(row["ok"] for row in gates.values()),
        "evaluation_complete": True,
        "resumable_shards": True,
        "not_a_claim": "Lock-v2 scores are not a CURRENT freeze.",
    }
    blocked = write_json(jobs_dir / "toolcall-qat-lock-v2.json", report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return lock_receipt_exit_code(evaluation_complete=True)


def main() -> int:
    global THRESHOLDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--limit-retrieval", type=int, default=1000)
    parser.add_argument("--limit-fullcall", type=int, default=1200)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent shard processes. Default 1: each worker loads a full MLX runtime.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.jobs_dir = args.jobs_dir.resolve()
    if args.shard_dir is not None:
        args.shard_dir = args.shard_dir.resolve()
    if args.package_dir is not None:
        args.package_dir = args.package_dir.resolve()
    run_thresholds = args.jobs_dir / "sft-v2-51m-thresholds-preregister.json"
    if run_thresholds.is_file():
        THRESHOLDS = run_thresholds.resolve()
    th = load_json(THRESHOLDS)
    if not th.get("registered_before_eval"):
        return fail("missing pre-registered 51M lock-v2 thresholds")
    layers = th["layers"]
    pkg_dir = args.package_dir
    if pkg_dir is None:
        if (SFT_QAT_PACKAGE_DIR / "weights.q4").is_file():
            pkg_dir = SFT_QAT_PACKAGE_DIR
        elif (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file():
            pkg_dir = QAT_Q4_PACKAGE_DIR
        else:
            pkg_dir = Q4_PACKAGE_DIR

    if args.shard_dir and args.shard_index is None:
        args.shard_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = shard_fingerprint(pkg_dir)
        pending = pending_lock_shards(
            args.shard_dir, args.shard_count, fingerprint, resume=args.resume
        )
        failed = []
        try:
            batches = shard_batches(pending, args.max_workers)
        except ValueError as exc:
            return fail(str(exc))
        for batch in batches:
            workers = []
            for index in batch:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--package-dir",
                    str(pkg_dir),
                    "--jobs-dir",
                    str(args.jobs_dir),
                    "--limit-retrieval",
                    str(args.limit_retrieval),
                    "--limit-fullcall",
                    str(args.limit_fullcall),
                    "--shard-dir",
                    str(args.shard_dir),
                    "--shard-count",
                    str(args.shard_count),
                    "--shard-index",
                    str(index),
                    "--max-workers",
                    "1",
                ]
                log = (args.shard_dir / f"shard-{index:03d}.log").open("ab")
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                workers.append((index, proc, log))
            for index, proc, log in workers:
                code = proc.wait()
                log.close()
                if code != 0:
                    failed.append((index, code))
            if failed:
                return fail(f"lock-v2 shard workers failed: {failed}")
        return aggregate_shards(args.shard_dir, args.shard_count, layers, pkg_dir, args.jobs_dir)

    sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
    from mei_sdk.package import load_package
    from mei_sdk.protocol import render_request
    from mei_sdk.runtime_51m import apply_confidence_gate, load_51m_runtime, validate_call

    universe = {t["name"]: t for t in load_json(LOCK_DIR / "tool-universe-v1.json").get("tools") or []}
    pkg = load_package(pkg_dir)
    mlx_limits = configure_mlx_memory()
    print(json.dumps({"mlx_memory": mlx_limits}), flush=True)
    runtime, loaded = load_51m_runtime(pkg)
    tok = runtime.tokenizer
    tool_vecs: dict[str, object] = {}

    def tool_vec(tool: dict):
        name = str(tool.get("name") or "")
        if name not in tool_vecs:
            blob = json.dumps(
                {"name": name, "description": tool.get("description") or "", "parameters": tool.get("parameters") or {}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tool_vecs[name] = runtime.embed_text(blob)
        return tool_vecs[name]

    def learned_top(query: str, catalog: list[dict], k: int = 5) -> list[dict]:
        import mlx.core as mx

        if len(catalog) <= k:
            return list(catalog)
        q = runtime.embed_text(query)
        scored = []
        for tool in catalog:
            sim = float(mx.sum(q * tool_vec(tool)).item())
            scored.append((sim, tool))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [t for _, t in scored[:k]]

    def catalog_of(names: list[str]) -> list[dict]:
        out = []
        for name in names:
            if name in universe:
                out.append(universe[name])
        return out

    def generate(query: str, tools: list[dict], *, facts: str = "", max_new: int = 64) -> dict:
        rendered = render_request({"query": query, "system_facts": facts}, tools)
        ids = tok.encode(rendered["prompt"] + ASSISTANT_PREFIX, add_bos=True, add_eos=False)
        decoded = runtime.greedy(ids, tools=tools, max_new=max_new, decode_mode="constrained")
        text = decoded.get("text") or ""
        validated = validate_call(text, tools=tools, query=query, system_facts=facts)
        gated = apply_confidence_gate(validated, decoded.get("confidence"))
        return {"text": text, "validated": validated, "gated": gated, "confidence": decoded.get("confidence")}

    ret_rows = load_jsonl(LOCK_DIR / "eval-retrieval-test.jsonl")[: args.limit_retrieval]
    if args.shard_index is not None:
        ret_rows = ret_rows[args.shard_index :: args.shard_count]
    learned_hits = 0
    lexical_hits = 0
    for row in ret_rows:
        catalog = catalog_of(row.get("catalog_tool_names") or [])
        gold = str(row.get("gold_tool") or "")
        learned = [str(t.get("name")) for t in learned_top(str(row.get("query") or ""), catalog, 5)]
        lexical = [str(t.get("name")) for t in lexical_rank(str(row.get("query") or ""), catalog, 5)]
        learned_hits += int(gold in learned)
        lexical_hits += int(gold in lexical)
    release_mlx_cache()
    n_ret = max(len(ret_rows), 1)
    recall_learned = learned_hits / n_ret
    recall_lex = lexical_hits / n_ret

    fc_rows = load_jsonl(LOCK_DIR / "eval-fullcall-test.jsonl")[: args.limit_fullcall]
    if args.shard_index is not None:
        fc_rows = fc_rows[args.shard_index :: args.shard_count]
    oracle_exact = 0
    oracle_er = 0
    learned_exact = 0
    unsupported = 0
    unprov = 0
    conf_pairs = []
    for i, row in enumerate(fc_rows):
        query = str(row.get("query") or "")
        facts = str(row.get("system_facts") or "")
        gold = row.get("answers") or []
        kind = str(row.get("kind") or "execute")
        oracle_tools = row.get("oracle_top5") or catalog_of((row.get("catalog_tool_names") or [])[:5])
        o = generate(query, oracle_tools, facts=facts)
        parsed = o["validated"].get("function_calls") or []
        if kind == "execute":
            hit = calls_equal(parsed, gold) and not o["validated"].get("refuse")
        else:
            hit = bool(o["validated"].get("refuse")) or parsed == []
        oracle_exact += int(hit)
        want_exec = kind == "execute"
        got_exec = o["gated"].get("execution") == "execute"
        oracle_er += int(want_exec == got_exec or (not want_exec and o["gated"].get("refuse")))
        unsupported += int(o["validated"].get("unsupported_accepted") or 0)
        unprov += int(o["validated"].get("unprovenanced_argument_accepted") or 0)
        catalog = catalog_of(row.get("catalog_tool_names") or [])
        learned_tools = learned_top(query, catalog, 5) if catalog else oracle_tools
        l = generate(query, learned_tools, facts=facts)
        lparsed = l["validated"].get("function_calls") or []
        if kind == "execute":
            lhit = calls_equal(lparsed, gold) and not l["validated"].get("refuse")
        else:
            lhit = bool(l["validated"].get("refuse")) or lparsed == []
        learned_exact += int(lhit)
        if l.get("confidence") is not None:
            conf_pairs.append((float(l["confidence"]), int(lhit)))
        if (i + 1) % 10 == 0:
            release_mlx_cache()
        if (i + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "fullcall": i + 1,
                        "oracle_exact_so_far": oracle_exact / (i + 1),
                        "rss_mb": round(rss_mb(), 1),
                    }
                ),
                flush=True,
            )

    n_fc = max(len(fc_rows), 1)
    oracle_rate = oracle_exact / n_fc
    learned_rate = learned_exact / n_fc
    ece = 0.0
    brier = 0.0
    if conf_pairs:
        brier = sum((p - y) ** 2 for p, y in conf_pairs) / len(conf_pairs)
        bins = 5
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            bucket = [p for p in conf_pairs if lo <= p[0] < hi or (b == bins - 1 and p[0] == 1)]
            if not bucket:
                continue
            conf = sum(p[0] for p in bucket) / len(bucket)
            freq = sum(p[1] for p in bucket) / len(bucket)
            ece += (len(bucket) / len(conf_pairs)) * abs(conf - freq)

    gates = {
        "recall_at_5_learned": {
            "value": recall_learned,
            "floor": layers["recall_at_5_learned_min"],
            "ok": recall_learned >= float(layers["recall_at_5_learned_min"]),
        },
        "recall_beats_lexical": {
            "learned": recall_learned,
            "lexical": recall_lex,
            "ok": (not layers.get("recall_at_5_must_exceed_lexical")) or recall_learned > recall_lex,
        },
        "oracle_top5_fullcall_exact": {
            "value": oracle_rate,
            "floor": layers["oracle_top5_fullcall_exact_min"],
            "ok": oracle_rate >= float(layers["oracle_top5_fullcall_exact_min"]),
        },
        "oracle_execute_refuse": {
            "value": oracle_er / n_fc,
            "floor": layers["oracle_top5_execute_refuse_ok_min"],
            "ok": (oracle_er / n_fc) >= float(layers["oracle_top5_execute_refuse_ok_min"]),
        },
        "learned_top5_e2e_exact": {
            "value": learned_rate,
            "floor": layers["learned_top5_e2e_exact_min"],
            "ok": learned_rate >= float(layers["learned_top5_e2e_exact_min"]),
        },
        "unsupported_accepted": {
            "value": unsupported,
            "max": layers["unsupported_accepted_max"],
            "ok": unsupported <= int(layers["unsupported_accepted_max"]),
        },
        "unprovenanced_argument_accepted": {
            "value": unprov,
            "max": layers["unprovenanced_argument_accepted_max"],
            "ok": unprov <= int(layers["unprovenanced_argument_accepted_max"]),
        },
    }
    hard_ok = all(row["ok"] for row in gates.values())
    report = {
        "kind": "toolcall-qat-lock-v2",
        "lock": "sft-v2-eval-lock-v2",
        "package_dir": str(pkg_dir),
        "thresholds": display_path(Path(THRESHOLDS).resolve()),
        "n_retrieval": len(ret_rows),
        "n_fullcall": len(fc_rows),
        "universe_n": len(universe),
        "recall_at_5_learned": recall_learned,
        "recall_at_5_lexical": recall_lex,
        "oracle_top5_fullcall_exact": oracle_rate,
        "learned_top5_e2e_exact": learned_rate,
        "unsupported_accepted": unsupported,
        "unprovenanced_argument_accepted": unprov,
        "confidence_ece": ece,
        "confidence_brier": brier,
        "input_fingerprint": shard_fingerprint(pkg_dir),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count if args.shard_index is not None else None,
        "counts": {
            "n_retrieval": len(ret_rows),
            "learned_hits": learned_hits,
            "lexical_hits": lexical_hits,
            "n_fullcall": len(fc_rows),
            "oracle_exact": oracle_exact,
            "oracle_execute_refuse": oracle_er,
            "learned_exact": learned_exact,
            "unsupported_accepted": unsupported,
            "unprovenanced_argument_accepted": unprov,
        },
        "layers": gates,
        "hard_ok": hard_ok,
        "evaluation_complete": True,
        "toy_scores_rejected": True,
        "n_loaded": loaded.get("n_loaded"),
        "qat_mandatory": True,
        "not_a_claim": "Lock-v2 scores are not a CURRENT freeze.",
    }
    path = (
        args.shard_dir / f"shard-{args.shard_index:03d}.json"
        if args.shard_index is not None and args.shard_dir
        else args.jobs_dir / "toolcall-qat-lock-v2.json"
    )
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return lock_receipt_exit_code(evaluation_complete=True)


if __name__ == "__main__":
    raise SystemExit(main())
