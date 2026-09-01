#!/usr/bin/env python3
"""Isolated mei_sdk MLX complete/embed benchmark. Does not mix retrieval catalog slices."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

PACKAGE = ROOT / "packages" / "mei-1.0-51m-base-scratch300m-v1"
WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class _KernelBenchSession:
    """Exercise model decode while preserving public candidate fail-closed policy.

    This adapter calls the numerical runtime directly and is only selected by
    the explicit ``--kernel-only`` benchmark flag.  It never changes the
    package manifest or the public Session API's raw/fixture restrictions.
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def complete(self, request: dict) -> dict:
        from mei_sdk.runtime_51m import complete_51m

        normalized = {
            "wire_version": "mei-runtime-wire-v2",
            "query": str(request.get("query") or ""),
            "context": {},
            "history": [],
            "evidence": [],
            "entities": [],
            "tool_results": [],
            "permissions": {},
            "state": {},
            "oracle_tools": list(request.get("oracle_tools") or []),
            "decode_mode": str(request.get("decode_mode") or "raw"),
            "max_new": int(request.get("max_new") or 128),
            "_kernel_benchmark_ignore_eos": True,
        }
        out = complete_51m(self.runtime, normalized)
        stats = dict(out.get("timings") or {})
        stats["prompt_tokens"] = int(out.get("prompt_tokens") or 0)
        stats["output_tokens"] = int(out.get("output_tokens") or 0)
        cache = dict(out.get("kv") or {})
        stats["kv_storage_dtype"] = cache.get("measured_cache_dtype")
        stats["activation_dtype"] = cache.get("activation_dtype")
        stats["incremental_decode_steps"] = int(
            cache.get("incremental_decode_steps") or 0
        )
        stats["recomputed_decode_steps"] = int(
            cache.get("recomputed_decode_steps") or 0
        )
        decode_ms = float(stats.get("decode_ms") or 0.0)
        if stats["output_tokens"] and decode_ms > 0:
            stats["output_tok_s"] = stats["output_tokens"] / (decode_ms / 1000.0)
        return {
            "ok": bool((out.get("validated") or {}).get("ok")),
            "refuse": bool((out.get("validated") or {}).get("refuse")),
            "raw_text": out.get("text") or "",
            "stats": stats,
        }

    def embed(self, text: str):
        return self.runtime.embed_text(str(text))


def _rss_mb() -> float | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024, 1)
    except Exception:
        return None


def _metal_peak_mb() -> float | None:
    try:
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        peak = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
        if clear:
            clear()
        if peak is None:
            return None
        return round(float(peak()) / (1024 * 1024), 1)
    except Exception:
        return None


def _host_meta() -> dict:
    try:
        import mlx
        import mlx.core as mx

        mlx_ver = getattr(mlx, "__version__", None) or getattr(mx, "__version__", "unknown")
        device = str(mx.default_device())
    except Exception:
        mlx_ver, device = "unavailable", None
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "mlx": mlx_ver,
        "device": device,
        "pid": os.getpid(),
    }


def _one(session, *, decode_mode: str, max_new: int, query: str) -> dict:
    t0 = time.perf_counter()
    out = session.complete(
        {
            "query": query,
            "oracle_tools": [WEATHER],
            "decode_mode": decode_mode,
            "max_new": max_new,
        }
    )
    wall = (time.perf_counter() - t0) * 1000
    stats = dict(out.get("stats") or {})
    stats.setdefault("wall_ms", wall)
    stats["ok"] = bool(out.get("ok"))
    stats["refuse"] = bool(out.get("refuse"))
    stats["n_raw"] = len(out.get("raw_text") or "")
    return stats


def run_matrix(session, *, warmup: int, repeats: int, modes: list[str], max_news: list[int]) -> list[dict]:
    query = "帮我查一下成都今天的天气"
    rows = []
    for decode_mode in modes:
        for max_new in max_news:
            cold = _one(session, decode_mode=decode_mode, max_new=max_new, query=query)
            for _ in range(max(0, warmup)):
                _one(session, decode_mode=decode_mode, max_new=max_new, query=query)
            warm = [_one(session, decode_mode=decode_mode, max_new=max_new, query=query) for _ in range(repeats)]
            tok_s = [float(x.get("output_tok_s") or 0.0) for x in warm]
            walls = [float(x.get("wall_ms") or 0.0) for x in warm]
            decodes = [float(x.get("decode_ms") or 0.0) for x in warm]
            n_out = [int(x.get("output_tokens") or 0) for x in warm]
            rows.append(
                {
                    "decode_mode": decode_mode,
                    "max_new": max_new,
                    "cold": cold,
                    "warm_n": len(warm),
                    "output_tokens_mean": round(statistics.mean(n_out), 2) if n_out else 0,
                    "output_tok_s_p50": round(statistics.median(tok_s), 2) if tok_s else 0,
                    "output_tok_s_p95": round(sorted(tok_s)[max(0, int(round(0.95 * (len(tok_s) - 1))))], 2) if tok_s else 0,
                    "wall_ms_p50": round(statistics.median(walls), 1) if walls else 0,
                    "decode_ms_p50": round(statistics.median(decodes), 1) if decodes else 0,
                    "prefill_ms_p50": round(statistics.median([float(x.get("prefill_ms") or 0) for x in warm]), 1),
                    "grammar_ms_p50": round(statistics.median([float(x.get("grammar_ms") or 0) for x in warm]), 1),
                    "validate_ms_p50": round(statistics.median([float(x.get("validate_ms") or 0) for x in warm]), 1),
                    "prompt_tokens": warm[0].get("prompt_tokens") if warm else None,
                    "rss_mb": _rss_mb(),
                    "metal_peak_mb": _metal_peak_mb(),
                    "warm_runs": warm,
                }
            )
    return rows


def eval_gate(rows: list[dict], *, min_raw_128_tok_s: float, min_speedup: float, baseline: dict | None) -> dict:
    raw128 = next((r for r in rows if r["decode_mode"] == "raw" and r["max_new"] == 128), None)
    reasons = []
    tok_s = float((raw128 or {}).get("output_tok_s_p50") or 0)
    if tok_s < min_raw_128_tok_s:
        reasons.append(f"raw128 tok/s {tok_s} < {min_raw_128_tok_s}")
    speedup = None
    if baseline and raw128:
        base = None
        for b in baseline.get("matrix") or []:
            if b.get("decode_mode") == "raw" and b.get("max_new") == 128:
                base = float(b.get("output_tok_s_p50") or 0)
        if base and base > 0:
            speedup = tok_s / base
            if speedup < min_speedup:
                reasons.append(f"speedup {speedup:.2f}x < {min_speedup}x vs preopt {base}")
    # later warm runs should not collapse vs first warm (recompile smell)
    if raw128 and len(raw128.get("warm_runs") or []) >= 3:
        series = [float(x.get("output_tok_s") or 0) for x in raw128["warm_runs"]]
        if series[0] > 1 and series[-1] < series[0] / 3:
            reasons.append("warm tok/s collapsed; possible stepwise recompile")
    return {
        "ok": not reasons,
        "raw128_tok_s": tok_s,
        "speedup_vs_baseline": speedup,
        "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument(
        "--backend",
        choices=("mlx-reference", "mlx-fused", "mlx-cq2"),
        default="mlx-reference",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--modes", default="raw,constrained")
    ap.add_argument("--max-new", default="32,128")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--min-raw-128-tok-s", type=float, default=300.0)
    ap.add_argument("--min-speedup", type=float, default=2.0)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--compare-reference", action="store_true")
    ap.add_argument("--embed-repeats", type=int, default=8)
    ap.add_argument(
        "--kernel-only",
        action="store_true",
        help="benchmark direct numerical decode without weakening candidate Session policy",
    )
    args = ap.parse_args()
    from mei_sdk import Engine
    from mei_sdk.mlx_backend import backend_revision

    t_load = time.perf_counter()
    engine = Engine.load(str(args.package), verify_hashes=True, backend=args.backend)
    load_ms = (time.perf_counter() - t_load) * 1000
    session = _KernelBenchSession(engine.runtime) if args.kernel_only else engine.create_session()
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    max_news = [int(x) for x in args.max_new.split(",") if x.strip()]
    matrix = run_matrix(session, warmup=args.warmup, repeats=args.repeats, modes=modes, max_news=max_news)
    embed_ms = []
    for _ in range(args.embed_repeats):
        t0 = time.perf_counter()
        session.embed("成都天气")
        embed_ms.append((time.perf_counter() - t0) * 1000)
    baseline = None
    if args.baseline and args.baseline.is_file():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    gate = eval_gate(matrix, min_raw_128_tok_s=args.min_raw_128_tok_s, min_speedup=args.min_speedup, baseline=baseline)
    comparison = None
    if args.compare_reference and args.backend in {"mlx-fused", "mlx-cq2"}:
        t_ref = time.perf_counter()
        reference_engine = Engine.load(
            str(args.package),
            verify_hashes=True,
            backend="mlx-reference",
        )
        reference_session = (
            _KernelBenchSession(reference_engine.runtime)
            if args.kernel_only
            else reference_engine.create_session()
        )
        reference_matrix = run_matrix(
            reference_session,
            warmup=args.warmup,
            repeats=args.repeats,
            modes=modes,
            max_news=max_news,
        )
        ratios = {}
        for fused_row in matrix:
            key = f"{fused_row['decode_mode']}:{fused_row['max_new']}"
            ref_row = next(
                (
                    row
                    for row in reference_matrix
                    if row["decode_mode"] == fused_row["decode_mode"]
                    and row["max_new"] == fused_row["max_new"]
                ),
                None,
            )
            ref_speed = float((ref_row or {}).get("output_tok_s_p50") or 0.0)
            if ref_speed > 0:
                ratios[key] = round(float(fused_row["output_tok_s_p50"]) / ref_speed, 3)
        comparison = {
            "same_process": True,
            "reference_elapsed_ms": round((time.perf_counter() - t_ref) * 1000, 1),
            "reference_revision": backend_revision("mlx-reference"),
            "reference_matrix": reference_matrix,
            "fused_over_reference": ratios,
        }
    report = {
        "id": "mei-sdk-mlx-complete-bench",
        "host": _host_meta(),
        "package": str(args.package),
        "backend": args.backend,
        "sdk_backend_revision": backend_revision(args.backend),
        "load_report": {
            k: (engine.load_report or {}).get(k)
            for k in ("n_loaded", "package_id", "sdk_backend_revision", "weights")
        },
        "load_ms": round(load_ms, 1),
        "execution_profile": {
            "weight_storage": (
                "resident-packed-cq2-cq4"
                if args.backend == "mlx-cq2"
                else "expanded-float32-from-cq2-package"
            ),
            "activation_quantization": (
                ((matrix[0].get("warm_runs") or [{}])[0]).get("activation_dtype")
                if matrix
                else None
            ),
            # Activation Q/DQ deliberately reconstructs into float for the
            # current MLX matmul kernels; only KV codes remain int8-resident.
            "activation_compute_dtype": "float32",
            "kv_storage": (
                ((matrix[0].get("warm_runs") or [{}])[0]).get("kv_storage_dtype")
                if matrix
                else None
            ),
            "compiled_decode": os.environ.get("MEI_SDK_NO_COMPILE") != "1",
            "decode_chunk": int(os.environ.get("MEI_SDK_DECODE_CHUNK", "24")),
            "qualification": (
                "native-v2-cq2-activation-qdq-int8-kv-performance"
                if args.backend == "mlx-cq2"
                else "float-reference-performance"
            ),
        },
        "matrix": matrix,
        "embed_ms_p50": round(statistics.median(embed_ms), 2) if embed_ms else None,
        "rss_mb": _rss_mb(),
        "metal_peak_mb": _metal_peak_mb(),
        "gate": gate,
        "comparison": comparison,
        "kernel_only": bool(args.kernel_only),
        "note": (
            "Direct numerical runtime benchmark; public candidate Session policy remains fail-closed."
            if args.kernel_only
            else "decode_mode=raw|constrained × max_new; not a retrieval catalog-size slice."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.gate and not gate["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
