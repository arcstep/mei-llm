#!/usr/bin/env python3
"""Measure the real Python/MLX v2 path on the frozen 51M CQ2 package.

The MLX loader reconstructs portable CQ2 tensors as float arrays and is a
numerical/Apple backend, not evidence for portable quantized resident memory.
This benchmark keeps cold-load, first-complete and warm-complete timings
separate so compilation and package reconstruction are not hidden.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from common.paths import ROOT


SDK_PYTHON = ROOT / "src/platform/python-sdk"


def memory_snapshot(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def catalog_from_package(package: Path) -> list[dict[str, Any]]:
    manifest = json.loads((package / "mei-model.json").read_text(encoding="utf-8"))
    index_rows = [
        row for row in manifest.get("files", []) if row.get("role") == "tool_index"
    ]
    if len(index_rows) != 1:
        raise RuntimeError("v2 benchmark requires exactly one frozen tool index")
    index = json.loads((package / index_rows[0]["path"]).read_text(encoding="utf-8"))
    return [row["schema"] for row in index["records"]]


def complete_once(engine: Any, mx: Any) -> dict[str, Any]:
    mx.reset_peak_memory()
    started = time.perf_counter()
    session = engine.create_session({"max_steps": 1})
    try:
        turn = session.complete(
            {
                "wire_version": "mei-runtime-wire-v2",
                "query": "打开厨房灯",
                "context": {},
                "evidence": [],
                "history": [],
                "tool_results": [],
                "permissions": {},
                "state": {},
                "decode_mode": "constrained",
                "max_new": 1,
            }
        )
    finally:
        session.close_session()
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "kind": turn.get("kind"),
        "stats": turn.get("stats") or {},
        "runtime_cache": turn.get("runtime_cache") or {},
        "mw_disposition": turn.get("mw_disposition") or {},
        "memory": memory_snapshot(mx),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(SDK_PYTHON))
    import mlx.core as mx
    from mei_sdk import Engine

    package = args.package.resolve()
    catalog = catalog_from_package(package)
    mx.set_memory_limit(8 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    mx.clear_cache()
    mx.reset_peak_memory()
    load_started = time.perf_counter()
    engine = Engine.load(str(package), backend=args.backend)
    mx.synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    load_memory = memory_snapshot(mx)
    register_started = time.perf_counter()
    engine.register_tools(catalog)
    register_ms = (time.perf_counter() - register_started) * 1000.0
    first = complete_once(engine, mx)
    warm = complete_once(engine, mx)
    capabilities = engine.capabilities()
    engine.close_engine()
    try:
        device = str(mx.default_device())
    except Exception:
        device = "unknown"
    return {
        "schema": "mei-51m-mlx-runtime-benchmark-v1",
        "package": str(package),
        "package_id": capabilities.get("package_id"),
        "backend": args.backend,
        "device": device,
        "platform": platform.platform(),
        "python": sys.version,
        "mlx_version": getattr(sys.modules.get("mlx"), "__version__", "unknown"),
        "measurement_profile": "full-catalog-prefill-plus-one-bounded-decode-v1",
        "portable_quantized_residency_claim": False,
        "load_ms": load_ms,
        "register_tools_ms": register_ms,
        "load_memory": load_memory,
        "first_complete": first,
        "warm_complete": warm,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("mlx-reference", "mlx-fused"), default="mlx-reference"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(benchmark(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
