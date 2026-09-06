"""Bounded MLX/Metal memory policy shared by every 51M productization stage."""

from __future__ import annotations

import gc
from typing import Any


POLICY_ID = "mei-51m-mlx-memory-v1-8g-256m"
MEMORY_LIMIT_BYTES = 8 * 1024**3
CACHE_LIMIT_BYTES = 256 * 1024**2


def configure_mlx_memory(
    mx: Any | None = None, *, reset_peak: bool = True
) -> tuple[Any, dict[str, Any]]:
    if mx is None:
        import mlx.core as mx

    previous_memory = int(mx.set_memory_limit(MEMORY_LIMIT_BYTES))
    previous_cache = int(mx.set_cache_limit(CACHE_LIMIT_BYTES))
    mx.clear_cache()
    if reset_peak:
        mx.reset_peak_memory()
    return mx, {
        "policy_id": POLICY_ID,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "cache_limit_bytes": CACHE_LIMIT_BYTES,
        "previous_memory_limit_bytes": previous_memory,
        "previous_cache_limit_bytes": previous_cache,
        "reset_peak": bool(reset_peak),
    }


def release_mlx_memory(
    mx: Any, *, collect_python: bool = False
) -> None:
    if collect_python:
        gc.collect()
    mx.clear_cache()


def mlx_memory_snapshot(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
