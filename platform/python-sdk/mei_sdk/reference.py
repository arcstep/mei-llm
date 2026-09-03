"""Explicit opt-in to the 51M Python+MLX numerical oracle."""

from __future__ import annotations

from typing import Any

def load_reference_runtime():
    from . import runtime_51m

    return runtime_51m


def reference_complete(runtime: Any, request: dict[str, Any]) -> Any:
    """Call the 51M oracle. Caller owns loading the runtime instance."""
    from .runtime_51m import complete_51m

    return complete_51m(runtime, request)
