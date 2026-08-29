"""Explicit opt-in to the historical Python+MLX reference runtime.

This module is the golden oracle for inference numerics. It is not the portable ABI.
The directory name `mei-1.0-58m-needle2-v2` is historical and is not part of the public SDK API.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from .errors import SdkError
from .version import SDK_ROOT

_REFERENCE_DIR = SDK_ROOT.parent / "runtime" / "mei-1.0-58m-needle2-v2"


def load_reference_runtime():
    path = _REFERENCE_DIR / "runtime_v2.py"
    if not path.is_file():
        raise SdkError("engine_unavailable", f"reference runtime missing: {path}")
    sys.path.insert(0, str(_REFERENCE_DIR))
    spec = importlib.util.spec_from_file_location("mei_runtime_reference_v2", path)
    if spec is None or spec.loader is None:
        raise SdkError("engine_unavailable", "cannot import reference runtime")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_complete(runtime: Any, request: dict[str, Any]) -> Any:
    """Call the MLX RuntimeV2.complete. Caller owns model loading."""
    return runtime.complete(
        query=request["query"],
        catalog=request.get("catalog") or [],
        oracle_tools=request.get("oracle_tools"),
        system_facts=request.get("system_facts") or "",
        history=request.get("history") or [],
        prior_tool_results=request.get("prior_tool_results") or [],
        decode_mode=request.get("decode_mode") or "raw",
        max_new=int(request.get("max_new") or 96),
    )
