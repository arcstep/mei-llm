"""MLX reference backend for MEI Runtime Python SDK.

This is the eval/inference path for `mei-1.0-51m` on Apple Silicon.
It loads MEI model packages and calls the Python+MLX numerical oracle. Shared
runtime semantics live in ``platform/_shared/runtime``; there is no model-specific
runtime directory in the public API.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .errors import SdkError
from .package import ModelPackage
from .version import PYTHON_SDK_ROOT, SDK_ROOT

_MEI_LLM = SDK_ROOT.parents[2]
_arch_env = os.environ.get("MEI_ARCHITECTURE_DIR")
_ARCH = Path(_arch_env).resolve() if _arch_env else (_MEI_LLM / "src/architecture/mei-1.2-51m")
_RUNTIME_SHARED = _MEI_LLM / "src/platform/_shared/runtime"

_REFERENCE_REVISION_FILES = (
    _ARCH / "architecture.py",
    _RUNTIME_SHARED / "kv_manager.py",
    _RUNTIME_SHARED / "decode.py",
    _RUNTIME_SHARED / "byte_grammar.py",
    _RUNTIME_SHARED / "schema_subset.py",
    _RUNTIME_SHARED / "provenance.py",
    _RUNTIME_SHARED / "tool_index.py",
    PYTHON_SDK_ROOT / "mei_sdk" / "runtime_51m.py",
    PYTHON_SDK_ROOT / "mei_sdk" / "mlx_backend.py",
)
_FUSED_REVISION_FILES = (
    *_REFERENCE_REVISION_FILES,
    _ARCH / "fused_ops.py",
)
_CQ2_REVISION_FILES = (
    *_FUSED_REVISION_FILES,
    _ARCH / "cq2_metal.py",
)


def backend_file_fingerprints(backend: str = "mlx-reference") -> dict[str, str]:
    out: dict[str, str] = {}
    files = (
        _CQ2_REVISION_FILES
        if backend == "mlx-cq2"
        else _FUSED_REVISION_FILES
        if backend == "mlx-fused"
        else _REFERENCE_REVISION_FILES
    )
    for path in files:
        rel = str(path.relative_to(_MEI_LLM))
        if not path.is_file():
            out[rel] = "missing"
            continue
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def backend_revision(backend: str = "mlx-reference") -> str:
    h = hashlib.sha256()
    h.update(backend.encode("utf-8"))
    h.update(b"\n")
    for rel, digest in sorted(backend_file_fingerprints(backend).items()):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_mlx_runtime(package: ModelPackage, *, backend: str = "mlx-reference"):
    if backend not in {"mlx-reference", "mlx-fused", "mlx-cq2"}:
        raise SdkError("invalid_argument", f"unknown MLX backend: {backend}")
    product = str(package.manifest.get("product") or "")
    if product == "mei-1.0-51m":
        from .runtime_51m import load_51m_runtime

        return load_51m_runtime(package, backend=backend)
    raise SdkError("package_invalid", f"unsupported product for MLX backend: {product}")


def complete_mlx(runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
    if runtime.__class__.__name__ == "Runtime51M":
        from .runtime_51m import complete_51m

        return complete_51m(runtime, request)
    oracle = request.get("oracle_tools")
    catalog = request.get("catalog") or []
    return runtime.complete(
        str(request.get("query") or ""),
        catalog=catalog or None,
        oracle_tools=oracle,
        system_facts=request.get("system_facts") or "",
        history=request.get("history") or [],
        prior_tool_results=request.get("prior_tool_results") or [],
        decode_mode=str(request.get("decode_mode") or "constrained"),
        max_new=int(request.get("max_new") or 128),
    )
