"""MLX reference backend for MEI Runtime Python SDK.

This is the eval/inference path for `mei-1.0-58m` on Apple Silicon.
It loads weights from a `mei-model-package-v1` directory and calls the
historical Python+MLX runtime. Public types stay MEI Runtime; the runtime
directory name is not part of the SDK API.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from .errors import SdkError
from .package import ModelPackage
from .version import SDK_ROOT

_MEI_LLM = SDK_ROOT.parent
_ARCH = _MEI_LLM / "architecture" / "mei-1.0-58m-arch-v1"
_RUNTIME = _MEI_LLM / "runtime" / "mei-1.0-58m-needle2-v2"
_RUNTIME_SHARED = _MEI_LLM / "runtime" / "_shared"
_CKPT = _MEI_LLM / "notebook" / "_tooling" / "model" / "mei-1.0-58m"
_ABANDONED = "notebook/archive/base/mei-1.0-58m-checkpoints"

_REFERENCE_REVISION_FILES = (
    _ARCH / "architecture.py",
    _RUNTIME / "kv_manager.py",
    _RUNTIME_SHARED / "decode.py",
    _RUNTIME / "byte_grammar.py",
    _RUNTIME / "runtime_v2.py",
    SDK_ROOT / "python" / "mei_sdk" / "mlx_backend.py",
)
_FUSED_REVISION_FILES = (
    *_REFERENCE_REVISION_FILES,
    _ARCH / "fused_ops.py",
)


def _ensure_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def backend_file_fingerprints(backend: str = "mlx-reference") -> dict[str, str]:
    out: dict[str, str] = {}
    files = _FUSED_REVISION_FILES if backend == "mlx-fused" else _REFERENCE_REVISION_FILES
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
    if backend not in {"mlx-reference", "mlx-fused"}:
        raise SdkError("invalid_argument", f"unknown MLX backend: {backend}")
    product = str(package.manifest.get("product") or "")
    if product == "mei-1.0-51m":
        from .runtime_51m import load_51m_runtime

        return load_51m_runtime(package, backend=backend)
    weights = (package.path / str(package.manifest["weights"]["file"])).resolve()
    tokenizer = (package.path / str(package.manifest["tokenizer"]["file"])).resolve()
    if _ABANDONED in str(weights):
        raise SdkError("package_invalid", "abandoned archive checkpoint is not a valid eval target")
    if not weights.is_file():
        raise SdkError("file_not_found", f"missing weights: {weights}")
    if not tokenizer.is_file():
        raise SdkError("file_not_found", f"missing tokenizer: {tokenizer}")
    for path in (_RUNTIME, _RUNTIME_SHARED, _ARCH, _CKPT):
        _ensure_path(path)
    from architecture import NeedleZh
    from checkpoint import load_params
    from config import NeedleZhConfig
    from tokenizer import ZhTokenizerV1

    from runtime_v2 import RuntimeV2

    tok = ZhTokenizerV1(tokenizer)
    cfg = NeedleZhConfig.from_target_v2()
    model = NeedleZh(cfg)
    report = load_params(
        model,
        weights,
        strict=False,
        allow_missing_prefixes=("contrastive", "mw", "confidence", "conf"),
        return_report=True,
    )
    import mlx.core as mx

    mx.eval(model.parameters())
    model.set_inference_backend(backend)
    runtime = RuntimeV2(model, tok, catalog=[], confidence_threshold=0.0)
    fingerprints = backend_file_fingerprints(backend)
    return runtime, {
        "n_loaded": (report or {}).get("n_loaded") if isinstance(report, dict) else report,
        "weights": str(weights),
        "tokenizer": str(tokenizer),
        "package_id": package.package_id,
        "backend": backend,
        "sdk_backend_revision": backend_revision(backend),
        "runtime_files_sha256": fingerprints,
    }


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
