from __future__ import annotations

import time
from typing import Any

from .errors import SdkError, error_from_id
from .package import ModelPackage, load_package
from .protocol import (
    MAX_SELECTED_TOOLS,
    leak_markers,
    parse_v2_text,
    render_request,
    request_leaks,
    schema_fingerprint,
)
from .version import sdk_versions

WIRE = sdk_versions()["wire_version"]


def _empty_confidence() -> dict[str, Any]:
    return {"available": False, "value": None, "source": None}


def _turn(
    *,
    ok: bool,
    refuse: bool,
    selected: list[str],
    fingerprint: str | None,
    calls: list[dict[str, Any]],
    raw_text: str | None,
    error: dict[str, Any] | None,
    provenance: dict[str, Any],
    capabilities: dict[str, Any],
    backend: str,
    wall_ms: float,
    decode_mode: str | None,
) -> dict[str, Any]:
    return {
        "wire_version": WIRE,
        "ok": ok,
        "error": error,
        "refuse": refuse,
        "selected_tools": selected,
        "schema_fingerprint": fingerprint,
        "function_calls": calls,
        "raw_text": raw_text,
        "confidence": _empty_confidence(),
        "provenance": provenance,
        "capabilities": capabilities,
        "stats": {"backend": backend, "wall_ms": wall_ms, "decode_mode": decode_mode},
    }


class Engine:
    def __init__(
        self,
        package: ModelPackage,
        *,
        runtime=None,
        load_report: dict[str, Any] | None = None,
        backend: str = "protocol",
    ):
        self.package = package
        self.closed = False
        self.runtime = runtime
        self.load_report = load_report or {}
        self.backend = backend

    @classmethod
    def load(
        cls,
        package_dir: str,
        *,
        verify_hashes: bool = True,
        backend: str = "protocol",
    ) -> "Engine":
        pkg = load_package(package_dir, verify_hashes=verify_hashes)
        if backend in {"auto", ""}:
            if pkg.packed_inference_ready() and str(pkg.manifest.get("product")) == "mei-1.0-51m":
                from .mlx_backend import load_mlx_runtime

                runtime, report = load_mlx_runtime(pkg, backend="mlx-reference")
                return cls(pkg, runtime=runtime, load_report=report, backend="mlx-reference")
            return cls(pkg, backend="protocol")
        if backend in {"protocol", "none"}:
            return cls(pkg, backend="protocol")
        if backend in {"mlx-reference", "mlx", "mlx-fused"}:
            from .mlx_backend import load_mlx_runtime

            selected = "mlx-reference" if backend == "mlx" else backend
            runtime, report = load_mlx_runtime(pkg, backend=selected)
            return cls(pkg, runtime=runtime, load_report=report, backend=selected)
        raise SdkError("invalid_argument", f"unknown backend: {backend}")

    def capabilities(self) -> dict[str, Any]:
        caps = self.package.capabilities()
        caps["versions"] = sdk_versions()
        caps["inference"] = self.runtime is not None
        caps["protocol"] = True
        caps["backend"] = self.backend
        if self.runtime is not None and self.package.packed_inference_ready():
            caps["quantized_only"] = True
        if self.load_report:
            caps["load_report"] = {
                k: self.load_report.get(k) for k in ("n_loaded", "package_id", "backend")
            }
        return caps

    def create_session(self, options: dict[str, Any] | None = None) -> "Session":
        if self.closed:
            raise SdkError("session_closed", "engine is closed")
        return Session(self, options or {})

    def close(self) -> None:
        self.closed = True


class Session:
    def __init__(self, engine: Engine, options: dict[str, Any]):
        self.engine = engine
        self.options = options
        self.closed = False
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        caps = self.engine.capabilities()
        decode_mode = str(request.get("decode_mode") or "constrained")
        if self.closed or self.engine.closed:
            raise SdkError("session_closed")
        if self.cancelled:
            return _turn(
                ok=False,
                refuse=True,
                selected=[],
                fingerprint=None,
                calls=[],
                raw_text=None,
                error=error_from_id("cancelled"),
                provenance={"validated": False, "ok": False, "detail": "cancelled"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        leaks = request_leaks(request)
        if leaks:
            return _turn(
                ok=False,
                refuse=True,
                selected=[],
                fingerprint=None,
                calls=[],
                raw_text=None,
                error=error_from_id("gold_leak", "forbidden markers: " + ",".join(leaks)),
                provenance={"validated": True, "ok": False, "detail": "gold_leak"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        if self.engine.runtime is not None and request.get("candidate_text") is None:
            return self._complete_mlx(request, caps, started, decode_mode)
        tools = request.get("oracle_tools")
        if tools is None:
            tools = request.get("catalog") or []
        if not isinstance(tools, list):
            return _turn(
                ok=False,
                refuse=True,
                selected=[],
                fingerprint=None,
                calls=[],
                raw_text=None,
                error=error_from_id("invalid_argument", "tools must be a list"),
                provenance={"validated": False, "ok": False, "detail": "invalid_tools"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        if len(tools) > MAX_SELECTED_TOOLS:
            return _turn(
                ok=False,
                refuse=True,
                selected=[],
                fingerprint=None,
                calls=[],
                raw_text=None,
                error=error_from_id("too_many_tools"),
                provenance={"validated": True, "ok": False, "detail": "too_many_tools"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        rendered = render_request(request, tools)
        selected = rendered["selected_tools"]
        fingerprint = rendered["schema_fingerprint"]
        candidate = request.get("candidate_text")
        if candidate is None:
            return _turn(
                ok=False,
                refuse=True,
                selected=selected,
                fingerprint=fingerprint,
                calls=[],
                raw_text=None,
                error=error_from_id(
                    "engine_unavailable",
                    "portable inference is not in this experimental SDK; pass candidate_text for protocol validation",
                ),
                provenance={"validated": False, "ok": False, "detail": "engine_unavailable"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        if leak_markers(str(candidate)):
            return _turn(
                ok=False,
                refuse=True,
                selected=selected,
                fingerprint=fingerprint,
                calls=[],
                raw_text=str(candidate),
                error=error_from_id("gold_leak", "generation contains forbidden markers"),
                provenance={"validated": True, "ok": False, "detail": "gold_leak"},
                capabilities=caps,
                backend="protocol",
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        parsed = parse_v2_text(str(candidate))
        calls = parsed["function_calls"]
        if calls:
            name = calls[0]["name"]
            if name not in selected:
                return _turn(
                    ok=False,
                    refuse=True,
                    selected=selected,
                    fingerprint=fingerprint,
                    calls=[],
                    raw_text=str(candidate),
                    error=error_from_id("protocol_violation", f"tool {name} not in selected_tools"),
                    provenance={"validated": True, "ok": False, "detail": "unknown_tool"},
                    capabilities=caps,
                    backend="protocol",
                    wall_ms=(time.perf_counter() - started) * 1000,
                    decode_mode=decode_mode,
                )
        ok = bool(parsed["ok"])
        error = None if ok else error_from_id("protocol_violation", str(parsed.get("error") or "protocol"))
        return _turn(
            ok=ok,
            refuse=bool(parsed["refuse"]),
            selected=selected,
            fingerprint=fingerprint,
            calls=calls if ok else [],
            raw_text=str(candidate),
            error=error,
            provenance={"validated": True, "ok": ok, "detail": parsed.get("error")},
            capabilities=caps,
            backend="protocol",
            wall_ms=(time.perf_counter() - started) * 1000,
            decode_mode=decode_mode,
        )

    def _complete_mlx(
        self,
        request: dict[str, Any],
        caps: dict[str, Any],
        started: float,
        decode_mode: str,
    ) -> dict[str, Any]:
        from .mlx_backend import complete_mlx
        from .protocol import schema_fingerprint

        oracle = request.get("oracle_tools")
        if oracle is not None and len(oracle) > MAX_SELECTED_TOOLS:
            return _turn(
                ok=False,
                refuse=True,
                selected=[],
                fingerprint=None,
                calls=[],
                raw_text=None,
                error=error_from_id("too_many_tools"),
                provenance={"validated": True, "ok": False, "detail": "too_many_tools"},
                capabilities=caps,
                backend=self.engine.backend,
                wall_ms=(time.perf_counter() - started) * 1000,
                decode_mode=decode_mode,
            )
        out = complete_mlx(self.engine.runtime, request)
        selected = list(out.get("selected_tools") or [])
        tools = oracle if oracle is not None else (request.get("catalog") or [])
        used = [t for t in tools if str(t.get("name") or "") in set(selected)] or tools[: len(selected)]
        validated = out.get("validated") or {}
        calls = list(validated.get("function_calls") or [])
        refuse = bool(validated.get("refuse"))
        ok = bool(validated.get("ok", True)) and not out.get("error")
        err = None
        if out.get("error"):
            err = error_from_id("protocol_violation", str(out["error"]))
            ok = False
        conf = out.get("confidence")
        turn = _turn(
            ok=ok,
            refuse=refuse or not calls,
            selected=selected,
            fingerprint=schema_fingerprint(used) if used else None,
            calls=calls if ok and not refuse else calls,
            raw_text=out.get("text"),
            error=err,
            provenance={
                "validated": True,
                "ok": bool(validated.get("ok", ok)),
                "detail": validated.get("error"),
            },
            capabilities=caps,
            backend=self.engine.backend,
            wall_ms=(time.perf_counter() - started) * 1000,
            decode_mode=str(out.get("decode", {}).get("mode") or decode_mode),
        )
        if conf is not None:
            turn["confidence"] = {
                "available": True,
                "value": float(conf) if not isinstance(conf, dict) else conf,
                "source": self.engine.backend,
                "version": "conf-head-v1",
            }
        if out.get("execution"):
            turn["execution"] = out["execution"]
            if out["execution"] != "execute":
                turn["refuse"] = True
                if out["execution"] == "escalate":
                    turn["ok"] = True
                    turn["function_calls"] = []
        stats = turn["stats"]
        timings = out.get("timings") or {}
        stats.update({k: float(v) for k, v in timings.items() if v is not None})
        n_prompt = out.get("prompt_tokens")
        n_out = out.get("output_tokens")
        if n_prompt is not None:
            stats["prompt_tokens"] = int(n_prompt)
        if n_out is not None:
            stats["output_tokens"] = int(n_out)
        decode_ms = float(stats.get("decode_ms") or 0.0)
        if n_out and decode_ms > 0:
            stats["output_tok_s"] = float(n_out) / (decode_ms / 1000.0)
        elif out.get("output_tok_s") is not None:
            stats["output_tok_s"] = float(out["output_tok_s"])
        report = self.engine.load_report or {}
        if report.get("sdk_backend_revision"):
            stats["sdk_backend_revision"] = report["sdk_backend_revision"]
        return turn

    def embed(self, text: str) -> list[float]:
        if self.engine.runtime is None:
            raise SdkError("engine_unavailable", "embed requires an inference backend")
        rt = self.engine.runtime
        if hasattr(rt, "embed_text"):
            vec = rt.embed_text(text)
        else:
            vec = rt._embed_text(text)
        if vec is None:
            raise SdkError("head_missing", "contrastive head did not return a vector")
        try:
            import mlx.core as mx

            mx.eval(vec)
        except Exception:
            pass
        data = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if data and isinstance(data[0], (list, tuple)):
            data = data[0]
        return [float(x) for x in data]

    def retrieve(self, query: str, catalog: list[dict[str, Any]], *, k: int = 5) -> dict[str, Any]:
        if self.engine.runtime is None:
            raise SdkError("engine_unavailable", "retrieve requires an inference backend")
        rt = self.engine.runtime
        if hasattr(rt, "search_top_k"):
            selected = rt.search_top_k(query, catalog, k=k)
        else:
            selected = rt._select(query, catalog)
        return {
            "selected_tools": [str(t.get("name") or "") for t in selected[:k]],
            "tools": selected[:k],
            "k": k,
        }

    def run(self, request: dict[str, Any], *, max_turns: int = 1) -> dict[str, Any]:
        turns: list[dict[str, Any]] = []
        stopped = "max_turns"
        for _ in range(max(1, int(max_turns))):
            turn = self.complete(request)
            turns.append(turn)
            if turn.get("error"):
                stopped = "error" if turn["error"]["id"] != "cancelled" else "cancelled"
                break
            if turn.get("refuse"):
                stopped = "refuse"
                break
            if turn.get("function_calls"):
                stopped = "call"
                break
        return {
            "wire_version": WIRE,
            "ok": bool(turns) and all(t.get("ok") for t in turns),
            "turns": turns,
            "stopped_reason": stopped,
        }
