from __future__ import annotations

import hashlib
import json
import threading
import time
from copy import deepcopy
from typing import Any, Callable

from .canonical import dumps_canonical, schema_fingerprint
from .errors import SdkError, error_from_id
from .package import ModelPackage, load_package
from .protocol import (
    DEFAULT_MAX_STEPS,
    HARD_MAX_STEPS,
    MAX_SELECTED_TOOLS,
    MAX_TOOL_RESULT_BYTES,
    leak_markers,
    normalize_request,
    render_request,
    request_leaks,
    validate_tool_result_v2,
)
from .shared import (
    NARRATION_ADAPTER_ID,
    NarrationProvider,
    UnsupportedSchemaError,
    catalog_fingerprint,
    narration_prompt_v2,
    validate_generated_call,
    validate_tools,
    verified_tool_result,
    verified_result_view,
)
from .version import sdk_versions

WIRE = sdk_versions()["wire_version"]
_SESSION_NONCE_LOCK = threading.Lock()
_NEXT_SESSION_NONCE = 1
_REFUSAL_ERRORS = {
    "provenance_missing",
    "permission_denied",
    "permission_not_granted",
    "permission_scope_missing",
    "permissions_invalid",
    "state_invalid",
    "state_conflict",
    "tool_state_contract_invalid",
    "mw_stop",
    "mw_constrained",
    "mw_invalid",
    "confidence_unavailable",
    "confidence_invalid",
}


def _allocate_session_nonce() -> str:
    global _NEXT_SESSION_NONCE
    with _SESSION_NONCE_LOCK:
        if _NEXT_SESSION_NONCE > 0xFFFFFFFF:
            raise SdkError("engine_unavailable", "session nonce space exhausted")
        value = _NEXT_SESSION_NONCE
        _NEXT_SESSION_NONCE += 1
    return f"s{value:08x}"


def load_model(
    package_dir: str, *, verify_hashes: bool = True, backend: str = "protocol"
) -> "Engine":
    return Engine.load(package_dir, verify_hashes=verify_hashes, backend=backend)


def _empty_confidence() -> dict[str, Any]:
    return {"available": False, "value": None, "source": None}


def _is_successful_respond(turn: dict[str, Any], tool_results: list[dict[str, Any]]) -> bool:
    """An empty model action becomes `respond` only after verified success.

    Deterministic gate refusals remain refusals even when a session has older
    results. This keeps `respond` a positive end-of-agent-loop signal rather
    than another spelling for refusal.
    """

    if not tool_results or any(row.get("status") != "ok" for row in tool_results):
        return False
    if turn.get("error") or turn.get("function_calls"):
        return False
    if any(gate.get("ok") is False for gate in (turn.get("provenance") or {}).get("gates", [])):
        return False
    try:
        return json.loads(str(turn.get("raw_text") or "")) == []
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _trusted_call_history(
    tool_results: list[dict[str, Any]],
    completed_calls: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render the session-owned call half of each accepted call/result pair.

    ``ToolResultV2`` intentionally does not repeat a tool name.  A continuation
    model therefore needs the corresponding accepted call in ordinary history;
    otherwise an opaque call_id and payload cannot reliably express chains such
    as ``book_flight -> pnr -> request_wheelchair``.  Results remain ordered by
    acceptance, while the call map is used only as immutable session evidence.
    """

    history: list[dict[str, Any]] = []
    for result in tool_results:
        call_id = str(result.get("call_id") or "")
        call = completed_calls.get(call_id)
        if not call:
            continue
        history.append(
            {
                "role": "assistant",
                "call_id": call_id,
                "content": dumps_canonical(
                    {
                        "call_id": call_id,
                        "name": str(call.get("name") or ""),
                        "arguments": deepcopy(call.get("arguments") or {}),
                    }
                ),
            }
        )
    return history


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
    gates: list[dict[str, Any]] | None = None,
    refusal_reason: str | None = None,
) -> dict[str, Any]:
    kind = "error" if error else ("call" if calls and not refuse else "refuse")
    provenance_out = dict(provenance)
    if gates:
        provenance_out["gates"] = list(gates)
    return {
        "wire_version": WIRE,
        "kind": kind,
        "call": calls[0] if kind == "call" and calls else None,
        "refusal": (
            {"reason": refusal_reason or "model_refusal", "detail": None}
            if kind == "refuse"
            else None
        ),
        "ok": ok,
        "error": error,
        "refuse": refuse,
        "selected_tools": selected,
        "schema_fingerprint": fingerprint,
        "function_calls": calls,
        "raw_text": raw_text,
        "confidence": _empty_confidence(),
        "provenance": provenance_out,
        "capabilities": capabilities,
        "state": {"step": 0, "pending_call_id": None, "cancelled": False},
        "stats": {"backend": backend, "wall_ms": wall_ms, "decode_mode": decode_mode},
    }


def _lexical_top5(query: str, catalog: list[dict[str, Any]], *, k: int = 5) -> list[dict[str, Any]]:
    """Deterministic degraded retrieval for protocol-only validation."""
    query_norm = (query or "").casefold()
    query_chars = set(query_norm)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for tool in catalog:
        name = str(tool.get("name") or "")
        text = (name + " " + str(tool.get("description") or "")).casefold()
        overlap = len(query_chars & set(text)) / max(1, len(query_chars))
        exact = 1.0 if name.casefold() and name.casefold() in query_norm else 0.0
        scored.append((exact * 10.0 + overlap, name, tool))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [tool for _, _, tool in scored[: min(int(k), len(scored))]]


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
        self.catalog: list[dict[str, Any]] = []

    @classmethod
    def load(
        cls,
        package_dir: str,
        *,
        verify_hashes: bool = True,
        backend: str = "protocol",
    ) -> "Engine":
        package = load_package(package_dir, verify_hashes=verify_hashes)
        if backend in {"auto", ""}:
            if package.packed_inference_ready() and str(package.manifest.get("product")) == "mei-1.0-51m":
                from .mlx_backend import load_mlx_runtime

                runtime, report = load_mlx_runtime(package, backend="mlx-reference")
                return cls(package, runtime=runtime, load_report=report, backend="mlx-reference")
            return cls(package, backend="protocol")
        if backend in {"protocol", "none"}:
            return cls(package, backend="protocol")
        if backend in {"mlx-reference", "mlx", "mlx-fused", "mlx-cq2"}:
            if not package.packed_inference_ready():
                raise SdkError(
                    "capability_missing",
                    "MLX inference requires a fully verified v2 package with every canonical head",
                )
            from .mlx_backend import load_mlx_runtime

            selected = "mlx-reference" if backend == "mlx" else backend
            runtime, report = load_mlx_runtime(package, backend=selected)
            return cls(package, runtime=runtime, load_report=report, backend=selected)
        raise SdkError("invalid_argument", f"unknown backend: {backend}")

    @staticmethod
    def version() -> dict[str, Any]:
        return sdk_versions()

    def capabilities(self) -> dict[str, Any]:
        capabilities = self.package.capabilities()
        capabilities["versions"] = sdk_versions()
        capabilities["inference"] = (
            self.runtime is not None and self.package.packed_inference_ready()
        )
        capabilities["backend_loaded"] = self.runtime is not None
        capabilities["protocol"] = True
        capabilities["backend"] = self.backend
        capabilities["stateful_tool_loop"] = True
        capabilities["max_steps_default"] = DEFAULT_MAX_STEPS
        capabilities["max_steps_hard"] = HARD_MAX_STEPS
        if self.runtime is not None and self.package.packed_inference_ready():
            capabilities["quantized_only"] = True
        if self.load_report:
            capabilities["load_report"] = {
                key: self.load_report.get(key) for key in ("n_loaded", "package_id", "backend")
            }
        return capabilities

    def register_tools(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        if self.closed:
            raise SdkError("session_closed", "engine is closed")
        try:
            validate_tools(catalog)
        except UnsupportedSchemaError as exc:
            raise SdkError("unsupported_schema", str(exc)) from exc
        self.catalog = [deepcopy(tool) for tool in catalog]
        return {
            "registered": len(self.catalog),
            "catalog_fingerprint": catalog_fingerprint(self.catalog),
        }

    def create_session(self, options: dict[str, Any] | None = None) -> "Session":
        if self.closed:
            raise SdkError("session_closed", "engine is closed")
        return Session(self, options or {})

    def close(self) -> None:
        self.closed = True

    close_engine = close


class Session:
    def __init__(self, engine: Engine, options: dict[str, Any]):
        self.engine = engine
        self.session_nonce = _allocate_session_nonce()
        self.options = dict(options)
        unknown_options = sorted(
            set(self.options) - {"max_steps", "max_tool_result_bytes"}
        )
        if unknown_options:
            raise SdkError(
                "invalid_argument", f"unknown session option: {unknown_options[0]}"
            )
        requested_steps = self.options.get("max_steps", DEFAULT_MAX_STEPS)
        if (
            isinstance(requested_steps, bool)
            or not isinstance(requested_steps, int)
            or requested_steps < 1
            or requested_steps > HARD_MAX_STEPS
        ):
            raise SdkError("invalid_argument", f"max_steps must be 1..{HARD_MAX_STEPS}")
        self.max_steps = requested_steps
        requested_result_bytes = self.options.get("max_tool_result_bytes", MAX_TOOL_RESULT_BYTES)
        if (
            isinstance(requested_result_bytes, bool)
            or not isinstance(requested_result_bytes, int)
            or not 1 <= requested_result_bytes <= 1024 * 1024
        ):
            raise SdkError("invalid_argument", "max_tool_result_bytes must be 1..1048576")
        self.max_tool_result_bytes = requested_result_bytes
        self.closed = False
        self.cancelled = False
        self.catalog = [deepcopy(tool) for tool in engine.catalog]
        self.pending_call: dict[str, Any] | None = None
        self.tool_results: list[dict[str, Any]] = []
        self.completed_calls: dict[str, dict[str, Any]] = {}
        self.step_index = 0
        self.narration = NarrationProvider()
        self.responded = False
        self.narration_query = ""

    def _require_open(self) -> None:
        if self.closed or self.engine.closed:
            raise SdkError("session_closed")

    def _terminalize_pending(
        self,
        *,
        status: str,
        source: str,
        error_code: str | None = None,
        message: str | None = None,
    ) -> None:
        """Settle a host-owned terminal path without leaving a poisoned call."""

        if self.pending_call is None:
            return
        call_id = str(self.pending_call.get("call_id") or "")
        row: dict[str, Any] = {
            "wire_version": WIRE,
            "call_id": call_id,
            "status": status,
            "payload": None,
            "provenance": {"source": source, "verified": True},
        }
        if status == "error":
            row["error"] = {
                "code": str(error_code or "host_error")[:256],
                "message": str(message or error_code or "host error")[:4096],
            }
        self.tool_results.append(row)
        self.completed_calls[call_id] = deepcopy(self.pending_call)
        self.pending_call = None

    def register_tools(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        self._require_open()
        try:
            validate_tools(catalog)
        except UnsupportedSchemaError as exc:
            raise SdkError("unsupported_schema", str(exc)) from exc
        self.catalog = [deepcopy(tool) for tool in catalog]
        return {
            "registered": len(self.catalog),
            "catalog_fingerprint": catalog_fingerprint(self.catalog),
        }

    def cancel(self) -> None:
        self._require_open()
        if self.cancelled:
            return
        self._terminalize_pending(
            status="cancelled", source="deterministic-policy:session-cancel"
        )
        self.cancelled = True

    def close(self) -> None:
        self.closed = True
        self.pending_call = None

    close_session = close

    def narrate(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_open()
        terminal_tool_result = bool(self.tool_results) and any(
            row.get("status") in {"error", "cancelled"} for row in self.tool_results
        )
        if not self.responded and not terminal_tool_result:
            raise SdkError("protocol_violation", "narration_requires_respond")
        options = dict(options or {})
        unknown = sorted(set(options) - {"mode", "call_ids", "locale"})
        if unknown:
            raise SdkError("invalid_argument", f"unknown narration option: {unknown[0]}")
        requested_mode = str(options.get("mode") or "deterministic")
        if requested_mode not in {"off", "deterministic", "adapter"}:
            raise SdkError("invalid_argument", "narration mode must be off|deterministic|adapter")
        locale = str(options.get("locale") or "zh-CN")
        if locale != "zh-CN":
            raise SdkError("invalid_argument", "only zh-CN narration is frozen")
        selected_ids = options.get("call_ids")
        if selected_ids is not None and (
            not isinstance(selected_ids, list)
            or any(not isinstance(value, str) for value in selected_ids)
        ):
            raise SdkError("invalid_argument", "narration call_ids must be a string array")
        allowed = set(selected_ids or [str(row.get("call_id") or "") for row in self.tool_results])
        views = []
        for accepted in self.tool_results:
            call_id = str(accepted.get("call_id") or "")
            if call_id not in allowed:
                continue
            call = self.completed_calls.get(call_id)
            if call is None:
                raise SdkError("protocol_violation", "narration_requires_session_result")
            view = deepcopy(accepted)
            view["tool_name"] = str(call.get("name") or "该工具")
            view["arguments"] = deepcopy(call.get("arguments") or {})
            views.append(view)
        if requested_mode != "off" and not views:
            raise SdkError("protocol_violation", "narration_requires_terminal_result")
        deterministic_text = "\n".join(
            self.narration.narrate(verified_result_view(view)) for view in views
        )
        runtime = self.engine.runtime
        adapter_ready = bool(
            runtime is not None
            and getattr(runtime, "narration_adapter", None) is not None
            and (self.engine.package.manifest.get("capabilities") or {}).get("narration") is True
        )
        adapter_text = None
        adapter_verified = False
        if requested_mode == "adapter" and adapter_ready:
            generated = runtime.generate_narration(
                narration_prompt_v2(self.narration_query, views), max_new=48
            )
            adapter_text = str(generated.get("text") or "").strip()
            # The first release gate accepts only the deterministic semantic
            # rendering. Any unsupported paraphrase falls back fail closed.
            adapter_verified = adapter_text == deterministic_text
        effective_mode = "adapter" if adapter_verified else (
            "deterministic" if requested_mode == "adapter" else requested_mode
        )
        fallback = requested_mode == "adapter" and not adapter_verified
        text = None
        if effective_mode != "off":
            text = adapter_text if adapter_verified else deterministic_text
        return {
            "wire_version": WIRE,
            "mode": effective_mode,
            "requested_mode": requested_mode,
            "locale": locale,
            "text": text,
            "grounded": True,
            "fallback_used": fallback,
            "provider_id": (
                NARRATION_ADAPTER_ID
                if adapter_verified
                else self.narration.capabilities()["provider_id"]
            ),
            "adapter_verified": adapter_verified,
            "call_ids": [str(view["call_id"]) for view in views],
        }

    def _error_turn(
        self,
        *,
        started: float,
        capabilities: dict[str, Any],
        decode_mode: str,
        error_id: str,
        detail: str | None = None,
        selected: list[str] | None = None,
        fingerprint: str | None = None,
        raw_text: str | None = None,
    ) -> dict[str, Any]:
        turn = _turn(
            ok=False,
            refuse=True,
            selected=list(selected or []),
            fingerprint=fingerprint,
            calls=[],
            raw_text=raw_text,
            error=error_from_id(error_id, detail),
            provenance={"validated": False, "ok": False, "detail": detail or error_id},
            capabilities=capabilities,
            backend=self.engine.backend,
            wall_ms=(time.perf_counter() - started) * 1000,
            decode_mode=decode_mode,
        )
        turn["state"] = {
            "step": self.step_index,
            "pending_call_id": self.pending_call.get("call_id") if self.pending_call else None,
            "cancelled": self.cancelled,
        }
        return turn

    def _finalize_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        calls = list(turn.get("function_calls") or [])
        if calls and not turn.get("refuse") and not turn.get("error"):
            if len(calls) != 1:
                turn["ok"] = False
                turn["refuse"] = True
                turn["kind"] = "error"
                turn["error"] = error_from_id("protocol_violation", "one call per step is required")
                turn["function_calls"] = []
                turn["call"] = None
                turn["refusal"] = None
            else:
                next_step = self.step_index + 1
                digest = hashlib.sha256(
                    (
                        self.session_nonce
                        + ":"
                        + str(next_step)
                        + ":"
                        + dumps_canonical(calls[0])
                    ).encode("utf-8")
                ).hexdigest()[:12]
                call_id = f"call-{self.session_nonce}-{next_step}-{digest}"
                call = dict(calls[0])
                call["call_id"] = call_id
                turn["function_calls"] = [call]
                turn["call"] = call
                turn["kind"] = "call"
                self.pending_call = call
        elif turn.get("error"):
            turn["kind"] = "error"
        elif _is_successful_respond(turn, self.tool_results):
            turn["kind"] = "respond"
            turn["ok"] = True
            turn["refuse"] = False
            turn["call"] = None
            turn["refusal"] = None
            turn["error"] = None
            turn["function_calls"] = []
            turn["execution"] = "respond"
            self.responded = True
        else:
            turn["kind"] = "refuse"
        self.step_index += 1
        turn["state"] = {
            "step": self.step_index,
            "pending_call_id": self.pending_call.get("call_id") if self.pending_call else None,
            "cancelled": self.cancelled,
        }
        return turn

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        capabilities = self.engine.capabilities()
        decode_mode = str(request.get("decode_mode") or "constrained") if isinstance(request, dict) else "constrained"
        self._require_open()
        if self.cancelled:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="cancelled",
            )
        if self.pending_call is not None:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="tool_result_required",
                detail="submit_tool_result() is required before the next complete()",
            )
        if self.step_index >= min(self.max_steps, HARD_MAX_STEPS):
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="max_steps",
                detail="max_steps_exceeded",
            )
        if (
            self.engine.package.manifest.get("package_format") == "mei-model-package-v2"
            and (not isinstance(request, dict) or request.get("wire_version") != WIRE)
        ):
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="invalid_argument",
                detail="native v2 sessions require wire_version=mei-runtime-wire-v2",
            )
        try:
            normalized = normalize_request(request)
        except (TypeError, ValueError) as exc:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="invalid_argument",
                detail=str(exc),
            )
        if not self.narration_query:
            self.narration_query = str(normalized.get("query") or "")
        if self.tool_results:
            normalized["history"] = list(normalized.get("history") or []) + _trusted_call_history(
                self.tool_results, self.completed_calls
            )
            normalized["tool_results"] = list(normalized.get("tool_results") or []) + list(self.tool_results)
        # Always overwrite the internal trust map, including with an empty
        # value. Caller-supplied ToolResult objects remain untrusted until this
        # Session has accepted them through submit_tool_result().
        normalized["_verified_result_map"] = {
            str(row.get("call_id") or ""): deepcopy(row)
            for row in self.tool_results
            if str(row.get("call_id") or "")
        }
        if normalized.get("catalog") is None and self.catalog:
            normalized["catalog"] = [deepcopy(tool) for tool in self.catalog]
        decode_mode = str(normalized.get("decode_mode") or "constrained")
        release_class = self.engine.package.manifest.get("release_class")
        production = release_class in {"candidate", "release"}
        if decode_mode == "raw" and production:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="decode_mode_forbidden",
                detail="candidate/release sessions require constrained decode",
            )
        fixture_fields = [
            field
            for field in ("candidate_text", "oracle_tools", "mw", "mw_disposition", "confidence")
            if normalized.get(field) is not None
        ]
        if production and fixture_fields:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="decode_mode_forbidden",
                detail=(
                    "candidate/release request overrides are forbidden: "
                    + ",".join(fixture_fields)
                ),
            )
        if production and normalized.get("enforce_confidence") is False:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="decode_mode_forbidden",
                detail="candidate/release sessions cannot disable confidence enforcement",
            )
        leaks = request_leaks(normalized)
        if leaks:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="gold_leak",
                detail="forbidden markers: " + ",".join(leaks),
            )
        oracle = normalized.get("oracle_tools")
        catalog = normalized.get("catalog")
        if catalog is None:
            catalog = self.catalog
        if oracle is not None and not isinstance(oracle, list):
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="invalid_argument",
                detail="oracle_tools must be an array",
            )
        if not isinstance(catalog, list):
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="invalid_argument",
                detail="catalog must be an array",
            )
        source_tools = oracle if oracle is not None else catalog
        try:
            validate_tools(source_tools)
        except UnsupportedSchemaError as exc:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="unsupported_schema",
                detail=str(exc),
            )
        if oracle is not None and len(oracle) > MAX_SELECTED_TOOLS:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="too_many_tools",
            )
        if self.engine.runtime is not None and normalized.get("candidate_text") is None:
            try:
                return self._finalize_turn(
                    self._complete_mlx(normalized, capabilities, started, decode_mode)
                )
            except UnsupportedSchemaError as exc:
                return self._error_turn(
                    started=started,
                    capabilities=capabilities,
                    decode_mode=decode_mode,
                    error_id="unsupported_schema",
                    detail=str(exc),
                )
            except SdkError as exc:
                return self._error_turn(
                    started=started,
                    capabilities=capabilities,
                    decode_mode=decode_mode,
                    error_id=exc.id,
                    detail=str(exc),
                )
            except ValueError as exc:
                detail = str(exc)
                error_id = (
                    "tool_schema_budget_exceeded"
                    if "tool_schema_budget_exceeded" in detail
                    else "invalid_argument"
                )
                return self._error_turn(
                    started=started,
                    capabilities=capabilities,
                    decode_mode=decode_mode,
                    error_id=error_id,
                    detail=detail,
                )
        tools = list(oracle) if oracle is not None else _lexical_top5(
            str(normalized.get("query") or ""), list(catalog), k=MAX_SELECTED_TOOLS
        )
        try:
            rendered = render_request(normalized, tools, already_normalized=True)
        except (UnsupportedSchemaError, ValueError) as exc:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="invalid_argument",
                detail=str(exc),
            )
        selected = rendered["selected_tools"]
        fingerprint = rendered["schema_fingerprint"]
        candidate = normalized.get("candidate_text")
        if candidate is None:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="engine_unavailable",
                detail="portable inference is unavailable; candidate_text only validates protocol semantics",
                selected=selected,
                fingerprint=fingerprint,
            )
        if leak_markers(str(candidate)):
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="gold_leak",
                detail="generation contains forbidden markers",
                selected=selected,
                fingerprint=fingerprint,
                raw_text=str(candidate),
            )
        validated = validate_generated_call(
            str(candidate),
            tools=tools,
            request=rendered["request"],
            confidence=normalized.get("confidence"),
            enforce_confidence=bool(normalized.get("enforce_confidence", False)),
        )
        gates = [{"gate": "retrieval", "ok": True, "selected_tools": selected}] + list(
            validated.get("gates") or []
        )
        validation_error = str(validated.get("error") or "")
        deterministic_refusal = validation_error in _REFUSAL_ERRORS or validation_error.startswith("state_requirement_missing:")
        error = None
        refusal_reason = None
        ok = bool(validated.get("ok"))
        if validation_error and deterministic_refusal:
            ok = True
            refusal_reason = validation_error
        elif validation_error:
            error = error_from_id("protocol_violation", validation_error)
        turn = _turn(
            ok=ok,
            refuse=bool(validated.get("refuse")) or deterministic_refusal,
            selected=selected,
            fingerprint=fingerprint,
            calls=list(validated.get("function_calls") or []) if error is None else [],
            raw_text=str(candidate),
            error=error,
            provenance={
                "validated": True,
                "ok": bool(validated.get("ok")),
                "arguments": validated.get("provenance") or {},
            },
            capabilities=capabilities,
            backend="protocol",
            wall_ms=(time.perf_counter() - started) * 1000,
            decode_mode=decode_mode,
            gates=gates,
            refusal_reason=refusal_reason,
        )
        confidence_value = validated.get("confidence_value")
        if confidence_value is not None:
            turn["confidence"] = {
                "available": True,
                "value": float(confidence_value),
                "source": "protocol-test",
            }
        if rendered.get("compatibility"):
            turn["provenance"]["request_compatibility"] = rendered["compatibility"]
        return self._finalize_turn(turn)

    def _complete_mlx(
        self,
        request: dict[str, Any],
        capabilities: dict[str, Any],
        started: float,
        decode_mode: str,
    ) -> dict[str, Any]:
        from .mlx_backend import complete_mlx

        oracle = request.get("oracle_tools")
        if oracle is not None and len(oracle) > MAX_SELECTED_TOOLS:
            return self._error_turn(
                started=started,
                capabilities=capabilities,
                decode_mode=decode_mode,
                error_id="too_many_tools",
            )
        out = complete_mlx(self.engine.runtime, request)
        selected = list(out.get("selected_tools") or [])
        tools = oracle if oracle is not None else (request.get("catalog") or self.catalog)
        used = [tool for tool in tools if str(tool.get("name") or "") in set(selected)]
        validated = out.get("validated") or {}
        calls = list(validated.get("function_calls") or [])
        validation_error = str(validated.get("error") or out.get("error") or "")
        deterministic_refusal = validation_error in _REFUSAL_ERRORS or validation_error.startswith("state_requirement_missing:")
        if not validation_error or deterministic_refusal:
            error = None
        elif validation_error == "tool_schema_budget_exceeded":
            error = error_from_id("tool_schema_budget_exceeded")
        else:
            error = error_from_id("protocol_violation", validation_error)
        turn = _turn(
            ok=bool(validated.get("ok", True)) if not deterministic_refusal else True,
            refuse=bool(validated.get("refuse")) or deterministic_refusal or not calls,
            selected=selected,
            fingerprint=schema_fingerprint(used) if used else None,
            calls=calls if error is None and not deterministic_refusal else [],
            raw_text=out.get("text"),
            error=error,
            provenance={
                "validated": True,
                "ok": bool(validated.get("ok", not error)),
                "arguments": validated.get("provenance") or {},
                "mw_disposition": validated.get("mw_audit"),
            },
            capabilities=capabilities,
            backend=self.engine.backend,
            wall_ms=(time.perf_counter() - started) * 1000,
            decode_mode=str(out.get("decode", {}).get("mode") or decode_mode),
            gates=[{"gate": "retrieval", "ok": True, "selected_tools": selected}]
            + list(validated.get("gates") or []),
            refusal_reason=validation_error if deterministic_refusal else None,
        )
        confidence = out.get("confidence")
        if confidence is not None:
            turn["confidence"] = {
                "available": True,
                "value": float(confidence) if not isinstance(confidence, dict) else confidence,
                "source": self.engine.backend,
                "version": "confidence-v2",
            }
        turn["stats"]["execution"] = validated.get("execution") or out.get("execution")
        stats = turn["stats"]
        stats.update(
            {key: float(value) for key, value in (out.get("timings") or {}).items() if value is not None}
        )
        if out.get("prompt_tokens") is not None:
            stats["prompt_tokens"] = int(out["prompt_tokens"])
        if out.get("output_tokens") is not None:
            stats["output_tokens"] = int(out["output_tokens"])
        decode_ms = float(stats.get("decode_ms") or 0.0)
        if stats.get("output_tokens") and decode_ms > 0:
            stats["output_tok_s"] = float(stats["output_tokens"]) / (decode_ms / 1000.0)
        if self.engine.load_report.get("sdk_backend_revision"):
            stats["sdk_backend_revision"] = self.engine.load_report["sdk_backend_revision"]
        if request.get("_compat"):
            turn["provenance"]["request_compatibility"] = request["_compat"]
        # Public Python v2 exposes the same executed sidecar/cache evidence as
        # Browser-WASM. These values come from the runtime, never the caller.
        turn["runtime_cache"] = dict(out.get("kv") or {})
        turn["mw_disposition"] = dict(out.get("mw_disposition") or {})
        return turn

    def submit_tool_result(self, result_or_call_id: Any, result: Any = None) -> dict[str, Any]:
        """Accept a ToolResultV2 and advance the pending-call state.

        ``submit_tool_result(call_id, payload)`` remains as a degraded Python
        compatibility spelling.  New callers should pass the same
        ToolResultV2 object used by Rust/C/Node/browser.
        """

        if self.closed or self.engine.closed:
            raise SdkError("session_closed")
        if self.pending_call is None:
            raise SdkError("stale_call_id", "there is no pending tool call")
        strict_v2 = result is None and isinstance(result_or_call_id, dict)
        if strict_v2:
            submitted = dict(result_or_call_id)
            try:
                validate_tool_result_v2(submitted)
            except (TypeError, ValueError) as exc:
                raise SdkError("invalid_argument", str(exc)) from exc
            call_id = str(submitted.get("call_id") or "")
            status = str(submitted.get("status") or "")
            payload = submitted.get("payload")
            provenance = submitted.get("provenance")
            submitted_error = submitted.get("error")
            if status == "error" and (
                not isinstance(submitted_error, dict)
                or set(submitted_error) != {"code", "message"}
                or not all(isinstance(submitted_error.get(key), str) for key in ("code", "message"))
            ):
                raise SdkError(
                    "invalid_argument", "error status requires error {code,message}"
                )
            if status == "ok" and submitted_error is not None:
                raise SdkError("invalid_argument", "ok status cannot carry error")
            if not isinstance(provenance, dict) or provenance.get("verified") is not True:
                raise SdkError("protocol_violation", "tool result provenance must be verified")
        else:
            call_id = str(result_or_call_id)
            if isinstance(result, dict) and "status" in result and (
                "payload" in result or "result" in result
            ):
                status = str(result.get("status"))
                payload = result.get("payload", result.get("result"))
                provenance = result.get("provenance") or {}
            else:
                status = "ok"
                payload = result
                provenance = {"source": "python_v1_adapter"}
        if str(call_id) != str(self.pending_call.get("call_id")):
            raise SdkError("stale_call_id", "stale or unknown call_id")
        if status not in {"ok", "error", "cancelled"}:
            raise SdkError("invalid_argument", "tool result status must be ok, error or cancelled")
        try:
            verified = (
                submitted
                if strict_v2
                else verified_tool_result(
                    call_id=str(call_id), status=status, payload=payload, provenance=provenance
                )
            )
            encoded = json.dumps(
                verified, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SdkError("invalid_argument", f"tool result is not JSON-safe: {exc}") from exc
        if len(encoded) > self.max_tool_result_bytes:
            raise SdkError(
                "tool_result_too_large",
                f"ToolResultV2 exceeds {self.max_tool_result_bytes} bytes",
            )
        # Freeze the exact JSON value that crossed the trust boundary.  A host
        # retaining and mutating its original nested payload must not rewrite
        # later provenance or narration state.
        verified = json.loads(encoded.decode("utf-8"))
        self.tool_results.append(verified)
        self.completed_calls[str(call_id)] = deepcopy(self.pending_call)
        self.pending_call = None
        if status == "cancelled":
            self.cancelled = True
        return {
            "wire_version": WIRE,
            "accepted": True,
            "call_id": str(call_id),
            "state": {
                "step": self.step_index,
                "pending_call_id": None,
                "cancelled": self.cancelled,
            },
        }

    def embed(self, text: str) -> list[float]:
        self._require_open()
        if self.engine.runtime is None:
            raise SdkError("engine_unavailable", "embed requires an inference backend")
        runtime = self.engine.runtime
        vector = runtime.embed_text(text) if hasattr(runtime, "embed_text") else runtime._embed_text(text)
        if vector is None:
            raise SdkError("head_missing", "contrastive head did not return a vector")
        try:
            import mlx.core as mx

            mx.eval(vector)
        except Exception:
            pass
        data = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if data and isinstance(data[0], (list, tuple)):
            data = data[0]
        return [float(value) for value in data]

    def retrieve(self, query: str, catalog: list[dict[str, Any]], *, k: int = 5) -> dict[str, Any]:
        self._require_open()
        try:
            validate_tools(catalog)
        except UnsupportedSchemaError as exc:
            raise SdkError("unsupported_schema", str(exc)) from exc
        if self.engine.runtime is None:
            selected = _lexical_top5(query, catalog, k=k)
            source = "degraded_lexical"
        else:
            runtime = self.engine.runtime
            selected = (
                runtime.search_top_k(query, catalog, k=k)
                if hasattr(runtime, "search_top_k")
                else runtime._select(query, catalog)
            )
            source = str(getattr(runtime, "retrieval_source", "contrastive"))
        return {
            "selected_tools": [str(tool.get("name") or "") for tool in selected[:k]],
            "tools": selected[:k],
            "k": k,
            "source": source,
            "schema_fingerprint": schema_fingerprint(catalog),
        }

    def run(
        self,
        request: dict[str, Any],
        *,
        executors: dict[str, Callable[[dict[str, Any]], Any]] | Callable[[dict[str, Any]], Any] | None = None,
        max_steps: int | None = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        requested = max_steps if max_steps is not None else max_turns
        limit = self.max_steps if requested is None else int(requested)
        if limit < 1 or limit > HARD_MAX_STEPS:
            raise SdkError("invalid_argument", f"max_steps must be 1..{HARD_MAX_STEPS}")
        turns: list[dict[str, Any]] = []
        stopped = "max_steps"
        base_request = dict(request)
        candidate_texts = list(base_request.pop("candidate_texts", []) or [])
        for index in range(limit):
            current = dict(base_request)
            if candidate_texts:
                current["candidate_text"] = candidate_texts[min(index, len(candidate_texts) - 1)]
            turn = self.complete(current)
            turns.append(turn)
            if turn.get("error"):
                if turn["error"]["id"] == "cancelled":
                    stopped = "cancelled"
                elif turn["error"]["id"] == "max_steps":
                    stopped = "max_steps"
                else:
                    stopped = "error"
                break
            if turn.get("kind") == "respond":
                stopped = "respond"
                break
            if turn.get("kind") == "refuse":
                stopped = "refuse"
                break
            calls = list(turn.get("function_calls") or [])
            if not calls:
                stopped = "error"
                break
            call = calls[0]
            executor: Callable[[dict[str, Any]], Any] | None
            if callable(executors):
                executor = executors
            elif isinstance(executors, dict):
                executor = executors.get(str(call.get("name") or ""))
            else:
                executor = None
            if executor is None:
                # Executor absence is a deterministic terminal ToolResult, not
                # a dangling pending call that poisons every later complete().
                self.submit_tool_result(
                    {
                        "wire_version": WIRE,
                        "call_id": str(call["call_id"]),
                        "status": "error",
                        "payload": None,
                        "error": {
                            "code": "executor_missing",
                            "message": f"no executor registered for {call.get('name')}",
                        },
                        "provenance": {
                            "source": "deterministic-policy",
                            "verified": True,
                        },
                    }
                )
                turns.append(
                    self._error_turn(
                        started=time.perf_counter(),
                        capabilities=self.engine.capabilities(),
                        decode_mode=str(current.get("decode_mode") or "constrained"),
                        error_id="executor_missing",
                        detail=f"no executor registered for {call.get('name')}",
                        selected=list(turn.get("selected_tools") or []),
                        fingerprint=turn.get("schema_fingerprint"),
                    )
                )
                stopped = "error"
                break
            try:
                payload = executor(call)
                result = {"status": "ok", "payload": payload, "provenance": {"source": "host_executor"}}
            except Exception as exc:  # Host exceptions become verified terminal results.
                result = {
                    "status": "error",
                    "payload": None,
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                    "provenance": {"source": "host_executor"},
                }
            submitted = {
                "wire_version": WIRE,
                "call_id": str(call["call_id"]),
                "status": str(result["status"]),
                "payload": result["payload"],
                "provenance": {
                    "source": f"host-executor:{call.get('name')}",
                    "verified": True,
                },
            }
            if result.get("error") is not None:
                submitted["error"] = result["error"]
            try:
                self.submit_tool_result(submitted)
            except SdkError as exc:
                self._terminalize_pending(
                    status="error",
                    source="deterministic-policy:result-rejected",
                    error_code=exc.id,
                    message=str(exc),
                )
                turns.append(
                    self._error_turn(
                        started=time.perf_counter(),
                        capabilities=self.engine.capabilities(),
                        decode_mode=str(current.get("decode_mode") or "constrained"),
                        error_id=exc.id,
                        detail=str(exc),
                        selected=list(turn.get("selected_tools") or []),
                        fingerprint=turn.get("schema_fingerprint"),
                    )
                )
                stopped = "cancelled" if exc.id == "cancelled" else "error"
                break
            if result["status"] != "ok":
                error_detail = result.get("error") or {}
                turns.append(
                    self._error_turn(
                        started=time.perf_counter(),
                        capabilities=self.engine.capabilities(),
                        decode_mode=str(current.get("decode_mode") or "constrained"),
                        error_id="protocol_violation",
                        detail=(
                            "executor_error:"
                            + str(error_detail.get("code") or "host_error")
                            + ":"
                            + str(error_detail.get("message") or "host executor failed")
                        ),
                        selected=list(turn.get("selected_tools") or []),
                        fingerprint=turn.get("schema_fingerprint"),
                    )
                )
                stopped = "cancelled" if result["status"] == "cancelled" else "error"
                break
            if self.cancelled:
                stopped = "cancelled"
                break
        else:
            stopped = "max_steps"
            turns.append(
                self._error_turn(
                    started=time.perf_counter(),
                    capabilities=self.engine.capabilities(),
                    decode_mode=str(base_request.get("decode_mode") or "constrained"),
                    error_id="max_steps",
                    detail="max_steps_exceeded",
                )
            )
        return {
            "wire_version": WIRE,
            "ok": bool(turns)
            and all(turn.get("ok") for turn in turns)
            and stopped == "respond",
            "turns": turns,
            "tool_results": list(self.tool_results),
            "stopped_reason": stopped,
        }
