"""Optional ctypes wrapper around the ``mei-runtime-abi-2`` C library.

The C surface deliberately uses opaque handles and stepwise JSON instead of a
language callback.  Keeping a :class:`NativeSession` alive across calls is
therefore important: a tool call returned by ``complete`` is resolved by a
later ``submit_tool_result`` on the same handle.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

from .canonical import dumps_canonical
from .errors import SdkError, lookup_id
from .json_schema_lite import loads_strict
from .version import SDK_ROOT


_REQUIRED_ABI2_SYMBOLS = (
    "mei_sdk_abi_version",
    "mei_sdk_version",
    "mei_sdk_engine_open",
    "mei_sdk_engine_register_tools",
    "mei_sdk_engine_close",
    "mei_sdk_session_open",
    "mei_sdk_session_complete",
    "mei_sdk_session_submit_tool_result",
    "mei_sdk_session_narrate",
    "mei_sdk_session_cancel",
    "mei_sdk_session_close",
    "mei_sdk_string_free",
    "mei_sdk_last_error",
)


def _lib_candidates() -> list[Path]:
    names = [
        "libmei_sdk.dylib",
        "libmei_sdk.so",
        "mei_sdk.dll",
        "libmei_sdk_ffi.dylib",
        "libmei_sdk_ffi.so",
    ]
    roots = [
        SDK_ROOT / "target" / "release",
        SDK_ROOT / "target" / "debug",
        SDK_ROOT / "c",
    ]
    extra = os.environ.get("MEI_SDK_LIB")
    out: list[Path] = []
    if extra:
        out.append(Path(extra))
    for root in roots:
        for name in names:
            out.append(root / name)
    return out


def load_native_library() -> ctypes.CDLL:
    last: Exception | None = None
    for path in _lib_candidates():
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            abi = lib.mei_sdk_abi_version
            abi.argtypes = []
            abi.restype = ctypes.c_int32
            found = int(abi())
            if found != 2:
                last = RuntimeError(f"{path} exposes ABI {found}, expected ABI 2")
                continue
            missing = [name for name in _REQUIRED_ABI2_SYMBOLS if not hasattr(lib, name)]
            if missing:
                last = RuntimeError(
                    f"{path} exposes incomplete ABI 2 symbols: {','.join(missing)}"
                )
                continue
            return lib
        except (AttributeError, OSError) as exc:
            last = exc
    raise SdkError("engine_unavailable", f"native libmei_sdk not found ({last})")


class _NativeJsonApi:
    """Shared error and owned-string handling for native wrappers."""

    lib: ctypes.CDLL

    def _check(self, code: int) -> None:
        if code == 0:
            return
        buf = ctypes.create_string_buffer(4096)
        self.lib.mei_sdk_last_error(buf, len(buf))
        raise SdkError(lookup_id(code), buf.value.decode("utf-8", "replace"))

    def _json_call(self, function: Any, handle: ctypes.c_void_p, payload: dict) -> dict:
        out = ctypes.c_char_p()
        try:
            encoded = dumps_canonical(payload).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise SdkError("invalid_json", str(exc)) from exc
        try:
            self._check(function(handle, encoded, ctypes.byref(out)))
            if not out.value:
                raise SdkError("protocol_violation", "native runtime returned an empty JSON result")
            try:
                value = loads_strict(ctypes.string_at(out).decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise SdkError("protocol_violation", f"invalid native JSON result: {exc}") from exc
            if not isinstance(value, dict):
                raise SdkError("protocol_violation", "native runtime result must be a JSON object")
            return value
        finally:
            if out.value:
                self.lib.mei_sdk_string_free(out)


class NativeEngine(_NativeJsonApi):
    def __init__(self, lib: ctypes.CDLL | None = None):
        self.lib = lib or load_native_library()
        self.lib.mei_sdk_abi_version.argtypes = []
        self.lib.mei_sdk_abi_version.restype = ctypes.c_int32
        found_abi = int(self.lib.mei_sdk_abi_version())
        if found_abi != 2:
            raise SdkError(
                "abi_version_mismatch",
                f"native runtime ABI {found_abi} is not mei-runtime-abi-2",
            )
        self.lib.mei_sdk_version.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.mei_sdk_version.restype = ctypes.c_int32
        self.lib.mei_sdk_engine_open.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.mei_sdk_engine_open.restype = ctypes.c_int32
        self.lib.mei_sdk_engine_register_tools.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.mei_sdk_engine_register_tools.restype = ctypes.c_int32
        self.lib.mei_sdk_engine_close.argtypes = [ctypes.c_void_p]
        self.lib.mei_sdk_engine_close.restype = ctypes.c_int32
        self.lib.mei_sdk_session_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.mei_sdk_session_open.restype = ctypes.c_int32
        self.lib.mei_sdk_session_complete.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.mei_sdk_session_complete.restype = ctypes.c_int32
        self.lib.mei_sdk_session_submit_tool_result.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.mei_sdk_session_submit_tool_result.restype = ctypes.c_int32
        self.lib.mei_sdk_session_narrate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.mei_sdk_session_narrate.restype = ctypes.c_int32
        self.lib.mei_sdk_session_cancel.argtypes = [ctypes.c_void_p]
        self.lib.mei_sdk_session_cancel.restype = ctypes.c_int32
        self.lib.mei_sdk_session_close.argtypes = [ctypes.c_void_p]
        self.lib.mei_sdk_session_close.restype = ctypes.c_int32
        self.lib.mei_sdk_string_free.argtypes = [ctypes.c_char_p]
        self.lib.mei_sdk_string_free.restype = ctypes.c_int32
        self.lib.mei_sdk_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.mei_sdk_last_error.restype = ctypes.c_int32
        self._engine = ctypes.c_void_p()

    @property
    def is_open(self) -> bool:
        return bool(self._engine.value)

    def abi_version(self) -> int:
        return int(self.lib.mei_sdk_abi_version())

    def version(self) -> dict:
        buf = ctypes.create_string_buffer(1024)
        self._check(self.lib.mei_sdk_version(buf, 1024))
        value = loads_strict(buf.value.decode("utf-8"))
        if not isinstance(value, dict):
            raise SdkError("protocol_violation", "native version must be a JSON object")
        return value

    def open(self, package_dir: str) -> None:
        if self.is_open:
            raise SdkError("invalid_argument", "native engine is already open")
        self._check(self.lib.mei_sdk_engine_open(package_dir.encode("utf-8"), ctypes.byref(self._engine)))

    def close(self) -> None:
        if self.is_open:
            self._check(self.lib.mei_sdk_engine_close(self._engine))
            self._engine = ctypes.c_void_p()

    def register_tools(self, tools: list[dict] | dict) -> dict:
        if not self.is_open:
            raise SdkError("engine_unavailable", "native engine is not open")
        payload = tools.get("tools") if isinstance(tools, dict) else tools
        if not isinstance(payload, list):
            raise SdkError("invalid_argument", "tools must be an array")
        return self._json_call(self.lib.mei_sdk_engine_register_tools, self._engine, payload)

    def create_session(self, options: dict | None = None) -> "NativeSession":
        if not self.is_open:
            raise SdkError("engine_unavailable", "native engine is not open")
        session = ctypes.c_void_p()
        try:
            payload = dumps_canonical(options or {}).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise SdkError("invalid_json", str(exc)) from exc
        self._check(self.lib.mei_sdk_session_open(self._engine, payload, ctypes.byref(session)))
        return NativeSession(self, session)

    def complete_json(self, request: dict) -> dict:
        """Compatibility one-shot helper.

        Multi-step tool execution must use :meth:`create_session`, otherwise
        the pending call state would be discarded when this method returns.
        """

        session = self.create_session()
        try:
            turn = session.complete(request)
            if turn.get("kind") == "call" or turn.get("call") is not None:
                session.cancel()
                raise SdkError(
                    "tool_result_required",
                    "one-shot complete_json cannot return a resumable call; use create_session()",
                )
            return turn
        finally:
            session.close()

    def __enter__(self) -> "NativeEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NativeSession(_NativeJsonApi):
    """Persistent ABI-2 session supporting the tool-result continuation loop."""

    def __init__(self, engine: NativeEngine, handle: ctypes.c_void_p):
        self.engine = engine
        self.lib = engine.lib
        self._session = handle

    @property
    def is_open(self) -> bool:
        return bool(self._session.value)

    def _require_open(self) -> None:
        if not self.is_open:
            raise SdkError("session_closed", "native session is closed")

    def complete(self, request: dict) -> dict:
        self._require_open()
        return self._json_call(self.lib.mei_sdk_session_complete, self._session, request)

    def submit_tool_result(self, result: dict) -> dict:
        self._require_open()
        return self._json_call(
            self.lib.mei_sdk_session_submit_tool_result,
            self._session,
            result,
        )

    def narrate(self, options: dict | None = None) -> dict:
        self._require_open()
        return self._json_call(
            self.lib.mei_sdk_session_narrate,
            self._session,
            options or {},
        )

    def cancel(self) -> None:
        self._require_open()
        self._check(self.lib.mei_sdk_session_cancel(self._session))

    def close(self) -> None:
        if self.is_open:
            self._check(self.lib.mei_sdk_session_close(self._session))
            self._session = ctypes.c_void_p()

    def __enter__(self) -> "NativeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
