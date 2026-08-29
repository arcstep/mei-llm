"""Optional ctypes wrapper around libmei_sdk. Semantics must match mei_sdk.engine."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from .errors import SdkError, lookup_id
from .version import SDK_ROOT


def _lib_candidates() -> list[Path]:
    names = [
        "libmei_sdk.dylib",
        "libmei_sdk.so",
        "mei_sdk.dll",
        "libmei_sdk_ffi.dylib",
        "libmei_sdk_ffi.so",
    ]
    roots = [
        SDK_ROOT / "target" / "debug",
        SDK_ROOT / "target" / "release",
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
            return ctypes.CDLL(str(path))
        except OSError as exc:
            last = exc
    raise SdkError("engine_unavailable", f"native libmei_sdk not found ({last})")


class NativeEngine:
    def __init__(self, lib: ctypes.CDLL | None = None):
        self.lib = lib or load_native_library()
        self.lib.mei_sdk_version.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.mei_sdk_version.restype = ctypes.c_int32
        self.lib.mei_sdk_engine_open.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.mei_sdk_engine_open.restype = ctypes.c_int32
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
        self.lib.mei_sdk_session_close.argtypes = [ctypes.c_void_p]
        self.lib.mei_sdk_session_close.restype = ctypes.c_int32
        self.lib.mei_sdk_string_free.argtypes = [ctypes.c_char_p]
        self.lib.mei_sdk_string_free.restype = ctypes.c_int32
        self.lib.mei_sdk_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.mei_sdk_last_error.restype = ctypes.c_int32
        self._engine = ctypes.c_void_p()

    def _check(self, code: int) -> None:
        if code == 0:
            return
        buf = ctypes.create_string_buffer(1024)
        self.lib.mei_sdk_last_error(buf, 1024)
        raise SdkError(lookup_id(code), buf.value.decode("utf-8", "replace"))

    def version(self) -> dict:
        buf = ctypes.create_string_buffer(1024)
        self._check(self.lib.mei_sdk_version(buf, 1024))
        return json.loads(buf.value.decode("utf-8"))

    def open(self, package_dir: str) -> None:
        self._check(self.lib.mei_sdk_engine_open(package_dir.encode("utf-8"), ctypes.byref(self._engine)))

    def close(self) -> None:
        if self._engine:
            self.lib.mei_sdk_engine_close(self._engine)
            self._engine = ctypes.c_void_p()

    def complete_json(self, request: dict) -> dict:
        session = ctypes.c_void_p()
        self._check(self.lib.mei_sdk_session_open(self._engine, b"{}", ctypes.byref(session)))
        out = ctypes.c_char_p()
        try:
            payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
            self._check(self.lib.mei_sdk_session_complete(session, payload, ctypes.byref(out)))
            text = ctypes.string_at(out).decode("utf-8")
            return json.loads(text)
        finally:
            if out:
                self.lib.mei_sdk_string_free(out)
            self.lib.mei_sdk_session_close(session)
