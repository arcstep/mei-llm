"""Schema-conditioned JSON prefix FSM for phase-1 [] | one call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from schema_render import schema_hash

_CACHE: dict[str, "CompiledSchema"] = {}


@dataclass(frozen=True)
class ArgSpec:
    name: str
    typ: str
    enum: tuple[Any, ...] | None
    required: bool


@dataclass(frozen=True)
class CompiledTool:
    name: str
    args: tuple[ArgSpec, ...]
    required: tuple[str, ...]
    arg_index: dict[str, ArgSpec]


@dataclass
class CompiledSchema:
    schema_hash: str
    tools: tuple[CompiledTool, ...]
    names: tuple[str, ...]
    by_name: dict[str, CompiledTool]


def _normalize_raw(toolset: Any) -> dict[str, Any]:
    if toolset is None:
        raise ValueError("toolset is required; no default VRM")
    if isinstance(toolset, dict) and "tools" in toolset:
        return toolset
    tools = []
    if isinstance(toolset, dict):
        for name, spec in toolset.items():
            if hasattr(spec, "properties"):
                tools.append(
                    {
                        "name": spec.name,
                        "parameters": {
                            "type": "object",
                            "properties": spec.properties,
                            "required": list(spec.required),
                        },
                    }
                )
            elif isinstance(spec, dict) and spec.get("name"):
                tools.append(spec)
            else:
                tools.append({"name": name, "parameters": spec if isinstance(spec, dict) else {}})
    return {"tools": tools}


def compile_schema(toolset: Any) -> CompiledSchema:
    raw = _normalize_raw(toolset)
    key = schema_hash(raw)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    tools: list[CompiledTool] = []
    for tool in raw.get("tools") or []:
        name = str(tool.get("name") or "")
        if not name:
            continue
        params = tool.get("parameters") or {}
        props = params.get("properties") or {}
        required = tuple(str(x) for x in (params.get("required") or []))
        args: list[ArgSpec] = []
        for arg_name, schema in props.items():
            schema = schema or {}
            enum = schema.get("enum")
            args.append(
                ArgSpec(
                    name=str(arg_name),
                    typ=str(schema.get("type") or "string"),
                    enum=tuple(enum) if enum is not None else None,
                    required=str(arg_name) in required,
                )
            )
        arg_t = tuple(args)
        tools.append(
            CompiledTool(
                name=name,
                args=arg_t,
                required=required,
                arg_index={a.name: a for a in arg_t},
            )
        )
    compiled = CompiledSchema(
        schema_hash=key,
        tools=tuple(tools),
        names=tuple(t.name for t in tools),
        by_name={t.name: t for t in tools},
    )
    _CACHE[key] = compiled
    return compiled


def _type_ok(value: Any, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_schema_calls(calls: Any, compiled: CompiledSchema) -> list[str]:
    if calls == [] or calls is None:
        return []
    if not isinstance(calls, list):
        return ["not a list"]
    if len(calls) > 1:
        return ["phase1 allows at most one call"]
    errors: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            errors.append("call not object")
            continue
        extra_keys = set(call) - {"name", "arguments"}
        if extra_keys:
            errors.append("unexpected call keys")
        name = str(call.get("name") or "")
        spec = compiled.by_name.get(name)
        if not spec:
            errors.append(f"unknown tool {name}")
            continue
        args = call.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            errors.append("arguments not object")
            continue
        for key in spec.required:
            if key not in args:
                errors.append(f"{name} missing {key}")
        for key, val in args.items():
            prop = spec.arg_index.get(key)
            if not prop:
                errors.append(f"{name} unexpected {key}")
                continue
            if prop.typ and not _type_ok(val, prop.typ):
                errors.append(f"{name}.{key} type")
            if prop.enum is not None and val not in prop.enum:
                errors.append(f"{name}.{key} not in enum")
    return errors


_WS = re.compile(r"[ \t\r\n]*")


class _P:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def eof(self) -> bool:
        return self.i >= self.n

    def peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def get(self) -> str:
        ch = self.peek()
        self.i += 1
        return ch


INCOMPLETE = "incomplete"
OK = "ok"
BAD = "bad"


def _prefix_of(s: str, options: tuple[str, ...] | list[str]) -> bool:
    return any(opt.startswith(s) for opt in options)


def _parse_string(p: _P) -> tuple[str, str | None]:
    """Return (status, value_or_none). value is None if unterminated."""
    if p.peek() != '"':
        return BAD, None
    p.get()
    out: list[str] = []
    while not p.eof():
        ch = p.get()
        if ch == "\\":
            if p.eof():
                return INCOMPLETE, None
            esc = p.get()
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
            out.append(mapping.get(esc, esc))
            continue
        if ch == '"':
            return OK, "".join(out)
        out.append(ch)
    return INCOMPLETE, "".join(out)


def _parse_bool_or_null_prefix(p: _P) -> str:
    start = p.i
    while not p.eof() and p.peek().isalpha():
        p.get()
    token = p.s[start : p.i]
    if not token:
        return INCOMPLETE if p.eof() else BAD
    if token in ("true", "false", "null"):
        return OK
    if "true".startswith(token) or "false".startswith(token) or "null".startswith(token):
        return INCOMPLETE if p.eof() else BAD
    return BAD


def _parse_number(p: _P) -> str:
    start = p.i
    if p.peek() == "-":
        p.get()
    if p.eof():
        return INCOMPLETE
    if not p.peek().isdigit():
        return BAD
    while not p.eof() and p.peek().isdigit():
        p.get()
    if not p.eof() and p.peek() == ".":
        p.get()
        if p.eof():
            return INCOMPLETE
        if not p.peek().isdigit():
            return BAD
        while not p.eof() and p.peek().isdigit():
            p.get()
    if start == p.i:
        return BAD
    return OK


def _parse_scalar(p: _P, spec: ArgSpec | None) -> str:
    p.skip_ws()
    if p.eof():
        return INCOMPLETE
    ch = p.peek()
    allowed_types = {spec.typ} if spec else {"string", "boolean", "integer", "number"}
    if ch == '"':
        if spec and spec.typ not in {"string", ""}:
            return BAD
        st, val = _parse_string(p)
        if st == BAD:
            return BAD
        if spec and spec.enum is not None:
            opts = tuple(str(x) for x in spec.enum)
            if st == INCOMPLETE:
                return INCOMPLETE if _prefix_of(val or "", opts) else BAD
            return OK if val in spec.enum or val in opts else BAD
        return st
    if ch in "tfn":
        if spec and spec.typ != "boolean" and spec.typ != "":
            # allow incomplete 't' only for boolean
            if spec.typ not in {"boolean"}:
                return BAD
        st = _parse_bool_or_null_prefix(p)
        if st != OK:
            return st
        # completed true/false
        return OK
    if ch == "-" or ch.isdigit():
        if spec and spec.typ not in {"integer", "number", ""}:
            return BAD
        return _parse_number(p)
    return BAD


def _parse_object_args(p: _P, tool: CompiledTool | None) -> str:
    if p.peek() != "{":
        return BAD
    p.get()
    seen: set[str] = set()
    p.skip_ws()
    if p.eof():
        return INCOMPLETE
    if p.peek() == "}":
        p.get()
        return OK
    while True:
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() != '"':
            return BAD
        st, key = _parse_string(p)
        if st == INCOMPLETE:
            names = tuple(a.name for a in tool.args) if tool else ()
            return INCOMPLETE if (not names or _prefix_of(key or "", names)) else BAD
        if st != OK or key is None:
            return BAD
        if tool and key not in tool.arg_index:
            return BAD
        if key in seen:
            return BAD
        seen.add(key)
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() != ":":
            return BAD
        p.get()
        spec = tool.arg_index.get(key) if tool else None
        stv = _parse_scalar(p, spec)
        if stv != OK:
            return stv
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() == ",":
            p.get()
            p.skip_ws()
            if p.eof():
                return INCOMPLETE
            continue
        if p.peek() == "}":
            p.get()
            return OK
        return BAD


def _parse_call_object(p: _P, compiled: CompiledSchema) -> str:
    if p.peek() != "{":
        return BAD
    p.get()
    name: str | None = None
    saw_args = False
    p.skip_ws()
    if p.eof():
        return INCOMPLETE
    if p.peek() == "}":
        return BAD
    while True:
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() != '"':
            return BAD
        st, key = _parse_string(p)
        if st == INCOMPLETE:
            return INCOMPLETE if _prefix_of(key or "", ("name", "arguments")) else BAD
        if st != OK or key is None:
            return BAD
        if key not in {"name", "arguments"}:
            return BAD
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() != ":":
            return BAD
        p.get()
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if key == "name":
            stn, val = _parse_string(p)
            if stn == INCOMPLETE:
                return INCOMPLETE if _prefix_of(val or "", compiled.names) else BAD
            if stn != OK or val is None:
                return BAD
            if val not in compiled.by_name:
                return BAD
            name = val
        else:
            saw_args = True
            tool = compiled.by_name.get(name) if name else None
            # arguments may appear before name; then tool is None and keys must still
            # be a prefix of the union of all arg names, or exact in some tool.
            if tool is None:
                st_args = _parse_object_args_union(p, compiled)
            else:
                st_args = _parse_object_args(p, tool)
            if st_args != OK:
                return st_args
        p.skip_ws()
        if p.eof():
            return INCOMPLETE
        if p.peek() == ",":
            p.get()
            continue
        if p.peek() == "}":
            p.get()
            return OK
        return BAD


def _parse_object_args_union(p: _P, compiled: CompiledSchema) -> str:
    names = tuple({a.name for t in compiled.tools for a in t.args})
    dummy = CompiledTool(name="", args=tuple(), required=tuple(), arg_index={n: ArgSpec(n, "", None, False) for n in names})
    dummy.arg_index  # noqa: B018
    # Rebuild with typ unknown
    dummy = CompiledTool(
        name="",
        args=tuple(ArgSpec(n, "", None, False) for n in names),
        required=tuple(),
        arg_index={n: ArgSpec(n, "", None, False) for n in names},
    )
    return _parse_object_args(p, dummy if names else None)


def prefix_status(text: str, compiled: CompiledSchema) -> str:
    s = (text or "").lstrip()
    if not s:
        return INCOMPLETE
    if s[0] != "[":
        return BAD
    p = _P(s)
    p.get()
    p.skip_ws()
    if p.eof():
        return INCOMPLETE
    if p.peek() == "]":
        p.get()
        p.skip_ws()
        return OK if p.eof() or p.peek() == "" else (OK if p.eof() else BAD)
    if p.peek() != "{":
        return BAD
    st = _parse_call_object(p, compiled)
    if st != OK:
        return st
    p.skip_ws()
    if p.eof():
        return INCOMPLETE
    if p.peek() == "]":
        p.get()
        return OK
    if p.peek() == ",":
        return BAD  # multi-call not in subset
    return BAD


def is_schema_legal_prefix(prefix: str, toolset: Any) -> bool:
    compiled = compile_schema(toolset)
    st = prefix_status(prefix, compiled)
    if st == BAD:
        return False
    if st == INCOMPLETE:
        return True
    try:
        obj = json.loads(prefix.lstrip())
    except json.JSONDecodeError:
        return True
    return not validate_schema_calls(obj, compiled)
