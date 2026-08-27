"""Byte-level grammar over selected schemas for compact [] | one tool call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from schema_render import compact_tools, dumps_canonical

NEG_INF = -1e9

_HEX_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def token_to_bytes(tokenizer, tok_id: int) -> bytes:
    if tok_id in {tokenizer.pad_id, tokenizer.bos_id}:
        return b""
    if tok_id == tokenizer.eos_id:
        return b""
    piece = tokenizer.sp.id_to_piece(int(tok_id))
    m = _HEX_PIECE.fullmatch(piece or "")
    if m:
        return bytes([int(m.group(1), 16)])
    if piece.startswith("▁"):
        piece = " " + piece[1:]
    try:
        return piece.encode("utf-8")
    except UnicodeEncodeError:
        return b""


def _json_str(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ArgConstraint:
    name: str
    typ: str
    required: bool
    enum: tuple[Any, ...] | None
    const: Any
    minimum: float | None
    maximum: float | None
    exclusive_minimum: float | None
    exclusive_maximum: float | None
    multiple_of: float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None


@dataclass
class CompiledByteGrammar:
    tools: tuple[dict[str, Any], ...]
    names: tuple[str, ...]
    args_by_tool: dict[str, tuple[ArgConstraint, ...]]
    token_bytes: dict[int, bytes] | None = None


def compile_byte_grammar(tools: list[dict[str, Any]]) -> CompiledByteGrammar:
    compact = compact_tools({"tools": tools})
    args_by_tool: dict[str, tuple[ArgConstraint, ...]] = {}
    names = []
    for tool in compact:
        name = str(tool.get("name") or "")
        if not name:
            continue
        names.append(name)
        params = tool.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(str(x) for x in (params.get("required") or []))
        args: list[ArgConstraint] = []
        for key, schema in props.items():
            schema = schema or {}
            enum = schema.get("enum")
            args.append(
                ArgConstraint(
                    name=str(key),
                    typ=str(schema.get("type") or "string"),
                    required=str(key) in required,
                    enum=tuple(enum) if enum is not None else None,
                    const=schema.get("const", None) if "const" in schema else None,
                    minimum=_num(schema.get("minimum")),
                    maximum=_num(schema.get("maximum")),
                    exclusive_minimum=_num(schema.get("exclusiveMinimum")),
                    exclusive_maximum=_num(schema.get("exclusiveMaximum")),
                    multiple_of=_num(schema.get("multipleOf")),
                    min_length=int(schema["minLength"]) if "minLength" in schema else None,
                    max_length=int(schema["maxLength"]) if "maxLength" in schema else None,
                    pattern=str(schema["pattern"]) if schema.get("pattern") else None,
                )
            )
        args_by_tool[name] = tuple(args)
    return CompiledByteGrammar(tools=tuple(compact), names=tuple(names), args_by_tool=args_by_tool)


def attach_token_bytes(grammar: CompiledByteGrammar, tokenizer) -> CompiledByteGrammar:
    table = {}
    for tok in range(tokenizer.vocab_size):
        table[tok] = token_to_bytes(tokenizer, tok)
    grammar.token_bytes = table
    return grammar


def _num(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def _valid_utf8_prefix(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as exc:
        return exc.reason == "unexpected end of data"


def is_legal_byte_prefix(data: bytes, grammar: CompiledByteGrammar) -> bool:
    if not _valid_utf8_prefix(data):
        return False
    try:
        text = data.decode("utf-8")
        incomplete = False
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
        incomplete = True
    return _legal_text_prefix(text, grammar, incomplete=incomplete)


def is_accept_bytes(data: bytes, grammar: CompiledByteGrammar) -> bool:
    if not _valid_utf8_prefix(data):
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _legal_text_prefix(text, grammar, incomplete=False, must_accept=True)


def _legal_text_prefix(text: str, grammar: CompiledByteGrammar, *, incomplete: bool, must_accept: bool = False) -> bool:
    s = text
    if not s:
        return not must_accept
    if s[0] != "[":
        return False
    rest = s[1:]
    if rest == "":
        return not must_accept
    if rest[0] == "]":
        return rest == "]" if must_accept else rest.startswith("]") and rest.strip() in {"]", ""}
    if not rest.startswith("{") and not "{".startswith(rest):
        return False
    if not rest.startswith("{"):
        return not must_accept
    # object
    obj = rest
    if obj == "{":
        return not must_accept
    expect = '{"name":'
    if not obj.startswith("{") :
        return False
    # walk compact canonical object: {"name":TOOL,"arguments":{...}}
    if not _startswith_or_prefix(obj, '{"name":', must_accept):
        return _is_prefix(obj, '{"name":') and not must_accept
    if len(obj) < len('{"name":'):
        return not must_accept
    after_name_key = obj[len('{"name":'):]
    name, consumed, name_done = _parse_json_string_prefix(after_name_key)
    if consumed == 0 and not after_name_key:
        return not must_accept
    if name is None and not name_done:
        if not after_name_key.startswith('"'):
            return after_name_key == "" or after_name_key.startswith('"')
        live = [n for n in grammar.names if n.startswith(after_name_key[1:])]
        return bool(live) and not must_accept
    if name is None:
        return False
    if name not in grammar.names:
        if any(n.startswith(name) for n in grammar.names) and not name_done:
            return not must_accept
        return False
    if not name_done:
        return not must_accept
    after_name = after_name_key[consumed:]
    mid = ',"arguments":'
    if not _startswith_or_prefix(after_name, mid, must_accept):
        return _is_prefix(after_name, mid) and not must_accept
    if len(after_name) < len(mid):
        return not must_accept
    after_mid = after_name[len(mid):]
    return _legal_args_prefix(after_mid, grammar.args_by_tool[name], must_accept=must_accept, incomplete=incomplete)


def _is_prefix(have: str, want: str) -> bool:
    return want.startswith(have)


def _startswith_or_prefix(have: str, want: str, must_accept: bool) -> bool:
    if have.startswith(want):
        return True
    return (not must_accept) and want.startswith(have)


def _parse_json_string_prefix(s: str) -> tuple[str | None, int, bool]:
    if not s:
        return None, 0, False
    if s[0] != '"':
        return None, 0, False
    out = []
    i = 1
    while i < len(s):
        ch = s[i]
        if ch == '"':
            return "".join(out), i + 1, True
        if ch == "\\":
            if i + 1 >= len(s):
                return "".join(out), i + 1, False
            esc = s[i + 1]
            mapping = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
            if esc in mapping:
                out.append(mapping[esc])
                i += 2
                continue
            return None, 0, False
        out.append(ch)
        i += 1
    return "".join(out), len(s), False


def _legal_args_prefix(s: str, args: tuple[ArgConstraint, ...], *, must_accept: bool, incomplete: bool) -> bool:
    if not s:
        return not must_accept
    if s[0] != "{":
        return (not must_accept) and (s == "" or "{".startswith(s))
    body = s[1:]
    seen: set[str] = set()
    idx = 0
    required = {a.name for a in args if a.required}
    by_name = {a.name: a for a in args}
    if body == "":
        return not must_accept
    if body[0] == "}":
        rest = body[1:]
        if required - seen:
            return False
        return _close_call(rest, must_accept=must_accept)
    while idx < len(body):
        name, consumed, done = _parse_json_string_prefix(body[idx:])
        if not done:
            live = [a.name for a in args if a.name not in seen and a.name.startswith(name or "")]
            if body[idx:] == "" or body[idx:].startswith('"'):
                return (not must_accept) and (bool(live) or body[idx:] in {'"', ""})
            return False
        if name not in by_name or name in seen:
            return False
        seen.add(name)
        idx += consumed
        if idx >= len(body):
            return not must_accept
        if body[idx] != ":":
            return body[idx:] == "" or ":".startswith(body[idx])
        idx += 1
        ok, used, val_done = _legal_value_prefix(body[idx:], by_name[name], must_accept=False)
        if not ok:
            return False
        idx += used
        if not val_done:
            return not must_accept
        if idx >= len(body):
            return not must_accept
        if body[idx] == ",":
            idx += 1
            continue
        if body[idx] == "}":
            if required - seen:
                return False
            return _close_call(body[idx + 1 :], must_accept=must_accept)
        return False
    return not must_accept


def _close_call(rest: str, *, must_accept: bool) -> bool:
    want = "}]"
    if rest == want:
        return True
    if must_accept:
        return rest == want
    return want.startswith(rest)


def _legal_value_prefix(s: str, spec: ArgConstraint, *, must_accept: bool) -> tuple[bool, int, bool]:
    if spec.const is not None:
        lit = _json_str(spec.const)
        if s.startswith(lit):
            return True, len(lit), True
        return lit.startswith(s), len(s), False
    if spec.enum is not None:
        lits = [_json_str(v) for v in spec.enum]
        for lit in lits:
            if s.startswith(lit):
                return True, len(lit), True
        return any(lit.startswith(s) for lit in lits), len(s), False
    if spec.typ == "boolean":
        for lit in ("true", "false"):
            if s.startswith(lit):
                return True, len(lit), True
            if lit.startswith(s):
                return True, len(s), False
        return False, 0, False
    if spec.typ in {"integer", "number"}:
        return _legal_number_prefix(s, spec)
    if spec.typ == "string" or spec.typ == "":
        return _legal_string_value_prefix(s, spec)
    return False, 0, False


def _legal_number_prefix(s: str, spec: ArgConstraint) -> tuple[bool, int, bool]:
    if not s:
        return True, 0, False
    m = re.match(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?", s)
    if not m:
        return s in {"-", "."} or s[0] in "0123456789-", 0 if not s else min(1, len(s)), False
    token = m.group(0)
    complete = token == s or (len(s) > len(token) and s[len(token)] in ",}]")
    if not complete:
        return True, len(token), False
    try:
        val = float(token) if spec.typ == "number" or "." in token or "e" in token.lower() else int(token)
    except ValueError:
        return True, len(token), False
    if spec.typ == "integer":
        if isinstance(val, float) and not float(val).is_integer():
            return False, 0, False
        val_n = float(val)
    else:
        val_n = float(val)
    if spec.minimum is not None and val_n < spec.minimum:
        return False, 0, False
    if spec.maximum is not None and val_n > spec.maximum:
        return False, 0, False
    if spec.exclusive_minimum is not None and val_n <= spec.exclusive_minimum:
        return False, 0, False
    if spec.exclusive_maximum is not None and val_n >= spec.exclusive_maximum:
        return False, 0, False
    if spec.multiple_of is not None and spec.multiple_of != 0:
        q = val_n / spec.multiple_of
        if abs(q - round(q)) > 1e-9:
            return False, 0, False
    return True, len(token), True


def _legal_string_value_prefix(s: str, spec: ArgConstraint) -> tuple[bool, int, bool]:
    name, consumed, done = _parse_json_string_prefix(s)
    if not done:
        ok = s == "" or s.startswith('"')
        return ok, consumed, False
    if spec.min_length is not None and len(name or "") < spec.min_length:
        return False, 0, False
    if spec.max_length is not None and len(name or "") > spec.max_length:
        return False, 0, False
    if spec.pattern:
        if re.fullmatch(spec.pattern, name or "") is None:
            return False, 0, False
    return True, consumed, True


def allowed_token_ids(
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    *,
    eos_ok: bool | None = None,
) -> list[int]:
    allowed: list[int] = []
    vocab = tokenizer.vocab_size
    for tok in range(vocab):
        if tok == tokenizer.pad_id or tok == tokenizer.bos_id:
            continue
        if tok == tokenizer.eos_id:
            if is_accept_bytes(prefix_bytes, grammar):
                allowed.append(tok)
            continue
        extra = token_to_bytes(tokenizer, tok)
        if not extra:
            continue
        trial = prefix_bytes + extra
        if is_legal_byte_prefix(trial, grammar):
            allowed.append(tok)
    return allowed


def allowed_token_ids_fast(
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    ranked_ids: list[int],
    *,
    limit: int = 256,
) -> int | None:
    if is_accept_bytes(prefix_bytes, grammar):
        return tokenizer.eos_id
    for tok in ranked_ids[:limit]:
        tok = int(tok)
        if tok in {tokenizer.pad_id, tokenizer.bos_id}:
            continue
        if tok == tokenizer.eos_id:
            if is_accept_bytes(prefix_bytes, grammar):
                return tok
            continue
        extra = token_to_bytes(tokenizer, tok)
        if extra and is_legal_byte_prefix(prefix_bytes + extra, grammar):
            return tok
    for tok in ranked_ids[limit:]:
        tok = int(tok)
        extra = _token_bytes(tokenizer, grammar, tok)
        if extra and is_legal_byte_prefix(prefix_bytes + extra, grammar):
            return tok
    return None


def _token_bytes(tokenizer, grammar: CompiledByteGrammar, tok: int) -> bytes:
    if grammar.token_bytes is not None and tok in grammar.token_bytes:
        return grammar.token_bytes[tok]
    return token_to_bytes(tokenizer, tok)


def mask_illegal_logits(
    logits,
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    ranked_ids: list[int],
    *,
    limit: int = 512,
):
    import mlx.core as mx

    vocab = int(logits.shape[-1])
    allowed: list[int] = []
    if is_accept_bytes(prefix_bytes, grammar):
        allowed.append(tokenizer.eos_id)
    def consider(tok: int) -> None:
        tok = int(tok)
        if tok in {tokenizer.pad_id, tokenizer.bos_id}:
            return
        if tok == tokenizer.eos_id:
            if is_accept_bytes(prefix_bytes, grammar):
                allowed.append(tok)
            return
        extra = _token_bytes(tokenizer, grammar, tok)
        if extra and is_legal_byte_prefix(prefix_bytes + extra, grammar):
            allowed.append(tok)

    for tok in ranked_ids[:limit]:
        consider(tok)
        if len(allowed) >= 8:
            break
    if not allowed:
        for tok in ranked_ids[limit:]:
            consider(tok)
            if allowed:
                break
    if not allowed:
        return logits, None
    idx = mx.array(sorted(set(allowed)), dtype=mx.int32)
    one = mx.sum(mx.equal(mx.arange(vocab)[:, None], idx[None, :]), axis=1)
    if logits.ndim == 2:
        one = one[None, :]
    masked = mx.where(one > 0, logits, mx.array(NEG_INF, dtype=logits.dtype))
    return masked, allowed[0]
