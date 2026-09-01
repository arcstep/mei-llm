"""UTF-8 byte grammar for ``[]`` or one schema-valid tool call.

This is the portable product grammar.  It constrains the exact compact wire
shape while allowing free scalar values covered by :mod:`schema_subset`.
Unsupported schemas fail during compilation instead of weakening decoding.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

try:
    from .canonical_json import MAX_SAFE_INTEGER, dumps_canonical
    from .schema_render import compact_tools
    from .schema_subset import validate_arguments, validate_tools, validate_value
except ImportError:
    from canonical_json import MAX_SAFE_INTEGER, dumps_canonical
    from schema_render import compact_tools
    from schema_subset import validate_arguments, validate_tools, validate_value

NEG_INF = -1e9
GRAMMAR_ID = "mei-utf8-byte-grammar-v2"
_HEX_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_TOKEN_BYTE_TABLES: dict[tuple[Any, ...], dict[int, bytes]] = {}
_GRAMMAR_CACHE: dict[tuple[Any, ...], "CompiledByteGrammar"] = {}


@dataclass(frozen=True)
class ArgConstraint:
    name: str
    schema: dict[str, Any]
    required: bool


@dataclass
class CompiledByteGrammar:
    tools: tuple[dict[str, Any], ...]
    names: tuple[str, ...]
    args_by_tool: dict[str, tuple[ArgConstraint, ...]]
    token_bytes: dict[int, bytes] | None = None


def token_to_bytes(tokenizer, tok_id: int, *, at_start: bool = False) -> bytes:
    if tok_id in {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        getattr(tokenizer, "unk_id", -1),
    }:
        return b""
    piece = tokenizer.sp.id_to_piece(int(tok_id))
    match = _HEX_PIECE.fullmatch(piece or "")
    if match:
        return bytes([int(match.group(1), 16)])
    if piece.startswith("▁"):
        # SentencePiece removes its dummy prefix only for the first generated
        # piece; later word-boundary markers decode to a real ASCII space.
        piece = ("" if at_start else " ") + piece[1:]
    try:
        return piece.encode("utf-8")
    except UnicodeEncodeError:
        return b""


def compile_byte_grammar(tools: list[dict[str, Any]]) -> CompiledByteGrammar:
    validate_tools(tools)
    compact = compact_tools({"tools": tools})
    names: list[str] = []
    args_by_tool: dict[str, tuple[ArgConstraint, ...]] = {}
    for tool in compact:
        name = str(tool["name"])
        names.append(name)
        params = tool.get("parameters") or {"type": "object", "properties": {}}
        required = set(params.get("required") or [])
        args_by_tool[name] = tuple(
            ArgConstraint(str(key), dict(schema), str(key) in required)
            for key, schema in (params.get("properties") or {}).items()
        )
    return CompiledByteGrammar(tuple(compact), tuple(names), args_by_tool)


def tokenizer_cache_key(tokenizer) -> tuple[Any, ...]:
    return (
        int(getattr(tokenizer, "vocab_size", 0) or 0),
        str(
            getattr(tokenizer, "model_sha256", "")
            or getattr(tokenizer, "model_path", "")
            or id(tokenizer)
        ),
    )


def token_bytes_table(tokenizer) -> dict[int, bytes]:
    key = tokenizer_cache_key(tokenizer)
    if key not in _TOKEN_BYTE_TABLES:
        _TOKEN_BYTE_TABLES[key] = {
            token: token_to_bytes(tokenizer, token, at_start=False)
            for token in range(int(tokenizer.vocab_size))
        }
    return _TOKEN_BYTE_TABLES[key]


def compile_byte_grammar_cached(tools: list[dict[str, Any]], tokenizer=None) -> CompiledByteGrammar:
    compact = compact_tools({"tools": tools})
    key = (dumps_canonical(compact), tokenizer_cache_key(tokenizer) if tokenizer is not None else None)
    hit = _GRAMMAR_CACHE.get(key)
    if hit is not None:
        return hit
    grammar = compile_byte_grammar(tools)
    if tokenizer is not None:
        grammar.token_bytes = token_bytes_table(tokenizer)
    _GRAMMAR_CACHE[key] = grammar
    return grammar


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
        incomplete_utf8 = False
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
        incomplete_utf8 = True
    return _legal_text_prefix(text, grammar, incomplete_utf8=incomplete_utf8)


def is_accept_bytes(data: bytes, grammar: CompiledByteGrammar) -> bool:
    if not _valid_utf8_prefix(data):
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _legal_text_prefix(text, grammar, incomplete_utf8=False, must_accept=True)


def parse_call_text(text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse and schema-check a complete candidate without importing an SDK."""
    try:
        grammar = compile_byte_grammar_cached(tools)
    except ValueError as exc:
        return {"ok": False, "refuse": True, "function_calls": [], "error": str(exc)}
    raw = text or ""
    if not raw or not _has_no_json_whitespace(raw):
        return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
    try:
        obj = json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            parse_int=_strict_int,
            parse_float=_strict_float,
        )
    except (json.JSONDecodeError, ValueError, UnicodeError):
        return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
    if obj == []:
        if not is_accept_bytes(raw.encode("utf-8"), grammar):
            return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
        return {"ok": True, "refuse": True, "function_calls": [], "error": None}
    if (
        not isinstance(obj, list)
        or len(obj) != 1
        or not isinstance(obj[0], dict)
        or list(obj[0]) != ["name", "arguments"]
        or not isinstance(obj[0].get("name"), str)
        or not isinstance(obj[0].get("arguments"), dict)
        or obj[0]["name"] not in grammar.names
    ):
        return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
    call = obj[0]
    tool = next(t for t in tools if str(t.get("name")) == call["name"])
    issues = validate_arguments(call["arguments"], tool.get("parameters") or {})
    if issues:
        return {
            "ok": False,
            "refuse": True,
            "function_calls": [],
            "error": "schema_validation",
            "issues": [issue.as_dict() for issue in issues],
        }
    # The structural parse above lets the caller report JSON-Schema failures
    # at the schema gate.  A schema-valid string must still be accepted by the
    # exact byte grammar; otherwise a decoder/validator drift fails closed.
    if not is_accept_bytes(raw.encode("utf-8"), grammar):
        return {"ok": False, "refuse": True, "function_calls": [], "error": "grammar"}
    return {"ok": True, "refuse": False, "function_calls": [call], "error": None}


def _strict_object_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _strict_int(value: str) -> int:
    parsed = int(value)
    if abs(parsed) > MAX_SAFE_INTEGER:
        raise ValueError("integer exceeds wire-v2 safe domain")
    return parsed


def _strict_float(value: str) -> float:
    parsed = float(value)
    if (
        not math.isfinite(parsed)
        or (parsed.is_integer() and abs(parsed) > MAX_SAFE_INTEGER)
    ):
        raise ValueError("number exceeds wire-v2 binary64 domain")
    return parsed


def _has_no_json_whitespace(text: str) -> bool:
    """The product wire is compact JSON; whitespace inside strings is data."""

    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in " \t\r\n":
            return False
    return True


def _legal_text_prefix(
    text: str,
    grammar: CompiledByteGrammar,
    *,
    incomplete_utf8: bool,
    must_accept: bool = False,
) -> bool:
    if not text:
        return not must_accept
    if not "[".startswith(text) and not text.startswith("["):
        return False
    if text == "[":
        return not must_accept
    rest = text[1:]
    if rest.startswith("]"):
        return rest == "]"
    object_open = "{"
    if not rest.startswith(object_open):
        return (not must_accept) and object_open.startswith(rest)
    obj = rest[1:]
    name_key = '"name":'
    if not obj.startswith(name_key):
        return (not must_accept) and name_key.startswith(obj)
    after_key = obj[len(name_key) :]
    name, used, done, valid = _parse_json_string_prefix(after_key)
    if not valid:
        return False
    if not done:
        return (not must_accept) and any(candidate.startswith(name or "") for candidate in grammar.names)
    if name not in grammar.names:
        return False
    after_name = after_key[used:]
    middle = ',"arguments":'
    if not after_name.startswith(middle):
        return (not must_accept) and middle.startswith(after_name)
    after_middle = after_name[len(middle) :]
    return _legal_args_prefix(
        after_middle,
        grammar.args_by_tool[name],
        must_accept=must_accept,
        incomplete_utf8=incomplete_utf8,
    )


def _parse_json_string_prefix(text: str) -> tuple[str | None, int, bool, bool]:
    if not text:
        return "", 0, False, True
    if text[0] != '"':
        return None, 0, False, False
    index = 1
    while index < len(text):
        char = text[index]
        if ord(char) < 0x20:
            return None, 0, False, False
        if char == '"':
            literal = text[: index + 1]
            try:
                value = json.loads(literal)
                if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
                    return None, 0, False, False
                return value, index + 1, True, True
            except json.JSONDecodeError:
                return None, 0, False, False
        if char != "\\":
            index += 1
            continue
        index += 1
        if index >= len(text):
            break
        escaped = text[index]
        if escaped in '"\\/bfnrt':
            index += 1
            continue
        if escaped != "u":
            return None, 0, False, False
        digits = text[index + 1 : index + 5]
        if any(ch not in "0123456789abcdefABCDEF" for ch in digits):
            return None, 0, False, False
        if len(digits) < 4:
            break
        index += 5
    # Decode the completed portion only; it is used for name/property prefix filtering.
    raw = text[1:index]
    try:
        value = json.loads('"' + raw.rstrip("\\") + '"')
    except json.JSONDecodeError:
        value = raw.split("\\", 1)[0]
    return value, len(text), False, True


def _legal_args_prefix(
    text: str,
    constraints: tuple[ArgConstraint, ...],
    *,
    must_accept: bool,
    incomplete_utf8: bool,
) -> bool:
    if not text:
        return not must_accept
    if text[0] != "{":
        return (not must_accept) and "{".startswith(text)
    body = text[1:]
    by_name = {constraint.name: constraint for constraint in constraints}
    required = {constraint.name for constraint in constraints if constraint.required}
    seen: set[str] = set()
    index = 0
    if body == "":
        return not must_accept
    if body.startswith("}"):
        return not required and _close_call(body[1:], must_accept=must_accept)
    while index < len(body):
        name, used, done, valid = _parse_json_string_prefix(body[index:])
        if not valid:
            return False
        if not done:
            live = [key for key in by_name if key not in seen and key.startswith(name or "")]
            return (not must_accept) and bool(live)
        if name not in by_name or name in seen:
            return False
        seen.add(name)
        constraint = by_name[name]
        index += used
        if index >= len(body):
            return not must_accept
        if body[index] != ":":
            return (not must_accept) and ":".startswith(body[index:])
        index += 1
        ok, consumed, value_done, value = _parse_value_prefix(body[index:], constraint.schema)
        if not ok:
            return False
        index += consumed
        if not value_done:
            return not must_accept
        if index >= len(body):
            # A number may still grow; constraints become final at a delimiter.
            return not must_accept
        if validate_value(value, constraint.schema, path=f"$.arguments.{name}"):
            return False
        if body[index] == ",":
            index += 1
            if index >= len(body):
                return not must_accept
            continue
        if body[index] == "}":
            if required - seen:
                return False
            return _close_call(body[index + 1 :], must_accept=must_accept)
        return False
    return not must_accept


def _close_call(rest: str, *, must_accept: bool) -> bool:
    wanted = "}]"
    if must_accept:
        return rest == wanted
    return wanted.startswith(rest)


def _parse_value_prefix(text: str, schema: dict[str, Any]) -> tuple[bool, int, bool, Any]:
    if "const" in schema:
        return _literal_prefix(text, schema["const"])
    if "enum" in schema:
        incomplete = False
        complete: list[tuple[bool, int, bool, Any]] = []
        for value in schema["enum"]:
            literal = dumps_canonical(value)
            if text.startswith(literal):
                complete.append((True, len(literal), True, value))
            incomplete = incomplete or literal.startswith(text)
        if complete:
            return max(complete, key=lambda row: row[1])
        return incomplete, len(text) if incomplete else 0, False, None
    raw_type = schema.get("type")
    types = tuple(raw_type) if isinstance(raw_type, list) else (str(raw_type),)
    candidates: list[tuple[bool, int, bool, Any]] = []
    for type_name in types:
        if type_name == "null":
            candidates.append(_literal_prefix(text, None))
        elif type_name == "boolean":
            candidates.extend((_literal_prefix(text, True), _literal_prefix(text, False)))
        elif type_name in {"integer", "number"}:
            candidates.append(_number_prefix(text, integer=type_name == "integer"))
        elif type_name == "string":
            value, used, done, valid = _parse_json_string_prefix(text)
            candidates.append((valid, used, done, value))
        elif type_name == "array":
            candidates.append(_array_prefix(text, schema))
    complete = [row for row in candidates if row[0] and row[2]]
    if complete:
        return max(complete, key=lambda row: row[1])
    partial = [row for row in candidates if row[0]]
    return max(partial, key=lambda row: row[1]) if partial else (False, 0, False, None)


def _literal_prefix(text: str, value: Any) -> tuple[bool, int, bool, Any]:
    literal = dumps_canonical(value)
    if text.startswith(literal):
        return True, len(literal), True, value
    if literal.startswith(text):
        return True, len(text), False, None
    return False, 0, False, None


def _number_prefix(text: str, *, integer: bool) -> tuple[bool, int, bool, Any]:
    if not text:
        return True, 0, False, None
    index = 0
    if text[index] == "-":
        index += 1
        if index == len(text):
            return True, index, False, None
    if index >= len(text) or not text[index].isdigit():
        return False, 0, False, None
    if text[index] == "0":
        index += 1
    else:
        while index < len(text) and text[index].isdigit():
            index += 1
    if integer:
        token = text[:index]
        try:
            return True, index, True, int(token)
        except ValueError:
            return False, 0, False, None
    if index < len(text) and text[index] == ".":
        index += 1
        if index == len(text):
            return True, index, False, None
        if not text[index].isdigit():
            return False, 0, False, None
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] in "eE":
        index += 1
        if index == len(text):
            return True, index, False, None
        if text[index] in "+-":
            index += 1
            if index == len(text):
                return True, index, False, None
        if not text[index].isdigit():
            return False, 0, False, None
        while index < len(text) and text[index].isdigit():
            index += 1
    token = text[:index]
    try:
        value: Any = float(token)
    except ValueError:
        return False, 0, False, None
    return True, index, True, value


def _array_prefix(text: str, schema: dict[str, Any]) -> tuple[bool, int, bool, Any]:
    if not text:
        return True, 0, False, None
    if text[0] != "[":
        return ("[".startswith(text), len(text) if "[".startswith(text) else 0, False, None)
    index = 1
    values: list[Any] = []
    if index >= len(text):
        return True, index, False, None
    if text[index] == "]":
        value = []
        return (not validate_value(value, schema), index + 1, True, value)
    item_schema = schema["items"]
    while index < len(text):
        ok, used, done, value = _parse_value_prefix(text[index:], item_schema)
        if not ok:
            return False, 0, False, None
        index += used
        if not done:
            return True, index, False, None
        if index >= len(text):
            return True, index, False, None
        if validate_value(value, item_schema, path=f"$[{len(values)}]"):
            return False, 0, False, None
        values.append(value)
        if text[index] == ",":
            index += 1
            if index >= len(text):
                return True, index, False, None
            continue
        if text[index] == "]":
            index += 1
            return (not validate_value(values, schema), index, True, values)
        return False, 0, False, None
    return True, index, False, None


def _token_bytes(
    tokenizer, grammar: CompiledByteGrammar, token: int, *, at_start: bool
) -> bytes:
    if not at_start and grammar.token_bytes is not None and token in grammar.token_bytes:
        return grammar.token_bytes[token]
    return token_to_bytes(tokenizer, token, at_start=at_start)


def _token_is_legal(
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix: bytes,
    token: int,
    *,
    generated_token_count: int,
) -> bool:
    token = int(token)
    if token in {tokenizer.pad_id, tokenizer.bos_id, getattr(tokenizer, "unk_id", -1)}:
        return False
    if token == tokenizer.eos_id:
        return is_accept_bytes(prefix, grammar)
    at_start = generated_token_count == 0
    extra = _token_bytes(tokenizer, grammar, token, at_start=at_start)
    if at_start and not extra and tokenizer.sp.id_to_piece(token) == "▁":
        # The tokenizer's initial dummy-prefix token consumes no UTF-8 bytes.
        # It is legal exactly once because the next call has count=1.
        return True
    return bool(extra) and is_legal_byte_prefix(prefix + extra, grammar)


def allowed_token_ids(
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    *,
    generated_token_count: int = 0,
) -> list[int]:
    return [
        token
        for token in range(int(tokenizer.vocab_size))
        if _token_is_legal(
            tokenizer,
            grammar,
            prefix_bytes,
            token,
            generated_token_count=generated_token_count,
        )
    ]


def select_legal_token(
    logits,
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    *,
    generated_token_count: int = 0,
) -> int | None:
    import mlx.core as mx

    if is_accept_bytes(prefix_bytes, grammar):
        return int(tokenizer.eos_id)
    row = logits[0] if logits.ndim == 2 else logits
    ranked = mx.argsort(row)[::-1]
    mx.eval(ranked)
    for token in ranked.tolist():
        if _token_is_legal(
            tokenizer,
            grammar,
            prefix_bytes,
            int(token),
            generated_token_count=generated_token_count,
        ):
            return int(token)
    return None


def mask_illegal_logits(
    logits,
    tokenizer,
    grammar: CompiledByteGrammar,
    prefix_bytes: bytes,
    *,
    generated_token_count: int = 0,
):
    import mlx.core as mx

    row = logits[0] if logits.ndim == 2 else logits
    allowed = allowed_token_ids(
        tokenizer,
        grammar,
        prefix_bytes,
        generated_token_count=generated_token_count,
    )
    if not allowed:
        return mx.full_like(row, NEG_INF), []
    indexes = mx.array(allowed, dtype=mx.int32)
    mask = mx.sum(mx.equal(mx.arange(int(row.shape[-1]))[:, None], indexes[None, :]), axis=1) > 0
    return mx.where(mask, row, mx.array(NEG_INF, dtype=row.dtype)), allowed
