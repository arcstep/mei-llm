"""Portable normalized-f16 tool index with complete provenance fingerprint."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from .canonical_json import dumps_canonical
    from .schema_render import compact_tools
    from .schema_subset import validate_tools
except ImportError:
    from canonical_json import dumps_canonical
    from schema_render import compact_tools
    from schema_subset import validate_tools

INDEX_FORMAT = "mei-tool-index-v2"
DEFAULT_SERIALIZER = "mei-tool-call-serializer-v2"
_INDEX_KEYS = {
    "format",
    "fingerprint",
    "catalog_sha256",
    "model_sha256",
    "head_sha256",
    "tokenizer_sha256",
    "serializer_id",
    "dtype",
    "normalized",
    "dimension",
    "records",
}
_RECORD_KEYS = {
    "tool_id",
    "schema",
    "schema_sha256",
    "embedding_f16_base64",
}


def _sha(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _catalog_view(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable catalog identity including deterministic safety contracts."""

    return [
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            "required_permissions": tool.get("required_permissions"),
            "required_state": tool.get("required_state"),
            "x-mei-permissions": tool.get("x-mei-permissions"),
            "x-mei-state": tool.get("x-mei-state"),
        }
        for tool in sorted(tools, key=lambda row: str(row.get("name") or ""))
    ]


def catalog_fingerprint(tools: list[dict[str, Any]]) -> str:
    validate_tools(tools)
    return _sha(dumps_canonical(_catalog_view(tools)))


def index_fingerprint(
    *,
    model_hash: str,
    head_hash: str,
    tokenizer_hash: str,
    serializer: str = DEFAULT_SERIALIZER,
    schema_hash: str,
    catalog_hash: str,
) -> str:
    payload = {
        "catalog_sha256": catalog_hash,
        "head_sha256": head_hash,
        "model_sha256": model_hash,
        "schema_sha256": schema_hash,
        "serializer_id": serializer,
        "tokenizer_sha256": tokenizer_hash,
    }
    return _sha(dumps_canonical(payload))


def render_tool_text(tool: dict[str, Any]) -> str:
    return dumps_canonical(
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
    )


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [float(value) for value in values]


def normalize(values: Any) -> list[float]:
    row = _as_float_list(values)
    norm = math.sqrt(sum(value * value for value in row))
    if not row or not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must be a finite non-zero vector")
    return [value / norm for value in row]


def _pack_f16(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<e", float(value)) for value in values)


def _unpack_f16(blob: bytes) -> list[float]:
    if len(blob) % 2:
        raise ValueError("invalid f16 embedding byte length")
    return [struct.unpack_from("<e", blob, offset)[0] for offset in range(0, len(blob), 2)]


@dataclass(frozen=True)
class ToolRecord:
    tool_id: str
    schema: dict[str, Any]
    embedding: tuple[float, ...]
    schema_hash: str


@dataclass(frozen=True)
class ScoredToolRecord:
    record: ToolRecord
    score: float

    @property
    def tool_id(self) -> str:
        return self.record.tool_id

    @property
    def schema(self) -> dict[str, Any]:
        return self.record.schema


class ToolIndex:
    def __init__(
        self,
        *,
        model_hash: str = "",
        head_hash: str = "",
        tokenizer_hash: str = "",
        serializer: str = DEFAULT_SERIALIZER,
    ):
        self.model_hash = model_hash
        self.head_hash = head_hash
        self.tokenizer_hash = tokenizer_hash
        self.serializer = serializer
        self.records: dict[str, ToolRecord] = {}
        self.catalog_hash = ""
        self.fingerprint = ""

    def invalidate_if_stale(
        self, *, model_hash: str, head_hash: str, tokenizer_hash: str, serializer: str | None = None
    ) -> bool:
        next_serializer = serializer or self.serializer
        changed = (
            self.model_hash != model_hash
            or self.head_hash != head_hash
            or self.tokenizer_hash != tokenizer_hash
            or self.serializer != next_serializer
        )
        if changed:
            self.records.clear()
            self.catalog_hash = ""
            self.fingerprint = ""
            self.model_hash = model_hash
            self.head_hash = head_hash
            self.tokenizer_hash = tokenizer_hash
            self.serializer = next_serializer
        return changed

    def build(self, tools: list[dict[str, Any]], embed: Callable[[str], Any]) -> None:
        validate_tools(tools)
        catalog_hash = catalog_fingerprint(tools)
        records: dict[str, ToolRecord] = {}
        dimension: int | None = None
        for tool in sorted(tools, key=lambda row: str(row.get("name") or "")):
            tool_id = str(tool["name"])
            # Quantize at build time so in-memory ranking and a reloaded
            # Python/Rust/WASM index observe the exact same f16 artifact.
            vector = _unpack_f16(_pack_f16(normalize(embed(render_tool_text(tool)))))
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("tool embeddings must have one dimension")
            schema_hash = _sha(dumps_canonical(compact_tools({"tools": [tool]})))
            records[tool_id] = ToolRecord(tool_id, copy.deepcopy(tool), tuple(vector), schema_hash)
        combined_schema = _sha(
            dumps_canonical({name: record.schema_hash for name, record in sorted(records.items())})
        )
        self.records = records
        self.catalog_hash = catalog_hash
        self.fingerprint = index_fingerprint(
            model_hash=self.model_hash,
            head_hash=self.head_hash,
            tokenizer_hash=self.tokenizer_hash,
            serializer=self.serializer,
            schema_hash=combined_schema,
            catalog_hash=catalog_hash,
        )

    def ranked(self, query_embedding: Any) -> list[ScoredToolRecord]:
        query = normalize(query_embedding)
        rows: list[tuple[float, str, ToolRecord]] = []
        for tool_id, record in self.records.items():
            if len(query) != len(record.embedding):
                raise ValueError("query embedding dimension mismatch")
            score = sum(left * right for left, right in zip(query, record.embedding))
            rows.append((score, tool_id, record))
        # Stable tool ID is the mandatory tie breaker across Python/Rust/WASM.
        rows.sort(key=lambda row: (-row[0], row[1]))
        return [ScoredToolRecord(record=record, score=score) for score, _, record in rows]

    def topk_scored(self, query_embedding: Any, *, k: int = 5) -> list[ScoredToolRecord]:
        if k < 1:
            return []
        rows = self.ranked(query_embedding)
        return rows[: min(int(k), len(rows))]

    def topk(self, query_embedding: Any, *, k: int = 5) -> list[ToolRecord]:
        return [row.record for row in self.topk_scored(query_embedding, k=k)]

    def select_tools(
        self, catalog: list[dict[str, Any]], query_embedding: Any, *, k: int = 5
    ) -> list[dict[str, Any]]:
        by_name = {str(tool.get("name")): tool for tool in catalog}
        return [
            by_name[record.tool_id]
            for record in self.topk(query_embedding, k=k)
            if record.tool_id in by_name
        ]

    def as_dict(self) -> dict[str, Any]:
        dimension = len(next(iter(self.records.values())).embedding) if self.records else 0
        return {
            "format": INDEX_FORMAT,
            "fingerprint": self.fingerprint,
            "catalog_sha256": self.catalog_hash,
            "model_sha256": self.model_hash,
            "head_sha256": self.head_hash,
            "tokenizer_sha256": self.tokenizer_hash,
            "serializer_id": self.serializer,
            "dtype": "float16",
            "normalized": True,
            "dimension": dimension,
            "records": [
                {
                    "tool_id": record.tool_id,
                    "schema": record.schema,
                    "schema_sha256": record.schema_hash,
                    "embedding_f16_base64": base64.b64encode(_pack_f16(record.embedding)).decode("ascii"),
                }
                for _, record in sorted(self.records.items())
            ],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(dumps_canonical(self.as_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
        expected_serializer: str = DEFAULT_SERIALIZER,
    ) -> "ToolIndex":
        def strict_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in values:
                if key in out:
                    raise ValueError(f"duplicate tool index key: {key}")
                out[key] = value
            return out

        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        if not isinstance(raw, dict) or set(raw) != _INDEX_KEYS:
            raise ValueError("tool index fields do not match v2")
        if raw.get("format") != INDEX_FORMAT or raw.get("dtype") != "float16" or raw.get("normalized") is not True:
            raise ValueError("unsupported tool index format")
        for field in (
            "fingerprint",
            "catalog_sha256",
            "model_sha256",
            "head_sha256",
            "tokenizer_sha256",
        ):
            if not _is_sha256(raw.get(field)):
                raise ValueError(f"tool index {field} must be lowercase sha256")
        if raw.get("serializer_id") != expected_serializer:
            raise ValueError("tool index serializer_id is not the canonical runtime serializer")
        if expected_fingerprint and raw.get("fingerprint") != expected_fingerprint:
            raise ValueError("tool index fingerprint mismatch")
        index = cls(
            model_hash=str(raw.get("model_sha256") or ""),
            head_hash=str(raw.get("head_sha256") or ""),
            tokenizer_hash=str(raw.get("tokenizer_sha256") or ""),
            serializer=str(raw.get("serializer_id") or DEFAULT_SERIALIZER),
        )
        dimension_raw = raw.get("dimension")
        if (
            not isinstance(dimension_raw, int)
            or isinstance(dimension_raw, bool)
            or dimension_raw < 0
        ):
            raise ValueError("tool index dimension must be non-negative")
        dimension = dimension_raw
        rows = raw.get("records")
        if not isinstance(rows, list):
            raise ValueError("tool index records must be an array")
        if rows and dimension == 0:
            raise ValueError("non-empty tool index must have a positive dimension")
        for row in rows:
            if not isinstance(row, dict) or set(row) != _RECORD_KEYS:
                raise ValueError("tool index record fields do not match v2")
            if not isinstance(row.get("tool_id"), str) or not row["tool_id"]:
                raise ValueError("tool index tool_id must be a non-empty string")
            tool_id = row["tool_id"]
            if tool_id in index.records:
                raise ValueError("duplicate tool ID in index")
            encoded_embedding = row.get("embedding_f16_base64")
            if not isinstance(encoded_embedding, str):
                raise ValueError("tool index embedding must be base64 text")
            try:
                blob = base64.b64decode(encoded_embedding, validate=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("tool index embedding is invalid base64") from exc
            embedding = _unpack_f16(blob)
            if len(embedding) != dimension:
                raise ValueError("tool index embedding dimension mismatch")
            norm = math.sqrt(sum(value * value for value in embedding))
            if not math.isfinite(norm) or not 0.99 <= norm <= 1.01:
                raise ValueError("tool index embedding is not normalized")
            if not isinstance(row.get("schema"), dict):
                raise ValueError("tool index schema must be an object")
            schema = dict(row["schema"])
            if not _is_sha256(row.get("schema_sha256")):
                raise ValueError("tool index schema_sha256 must be lowercase sha256")
            validate_tools([schema])
            if str(schema.get("name") or "") != tool_id:
                raise ValueError("tool index ID/schema mismatch")
            expected_schema = _sha(dumps_canonical(compact_tools({"tools": [schema]})))
            if row.get("schema_sha256") != expected_schema:
                raise ValueError("tool index schema hash mismatch")
            index.records[tool_id] = ToolRecord(tool_id, schema, tuple(embedding), expected_schema)
        index.catalog_hash = str(raw.get("catalog_sha256") or "")
        reconstructed_catalog = [record.schema for _, record in sorted(index.records.items())]
        if catalog_fingerprint(reconstructed_catalog) != index.catalog_hash:
            raise ValueError("tool index catalog hash mismatch")
        index.fingerprint = str(raw.get("fingerprint") or "")
        combined_schema = _sha(
            dumps_canonical(
                {name: record.schema_hash for name, record in sorted(index.records.items())}
            )
        )
        expected = index_fingerprint(
            model_hash=index.model_hash,
            head_hash=index.head_hash,
            tokenizer_hash=index.tokenizer_hash,
            serializer=index.serializer,
            schema_hash=combined_schema,
            catalog_hash=index.catalog_hash,
        )
        if expected != index.fingerprint:
            raise ValueError("tool index provenance fingerprint mismatch")
        return index
