#!/usr/bin/env python3
"""Structured-source recordizers for natural-corpus sourcing v2.

Record-mode admission: structure-sensitive formats (jsonl/json/xml/yaml/
sqllogictest/parquet code files) are split into records whose canonical bytes
drive dedup and whose text is tokenized WITHOUT whitespace folding, so JSON
key order, XML serialization, and YAML indentation keep their semantics.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

TEXT_KEYS = ("text", "content", "document", "body")


class StructuredError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _record(
    key: str | None, canonical_bytes: bytes, text: str, *, valid: bool = True
) -> dict[str, Any]:
    return {
        "valid": valid,
        "key": key,
        "canonical_bytes": canonical_bytes,
        "text": text,
    }


def _key_for(value: Any, key_field: str | None, index: int) -> str:
    if key_field and isinstance(value, dict):
        found = value.get(key_field)
        if found is not None:
            return str(found)
    return str(index)


def _iter_json_lines(path: Path, key_field: str | None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield _record(None, f"invalid-line-{line_number}".encode(), "", valid=False)
                continue
            key = _key_for(value, key_field, line_number)
            payload = canonical(value)
            yield _record(key, payload, payload.decode("utf-8"))


def _iter_json_documents(path: Path, key_field: str | None) -> Iterator[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        yield _record(None, b"invalid-json-file", "", valid=False)
        return
    rows = value if isinstance(value, list) else [value]
    for index, row in enumerate(rows, 1):
        key = _key_for(row, key_field, index)
        payload = canonical(row)
        yield _record(key, payload, payload.decode("utf-8"))


def _iter_xml_records(path: Path, record_elements: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    index = 0
    try:
        for _event, element in ET.iterparse(path, events=("end",)):
            if element.tag in record_elements:
                index += 1
                payload = ET.tostring(element, encoding="unicode")
                yield _record(str(index), payload.encode("utf-8"), payload)
                element.clear()
    except ET.ParseError:
        yield _record(None, b"invalid-xml", "", valid=False)


def _iter_yaml_documents(path: Path, max_record_bytes: int) -> Iterator[dict[str, Any]]:
    try:
        import yaml
    except ImportError as error:
        raise StructuredError("yaml recordizer requires pyyaml") from error
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        chunks = handle.read().split("\n---\n")
    for index, chunk in enumerate(chunks, 1):
        text = chunk.strip("\n")
        if not text.strip():
            continue
        if len(text.encode("utf-8")) > max_record_bytes:
            yield _record(str(index), text.encode("utf-8"), text, valid=False)
            continue
        try:
            yaml.safe_load(text)
        except yaml.YAMLError:
            yield _record(str(index), text.encode("utf-8"), text, valid=False)
            continue
        yield _record(str(index), text.encode("utf-8"), text)


def _iter_sqllogictest_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        chunks = handle.read().split("\n\n")
    for index, chunk in enumerate(chunks, 1):
        text = chunk.strip()
        if not text:
            continue
        yield _record(str(index), text.encode("utf-8"), text)


def _iter_parquet_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise StructuredError("parquet recordizer requires pyarrow") from error
    source = parquet.ParquetFile(path)
    names = source.schema_arrow.names
    column = next((key for key in ("content", "text", "code") if key in names), None)
    if not column:
        raise StructuredError(f"{path}: no supported code text column")
    key_column = next((key for key in ("path", "filename", "id") if key in names), None)
    for batch in source.iter_batches(batch_size=256):
        keys = batch.column(key_column).to_pylist() if key_column else []
        for row_index, value in enumerate(batch.column(column).to_pylist()):
            if not isinstance(value, str) or not value.strip():
                continue
            key = str(keys[row_index]) if keys else str(row_index)
            yield _record(key, value.encode("utf-8"), value)


def iter_records(
    path: Path,
    *,
    record_elements: tuple[str, ...] = (),
    max_record_bytes: int = 1 << 20,
    key_field: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one dict per record: {valid, key, canonical_bytes, text}."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_json_lines(path, key_field)
    elif suffix == ".json":
        yield from _iter_json_documents(path, key_field)
    elif suffix == ".xml":
        yield from _iter_xml_records(path, record_elements)
    elif suffix in {".yaml", ".yml"}:
        yield from _iter_yaml_documents(path, max_record_bytes)
    elif suffix in {".test", ".slt"}:
        yield from _iter_sqllogictest_records(path)
    elif suffix == ".parquet":
        yield from _iter_parquet_records(path)
    else:
        raise StructuredError(f"no recordizer for format: {path}")
