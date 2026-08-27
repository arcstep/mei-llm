"""Canonical tool embedding index with fingerprint-based invalidation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from prompt_v2 import schema_fingerprint
from schema_render import compact_tools, dumps_canonical


def index_fingerprint(*, model_hash: str, head_hash: str, tokenizer_hash: str, serializer: str, schema_hash: str) -> str:
    blob = "|".join([model_hash, head_hash, tokenizer_hash, serializer, schema_hash])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ToolRecord:
    tool_id: str
    schema: dict[str, Any]
    embedding: Any
    schema_hash: str
    fingerprint: str


class ToolIndex:
    def __init__(self):
        self.records: dict[str, ToolRecord] = {}
        self.model_hash = ""
        self.head_hash = ""
        self.tokenizer_hash = ""
        self.serializer = "mei-tool-call-serializer-v2"

    def invalidate_if_stale(self, *, model_hash: str, head_hash: str, tokenizer_hash: str) -> bool:
        changed = (
            self.model_hash != model_hash
            or self.head_hash != head_hash
            or self.tokenizer_hash != tokenizer_hash
        )
        if changed:
            self.records.clear()
            self.model_hash = model_hash
            self.head_hash = head_hash
            self.tokenizer_hash = tokenizer_hash
        return changed

    def upsert(self, tool: dict[str, Any], embedding, *, fingerprint: str | None = None) -> ToolRecord:
        name = str(tool.get("name") or "")
        schema_h = schema_fingerprint({"tools": [tool]})
        fp = fingerprint or index_fingerprint(
            model_hash=self.model_hash,
            head_hash=self.head_hash,
            tokenizer_hash=self.tokenizer_hash,
            serializer=self.serializer,
            schema_hash=schema_h,
        )
        rec = ToolRecord(tool_id=name, schema=tool, embedding=embedding, schema_hash=schema_h, fingerprint=fp)
        self.records[name] = rec
        return rec

    def drop_schema(self, tool_id: str) -> None:
        self.records.pop(tool_id, None)

    def topk(self, query_emb, k: int = 5) -> list[ToolRecord]:
        rows = list(self.records.values())
        if not rows:
            return []
        if len(rows) <= k:
            return rows
        q = query_emb.astype(mx.float32)
        mat = mx.stack([r.embedding.astype(mx.float32) for r in rows], axis=0)
        scores = mat @ q
        order = mx.argsort(scores)[::-1]
        idx = [int(i) for i in order[:k].tolist()]
        return [rows[i] for i in idx]

    def select_tools(self, catalog: list[dict[str, Any]], query_emb, k: int = 5) -> list[dict[str, Any]]:
        if len(catalog) <= 5:
            return list(catalog)
        by_name = {str(t.get("name")): t for t in catalog}
        picked = self.topk(query_emb, k=k)
        out = []
        for rec in picked:
            if rec.tool_id in by_name:
                out.append(by_name[rec.tool_id])
        return out[:k]
