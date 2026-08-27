"""Shared RAG retrieval protocol for RuntimeV2, evaluators, and external adapters.

Canonical pipeline:
  render_tool(tool) → encode_catalog → build/cache index → encode raw query → topk(k=5) → selected schemas

58M ContrastiveHead and external embedding models share the same canonical tool text
and raw-query encoding contract. Tokenizer/max-length/pooling of external models stay native
and must be recorded by the adapter.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Sequence

from prompt_v2 import SERIALIZER_ID, render_tools_block, schema_fingerprint

K_DEFAULT = 5
QUERY_MAX_TOKENS = 48
QUERY_MIN_TOKENS = 8
EMBED_DIM_58M = 128
PROTOCOL_ID = "mei-retrieval-rag-protocol-v2"


def render_tool_text(tool: dict[str, Any]) -> str:
    """Canonical compact schema text used as the catalog document."""
    return render_tools_block([tool])


def catalog_schema_fingerprint(tools: Sequence[dict[str, Any]]) -> str:
    return schema_fingerprint({"tools": list(tools)})


def encode_text_ids(tokenizer, text: str, *, max_len: int = QUERY_MAX_TOKENS, min_len: int = QUERY_MIN_TOKENS) -> list[int]:
    """58M ContrastiveHead tokenization: BOS, truncate, pad to min_len."""
    ids = list(tokenizer.encode(text, add_bos=True)[:max_len])
    if len(ids) < min_len:
        pad_id = int(getattr(tokenizer, "pad_id", 0) or 0)
        ids = ids + [pad_id] * (min_len - len(ids))
    return ids


def l2_normalize_list(vec: Sequence[float]) -> list[float]:
    acc = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / acc for x in vec]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return sum(float(a[i]) * float(b[i]) for i in range(n))


def rank_by_dot(
    query_vec: Sequence[float],
    items: Sequence[tuple[str, Sequence[float]]],
    *,
    k: int = K_DEFAULT,
) -> list[tuple[str, float]]:
    """Exact quality ranking: L2-dot. Ties keep original catalog order (no name tie-break)."""
    q = l2_normalize_list(query_vec)
    scored: list[tuple[int, float, str]] = []
    for i, (name, vec) in enumerate(items):
        scored.append((i, dot(q, l2_normalize_list(vec)), name))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return [(name, score) for _i, score, name in scored[:k]]


def select_topk_tools(
    catalog: Sequence[dict[str, Any]],
    ranked_names: Sequence[str],
    *,
    k: int = K_DEFAULT,
) -> list[dict[str, Any]]:
    if len(catalog) <= k:
        return list(catalog)
    by_name = {str(t.get("name") or ""): t for t in catalog}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in ranked_names:
        if name in seen or name not in by_name:
            continue
        out.append(by_name[name])
        seen.add(name)
        if len(out) >= k:
            break
    return out


def protocol_fingerprint(
    *,
    model_id: str,
    head_id: str,
    tokenizer_hash: str,
    schema_hash: str,
    pooling: str = "contrastive_head",
    max_len: int = QUERY_MAX_TOKENS,
) -> str:
    blob = "|".join(
        [
            PROTOCOL_ID,
            SERIALIZER_ID,
            model_id,
            head_id,
            tokenizer_hash,
            schema_hash,
            pooling,
            str(max_len),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CatalogIndex:
    """Backend-agnostic catalog index. Encoder is injected."""

    def __init__(self, encode_fn: Callable[[str], Sequence[float]], *, k: int = K_DEFAULT):
        self.encode_fn = encode_fn
        self.k = k
        self.tools: list[dict[str, Any]] = []
        self.names: list[str] = []
        self.vectors: list[list[float]] = []
        self.schema_fp = ""
        self.docs: list[str] = []

    def build(self, catalog: Sequence[dict[str, Any]]) -> None:
        fp = catalog_schema_fingerprint(catalog)
        if fp == self.schema_fp and self.vectors and len(self.tools) == len(catalog):
            return
        self.tools = list(catalog)
        self.names = [str(t.get("name") or "") for t in self.tools]
        self.docs = [render_tool_text(t) for t in self.tools]
        self.vectors = [l2_normalize_list(self.encode_fn(doc)) for doc in self.docs]
        self.schema_fp = fp

    def invalidate(self) -> None:
        self.tools = []
        self.names = []
        self.vectors = []
        self.docs = []
        self.schema_fp = ""

    def search(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        top_k = int(k or self.k)
        if len(self.tools) <= top_k:
            return list(self.tools)
        q = self.encode_fn(query)
        ranked = rank_by_dot(q, list(zip(self.names, self.vectors)), k=top_k)
        return select_topk_tools(self.tools, [name for name, _ in ranked], k=top_k)

    def rank_all(self, query: str) -> list[tuple[str, float]]:
        if not self.tools:
            return []
        q = self.encode_fn(query)
        return rank_by_dot(q, list(zip(self.names, self.vectors)), k=len(self.names))
