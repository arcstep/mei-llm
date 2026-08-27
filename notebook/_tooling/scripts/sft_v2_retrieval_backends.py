"""Sparse and dense retrieval backends sharing retrieval_rag_protocol.

Quality ranking is exact L2-dot/cosine. HNSW is an optional ANN efficiency column.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Any, Sequence

from sft_canonical_lib import compact_tools

PROTOCOL_DIR_HINT = "runtime/mei-1.0-58m-needle2-v2"


def _protocol():
    import sys
    from pathlib import Path

    from repo_paths import ROOT

    model_dir = ROOT / "notebook/_tooling/model/mei-1.0-58m"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from retrieval_rag_protocol import (  # noqa: E402
        K_DEFAULT,
        CatalogIndex,
        render_tool_text,
        select_topk_tools,
    )

    return K_DEFAULT, CatalogIndex, render_tool_text, select_topk_tools


def tokenize(text: str) -> list[str]:
    raw = (text or "").lower()
    toks: list[str] = []
    buf = []
    for ch in raw:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                toks.append("".join(buf))
                buf = []
            toks.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                toks.append("".join(buf))
                buf = []
    if buf:
        toks.append("".join(buf))
    return [t for t in toks if t]


def char_ngrams(text: str, ns: tuple[int, ...] = (2, 3)) -> list[str]:
    s = (text or "").replace(" ", "")
    grams: list[str] = []
    for n in ns:
        if len(s) < n:
            continue
        grams.extend(s[i : i + n] for i in range(len(s) - n + 1))
    return grams or list(s)


class SparseRetriever:
    kind = "sparse"

    def __init__(self, *, mode: str = "bm25", k: int = 5):
        self.mode = mode
        self.k = k
        self.tools: list[dict[str, Any]] = []
        self.names: list[str] = []
        self.docs: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.df: Counter[str] = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.schema_fp = ""

    def _tok(self, text: str) -> list[str]:
        return tokenize(text) if self.mode == "bm25" else char_ngrams(text)

    def build(self, catalog: Sequence[dict[str, Any]]) -> None:
        _, _, render_tool_text, _ = _protocol()
        self.tools = [compact_tools([t])[0] for t in catalog]
        self.names = [str(t["name"]) for t in self.tools]
        self.docs = [render_tool_text(t) for t in self.tools]
        self.doc_tokens = [self._tok(doc) for doc in self.docs]
        self.df = Counter()
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.df[tok] += 1
        n = max(1, len(self.doc_tokens))
        self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()}
        self.avgdl = sum(len(t) for t in self.doc_tokens) / n

    def _score(self, query: str, doc_i: int) -> float:
        q_toks = self._tok(query)
        d_toks = self.doc_tokens[doc_i]
        if not q_toks or not d_toks:
            return 0.0
        tf = Counter(d_toks)
        if self.mode == "bm25":
            k1, b = 1.5, 0.75
            dl = len(d_toks)
            score = 0.0
            for tok in q_toks:
                if tok not in tf:
                    continue
                idf = self.idf.get(tok, 0.0)
                freq = tf[tok]
                score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / (self.avgdl or 1)))
            return score
        # char tf-idf cosine
        qtf = Counter(q_toks)
        q_w = {t: qtf[t] * self.idf.get(t, 0.0) for t in qtf}
        d_w = {t: tf[t] * self.idf.get(t, 0.0) for t in tf}
        keys = set(q_w) | set(d_w)
        num = sum(q_w.get(k, 0.0) * d_w.get(k, 0.0) for k in keys)
        qn = math.sqrt(sum(v * v for v in q_w.values())) or 1.0
        dn = math.sqrt(sum(v * v for v in d_w.values())) or 1.0
        return num / (qn * dn)

    def rank_all(self, query: str) -> list[tuple[str, float]]:
        return self.rank_subset(query, self.names)

    def rank_subset(self, query: str, names: Sequence[str]) -> list[tuple[str, float]]:
        want = set(names)
        scored: list[tuple[int, float, str]] = []
        q_toks_cache = self._tok(query)
        orig_tok = self._tok

        def _tok_once(_text: str) -> list[str]:
            return q_toks_cache

        self._tok = _tok_once  # type: ignore[method-assign]
        try:
            for i, name in enumerate(self.names):
                if name not in want:
                    continue
                scored.append((i, self._score(query, i), name))
        finally:
            self._tok = orig_tok  # type: ignore[method-assign]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return [(name, score) for _i, score, name in scored]

    def search(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        _, _, _, select_topk_tools = _protocol()
        top_k = int(k or self.k)
        if len(self.tools) <= top_k:
            return list(self.tools)
        ranked = [name for name, _ in self.rank_all(query)[:top_k]]
        return select_topk_tools(self.tools, ranked, k=top_k)


class DenseExactRetriever:
    kind = "dense_exact"

    def __init__(self, encode_fn, *, name: str, k: int = 5, meta: dict | None = None):
        self.name = name
        self.k = k
        self.meta = dict(meta or {})
        _, CatalogIndex, _, _ = _protocol()
        self.index = CatalogIndex(encode_fn, k=k)

    def build(self, catalog: Sequence[dict[str, Any]]) -> None:
        self.index.build(catalog)

    def search(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        return self.index.search(query, k=k or self.k)

    def rank_all(self, query: str) -> list[tuple[str, float]]:
        return self.index.rank_all(query)


class HnswRetriever:
    kind = "hnsw"

    def __init__(self, encode_fn, *, name: str, k: int = 5, meta: dict | None = None):
        self.encode_fn = encode_fn
        self.name = name
        self.k = k
        self.meta = dict(meta or {})
        self.tools: list[dict[str, Any]] = []
        self.names: list[str] = []
        self.index = None
        self.dim = 0
        self.build_ms = None
        self.available = True
        self.error = ""

    def build(self, catalog: Sequence[dict[str, Any]]) -> None:
        try:
            import hnswlib
            import numpy as np
        except ImportError as exc:
            self.available = False
            self.error = f"missing:{exc}"
            self.tools = [compact_tools([t])[0] for t in catalog]
            return
        _, _, render_tool_text, _ = _protocol()
        self.tools = [compact_tools([t])[0] for t in catalog]
        self.names = [str(t["name"]) for t in self.tools]
        docs = [render_tool_text(t) for t in self.tools]
        t0 = time.perf_counter()
        vecs = [list(self.encode_fn(doc)) for doc in docs]
        self.dim = len(vecs[0]) if vecs else 0
        mat = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(max_elements=len(mat), ef_construction=200, M=16)
        idx.add_items(mat, list(range(len(mat))))
        idx.set_ef(64)
        self.index = idx
        self.build_ms = round((time.perf_counter() - t0) * 1000, 2)

    def search(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        _, _, _, select_topk_tools = _protocol()
        top_k = int(k or self.k)
        if not self.available or self.index is None:
            return list(self.tools)[:top_k]
        if len(self.tools) <= top_k:
            return list(self.tools)
        import numpy as np

        q = np.asarray(list(self.encode_fn(query)), dtype="float32")
        n = float(np.linalg.norm(q)) or 1.0
        q = q / n
        labels, _dists = self.index.knn_query(q, k=top_k)
        names = [self.names[int(i)] for i in labels[0]]
        return select_topk_tools(self.tools, names, k=top_k)


def try_sentence_transformer(model_id: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, {"status": "unavailable", "error": "sentence_transformers_missing", "hf_id": model_id}
    try:
        model = SentenceTransformer(model_id)
    except Exception as exc:  # noqa: BLE001
        return None, {"status": "unavailable", "error": str(exc.__class__.__name__), "hf_id": model_id}

    def encode_fn(text: str):
        vec = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    tok = getattr(model, "tokenizer", None)
    max_len = int(getattr(model, "max_seq_length", 0) or 0)
    return encode_fn, {
        "status": "ready",
        "hf_id": model_id,
        "max_seq_length": max_len,
        "tokenizer": getattr(tok, "name_or_path", None),
        "pooling": "sentence_transformers_default",
        "normalize": True,
    }


def mei58m_encode_fn(model, tokenizer):
    import mlx.core as mx

    from retrieval_rag_protocol import encode_text_ids

    def encode_fn(text: str):
        ids = encode_text_ids(tokenizer, text)
        arr = mx.array([ids], dtype=mx.int32)
        out = model(arr, return_contrastive=True, return_cells=True)
        vec = out["contrastive"][0].astype(mx.float32)
        norm = mx.sqrt(mx.sum(vec * vec)) + mx.array(1e-8, dtype=mx.float32)
        return [float(x) for x in (vec / norm).tolist()]

    return encode_fn
