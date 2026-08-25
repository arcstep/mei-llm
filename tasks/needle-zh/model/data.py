"""Real-tokenizer packing for pretrain windows and assistant-only SFT masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from grammar import dump_calls
from tokenizer import ZhTokenizerV1

RUNG_TOKEN_BUDGET = {
    "smoke": 1,
    "pilot-1m": 1_000_000,
    "pilot-5m": 5_000_000,
    "100m": 100_000_000,
    "300m": 300_000_000,
    "1b": 1_000_000_000,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_text_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def document_ids(tok: ZhTokenizerV1, text: str) -> list[int]:
    return tok.encode_document(str(text or ""))


def pack_windows(
    doc_streams: Iterable[list[int]],
    seq_len: int,
    pad_id: int = 0,
) -> list[dict[str, list]]:
    stream: list[int] = []
    for doc in doc_streams:
        stream.extend(doc)
    out: list[dict[str, list]] = []
    step = seq_len
    for i in range(0, max(0, len(stream) - 1), step):
        chunk = stream[i : i + seq_len + 1]
        if len(chunk) < 2:
            continue
        if len(chunk) < seq_len + 1:
            chunk = chunk + [pad_id] * (seq_len + 1 - len(chunk))
        x, y = chunk[:-1], chunk[1:]
        mask = [0.0 if t == pad_id else 1.0 for t in y]
        if sum(mask) <= 0:
            continue
        out.append({"x": x, "y": y, "mask": mask})
    return out


def count_tokens(tok: ZhTokenizerV1, texts: Iterable[str]) -> int:
    return sum(len(tok.encode_document(t)) for t in texts)


def refuse_if_short(
    rung: str,
    n_tokens: int,
    *,
    smoke: bool,
    cap: int | None = None,
) -> None:
    key = "smoke" if smoke else rung
    need = RUNG_TOKEN_BUDGET.get(key)
    if need is None:
        raise ValueError(f"unknown rung {rung}")
    if cap is not None:
        need = min(need, int(cap))
    if n_tokens < need:
        raise ValueError(
            f"rung {key} needs >= {need} tokens, packed {n_tokens}; refuse rather than alias the name"
        )


def encode_sft_row(
    tok: ZhTokenizerV1,
    row: dict[str, Any],
    seq_len: int,
) -> dict[str, Any]:
    query = str(row.get("query") or "")
    answers = row.get("answers") or []
    answer_text = dump_calls(answers)
    packed = tok.encode_chat(query, answer_text)
    ids = list(packed["ids"])
    n_prompt = int(packed["n_prompt"])
    if len(ids) > seq_len:
        overflow = len(ids) - seq_len
        keep_prompt = max(1, n_prompt - overflow)
        ids = packed["prompt_ids"][-keep_prompt:] + packed["answer_ids"]
        n_prompt = keep_prompt
        ids = ids[:seq_len]
    pad_id = tok.pad_id
    if len(ids) < seq_len:
        ids = ids + [pad_id] * (seq_len - len(ids))
    x, y = ids[:-1], ids[1:]
    # Loss on tokens that predict assistant content (incl. first assistant token).
    mask = []
    for t, target in enumerate(y):
        in_assistant = (t + 1) >= n_prompt and target != pad_id
        mask.append(1.0 if in_assistant else 0.0)
    conf = float(row.get("confidence_label") if row.get("confidence_label") is not None else (1.0 if answers else 0.0))
    return {
        "x": x,
        "y": y,
        "mask": mask,
        "confidence": conf,
        "n_prompt": n_prompt,
        "n_unmasked": int(sum(mask)),
    }


def token_jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def exact_dedup(texts: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out: list[str] = []
    dropped = 0
    for t in texts:
        h = sha256_text(t)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(t)
    return out, dropped


def hash_split(texts: list[str], *, valid_frac: float = 0.05) -> tuple[list[str], list[str]]:
    train, valid = [], []
    cut = max(1, int(valid_frac * 10_000))
    for t in texts:
        bucket = int(sha256_text(t)[:8], 16) % 10_000
        (valid if bucket < cut else train).append(t)
    if not valid and train:
        valid = [train[-1]]
        train = train[:-1]
    return train, valid


def token_near_dups(
    left: list[tuple[str, list[int]]],
    right: list[tuple[str, list[int]]],
    *,
    threshold: float = 0.9,
    limit: int = 32,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for sid, a in left:
        if len(a) < 4:
            continue
        for eid, b in right:
            if len(b) < 4:
                continue
            score = token_jaccard(a, b)
            if score >= threshold:
                hits.append({"left": sid, "right": eid, "jaccard": round(score, 4)})
                if len(hits) >= limit:
                    return hits
    return hits


def corpus_stats(tok: ZhTokenizerV1, rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(r.get("text") or "") for r in rows]
    sources: dict[str, int] = {}
    for r in rows:
        src = str(r.get("source") or "unknown")
        sources[src] = sources.get(src, 0) + 1
    n_tok = count_tokens(tok, texts)
    return {
        "n_docs": len(rows),
        "n_tokens": n_tok,
        "sources": sources,
        "sha256": hashlib.sha256(
            "\n".join(sha256_text(t) for t in texts).encode("utf-8")
        ).hexdigest(),
    }


def normalize_document(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def hash_split_label(text: str, *, valid_frac: float = 0.05) -> str:
    cut = max(1, int(valid_frac * 10_000))
    bucket = int(sha256_text(text)[:8], 16) % 10_000
    return "valid" if bucket < cut else "train"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_shard_rows(paths: Iterable[Path], *, split: str | None = None) -> Iterator[dict[str, Any]]:
    for path in paths:
        for row in iter_jsonl(path):
            if split is None or str(row.get("split") or "") == split:
                yield row


def window_from_chunk(chunk: list[int], seq_len: int, pad_id: int) -> dict[str, list]:
    if len(chunk) < seq_len + 1:
        chunk = chunk + [pad_id] * (seq_len + 1 - len(chunk))
    x, y = chunk[:-1], chunk[1:]
    mask = [0.0 if t == pad_id else 1.0 for t in y]
    return {"x": x, "y": y, "mask": mask}


def iter_packed_windows(
    doc_streams: Iterable[list[int]],
    seq_len: int,
    pad_id: int = 0,
    *,
    start_window: int = 0,
) -> Iterator[dict[str, list]]:
    buf: list[int] = []
    produced = 0
    for doc in doc_streams:
        buf.extend(doc)
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len:]
            if produced >= start_window:
                win = window_from_chunk(chunk, seq_len, pad_id)
                if sum(win["mask"]) > 0:
                    yield win
            produced += 1
    if len(buf) >= 2:
        if produced >= start_window:
            win = window_from_chunk(buf, seq_len, pad_id)
            if sum(win["mask"]) > 0:
                yield win


def iter_split_documents(
    paths: Iterable[Path],
    tok: ZhTokenizerV1,
    *,
    split: str,
) -> Iterator[list[int]]:
    for row in iter_shard_rows(paths, split=split):
        text = normalize_document(str(row.get("text") or ""))
        if text:
            yield document_ids(tok, text)


def leak_strings_from_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    keys = ("query", "input", "target", "text")
    for row in rows:
        for key in keys:
            val = row.get(key)
            if isinstance(val, str):
                s = normalize_document(val)
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def document_leaks_eval(text: str, leak_strings: list[str]) -> str | None:
    """Return the leak string if this pretrain doc must be dropped."""
    t = normalize_document(text)
    if not t:
        return None
    for s in leak_strings:
        if t == s:
            return s
        # Digit-only probes like "3500" match too much wiki; require exact doc equality.
        if s.isdigit():
            continue
        latin = all(ord(c) < 128 for c in s)
        if latin and len(s) < 12:
            continue
        if len(s) >= 8 and s in t:
            return s
        if 4 <= len(s) < 8:
            padded = f" {t} "
            if f" {s} " in padded:
                return s
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_pretrain_shards(root: Path, *, smoke: bool = False) -> list[Path]:
    shard_dir = Path(root) / "shards"
    if smoke:
        p = shard_dir / "smoke.jsonl"
        return [p] if p.is_file() else []
    named = []
    for pat in ("train-*.jsonl", "valid-*.jsonl", "mix-v0.jsonl"):
        named.extend(sorted(shard_dir.glob(pat)))
    return named


def list_raw_pages(root: Path) -> list[Path]:
    raw_dir = Path(root) / "raw"
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.glob("pages-*.jsonl") if p.is_file())


list_raw_pages = list_raw_pages


def list_token_shards(root: Path, split: str) -> list[Path]:
    token_dir = Path(root) / "tokens"
    if not token_dir.is_dir():
        return []
    return sorted(token_dir.glob(f"{split}-*.bin"))


def write_uint16_tokens(path: Path, ids: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(ids, dtype="<u2")
    if arr.size and int(arr.max()) > 65535:
        raise ValueError("token id exceeds uint16")
    tmp = path.with_name(path.name + ".tmp")
    arr.tofile(tmp)
    tmp.replace(path)


def _bisect_prefix(prefix: list[int], pos: int) -> int:
    lo, hi = 0, len(prefix)
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


class PackedTokenSource:
    """Random-access packed LM windows over concatenated uint16 mmap shards."""

    def __init__(self, paths: list[Path], seq_len: int, pad_id: int = 0):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("no token shards")
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.maps = [np.memmap(p, dtype="<u2", mode="r") for p in self.paths]
        self.lengths = [int(m.shape[0]) for m in self.maps]
        self.prefix: list[int] = []
        acc = 0
        for n in self.lengths:
            acc += n
            self.prefix.append(acc)
        self.n_tokens = acc
        usable = max(0, self.n_tokens - 1)
        self.n_windows = (usable + self.seq_len - 1) // self.seq_len if usable else 0

    @property
    def n_predictable_tokens(self) -> int:
        """Unique next-token targets in one mmap pass: every token except the first."""
        return max(0, int(self.n_tokens) - 1)

    def __len__(self) -> int:
        return self.n_windows

    def _token_at(self, pos: int) -> int:
        if pos < 0 or pos >= self.n_tokens:
            raise IndexError(pos)
        si = _bisect_prefix(self.prefix, pos)
        prev = 0 if si == 0 else self.prefix[si - 1]
        return int(self.maps[si][pos - prev])

    def _gather(self, start: int, length: int) -> list[int]:
        out: list[int] = []
        pos = start
        remain = length
        while remain > 0 and pos < self.n_tokens:
            si = _bisect_prefix(self.prefix, pos)
            prev = 0 if si == 0 else self.prefix[si - 1]
            local = pos - prev
            take = min(remain, self.lengths[si] - local)
            chunk = np.asarray(self.maps[si][local : local + take], dtype=np.int32)
            out.extend(int(x) for x in chunk.tolist())
            pos += take
            remain -= take
        return out

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if idx < 0:
            idx += self.n_windows
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(idx)
        start = idx * self.seq_len
        chunk = self._gather(start, self.seq_len + 1)
        return window_from_chunk(chunk, self.seq_len, self.pad_id)


def windows_from_ids(ids: list[int], seq_len: int, pad_id: int = 0) -> list[dict[str, list]]:
    return pack_windows([ids], seq_len, pad_id)


PackedTokenSource = PackedTokenSource
write_uint16_tokens = write_uint16_tokens
list_raw_pages = list_raw_pages
