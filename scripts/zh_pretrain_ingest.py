#!/usr/bin/env python3
"""Shared ingest helpers for Needle-zh pretrain packs (v1/v2)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from data import (  # noqa: E402
    document_leaks_eval,
    file_sha256,
    normalize_document,
    sha256_text,
    write_uint16_tokens,
)

UNK_TOKEN_MAX = 0.005
UNK_DOC_MAX = 0.05
MIN_CHARS = 64
MAX_CHARS = 100_000
CHUNK_CHARS = 4096
MAX_STRUCT_CHARS = 32_768
TOKENS_PER_SHARD = 8_000_000
VALID_FRAC = 0.01

CJK_RE = re.compile(r"[\u3400-\u9fff]")
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
NAV_RE = re.compile(r"(点击这里|免责声明|版权所有|登录|注册|首页\s*[>|»]|Cookie|javascript:)", re.I)
WIKIISH_RE = re.compile(r"(维基百科|本条目|参考文献|外部链接|参见\[编辑\])")
SPOKEN_RE = re.compile(
    r"(吗[？?]|呢[？?]|吧[。！!]|啊[。！!]|怎么|为什么|有没有|我觉得|其实|哈哈|这个|那个|咱们|啥)"
)
COLLOQUIAL_KEEP_LABELS = {
    "dialogue",
    "dialog",
    "social",
    "forum",
    "blog",
    "comment",
    "review",
    "qa",
    "question",
    "entertainment",
    "story",
    "novel",
    "life",
    "chat",
    "bbs",
}
COLLOQUIAL_DROP_LABELS = {
    "encyclopedia",
    "academic",
    "paper",
    "wiki",
    "government",
    "legal",
    "patent",
}
COLLOQUIAL_DOMAIN_RE = re.compile(
    r"(forum|bbs|tieba|zhihu|weibo|blog|qa|问答|论坛|博客|评论|生活|social|review|dialogue|dialog)",
    re.I,
)
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "apache-2.0-license",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-2-clause-patent",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
    "zlib",
    "mpl-2.0",
    "postgresql",
}
STRUCT_LANGS = {"json", "json5", "yaml", "yml", "toml", "ini", "cfg", "conf"}
STRUCT_SUFFIXES = (".json", ".json5", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")
DROP_PATH_RE = re.compile(
    r"(^|/)(vendor|node_modules|third_party|dist|build|lock|\.min\.)(/|$)|package-lock|yarn\.lock|poetry\.lock",
    re.I,
)
FUNC_BODY_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:function|class|def|interface|type|enum)\b",
    re.M,
)


def rel(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_hash_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            h = line.strip()
            if h:
                out.add(h)
    return out


def append_hash(path: Path, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(digest + "\n")


def hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == src.stat().st_size:
        return
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def iter_chunks(text: str, size: int = CHUNK_CHARS, min_chars: int = MIN_CHARS) -> Iterator[str]:
    if len(text) <= size:
        if len(text) >= min_chars:
            yield text
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut <= start + min_chars:
                cut = text.rfind("\n", start, end)
            if cut > start + min_chars:
                end = cut
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            yield chunk
        if end <= start:
            break
        start = end


def hash_bucket(digest: str, *, valid_frac: float = VALID_FRAC) -> str:
    cut = max(1, int(valid_frac * 10_000))
    bucket = int(digest[:8], 16) % 10_000
    return "valid" if bucket < cut else "train"


def simhash64(text: str) -> int:
    v = [0] * 64
    n = len(text)
    if n < 3:
        h = int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)
        return h
    step = 1 if n < 4096 else max(1, n // 2048)
    for i in range(0, n - 2, step):
        g = text[i : i + 3].encode("utf-8")
        h = int(hashlib.blake2b(g, digest_size=8).hexdigest(), 16)
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if v[b] >= 0:
            out |= 1 << b
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class NearDupIndex:
    """4×16-bit band LSH over 64-bit simhash."""

    def __init__(self, max_hamming: int = 3):
        self.max_hamming = int(max_hamming)
        self.bands: list[dict[int, list[int]]] = [dict() for _ in range(4)]
        self.n = 0

    def _keys(self, value: int) -> list[int]:
        return [(value >> (16 * i)) & 0xFFFF for i in range(4)]

    def near(self, value: int) -> bool:
        seen: set[int] = set()
        for i, key in enumerate(self._keys(value)):
            for other in self.bands[i].get(key, ()):
                if other in seen:
                    continue
                seen.add(other)
                if hamming64(value, other) <= self.max_hamming:
                    return True
        return False

    def add(self, value: int) -> None:
        for i, key in enumerate(self._keys(value)):
            self.bands[i].setdefault(key, []).append(value)
        self.n += 1


def empty_stats() -> dict[str, int]:
    return {
        "n_in": 0,
        "n_keep": 0,
        "n_short": 0,
        "n_huge": 0,
        "n_no_cjk": 0,
        "n_garbled": 0,
        "n_wiki": 0,
        "n_dup": 0,
        "n_near": 0,
        "n_leak": 0,
        "n_pii": 0,
        "n_nav": 0,
        "n_unk_drop": 0,
        "n_skip_style": 0,
        "n_tokens": 0,
        "n_unk": 0,
        "n_chars": 0,
        "n_train_tokens": 0,
        "n_valid_tokens": 0,
    }


class TokenBinWriter:
    def __init__(self, out_dir: Path, split: str, tokens_per_shard: int, start_idx: int = 0, root: Path | None = None):
        self.out_dir = out_dir
        self.split = split
        self.tokens_per_shard = max(1, tokens_per_shard)
        self.idx = start_idx
        self.root = root
        self.parts: list[np.ndarray] = []
        self.buf_size = 0
        self.n_docs = 0
        self.n_tokens = 0
        self.n_unk = 0
        self.meta: list[dict] = []
        self.index_rows: list[dict] = []

    def _rel(self, path: Path) -> str:
        if self.root is None:
            return str(path)
        return rel(path, self.root)

    def _flush(self) -> None:
        if self.buf_size == 0:
            return
        path = self.out_dir / f"{self.split}-{self.idx:04d}.bin"
        buf = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
        write_uint16_tokens(path, buf)
        idx_path = self.out_dir / f"{self.split}-{self.idx:04d}.idx.jsonl"
        idx_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.index_rows),
            encoding="utf-8",
        )
        self.meta.append(
            {
                "path": self._rel(path),
                "index": self._rel(idx_path),
                "split": self.split,
                "n_docs": self.n_docs,
                "n_tokens": self.n_tokens,
                "n_unk": self.n_unk,
                "sha256": file_sha256(path),
            }
        )
        self.idx += 1
        self.parts = []
        self.buf_size = 0
        self.n_docs = 0
        self.n_tokens = 0
        self.n_unk = 0
        self.index_rows = []

    def write(self, ids: list[int] | np.ndarray, row: dict, unk_id: int) -> None:
        arr = np.asarray(ids, dtype="<u2")
        if self.buf_size and self.buf_size + arr.size > self.tokens_per_shard:
            self._flush()
        offset = int(self.buf_size)
        self.parts.append(arr.copy())
        self.buf_size += int(arr.size)
        n_unk = int(np.sum(arr == unk_id))
        self.n_docs += 1
        self.n_tokens += int(arr.size)
        self.n_unk += n_unk
        self.index_rows.append(
            {
                "offset": offset,
                "n_tokens": int(arr.size),
                "n_unk": n_unk,
                "page_id": row.get("page_id") or row.get("id") or "",
                "chunk_id": row.get("chunk_id", 0),
                "title": row.get("title") or "",
                "sha256": row.get("sha256") or "",
            }
        )

    def flush(self) -> None:
        self._flush()

    def close(self) -> None:
        self._flush()


class SplitWriters:
    def __init__(self, out_dir: Path, prefix: str, tokens_per_shard: int, root: Path):
        self.train = TokenBinWriter(out_dir, f"{prefix}-train", tokens_per_shard, root=root)
        self.valid = TokenBinWriter(out_dir, f"{prefix}-valid", tokens_per_shard, root=root)

    def close(self) -> None:
        self.train.close()
        self.valid.close()


def pii_or_nav(text: str) -> str | None:
    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        return "pii"
    if NAV_RE.search(text):
        return "nav"
    return None


def domain_labels(domain: Any) -> list[str]:
    if domain is None:
        return []
    if isinstance(domain, str):
        return [domain]
    if isinstance(domain, dict):
        out: list[str] = []
        sl = domain.get("single_label") or domain.get("label")
        if sl:
            out.append(str(sl))
        ml = domain.get("multi_label") or domain.get("labels") or []
        if isinstance(ml, str):
            out.append(ml)
        elif isinstance(ml, (list, tuple)):
            out.extend(str(x) for x in ml if x)
        return out
    return [str(domain)]


def colloquial_keep(
    text: str,
    *,
    domain: Any = None,
    quality: float | None = None,
    toxicity: float | None = None,
) -> bool:
    if quality is not None and quality < 0.25:
        return False
    if toxicity is not None and toxicity > 0.5:
        return False
    labels = [x.strip().lower() for x in domain_labels(domain) if str(x).strip()]
    if any(x in COLLOQUIAL_DROP_LABELS for x in labels):
        return False
    if WIKIISH_RE.search(text):
        return False
    spoken = len(SPOKEN_RE.findall(text))
    blob = " ".join(labels)
    domain_hit = any(x in COLLOQUIAL_KEEP_LABELS for x in labels) or bool(
        blob and COLLOQUIAL_DOMAIN_RE.search(blob)
    )
    if domain_hit:
        return True
    if spoken >= 2:
        return True
    if ("？" in text or "?" in text) and spoken >= 1:
        return True
    return False


def license_ok(value: str | None) -> bool:
    if not value:
        return False
    key = str(value).strip().lower().replace(" ", "-")
    if key in PERMISSIVE_LICENSES:
        return True
    for part in re.split(r"[|,;/]", key):
        if part.strip() in PERMISSIVE_LICENSES:
            return True
    return False


SCHEMA_HINT_RE = re.compile(
    r'("\$schema"|openapi\s*:|"properties"\s*:|"required"\s*:|"enum"\s*:|JSON Schema|OpenAPI)',
    re.I,
)
SIG_KEEP_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:function|class|def|interface|type|enum)\b"
)


def is_struct_path(path: str, language: str | None) -> bool:
    p = str(path or "").replace("\\", "/").lower()
    lang = str(language or "").lower()
    if DROP_PATH_RE.search(p):
        return False
    if lang in STRUCT_LANGS or p.endswith(STRUCT_SUFFIXES):
        return True
    return False


def is_clean_structure_text(text: str) -> bool:
    raw = str(text or "").strip()
    if len(raw) < MIN_CHARS:
        return False
    if not SCHEMA_HINT_RE.search(raw):
        return False
    comment_lines = 0
    code_body = 0
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("//") or s.startswith("*"):
            comment_lines += 1
        if FUNC_BODY_RE.match(s) and "{" in s:
            code_body += 1
    if comment_lines >= 8:
        return False
    if code_body >= 3:
        return False
    return True


def extract_signatures(text: str, *, max_chars: int = 2048) -> str:
    """Keep schema/OpenAPI and declaration lines; drop comment runs and function bodies."""
    raw = str(text or "")
    if SCHEMA_HINT_RE.search(raw):
        lines: list[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//") or s.startswith("*"):
                continue
            lines.append(line[:240])
            if sum(len(x) + 1 for x in lines) >= max_chars:
                break
        return "\n".join(lines).strip()
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith('"""') or s.startswith("'''"):
            continue
        if SIG_KEEP_RE.match(s) or s.startswith("@"):
            if "{" in s:
                s = s.split("{", 1)[0].rstrip()
            if not s:
                continue
            lines.append(s[:240])
        if sum(len(x) + 1 for x in lines) >= max_chars:
            break
    kept = "\n".join(lines).strip()
    if kept and not SCHEMA_HINT_RE.search(kept) and len(kept) < 80:
        return ""
    return kept


class Ingestor:
    def __init__(
        self,
        tok,
        writers: SplitWriters,
        seen: set[str],
        wiki: set[str],
        leaks: list[str],
        hash_path: Path,
        near: NearDupIndex | None,
        *,
        require_cjk: bool,
        max_chars: int,
        style: str = "any",
    ):
        self.tok = tok
        self.writers = writers
        self.seen = seen
        self.wiki = wiki
        self.leaks = leaks
        self.hash_path = hash_path
        self.near = near
        self.require_cjk = require_cjk
        self.max_chars = max_chars
        self.style = style
        self.stats = empty_stats()
        self.reviews: list[dict[str, Any]] = []

    def accept_text(
        self,
        text: str,
        *,
        page_id: str,
        title: str,
        domain: str | None = None,
        quality: float | None = None,
        toxicity: float | None = None,
    ) -> int:
        stats = self.stats
        stats["n_in"] += 1
        raw = str(text or "")
        if self.style == "colloquial":
            if not colloquial_keep(raw, domain=domain, quality=quality, toxicity=toxicity):
                stats["n_skip_style"] += 1
                return 0
        if len(raw) > self.max_chars:
            if self.style == "structure":
                raw = raw[: self.max_chars]
            else:
                stats["n_huge"] += 1
                return 0
        added = 0
        for chunk_id, chunk in enumerate(iter_chunks(normalize_document(raw))):
            if self.require_cjk and CJK_RE.search(chunk) is None:
                stats["n_no_cjk"] += 1
                continue
            if "�" in chunk:
                stats["n_garbled"] += 1
                continue
            flag = pii_or_nav(chunk)
            if flag == "pii":
                stats["n_pii"] += 1
                continue
            if flag == "nav":
                stats["n_nav"] += 1
                continue
            digest = sha256_text(chunk)
            if digest in self.wiki:
                stats["n_wiki"] += 1
                continue
            if digest in self.seen:
                stats["n_dup"] += 1
                continue
            leak = document_leaks_eval(chunk, self.leaks)
            if leak is not None:
                stats["n_leak"] += 1
                continue
            fp = simhash64(chunk)
            if self.near is not None and self.near.near(fp):
                stats["n_near"] += 1
                continue
            ids = self.tok.encode_document(chunk)
            if len(ids) < 4:
                stats["n_short"] += 1
                continue
            n_unk = sum(1 for t in ids if t == self.tok.unk_id)
            if n_unk / len(ids) > UNK_DOC_MAX:
                stats["n_unk_drop"] += 1
                continue
            split = hash_bucket(digest)
            self.seen.add(digest)
            append_hash(self.hash_path, digest)
            if self.near is not None:
                self.near.add(fp)
            writer = self.writers.valid if split == "valid" else self.writers.train
            writer.write(
                ids,
                {
                    "page_id": page_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "sha256": digest,
                },
                self.tok.unk_id,
            )
            stats["n_keep"] += 1
            stats["n_tokens"] += len(ids)
            stats["n_unk"] += n_unk
            stats["n_chars"] += len(chunk)
            if split == "valid":
                stats["n_valid_tokens"] += len(ids)
            else:
                stats["n_train_tokens"] += len(ids)
            added += len(ids)
            if len(self.reviews) < 40 and stats["n_keep"] % 17 == 0:
                self.reviews.append(
                    {
                        "id": page_id,
                        "n_chars": len(chunk),
                        "split": split,
                        "preview": chunk[:240],
                    }
                )
        return added


def next_shard_idx(token_dir: Path, prefix: str) -> int:
    existing = sorted(token_dir.glob(f"{prefix}-*.bin"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("-")[-1]) + 1


def shard_token_count(bin_path: Path) -> int:
    idx = bin_path.with_name(bin_path.stem + ".idx.jsonl")
    if not idx.is_file():
        return bin_path.stat().st_size // 2
    from data import iter_jsonl

    return sum(int(row.get("n_tokens") or 0) for row in iter_jsonl(idx))


def split_existing_bins(
    src_bins: Iterable[Path],
    dest_dir: Path,
    prefix: str,
    *,
    root: Path,
    valid_frac: float = VALID_FRAC,
) -> dict[str, int]:
    from data import iter_jsonl

    dest_dir.mkdir(parents=True, exist_ok=True)
    writers = SplitWriters(dest_dir, prefix, TOKENS_PER_SHARD, root)
    n_train = n_valid = n_docs = 0
    for bin_path in src_bins:
        idx_path = bin_path.with_name(bin_path.stem + ".idx.jsonl")
        mmap = np.memmap(bin_path, dtype="<u2", mode="r")
        if idx_path.is_file():
            rows = list(iter_jsonl(idx_path))
        else:
            rows = [{"offset": 0, "n_tokens": int(mmap.shape[0]), "n_unk": 0, "sha256": file_sha256(bin_path)}]
        for row in rows:
            n = int(row.get("n_tokens") or 0)
            off = int(row.get("offset") or 0)
            if n <= 0:
                continue
            digest = str(row.get("sha256") or sha256_text(f"{bin_path}:{off}:{n}"))
            split = hash_bucket(digest, valid_frac=valid_frac)
            ids = np.asarray(mmap[off : off + n], dtype="<u2")
            writer = writers.valid if split == "valid" else writers.train
            writer.write(ids, row, unk_id=3)
            n_docs += 1
            if split == "valid":
                n_valid += n
            else:
                n_train += n
    writers.close()
    return {"n_docs": n_docs, "n_train_tokens": n_train, "n_valid_tokens": n_valid}


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    cols = [c for c in (columns or names) if c in names]
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        pydict = batch.to_pydict()
        n = batch.num_rows
        keys = list(pydict)
        for i in range(n):
            yield {k: pydict[k][i] for k in keys}


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
