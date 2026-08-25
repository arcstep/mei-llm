#!/usr/bin/env python3
"""Extract untruncated zhwiki article text for pretrain, resumable via SQLite.

Reuses wikitext stripping from fetch_zhwiki_dump.py. Does not cap pages at 256
chars and does not early-stop on a global char budget. Long articles are split
on paragraph boundaries. Output lives under gitignored corpora/zh-pretrain-v0/raw/.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN, CORPUS_ZH_VOCAB, ROOT

from fetch_zhwiki_dump import WS, localname, strip_wikitext

CLEANER_ID = "zhwiki-pretrain-v1-paragraph"
REDIRECT_RE = re.compile(r"^#\s*(?:redirect|重定向)\b", re.I)
DEFAULT_DUMP = CORPUS_ZH_VOCAB / "dumps" / "zhwiki-latest-pages-articles.xml.bz2"
LENGTH_BINS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def length_bin(n: int) -> str:
    for edge in LENGTH_BINS:
        if n <= edge:
            return f"le_{edge}"
    return f"gt_{LENGTH_BINS[-1]}"


def clean_pretrain_text(raw: str) -> str:
    text = strip_wikitext(raw)
    text = text.replace("\x00", " ")
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        line = WS.sub(" ", line).strip()
        if not line:
            if buf:
                paragraphs.append("".join(buf))
                buf = []
            continue
        if line.startswith(("=", "#", "*", "{", "|", "!")):
            continue
        if "目录" in line and len(line) < 8:
            continue
        buf.append(line)
    if buf:
        paragraphs.append("".join(buf))
    kept = [p for p in paragraphs if p and not REDIRECT_RE.match(p)]
    return "\n".join(kept)


def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not paras:
        return []
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for p in paras:
        extra = len(p) + (1 if buf else 0)
        if buf and n + extra > max_chars:
            out.append("\n".join(buf))
            buf = [p]
            n = len(p)
        else:
            buf.append(p)
            n += extra
        while buf and n > max_chars:
            cur = "\n".join(buf)
            out.append(cur[:max_chars])
            rest = cur[max_chars:].lstrip("\n")
            buf = [rest] if rest else []
            n = len(rest)
    if buf:
        joined = "\n".join(buf)
        if joined:
            out.append(joined)
    return [c for c in out if c.strip()]


class ExtractState:
    def __init__(self, path: Path, dump_sha: str, cleaner_id: str, max_chunk: int, min_chars: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS seen (sha TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS shards (
              idx INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              n_docs INTEGER NOT NULL,
              n_chars INTEGER NOT NULL,
              sha256 TEXT,
              closed INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.conn.commit()
        existing = self.get_meta("dump_sha256")
        if existing and existing != dump_sha:
            raise ValueError(f"dump hash changed: state={existing} file={dump_sha}; refuse resume")
        cleaner = self.get_meta("cleaner_id")
        if cleaner and cleaner != cleaner_id:
            raise ValueError(f"cleaner changed: state={cleaner} now={cleaner_id}; refuse resume")
        self.set_meta("dump_sha256", dump_sha)
        self.set_meta("cleaner_id", cleaner_id)
        self.set_meta("max_chunk_chars", str(max_chunk))
        self.set_meta("min_chars", str(min_chars))
        self.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )

    def last_ordinal(self) -> int:
        return int(self.get_meta("last_ordinal") or 0)

    def seen(self, sha: str) -> bool:
        return self.conn.execute("SELECT 1 FROM seen WHERE sha=?", (sha,)).fetchone() is not None

    def add_seen(self, sha: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO seen(sha) VALUES(?)", (sha,))

    def closed_shards(self) -> set[int]:
        rows = self.conn.execute("SELECT idx FROM shards WHERE closed=1").fetchall()
        return {int(r[0]) for r in rows}

    def next_shard_idx(self) -> int:
        closed = self.closed_shards()
        return (max(closed) + 1) if closed else 0

    def record_shard(
        self, idx: int, name: str, n_docs: int, n_chars: int, sha: str | None, closed: bool
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO shards(idx, name, n_docs, n_chars, sha256, closed)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(idx) DO UPDATE SET
              name=excluded.name, n_docs=excluded.n_docs, n_chars=excluded.n_chars,
              sha256=excluded.sha256, closed=excluded.closed
            """,
            (idx, name, n_docs, n_chars, sha, 1 if closed else 0),
        )

    def closed_shard_rows(self) -> list[dict]:
        out = []
        for row in self.conn.execute(
            "SELECT idx, name, n_docs, n_chars, sha256 FROM shards WHERE closed=1 ORDER BY idx"
        ):
            out.append(
                {"idx": row[0], "name": row[1], "n_docs": row[2], "n_chars": row[3], "sha256": row[4]}
            )
        return out

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


class ShardWriter:
    def __init__(self, raw_dir: Path, idx: int, docs_per_shard: int, state: ExtractState):
        if idx in state.closed_shards():
            raise RuntimeError(f"refusing to overwrite closed shard pages-{idx:04d}.jsonl")
        self.state = state
        self.idx = idx
        self.docs_per_shard = docs_per_shard
        self.name = f"pages-{idx:04d}.jsonl"
        self.final = raw_dir / self.name
        self.tmp = raw_dir / f"{self.name}.tmp"
        if self.tmp.is_file():
            self.tmp.unlink()
        self.fh = self.tmp.open("w", encoding="utf-8")
        self.n_docs = 0
        self.n_chars = 0

    def write(self, row: dict) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n_docs += 1
        self.n_chars += len(str(row.get("text") or ""))

    def full(self) -> bool:
        return self.n_docs >= self.docs_per_shard

    def close(self, *, finalize: bool) -> None:
        self.fh.flush()
        self.fh.close()
        if finalize and self.n_docs > 0:
            sha = sha256_file(self.tmp)
            self.tmp.replace(self.final)
            self.state.record_shard(self.idx, self.name, self.n_docs, self.n_chars, sha, True)
        else:
            if self.tmp.is_file():
                self.tmp.unlink()
            self.state.record_shard(self.idx, self.name, 0, 0, None, False)


def iter_dump_pages(dump_path: Path):
    with bz2.open(dump_path, "rb") as xml_stream:
        for _event, elem in ET.iterparse(xml_stream, events=("end",)):
            if localname(elem.tag) != "page":
                continue
            ns = title = page_id = text = ""
            for child in elem:
                n = localname(child.tag)
                if n == "title":
                    title = (child.text or "").strip()
                elif n == "ns":
                    ns = (child.text or "").strip()
                elif n == "id" and not page_id:
                    page_id = (child.text or "").strip()
                elif n == "revision":
                    for rev in child:
                        if localname(rev.tag) == "text":
                            text = rev.text or ""
            yield {"id": page_id, "title": title, "ns": ns, "text": text}
            elem.clear()


def extract(
    dump_path: Path,
    raw_dir: Path,
    *,
    max_chunk_chars: int = 4096,
    min_chars: int = 64,
    docs_per_shard: int = 20000,
    limit_pages: int | None = None,
) -> dict:
    if not dump_path.is_file():
        raise FileNotFoundError(f"missing dump {dump_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dump_sha = sha256_file(dump_path)
    state = ExtractState(
        raw_dir / "extract-state.sqlite",
        dump_sha,
        CLEANER_ID,
        max_chunk_chars,
        min_chars,
    )
    last = state.last_ordinal()
    persist_keys = (
        "n_ns",
        "n_redirect",
        "n_short",
        "n_dup",
        "n_short_chunk",
        "n_chunks",
        "n_chars",
        "pages_kept",
    )
    stats: Counter[str] = Counter({k: int(state.get_meta(f"stat_{k}") or 0) for k in persist_keys})
    try:
        length_hist = Counter(json.loads(state.get_meta("length_hist") or "{}"))
    except json.JSONDecodeError:
        length_hist = Counter()
    baseline_processed = stats["n_ns"] + stats["n_redirect"] + stats["n_short"] + stats["pages_kept"]

    def persist_stats() -> None:
        for k in persist_keys:
            state.set_meta(f"stat_{k}", str(int(stats[k])))
        state.set_meta("length_hist", json.dumps(dict(length_hist), ensure_ascii=False))

    writer: ShardWriter | None = None
    next_idx = state.next_shard_idx()
    ordinal = 0
    shard_rows: list[dict] = []
    try:
        for page in iter_dump_pages(dump_path):
            ordinal += 1
            stats["pages_in"] += 1
            if ordinal <= last:
                stats["skipped"] += 1
                continue
            processed_now = (
                stats["n_ns"]
                + stats["n_redirect"]
                + stats["n_short"]
                + stats["pages_kept"]
                - baseline_processed
            )
            if limit_pages is not None and processed_now >= limit_pages:
                break

            skip_reason = None
            raw = page["text"] or ""
            if page["ns"] != "0" or not page["title"]:
                skip_reason = "n_ns"
            elif REDIRECT_RE.match(raw.lstrip()):
                skip_reason = "n_redirect"
            else:
                body = clean_pretrain_text(raw)
                if not body or REDIRECT_RE.match(body):
                    skip_reason = "n_redirect"
                elif len(body) < min_chars:
                    skip_reason = "n_short"
                else:
                    chunks = chunk_paragraphs(body, max_chunk_chars)
                    wrote = 0
                    for chunk_id, chunk in enumerate(chunks):
                        if len(chunk) < min_chars:
                            stats["n_short_chunk"] += 1
                            continue
                        digest = sha256_text(chunk)
                        if state.seen(digest):
                            stats["n_dup"] += 1
                            continue
                        if writer is None:
                            writer = ShardWriter(raw_dir, next_idx, docs_per_shard, state)
                        writer.write(
                            {
                                "id": f"{page['id']}:{chunk_id}",
                                "page_id": page["id"],
                                "chunk_id": chunk_id,
                                "title": page["title"],
                                "text": chunk,
                                "n_chars": len(chunk),
                                "sha256": digest,
                                "source": "zhwiki-pages-articles",
                                "cleaner_id": CLEANER_ID,
                            }
                        )
                        state.add_seen(digest)
                        stats["n_chunks"] += 1
                        stats["n_chars"] += len(chunk)
                        length_hist[length_bin(len(chunk))] += 1
                        wrote += 1
                        if writer.full():
                            writer.close(finalize=True)
                            next_idx += 1
                            writer = None
                    if wrote:
                        stats["pages_kept"] += 1
                    elif not chunks:
                        skip_reason = "n_short"

            if skip_reason:
                stats[skip_reason] += 1
            state.set_meta("last_ordinal", str(ordinal))
            last = ordinal
            if ordinal % 500 == 0:
                persist_stats()
                state.commit()
                print(
                    json.dumps(
                        {
                            "ordinal": ordinal,
                            "pages_kept": stats["pages_kept"],
                            "chunks": stats["n_chunks"],
                            "chars": stats["n_chars"],
                            "dup": stats["n_dup"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if writer is not None:
            finalize = writer.n_docs > 0
            writer.close(finalize=finalize)
            writer = None
        persist_stats()
        state.set_meta("complete", "1" if limit_pages is None else "0")
        state.commit()
        shard_rows = state.closed_shard_rows()
    finally:
        if writer is not None:
            writer.close(finalize=False)
        state.close()

    for row in shard_rows:
        row["path"] = relpath(raw_dir / row["name"])
    le256 = int(length_hist.get("le_256", 0))
    total_chunks = int(stats["n_chunks"])
    man = {
        "dump": relpath(dump_path),
        "dump_sha256": dump_sha,
        "dump_bytes": dump_path.stat().st_size,
        "cleaner_id": CLEANER_ID,
        "max_chunk_chars": max_chunk_chars,
        "min_chars": min_chars,
        "docs_per_shard": docs_per_shard,
        "last_ordinal": last,
        "n_pages_in": int(stats["pages_in"]),
        "n_pages_kept": int(stats["pages_kept"]),
        "n_chunks": total_chunks,
        "n_chars": int(stats["n_chars"]),
        "n_dup": int(stats["n_dup"]),
        "n_short": int(stats["n_short"]),
        "n_ns": int(stats["n_ns"]),
        "n_redirect": int(stats["n_redirect"]),
        "n_skipped_resume": int(stats["skipped"]),
        "length_hist": dict(length_hist),
        "frac_le_256": round(le256 / total_chunks, 6) if total_chunks else None,
        "shards": shard_rows,
        "complete": limit_pages is None,
    }
    payload = json.dumps(man, ensure_ascii=False, indent=2) + "\n"
    (raw_dir / "extract-manifest.json").write_text(payload, encoding="utf-8")
    CORPUS_ZH_PRETRAIN.mkdir(parents=True, exist_ok=True)
    (CORPUS_ZH_PRETRAIN / "extract-manifest.json").write_text(payload, encoding="utf-8")
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--raw-dir", type=Path, default=CORPUS_ZH_PRETRAIN / "raw")
    ap.add_argument("--max-chunk-chars", type=int, default=4096)
    ap.add_argument("--min-chars", type=int, default=64)
    ap.add_argument("--docs-per-shard", type=int, default=20000)
    ap.add_argument("--limit-pages", type=int, default=None)
    args = ap.parse_args()
    try:
        man = extract(
            args.dump,
            args.raw_dir,
            max_chunk_chars=args.max_chunk_chars,
            min_chars=args.min_chars,
            docs_per_shard=args.docs_per_shard,
            limit_pages=args.limit_pages,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(man, ensure_ascii=False, indent=2))
    if not man["n_chunks"]:
        print("no chunks extracted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
