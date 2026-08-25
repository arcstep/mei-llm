#!/usr/bin/env python3
"""Download zhwiki pages-articles dump and extract a SentencePiece sample.

Streams bz2 from dumps.wikimedia.org, writes dump (optional), wiki jsonl, and
spm-sample.txt. Stops sample at --max-chars. Use --sample-only to close HTTP
once the sample is full (dump file may be incomplete).
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import ParseError

from repo_paths import CORPUS_ZH_VOCAB

DUMP_URL = (
    "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
)
MIRROR_HINT = (
    "resume: python3 scripts/fetch_zhwiki_dump.py --resume\n"
    "mirror: https://dumps.wikimedia.org/zhwiki/latest/\n"
    "or copy an existing *.xml.bz2 onto corpora/zh-vocab-v0/dumps/"
)

WIKI_NOISE = re.compile(
    r"(?is)\{\{.*?}}"
    r"|\[\[(?:File|Image|Category|分类|文件|图像):.*?\]\]"
    r"|<ref\b.*?</ref>"
    r"|<[^>]+>"
)
WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
WIKI_BOLD = re.compile(r"'{2,}")
WS = re.compile(r"[ \t\u3000]+")


def strip_wikitext(raw: str) -> str:
    try:
        import mwparserfromhell  # type: ignore

        return str(mwparserfromhell.parse(raw).strip_code())
    except Exception:
        text = WIKI_NOISE.sub(" ", raw)
        text = WIKI_LINK.sub(r"\1", text)
        text = WIKI_BOLD.sub("", text)
        return text


def clean_text(raw: str, max_len: int = 256) -> str:
    text = strip_wikitext(raw)
    text = text.replace("\x00", " ")
    lines = []
    for line in text.splitlines():
        line = WS.sub(" ", line).strip()
        if not line or line.startswith(("=", "#", "*", "{", "|", "!")):
            continue
        if "目录" in line and len(line) < 8:
            continue
        lines.append(line)
    joined = "".join(lines) if lines else WS.sub("", text)
    joined = re.sub(r"\s+", "", joined)
    return joined[:max_len]


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class SampleWriter:
    def __init__(self, jsonl_path: Path, sample_path: Path, max_chars: int) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = jsonl_path.open("w", encoding="utf-8")
        self.sample = sample_path.open("w", encoding="utf-8")
        self.max_chars = max_chars
        self.n_chars = 0
        self.n_pages = 0
        self.seen: set[str] = set()

    def add(self, page_id: str, title: str, text: str) -> bool:
        if self.n_chars >= self.max_chars:
            return False
        body = clean_text(text)
        if len(body) < 20:
            return True
        key = body[:80]
        if key in self.seen:
            return True
        self.seen.add(key)
        rec = {"id": page_id, "title": title, "text": body}
        self.jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.sample.write(body + "\n")
        self.n_chars += len(body)
        self.n_pages += 1
        if self.n_pages % 500 == 0:
            print(f"... pages={self.n_pages} chars={self.n_chars}", flush=True)
        return self.n_chars < self.max_chars

    def close(self) -> None:
        self.jsonl.close()
        self.sample.close()


def parse_pages(xml_stream, writer: SampleWriter) -> None:
    try:
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
            if ns == "0" and title:
                if not writer.add(page_id, title, text):
                    elem.clear()
                    return
            elem.clear()
    except ParseError as exc:
        if writer.n_chars >= min(1_000_000, writer.max_chars // 10):
            print(f"xml ended early after {writer.n_chars} chars ({exc})", flush=True)
            return
        raise


class TeeReader:
    """File-like: write compressed bytes to dump while feeding decompressor."""

    def __init__(self, resp, dump_fp, decompressor: bz2.BZ2Decompressor):
        self.resp = resp
        self.dump_fp = dump_fp
        self.dec = decompressor
        self.buf = b""
        self.eof = False

    def read(self, n: int = 8192) -> bytes:
        while len(self.buf) < n and not self.eof:
            chunk = self.resp.read(256 * 1024)
            if not chunk:
                self.eof = True
                leftover = self.dec.flush() if hasattr(self.dec, "flush") else b""
                if leftover:
                    self.buf += leftover
                break
            if self.dump_fp is not None:
                self.dump_fp.write(chunk)
            try:
                self.buf += self.dec.decompress(chunk)
            except OSError:
                self.eof = True
                break
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


def fetch_stream(url: str, resume_from: int = 0):
    headers = {"User-Agent": "mei-llm-zh-vocab/0.1 (corpus sample)"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DUMP_URL)
    ap.add_argument("--dump-dir", type=Path, default=CORPUS_ZH_VOCAB / "dumps")
    ap.add_argument("--raw-dir", type=Path, default=CORPUS_ZH_VOCAB / "raw")
    ap.add_argument("--max-chars", type=int, default=80_000_000)
    ap.add_argument("--sample-only", action="store_true", help="stop HTTP after sample is full")
    ap.add_argument("--from-dump", type=Path, default=None, help="extract from local bz2")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dump_path = args.dump_dir / "zhwiki-latest-pages-articles.xml.bz2"
    jsonl_path = args.raw_dir / "wiki-text" / "pages.jsonl"
    sample_path = args.raw_dir / "spm-sample.txt"
    writer = SampleWriter(jsonl_path, sample_path, args.max_chars)

    try:
        if args.from_dump:
            src = args.from_dump
            if not src.is_file():
                print(f"missing dump: {src}", file=sys.stderr)
                return 1
            print(f"extract {src}", flush=True)
            with bz2.open(src, "rb") as xml_stream:
                parse_pages(xml_stream, writer)
        else:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            resume_from = dump_path.stat().st_size if args.resume and dump_path.is_file() else 0
            print(f"GET {args.url} resume={resume_from}", flush=True)
            try:
                resp = fetch_stream(args.url, resume_from)
            except Exception as exc:  # noqa: BLE001
                print(f"download failed: {exc}\n{MIRROR_HINT}", file=sys.stderr)
                return 1
            mode = "ab" if resume_from else "wb"
            dump_fp = dump_path.open(mode)
            tee = TeeReader(resp, dump_fp, bz2.BZ2Decompressor())
            try:
                parse_pages(tee, writer)
                if not args.sample_only and not tee.eof:
                    print("sample full; finishing dump file...", flush=True)
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        dump_fp.write(chunk)
            finally:
                dump_fp.close()
                resp.close()
    finally:
        writer.close()

    print(
        json.dumps(
            {
                "pages": writer.n_pages,
                "chars": writer.n_chars,
                "jsonl": str(jsonl_path),
                "sample": str(sample_path),
                "dump": str(dump_path) if dump_path.is_file() else None,
            },
            ensure_ascii=False,
        )
    )
    if writer.n_chars < min(1_000_000, args.max_chars):
        print("sample too small", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
