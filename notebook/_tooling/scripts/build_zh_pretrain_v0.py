#!/usr/bin/env python3
"""Build Chinese pretrain shards with real token counts, hashes, split, and budget gates.

Non-smoke path streams local zhwiki pages.jsonl. Does not download external corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_PRETRAIN_PROBES,
    CORPUS_ZH_PRETRAIN,
    CORPUS_ZH_VOCAB,
    LANGUAGE_WORK_V0,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    all_eval_jsonl,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from data import (  # noqa: E402
    corpus_stats,
    document_leaks_eval,
    exact_dedup,
    file_sha256,
    hash_split,
    hash_split_label,
    iter_jsonl,
    leak_strings_from_rows,
    list_raw_pages,
    normalize_document,
    refuse_if_short,
    token_near_dups,
    write_uint16_tokens,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

PAGES_DEFAULT = CORPUS_ZH_VOCAB / "raw" / "wiki-text" / "pages.jsonl"

INSTRUCTIONS = [
    "把下面句子原样抄写：",
    "只输出数字：",
    "把中文数字改成阿拉伯数字：",
    "按 JSON 输出键值：",
    "判断这句话是否完整：",
]
JSON_TEMPLATES = [
    '{"city":"成都","day":"今天"}',
    '{"room":"客厅","on":true}',
    '{"shop":"兰州拉面","dish":"牛肉面"}',
    '{"name":"nod","arguments":{}}',
    "[]",
]
COPY_SRC = [
    "请把门关上。",
    "厨房灯已经打开。",
    "订单已经取消。",
    "用户站在门口。",
    "把垃圾拿出去。",
]
DATES = ["2026-08-24", "2024年1月3日", "二零二六年八月二十四日", "3500", "三千五百", "12:30"]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def wiki_lines(limit: int) -> list[str]:
    sample = CORPUS_ZH_VOCAB / "raw" / "spm-sample.txt"
    if not sample.is_file():
        return []
    out = []
    with sample.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if 8 <= len(s) <= 240:
                out.append(s)
            if len(out) >= limit:
                break
    return out


def synth(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        kind = rng.choice(["copy", "json", "num", "instr", "short"])
        if kind == "copy":
            s = rng.choice(COPY_SRC)
            rows.append(f"{rng.choice(INSTRUCTIONS)}{s}\n{s}")
        elif kind == "json":
            rows.append("输出合法 JSON：\n" + rng.choice(JSON_TEMPLATES))
        elif kind == "num":
            rows.append(f"数字或日期：{rng.choice(DATES)}")
        elif kind == "instr":
            rows.append(rng.choice(INSTRUCTIONS) + rng.choice(COPY_SRC))
        else:
            rows.append(rng.choice(COPY_SRC))
    return rows


def probes() -> list[dict]:
    return [
        {"probe_id": "PROBE-UTF-001", "family": "utf8_roundtrip", "input": "把「龙」字原样输出", "target": "龙"},
        {"probe_id": "PROBE-COPY-001", "family": "copy", "input": "原样抄写：厨房灯打开", "target": "厨房灯打开"},
        {"probe_id": "PROBE-NUM-001", "family": "number", "input": "三千五百写成阿拉伯数字", "target": "3500"},
        {"probe_id": "PROBE-DATE-001", "family": "date", "input": "二零二六年八月二十四日写成 ISO 日期", "target": "2026-08-24"},
        {"probe_id": "PROBE-JSON-001", "family": "json", "input": "空工具列表", "target": "[]"},
        {"probe_id": "PROBE-JSON-002", "family": "json", "input": "点头的 JSON", "target": '{"name":"nod","arguments":{}}'},
        {"probe_id": "PROBE-INSTR-001", "family": "instruction", "input": "只回答：收到", "target": "收到"},
    ]


def collect_leak_strings() -> list[str]:
    rows: list[dict] = []
    for path in all_eval_jsonl():
        rel = str(path)
        if "pretrain-probes" in rel:
            continue
        rows.extend(iter_jsonl(path))
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        rows.extend(iter_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
    else:
        rows.extend(probes())
    return leak_strings_from_rows(rows)


class ShardWriter:
    def __init__(self, out_dir: Path, split: str, docs_per_shard: int):
        self.out_dir = out_dir
        self.split = split
        self.docs_per_shard = max(1, docs_per_shard)
        self.index = 0
        self.count_in_shard = 0
        self.handle = None
        self.path: Path | None = None
        self.meta: list[dict] = []
        self.hasher = None

    def _open(self) -> None:
        self.path = self.out_dir / f"{self.split}-{self.index:04d}.jsonl"
        self.handle = self.path.open("w", encoding="utf-8")
        self.hasher = hashlib.sha256()
        self.count_in_shard = 0
        self.tokens_in_shard = 0

    def _close(self) -> None:
        if self.handle is None or self.path is None:
            return
        self.handle.close()
        self.meta.append(
            {
                "path": str(self.path.relative_to(ROOT)),
                "split": self.split,
                "n_docs": self.count_in_shard,
                "n_tokens": self.tokens_in_shard,
                "sha256": file_sha256(self.path),
            }
        )
        self.handle = None
        self.index += 1

    def write(self, row: dict) -> None:
        if self.handle is None:
            self._open()
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.handle.write(line)
        self.count_in_shard += 1
        self.tokens_in_shard += int(row.get("n_tokens") or 0)
        if self.count_in_shard >= self.docs_per_shard:
            self._close()

    def close(self) -> None:
        self._close()


UNK_TOKEN_MAX = 0.005


class TokenBinWriter:
    def __init__(self, out_dir: Path, split: str, tokens_per_shard: int, idx_dir: Path | None = None):
        self.out_dir = out_dir
        self.idx_dir = idx_dir or out_dir
        self.split = split
        self.tokens_per_shard = max(1, tokens_per_shard)
        self.idx = 0
        self.buf: list[int] = []
        self.n_docs = 0
        self.n_tokens = 0
        self.n_unk = 0
        self.meta: list[dict] = []
        self.index_rows: list[dict] = []

    def _flush(self) -> None:
        if not self.buf:
            return
        path = self.out_dir / f"{self.split}-{self.idx:04d}.bin"
        write_uint16_tokens(path, self.buf)
        self.idx_dir.mkdir(parents=True, exist_ok=True)
        idx_path = self.idx_dir / f"{self.split}-{self.idx:04d}.idx.jsonl"
        idx_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.index_rows),
            encoding="utf-8",
        )
        self.meta.append(
            {
                "path": str(path.relative_to(ROOT)),
                "index": str(idx_path.relative_to(ROOT)),
                "split": self.split,
                "n_docs": self.n_docs,
                "n_tokens": self.n_tokens,
                "n_unk": self.n_unk,
                "sha256": file_sha256(path),
            }
        )
        self.idx += 1
        self.buf = []
        self.n_docs = 0
        self.n_tokens = 0
        self.n_unk = 0
        self.index_rows = []

    def write(self, ids: list[int], row: dict, unk_id: int) -> None:
        if self.buf and len(self.buf) + len(ids) > self.tokens_per_shard:
            self._flush()
        offset = len(self.buf)
        self.buf.extend(ids)
        n_unk = sum(1 for t in ids if t == unk_id)
        self.n_docs += 1
        self.n_tokens += len(ids)
        self.n_unk += n_unk
        self.index_rows.append(
            {
                "offset": offset,
                "n_tokens": len(ids),
                "n_unk": n_unk,
                "page_id": row.get("page_id") or row.get("id") or "",
                "chunk_id": row.get("chunk_id", 0),
                "title": row.get("title") or "",
                "sha256": row.get("sha256")
                or hashlib.sha256(str(row.get("text") or "").encode("utf-8")).hexdigest(),
            }
        )

    def close(self) -> None:
        self._flush()


def iter_raw_records(raw_dir: Path):
    pages = sorted(raw_dir.glob("pages-*.jsonl"))
    if not pages:
        pages = list_raw_pages(raw_dir.parent)
    for path in pages:
        yield from iter_jsonl(path)


def build_from_raw(
    tok: ZhTokenizerV1,
    raw_dir: Path,
    *,
    tokens_per_shard: int,
    min_chars: int,
    valid_frac: float,
) -> dict:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw dir {raw_dir}")
    leaks = collect_leak_strings()
    token_dir = CORPUS_ZH_PRETRAIN / "tokens"
    idx_dir = LANGUAGE_WORK_V0 / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)
    for old in list(token_dir.glob("train-*.bin")) + list(token_dir.glob("valid-*.bin")):
        old.unlink()
    writers = {
        "train": TokenBinWriter(token_dir, "train", tokens_per_shard, idx_dir=idx_dir),
        "valid": TokenBinWriter(token_dir, "valid", max(1, tokens_per_shard // 4), idx_dir=idx_dir),
    }
    extract_man = {}
    em = LANGUAGE_WORK_V0 / "extract-manifest.json"
    if em.is_file():
        extract_man = json.loads(em.read_text(encoding="utf-8"))
    seen_hash: set[str] = set()
    n_in = n_keep = n_dup = n_short = n_leak = 0
    n_train = n_valid = 0
    tok_train = tok_valid = 0
    unk_train = unk_valid = 0
    n_chars = 0
    leak_examples: list[str] = []
    decode_checks: list[dict] = []
    for raw in iter_raw_records(raw_dir):
        n_in += 1
        text = normalize_document(str(raw.get("text") or ""))
        if len(text) < min_chars:
            n_short += 1
            continue
        h = str(raw.get("sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest())
        if h in seen_hash:
            n_dup += 1
            continue
        seen_hash.add(h)
        leak = document_leaks_eval(text, leaks)
        if leak is not None:
            n_leak += 1
            if len(leak_examples) < 8:
                leak_examples.append(leak[:80])
            continue
        split = hash_split_label(text, valid_frac=valid_frac)
        ids = tok.encode_document(text)
        n_tok = len(ids)
        n_unk = sum(1 for t in ids if t == tok.unk_id)
        n_chars += len(text)
        row = {
            "text": text,
            "page_id": str(raw.get("page_id") or raw.get("id") or ""),
            "chunk_id": raw.get("chunk_id", 0),
            "title": str(raw.get("title") or ""),
            "sha256": h,
        }
        writers[split].write(ids, row, tok.unk_id)
        if split == "valid":
            n_valid += 1
            tok_valid += n_tok
            unk_valid += n_unk
        else:
            n_train += 1
            tok_train += n_tok
            unk_train += n_unk
        n_keep += 1
        if len(decode_checks) < 8 and n_tok >= 8:
            mid = ids[2:-1] if len(ids) > 4 else ids
            decode_checks.append(
                {
                    "page_id": row["page_id"],
                    "roundtrip_prefix": tok.decode(mid)[:40],
                    "n_tokens": n_tok,
                }
            )
    writers["train"].close()
    writers["valid"].close()
    n_tokens = tok_train + tok_valid
    n_unk = unk_train + unk_valid
    unk_rate = (n_unk / n_tokens) if n_tokens else 0.0
    chars_per_token = (n_chars / n_tokens) if n_tokens else None
    return {
        "source": "zhwiki-full-raw",
        "raw_dir": str(raw_dir.relative_to(ROOT)) if raw_dir.is_relative_to(ROOT) else str(raw_dir),
        "dump_sha256": extract_man.get("dump_sha256"),
        "cleaner_id": extract_man.get("cleaner_id"),
        "extract_manifest": str(em.relative_to(ROOT)) if em.is_file() else None,
        "n_pages_in": n_in,
        "n_docs": n_keep,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_tokens": n_tokens,
        "n_train_tokens": tok_train,
        "n_valid_tokens": tok_valid,
        "n_unk_tokens": n_unk,
        "unk_token_rate": round(unk_rate, 8),
        "n_chars": n_chars,
        "chars_per_token": round(chars_per_token, 4) if chars_per_token else None,
        "dedup_dropped": n_dup,
        "short_dropped": n_short,
        "leak_dropped": n_leak,
        "leak_examples": leak_examples,
        "decode_checks": decode_checks,
        "tokenizer_sha256": tok.model_sha256,
        "token_shards": writers["train"].meta + writers["valid"].meta,
        "valid_frac": valid_frac,
        "min_chars": min_chars,
        "tokens_per_shard": tokens_per_shard,
        "binary": True,
        "byte_fallback": False,
        "unk_gate_ok": unk_rate <= UNK_TOKEN_MAX,
        "unk_token_max": UNK_TOKEN_MAX,
        "gap_to_100m": max(0, 100_000_000 - n_tokens),
        "gap_to_1b": max(0, 1_000_000_000 - n_tokens),
    }


def build_from_pages(
    tok: ZhTokenizerV1,
    pages: Path,
    *,
    count_only: bool,
    docs_per_shard: int,
    min_chars: int,
    valid_frac: float,
    target_tokens: int | None,
) -> dict:
    if not pages.is_file():
        raise FileNotFoundError(f"missing wiki pages {pages}")
    leaks = collect_leak_strings()
    shard_dir = LANGUAGE_WORK_V0 / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    writers = None
    if not count_only:
        for old in shard_dir.glob("train-*.jsonl"):
            old.unlink()
        for old in shard_dir.glob("valid-*.jsonl"):
            old.unlink()
        writers = {
            "train": ShardWriter(shard_dir, "train", docs_per_shard),
            "valid": ShardWriter(shard_dir, "valid", max(1, docs_per_shard // 4)),
        }
    seen_hash: set[str] = set()
    n_in = n_keep = n_dup = n_short = n_leak = 0
    n_train = n_valid = 0
    tok_train = tok_valid = 0
    leak_examples: list[str] = []
    for raw in iter_jsonl(pages):
        n_in += 1
        text = normalize_document(str(raw.get("text") or ""))
        if len(text) < min_chars:
            n_short += 1
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hash:
            n_dup += 1
            continue
        seen_hash.add(h)
        leak = document_leaks_eval(text, leaks)
        if leak is not None:
            n_leak += 1
            if len(leak_examples) < 8:
                leak_examples.append(leak[:80])
            continue
        split = hash_split_label(text, valid_frac=valid_frac)
        n_tok = len(tok.encode_document(text))
        if split == "valid":
            n_valid += 1
            tok_valid += n_tok
        else:
            n_train += 1
            tok_train += n_tok
        n_keep += 1
        if writers is not None:
            writers[split].write(
                {
                    "text": text,
                    "source": "zhwiki-pages",
                    "split": split,
                    "title": str(raw.get("title") or ""),
                    "page_id": str(raw.get("id") or ""),
                    "n_tokens": n_tok,
                }
            )
        total = tok_train + tok_valid
        if target_tokens is not None and total >= target_tokens and n_valid > 0 and n_train > 0:
            break
    if writers is not None:
        writers["train"].close()
        writers["valid"].close()
    n_tokens = tok_train + tok_valid
    shard_meta = []
    if writers is not None:
        shard_meta = writers["train"].meta + writers["valid"].meta
    man = {
        "pages": str(pages.relative_to(ROOT)) if pages.is_relative_to(ROOT) else str(pages),
        "pages_sha256": file_sha256(pages),
        "n_pages_in": n_in,
        "n_docs": n_keep,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_tokens": n_tokens,
        "n_train_tokens": tok_train,
        "n_valid_tokens": tok_valid,
        "dedup_dropped": n_dup,
        "short_dropped": n_short,
        "leak_dropped": n_leak,
        "leak_examples": leak_examples,
        "tokenizer_sha256": tok.model_sha256,
        "shards": shard_meta,
        "valid_frac": valid_frac,
        "min_chars": min_chars,
        "count_only": count_only,
        "byte_fallback": False,
        "gap_to_100m": max(0, 100_000_000 - n_tokens),
        "source": "zhwiki-pages.jsonl",
    }
    return man


def build_smoke(tok: ZhTokenizerV1, wiki_limit: int) -> dict:
    wiki = wiki_lines(wiki_limit)
    syn = synth(800)
    texts, n_dup = exact_dedup(wiki + syn)
    train_texts, valid_texts = hash_split(texts, valid_frac=0.05)
    train_rows = [{"text": t, "source": "zh-pretrain-v0", "split": "train"} for t in train_texts]
    valid_rows = [{"text": t, "source": "zh-pretrain-v0", "split": "valid"} for t in valid_texts]
    all_rows = train_rows + valid_rows
    n_tok = sum(len(tok.encode_document(t)) for t in texts)
    refuse_if_short("100m", n_tok, smoke=True)
    train_ids = [(f"train-{i}", tok.encode(t)) for i, t in enumerate(train_texts[:256])]
    valid_ids = [(f"valid-{i}", tok.encode(t)) for i, t in enumerate(valid_texts[:256])]
    near = token_near_dups(train_ids, valid_ids, threshold=0.9, limit=16)
    shard = CORPUS_ZH_PRETRAIN / "shards" / "smoke.jsonl"
    dump_jsonl(shard, all_rows)
    dump_jsonl(BANK_NEEDLE_PRETRAIN_PROBES, probes())
    stats = corpus_stats(tok, all_rows)
    man = {
        "shard": str(shard.relative_to(ROOT)),
        "n_docs": len(all_rows),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "wiki_n": len(wiki),
        "synth_n": len(syn),
        "dedup_dropped": n_dup,
        "n_tokens": stats["n_tokens"],
        "sources": stats["sources"],
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        "tokenizer_sha256": tok.model_sha256,
        "token_near_dups": near,
        "probes": str(BANK_NEEDLE_PRETRAIN_PROBES.relative_to(ROOT)),
        "rung": "smoke",
        "smoke": True,
        "byte_fallback": False,
        "gap_to_100m": max(0, 100_000_000 - stats["n_tokens"]),
    }
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rung", default="100m", choices=["pilot-1m", "pilot-5m", "100m", "300m", "1b"])
    ap.add_argument("--wiki-lines", type=int, default=20000)
    ap.add_argument("--pages", type=Path, default=PAGES_DEFAULT)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--docs-per-shard", type=int, default=20000)
    ap.add_argument("--min-chars", type=int, default=32)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--target-tokens", type=int, default=None)
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--raw-dir", type=Path, default=LANGUAGE_WORK_V0 / "raw")
    ap.add_argument("--tokens-per-shard", type=int, default=8_000_000)
    args = ap.parse_args()
    tok = ZhTokenizerV1()
    CORPUS_ZH_PRETRAIN.mkdir(parents=True, exist_ok=True)
    dump_jsonl(BANK_NEEDLE_PRETRAIN_PROBES, probes())
    if args.smoke:
        man = build_smoke(tok, 2000 if args.smoke else args.wiki_lines)
    else:
        raw_dir = args.raw_dir
        use_raw = args.from_raw or bool(list(raw_dir.glob("pages-*.jsonl")))
        try:
            if use_raw and not args.count_only:
                man = build_from_raw(
                    tok,
                    raw_dir,
                    tokens_per_shard=args.tokens_per_shard,
                    min_chars=args.min_chars,
                    valid_frac=args.valid_frac,
                )
            else:
                man = build_from_pages(
                    tok,
                    args.pages,
                    count_only=args.count_only,
                    docs_per_shard=args.docs_per_shard,
                    min_chars=args.min_chars,
                    valid_frac=args.valid_frac,
                    target_tokens=args.target_tokens,
                )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        man["rung"] = args.rung
        man["smoke"] = False
        if man.get("binary") and man.get("unk_gate_ok") is False:
            man["budget_ok"] = False
            (CORPUS_ZH_PRETRAIN / "manifest.json").write_text(
                json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(man, ensure_ascii=False, indent=2))
            print(
                f"UNK token rate {man.get('unk_token_rate')} exceeds {UNK_TOKEN_MAX}; "
                "do not start official training; tokenizer v2 is not auto-built.",
                file=sys.stderr,
            )
            return 3
        try:
            refuse_if_short(args.rung, int(man["n_tokens"]), smoke=False)
            man["budget_ok"] = True
        except ValueError as exc:
            man["budget_ok"] = False
            man["budget_error"] = str(exc)
            (CORPUS_ZH_PRETRAIN / "manifest.json").write_text(
                json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(man, ensure_ascii=False, indent=2))
            print(str(exc), file=sys.stderr)
            # Unique wiki may be short of a named rung; record the gap and continue
            # so later unique-token training can still consume the real corpus.
            if use_raw:
                return 0
            return 2
    (CORPUS_ZH_PRETRAIN / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
