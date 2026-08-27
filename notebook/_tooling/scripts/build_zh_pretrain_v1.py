#!/usr/bin/env python3
"""Assemble zh-pretrain-v1: frozen wiki v0 + FineWeb2-HQ + SchemaStore + OAI examples.

Does not rewrite zh-pretrain-v0. Wiki shards are referenced via mix.json, not copied.
Raises --rung 1b/3b by consuming more FineWeb2-HQ cmn_Hani parquet files from the same cursor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_PRETRAIN_PROBES,
    CORPUS_ZH_PRETRAIN,
    CORPUS_ZH_PRETRAIN_V1,
    LANGUAGE_WORK_V0,
    LANGUAGE_WORK_V1,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    all_eval_jsonl,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from data import (  # noqa: E402
    document_leaks_eval,
    file_sha256,
    iter_jsonl,
    leak_strings_from_rows,
    normalize_document,
    sha256_text,
    write_uint16_tokens,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

HF_REPO = "epfml/FineWeb2-HQ"
HF_SUBSET = "cmn_Hani"
WIKI_TRAIN_TOKENS = 649_904_474
WIKI_VALID_TOKENS = 34_610_229
UNK_TOKEN_MAX = 0.005
UNK_DOC_MAX = 0.05
MIN_CHARS = 64
MAX_CHARS = 100_000
CHUNK_CHARS = 4096
MAX_SCHEMA_CHARS = 32_768
TOKENS_PER_SHARD = 8_000_000
CJK_RE = re.compile(r"[\u3400-\u9fff]")
RUNG_TRAIN = {
    "1b": 1_000_000_000,
    "3b": 3_000_000_000,
}
OAI_FILES = (
    "api-with-examples.yaml",
    "petstore.yaml",
    "callback-example.yaml",
)
OAI_URL = (
    "https://raw.githubusercontent.com/OAI/OpenAPI-Specification/3.0.3/examples/v3.0/{name}"
)
PROBE_HQ = ROOT / "notebook/archive/corpus/_probe/cache/fineweb2-hq-cmn" / HF_SUBSET
PROBE_SCHEMA = ROOT / "notebook/archive/corpus/_probe/cache/github/schemastore/src/schemas/json"
PROBE_OAI = ROOT / "notebook/archive/corpus/_probe/cache/oai-examples"
PROBE_WIKI_HASHES = ROOT / "notebook/archive/corpus/_probe/wiki-hashes/wiki-norm.sha256"


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def collect_leak_strings() -> list[str]:
    rows: list[dict] = []
    for path in all_eval_jsonl():
        if "pretrain-probes" in str(path):
            continue
        rows.extend(iter_jsonl(path))
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        rows.extend(iter_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
    return leak_strings_from_rows(rows)


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


def load_wiki_hashes() -> set[str]:
    if PROBE_WIKI_HASHES.is_file():
        return load_hash_file(PROBE_WIKI_HASHES)
    hashes: set[str] = set()
    token_dir = CORPUS_ZH_PRETRAIN / "tokens"
    idx_dir = LANGUAGE_WORK_V0 / "tokens"
    for idx in idx_dir.glob("*.idx.jsonl"):
        for row in iter_jsonl(idx):
            h = str(row.get("sha256") or "")
            if h:
                hashes.add(h)
    if not hashes:
        raise FileNotFoundError(
            f"missing wiki hashes at {PROBE_WIKI_HASHES} and empty {idx_dir}"
        )
    return hashes


def list_wiki_bins(split: str) -> list[Path]:
    token_dir = CORPUS_ZH_PRETRAIN / "tokens"
    return sorted(token_dir.glob(f"{split}-*.bin"))


def next_shard_idx(token_dir: Path, prefix: str) -> int:
    existing = sorted(token_dir.glob(f"{prefix}-*.bin"))
    if not existing:
        return 0
    stem = existing[-1].stem
    return int(stem.split("-")[-1]) + 1


def shard_token_count(bin_path: Path) -> int:
    idx = bin_path.with_name(bin_path.stem + ".idx.jsonl")
    if not idx.is_file():
        return bin_path.stat().st_size // 2
    return sum(int(row.get("n_tokens") or 0) for row in iter_jsonl(idx))


class TokenBinWriter:
    def __init__(self, out_dir: Path, split: str, tokens_per_shard: int, start_idx: int = 0, idx_dir: Path | None = None):
        self.out_dir = out_dir
        self.idx_dir = idx_dir or out_dir
        self.split = split
        self.tokens_per_shard = max(1, tokens_per_shard)
        self.idx = start_idx
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
                "path": rel(path),
                "index": rel(idx_path),
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
                "sha256": row.get("sha256") or "",
            }
        )

    def flush(self) -> None:
        self._flush()

    def close(self) -> None:
        self._flush()


def iter_chunks(text: str, size: int = CHUNK_CHARS, min_chars: int = MIN_CHARS):
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
            if cut > start + min_chars:
                end = cut
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            yield chunk
        if end <= start:
            break
        start = end


def iter_parquet_texts(path: Path):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    text_col = next((c for c in ("text", "content", "document") if c in names), None)
    if text_col is None:
        raise ValueError(f"no text column in {path}: {names}")
    for batch in pf.iter_batches(batch_size=256, columns=[text_col]):
        col = batch.column(text_col)
        for i in range(batch.num_rows):
            val = col[i].as_py()
            if isinstance(val, str) and val.strip():
                yield val


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


def parquet_path(raw_hq: Path, name: str) -> Path:
    return raw_hq / HF_SUBSET / name


def ensure_parquet(raw_hq: Path, name: str) -> Path:
    dest = parquet_path(raw_hq, name)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    probe = PROBE_HQ / name
    if probe.is_file():
        hardlink_or_copy(probe, dest)
        return dest
    from huggingface_hub import hf_hub_download

    print(f"download {HF_REPO} {HF_SUBSET}/{name}", flush=True)
    hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"{HF_SUBSET}/{name}",
        local_dir=str(raw_hq),
    )
    if not dest.is_file():
        raise FileNotFoundError(f"download missed {dest}")
    return dest


def list_hq_names() -> list[str]:
    fallback = [f"000_{i:05d}.parquet" for i in range(975)]
    try:
        from huggingface_hub import list_repo_files

        files = [
            Path(f).name
            for f in list_repo_files(HF_REPO, repo_type="dataset")
            if f.startswith(f"{HF_SUBSET}/") and f.endswith(".parquet")
        ]
        return sorted(files) if files else fallback
    except Exception as exc:
        print(f"list_repo_files fallback: {exc}", flush=True)
        return fallback


def ensure_schemastore(dest_json: Path) -> Path:
    if dest_json.is_dir() and any(dest_json.glob("*.json")):
        return dest_json
    if PROBE_SCHEMA.is_dir() and any(PROBE_SCHEMA.glob("*.json")):
        dest_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PROBE_SCHEMA, dest_json, dirs_exist_ok=True)
        return dest_json
    raise FileNotFoundError(
        f"SchemaStore json missing at {dest_json} and {PROBE_SCHEMA}; "
        "sparse-clone SchemaStore src/schemas/json first"
    )


def ensure_oai(dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name in OAI_FILES:
        dest = dest_dir / name
        if not dest.is_file():
            probe = PROBE_OAI / name
            if probe.is_file():
                shutil.copy2(probe, dest)
            else:
                import urllib.request

                url = OAI_URL.format(name=name)
                print(f"download {url}", flush=True)
                urllib.request.urlretrieve(url, dest)
        out.append(dest)
    return out


def append_hash(path: Path, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(digest + "\n")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        "n_leak": 0,
        "n_unk_drop": 0,
        "n_tokens": 0,
        "n_unk": 0,
        "n_chars": 0,
    }


class Ingestor:
    def __init__(
        self,
        tok: ZhTokenizerV1,
        writer: TokenBinWriter,
        seen: set[str],
        wiki: set[str],
        leaks: list[str],
        hash_path: Path,
        *,
        require_cjk: bool,
        max_chars: int,
    ):
        self.tok = tok
        self.writer = writer
        self.seen = seen
        self.wiki = wiki
        self.leaks = leaks
        self.hash_path = hash_path
        self.require_cjk = require_cjk
        self.max_chars = max_chars
        self.stats = empty_stats()

    def accept_text(self, text: str, *, page_id: str, title: str) -> int:
        stats = self.stats
        stats["n_in"] += 1
        raw = str(text or "")
        if len(raw) > self.max_chars:
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
            ids = self.tok.encode_document(chunk)
            if len(ids) < 4:
                stats["n_short"] += 1
                continue
            n_unk = sum(1 for t in ids if t == self.tok.unk_id)
            if n_unk / len(ids) > UNK_DOC_MAX:
                stats["n_unk_drop"] += 1
                continue
            self.seen.add(digest)
            append_hash(self.hash_path, digest)
            self.writer.write(
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
            added += len(ids)
        return added


def write_mix(
    work: Path,
    *,
    wiki_train: list[Path],
    wiki_valid: list[Path],
    extra_train: list[Path],
    n_train_tokens: int,
    n_valid_tokens: int,
    n_wiki_train: int,
    n_supplement: int,
    target: int,
    cursor: dict,
    sources: dict,
) -> dict:
    mix = {
        "id": "zh-pretrain-v1",
        "tokenizer": "zh-24k-v1",
        "valid_source": "zh-pretrain-v0 wiki valid (frozen, for 100M/300M curve comparability)",
        "n_train_tokens": n_train_tokens,
        "n_valid_tokens": n_valid_tokens,
        "n_wiki_train_tokens": n_wiki_train,
        "n_supplement_train_tokens": n_supplement,
        "target_train_tokens": target,
        "reached_target": n_train_tokens >= target,
        "train_shards": [rel(p) for p in wiki_train + extra_train],
        "valid_shards": [rel(p) for p in wiki_valid],
        "sources": sources,
        "cursor": rel(work / "ingest-cursor.json"),
    }
    dump_json(work / "mix.json", mix)
    dump_json(work / "ingest-cursor.json", cursor)
    return mix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=["1b", "3b"], default="1b")
    ap.add_argument("--target-train-tokens", type=int, default=None)
    ap.add_argument("--max-parquets", type=int, default=None)
    ap.add_argument("--out", type=Path, default=CORPUS_ZH_PRETRAIN_V1)
    ap.add_argument("--work", type=Path, default=LANGUAGE_WORK_V1)
    args = ap.parse_args()

    target = int(args.target_train_tokens or RUNG_TRAIN[args.rung])
    corpus = args.out if args.out.is_absolute() else ROOT / args.out
    work = args.work if args.work.is_absolute() else ROOT / args.work
    wiki_train = list_wiki_bins("train")
    wiki_valid = list_wiki_bins("valid")
    if not wiki_train or not wiki_valid:
        print(f"missing frozen v0 shards under {CORPUS_ZH_PRETRAIN / 'tokens'}", file=sys.stderr)
        return 2

    tok = ZhTokenizerV1()
    leaks = collect_leak_strings()
    wiki = load_wiki_hashes()
    print(f"wiki hashes {len(wiki)} leak_strings {len(leaks)}", flush=True)

    token_dir = corpus / "tokens"
    idx_dir = work / "tokens"
    raw_hq = work / "raw" / "fineweb2-hq"
    raw_schema = work / "raw" / "schemastore" / "src" / "schemas" / "json"
    raw_oai = work / "raw" / "oai-examples"
    hash_path = work / "hashes" / "supplement.sha256"
    token_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)
    (work / "hashes").mkdir(parents=True, exist_ok=True)

    cursor_path = work / "ingest-cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8")) if cursor_path.is_file() else {}
    seen = set(wiki)
    seen |= load_hash_file(hash_path)

    stats_path = work / "ingest-stats.json"
    stats_all = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}

    def recount(prefix: str) -> int:
        return sum(shard_token_count(p) for p in sorted(token_dir.glob(f"{prefix}-*.bin")))

    cursor["n_schema_train_tokens"] = recount("schema-train")
    cursor["n_oai_train_tokens"] = recount("oai-train")
    cursor["n_hq_train_tokens"] = recount("hq-train")
    n_supplement = (
        int(cursor["n_schema_train_tokens"])
        + int(cursor["n_oai_train_tokens"])
        + int(cursor["n_hq_train_tokens"])
    )
    n_train = WIKI_TRAIN_TOKENS + n_supplement
    t0 = time.time()

    def extra_bins() -> list[Path]:
        return (
            sorted(token_dir.glob("schema-train-*.bin"))
            + sorted(token_dir.glob("oai-train-*.bin"))
            + sorted(token_dir.glob("hq-train-*.bin"))
        )

    def persist(extra_note: dict | None = None) -> dict:
        sources = {
            "wiki_v0": {
                "band": "A",
                "n_train_tokens": WIKI_TRAIN_TOKENS,
                "path": rel(CORPUS_ZH_PRETRAIN),
            },
            "fineweb2_hq_cmn_hani": {
                "band": "C",
                "license": "ODC-By-1.0 + Common-Crawl-ToU",
                "repo": HF_REPO,
                "subset": HF_SUBSET,
                "consumed_parquets": cursor.get("consumed_parquets") or [],
                "n_train_tokens": int(cursor.get("n_hq_train_tokens") or 0),
            },
            "schemastore_json": {
                "band": "B",
                "n_train_tokens": int(cursor.get("n_schema_train_tokens") or 0),
            },
            "oai_openapi_examples_3_0_3": {
                "band": "B",
                "n_train_tokens": int(cursor.get("n_oai_train_tokens") or 0),
            },
        }
        payload = {
            "rung": args.rung,
            "target_train_tokens": target,
            "consumed_parquets": cursor.get("consumed_parquets") or [],
            "schema_done": bool(cursor.get("schema_done")),
            "oai_done": bool(cursor.get("oai_done")),
            "n_wiki_train_tokens": WIKI_TRAIN_TOKENS,
            "n_hq_train_tokens": int(cursor.get("n_hq_train_tokens") or 0),
            "n_schema_train_tokens": int(cursor.get("n_schema_train_tokens") or 0),
            "n_oai_train_tokens": int(cursor.get("n_oai_train_tokens") or 0),
            "n_supplement_train_tokens": n_supplement,
            "n_train_tokens": n_train,
            "n_valid_tokens": WIKI_VALID_TOKENS,
            "next_hq_shard_idx": next_shard_idx(token_dir, "hq-train"),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if extra_note:
            payload.update(extra_note)
        cursor.update(payload)
        mix = write_mix(
            work,
            wiki_train=wiki_train,
            wiki_valid=wiki_valid,
            extra_train=extra_bins(),
            n_train_tokens=n_train,
            n_valid_tokens=WIKI_VALID_TOKENS,
            n_wiki_train=WIKI_TRAIN_TOKENS,
            n_supplement=n_supplement,
            target=target,
            cursor=cursor,
            sources=sources,
        )
        dump_json(stats_path, stats_all)
        return mix

    if not cursor.get("schema_done"):
        schema_dir = ensure_schemastore(raw_schema)
        writer = TokenBinWriter(
            token_dir, "schema-train", TOKENS_PER_SHARD, start_idx=next_shard_idx(token_dir, "schema-train"), idx_dir=idx_dir
        )
        ing = Ingestor(
            tok, writer, seen, wiki, leaks, hash_path, require_cjk=False, max_chars=MAX_SCHEMA_CHARS
        )
        files = sorted(p for p in schema_dir.glob("*.json") if p.is_file())
        print(f"schema files {len(files)}", flush=True)
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            ing.accept_text(text, page_id=path.stem, title=path.name)
        writer.close()
        cursor["schema_done"] = True
        cursor["n_schema_train_tokens"] = recount("schema-train")
        n_supplement = (
            int(cursor["n_schema_train_tokens"])
            + int(cursor.get("n_oai_train_tokens") or 0)
            + int(cursor.get("n_hq_train_tokens") or 0)
        )
        n_train = WIKI_TRAIN_TOKENS + n_supplement
        stats_all["schema"] = ing.stats
        persist()
        print(json.dumps({"schema": ing.stats}, ensure_ascii=False), flush=True)

    if not cursor.get("oai_done"):
        oai_files = ensure_oai(raw_oai)
        writer = TokenBinWriter(
            token_dir, "oai-train", TOKENS_PER_SHARD, start_idx=next_shard_idx(token_dir, "oai-train"), idx_dir=idx_dir
        )
        ing = Ingestor(
            tok, writer, seen, wiki, leaks, hash_path, require_cjk=False, max_chars=MAX_SCHEMA_CHARS
        )
        for path in oai_files:
            text = path.read_text(encoding="utf-8", errors="replace")
            ing.accept_text(text, page_id=path.stem, title=path.name)
        writer.close()
        cursor["oai_done"] = True
        cursor["n_oai_train_tokens"] = recount("oai-train")
        n_supplement = (
            int(cursor.get("n_schema_train_tokens") or 0)
            + int(cursor["n_oai_train_tokens"])
            + int(cursor.get("n_hq_train_tokens") or 0)
        )
        n_train = WIKI_TRAIN_TOKENS + n_supplement
        stats_all["oai"] = ing.stats
        persist()
        print(json.dumps({"oai": ing.stats}, ensure_ascii=False), flush=True)

    consumed = list(cursor.get("consumed_parquets") or [])
    names = list_hq_names()
    pending = [n for n in names if n not in consumed]
    writer = TokenBinWriter(
        token_dir, "hq-train", TOKENS_PER_SHARD, start_idx=next_shard_idx(token_dir, "hq-train"), idx_dir=idx_dir
    )
    n_parquet = 0
    try:
        for name in pending:
            if n_train >= target:
                break
            if args.max_parquets is not None and n_parquet >= args.max_parquets:
                break
            path = ensure_parquet(raw_hq, name)
            print(
                f"ingest {name} size={path.stat().st_size} train={n_train} target={target}",
                flush=True,
            )
            ing = Ingestor(
                tok, writer, seen, wiki, leaks, hash_path, require_cjk=True, max_chars=MAX_CHARS
            )
            for i, text in enumerate(iter_parquet_texts(path), start=1):
                ing.accept_text(text, page_id=f"{name}:{i}", title=name)
                if i % 5000 == 0:
                    print(
                        f"  {name} docs={i} keep={ing.stats['n_keep']} tok={ing.stats['n_tokens']}",
                        flush=True,
                    )
            consumed.append(name)
            cursor["consumed_parquets"] = consumed
            writer.flush()
            cursor["n_hq_train_tokens"] = recount("hq-train")
            n_supplement = (
                int(cursor.get("n_schema_train_tokens") or 0)
                + int(cursor.get("n_oai_train_tokens") or 0)
                + int(cursor["n_hq_train_tokens"])
            )
            n_train = WIKI_TRAIN_TOKENS + n_supplement
            stats_all[name] = ing.stats
            persist()
            n_parquet += 1
            print(json.dumps({"parquet": name, **ing.stats, "n_train_tokens": n_train}, ensure_ascii=False), flush=True)
    finally:
        writer.close()
        cursor["n_hq_train_tokens"] = recount("hq-train")
        n_supplement = (
            int(cursor.get("n_schema_train_tokens") or 0)
            + int(cursor.get("n_oai_train_tokens") or 0)
            + int(cursor["n_hq_train_tokens"])
        )
        n_train = WIKI_TRAIN_TOKENS + n_supplement
        persist()

    extra = extra_bins()
    n_unk = 0
    n_tok_extra = 0
    for path in extra:
        idx = idx_dir / f"{path.stem}.idx.jsonl"
        if not idx.is_file():
            continue
        for row in iter_jsonl(idx):
            n_unk += int(row.get("n_unk") or 0)
            n_tok_extra += int(row.get("n_tokens") or 0)
    wiki_unk = 295_816
    n_tokens_all = n_train + WIKI_VALID_TOKENS
    unk_rate = (wiki_unk + n_unk) / n_tokens_all if n_tokens_all else 0.0
    man = {
        "id": "zh-pretrain-v1",
        "source": "wiki-v0 + fineweb2-hq-cmn_hani + schemastore + oai-3.0.3",
        "wiki_manifest": rel(CORPUS_ZH_PRETRAIN / "manifest.json"),
        "n_wiki_train_tokens": WIKI_TRAIN_TOKENS,
        "n_valid_tokens": WIKI_VALID_TOKENS,
        "n_supplement_train_tokens": n_supplement,
        "n_train_tokens": n_train,
        "n_tokens": n_tokens_all,
        "n_unk_tokens_supplement": n_unk,
        "unk_token_rate_approx": round(unk_rate, 8),
        "unk_gate_ok": unk_rate <= UNK_TOKEN_MAX,
        "unk_token_max": UNK_TOKEN_MAX,
        "tokenizer_sha256": tok.model_sha256,
        "binary": True,
        "mix": rel(work / "mix.json"),
        "reached_target": n_train >= target,
        "target_train_tokens": target,
        "rung": args.rung,
        "gap_to_target": max(0, target - n_train),
        "supplement_shards": [rel(p) for p in extra],
        "consumed_parquets": cursor.get("consumed_parquets") or [],
    }
    dump_json(corpus / "manifest.json", man)
    persist()
    print(json.dumps({
        "n_train_tokens": n_train,
        "n_supplement_train_tokens": n_supplement,
        "reached_target": n_train >= target,
        "gap_to_target": max(0, target - n_train),
        "unk_token_rate_approx": man["unk_token_rate_approx"],
        "consumed_parquets": cursor.get("consumed_parquets") or [],
        "elapsed_s": round(time.time() - t0, 1),
    }, ensure_ascii=False), flush=True)
    if n_train < target:
        print(f"short of {args.rung} unique train: {n_train} < {target}", file=sys.stderr)
        return 2
    if unk_rate > UNK_TOKEN_MAX:
        print(f"UNK token rate {unk_rate} exceeds {UNK_TOKEN_MAX}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
