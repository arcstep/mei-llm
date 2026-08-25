#!/usr/bin/env python3
"""One-shot Needle-zh corpus source probe. Does not modify the v0 builder or trainer.

Usage:
  python3 scripts/probe_zh_corpus_sources.py index-wiki
  python3 scripts/probe_zh_corpus_sources.py analyze --source-id NAME --path FILE
  python3 scripts/probe_zh_corpus_sources.py summarize
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "tasks/needle-zh/model"
for p in (SCRIPTS, MODEL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from repo_paths import CORPUS_ZH_PRETRAIN, TOKENIZER_ZH_V1  # noqa: E402
from data import (  # noqa: E402
    document_leaks_eval,
    leak_strings_from_rows,
    normalize_document,
    sha256_text,
    token_jaccard,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

PROBE_ROOT = ROOT / "corpora/_probe"
CARDS = PROBE_ROOT / "cards"
WIKI_HASH_DIR = PROBE_ROOT / "wiki-hashes"
UNK_ID = 3
UNK_TOKEN_MAX = 0.005
CJK_RE = re.compile(r"[\u3400-\u9fff]")
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
URL_RE = re.compile(r"https?://[^\s\"']+", re.I)
NAV_RE = re.compile(r"(点击这里|免责声明|版权所有|登录|注册|首页\s*[>|»]|Cookie|javascript:)", re.I)
JSONISH_RE = re.compile(r"^\s*[\[{]")
CMD_RE = re.compile(r"(帮我|请|不要|别|取消|查询|打开|关闭|设置|预约|呼叫)")
NEG_RE = re.compile(r"(不要|别|不用|取消|不是|并未|没有)")
ELLIPSIS_RE = re.compile(r"(那个|这[个次]|刚才|上次|还是|同上)")


def load_eval_leaks() -> list[str]:
    from repo_paths import all_eval_jsonl, BANK_NEEDLE_PRETRAIN_PROBES
    from data import iter_jsonl

    rows: list[dict] = []
    for path in all_eval_jsonl():
        if "pretrain-probes" in str(path):
            continue
        rows.extend(iter_jsonl(path))
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        rows.extend(iter_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
    return leak_strings_from_rows(rows)


def index_wiki() -> dict[str, Any]:
    raw_dir = CORPUS_ZH_PRETRAIN / "raw"
    files = sorted(raw_dir.glob("pages-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"missing wiki raw shards in {raw_dir}")
    WIKI_HASH_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = WIKI_HASH_DIR / "wiki-raw.sha256"
    norm_path = WIKI_HASH_DIR / "wiki-norm.sha256"
    n = 0
    with raw_path.open("w") as raw_f, norm_path.open("w") as norm_f:
        for path in files:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = str(row.get("text") or "")
                    raw = str(row.get("sha256") or sha256_text(text))
                    raw_f.write(raw + "\n")
                    norm_f.write(sha256_text(normalize_document(text)) + "\n")
                    n += 1
                    if n % 200_000 == 0:
                        print(f"indexed {n} wiki chunks", flush=True)
    man = {"n_chunks": n, "raw": str(raw_path.relative_to(ROOT)), "norm": str(norm_path.relative_to(ROOT))}
    (WIKI_HASH_DIR / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=2))
    return man


def load_hash_set(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            h = line.strip()
            if h:
                out.add(h)
    return out


def iter_json_blob(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            yield s
        return
    if isinstance(obj, dict):
        sample_vals = list(obj.values())[:5]
        if sample_vals and all(isinstance(v, dict) and ("messages" in v or "dialogue" in v) for v in sample_vals):
            for v in obj.values():
                yield from iter_json_blob(v)
            return
        for key in ("text", "content", "document", "data", "body", "utterance", "query", "instruction"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip() and key != "data":
                yield val
                return
        if isinstance(obj.get("data"), str) and obj["data"].strip():
            yield obj["data"]
            return
        msgs = obj.get("messages") or obj.get("dialogue") or obj.get("utterances") or obj.get("turns")
        if isinstance(msgs, list) and msgs:
            parts = []
            for m in msgs:
                if isinstance(m, str):
                    parts.append(m)
                elif isinstance(m, dict):
                    role = str(m.get("role") or m.get("speaker") or "").lower()
                    if role in {"sys", "system", "bot", "assistant"}:
                        continue
                    for k in ("text", "utterance", "content", "message"):
                        if isinstance(m.get(k), str):
                            parts.append(m[k])
                            break
            if parts:
                yield "\n".join(parts)
                return
        dumped = json.dumps(obj, ensure_ascii=False)
        if dumped not in ("{}", "[]", "null"):
            yield dumped
        return
    if isinstance(obj, list):
        for item in obj:
            yield from iter_json_blob(item)


def iter_jsonl_texts(path: Path, *, max_bytes: int | None = None) -> Iterator[str]:
    consumed = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            consumed += len(line.encode("utf-8", errors="replace"))
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if line:
                    yield line
                continue
            for text in iter_json_blob(obj):
                yield text
            if max_bytes is not None and consumed >= max_bytes:
                return


def iter_parquet_texts(path: Path, *, max_docs: int | None = None) -> Iterator[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    text_col = next((c for c in ("text", "content", "document", "data") if c in names), None)
    cols_wanted = [text_col] if text_col else None
    seen = 0
    for batch in pf.iter_batches(batch_size=256, columns=cols_wanted):
        cols = {name: batch.column(name) for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            if text_col:
                val = cols[text_col][i].as_py()
                if isinstance(val, str) and val.strip():
                    yield val
                elif val is not None:
                    yield from iter_json_blob(val)
            else:
                row = {name: cols[name][i].as_py() for name in cols}
                yield from iter_json_blob(row)
            seen += 1
            if max_docs is not None and seen >= max_docs:
                return


def iter_json_file_texts(path: Path) -> Iterator[str]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    yield from iter_json_blob(obj)


def iter_plain_or_yaml(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        yield text


def iter_source_texts(path: Path, *, max_bytes: int | None, max_docs: int | None) -> Iterator[str]:
    if path.is_dir():
        files = []
        for ext in ("*.jsonl", "*.json", "*.yaml", "*.yml"):
            files.extend(path.rglob(ext))
        files = [p for p in files if p.is_file() and ".git" not in p.parts]
        random.Random(0).shuffle(files)
        n = 0
        for fp in files:
            for text in iter_source_texts(fp, max_bytes=None, max_docs=None):
                yield text
                n += 1
                if max_docs is not None and n >= max_docs:
                    return
        return
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from iter_parquet_texts(path, max_docs=max_docs)
        return
    if suffix == ".jsonl":
        n = 0
        for text in iter_jsonl_texts(path, max_bytes=max_bytes):
            yield text
            n += 1
            if max_docs is not None and n >= max_docs:
                return
        return
    if suffix == ".json":
        n = 0
        for text in iter_json_file_texts(path):
            yield text
            n += 1
            if max_docs is not None and n >= max_docs:
                return
        return
    if suffix in {".yaml", ".yml", ".md", ".txt"}:
        yield from iter_plain_or_yaml(path)
        return
    raise ValueError(f"unsupported probe path {path}")


def char_class_stats(text: str) -> dict[str, int]:
    n_cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    n_lat = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    n_dig = sum(1 for ch in text if ch.isdigit())
    return {"n_cjk": n_cjk, "n_latin": n_lat, "n_digit": n_dig}


def heuristic_flags(text: str) -> dict[str, bool]:
    return {
        "nav_or_ad": bool(NAV_RE.search(text)),
        "email": bool(EMAIL_RE.search(text)),
        "phone": bool(PHONE_RE.search(text)),
        "has_url": bool(URL_RE.search(text)),
        "jsonish": bool(JSONISH_RE.match(text)),
        "commandish": bool(CMD_RE.search(text)),
        "negation": bool(NEG_RE.search(text)),
        "ellipsisish": bool(ELLIPSIS_RE.search(text)),
        "garbled": text.count("�") > 0 or (len(text) > 80 and CJK_RE.search(text) is None and "{" not in text[:40]),
    }


def stratified_sample(rows: list[dict[str, Any]], k: int, seed: int = 0) -> list[dict[str, Any]]:
    if len(rows) <= k:
        return rows
    lens = [r["n_chars"] for r in rows]
    qs = [sorted(lens)[max(0, int(len(lens) * q) - 1)] for q in (0.33, 0.66)]
    buckets = {"short": [], "mid": [], "long": []}
    for r in rows:
        if r["n_chars"] <= qs[0]:
            buckets["short"].append(r)
        elif r["n_chars"] <= qs[1]:
            buckets["mid"].append(r)
        else:
            buckets["long"].append(r)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    per = max(1, k // 3)
    for name in ("short", "mid", "long"):
        pool = buckets[name]
        rng.shuffle(pool)
        out.extend(pool[:per])
    if len(out) < k:
        rest = [r for r in rows if r not in out]
        rng.shuffle(rest)
        out.extend(rest[: k - len(out)])
    return out[:k]


def analyze(source_id: str, path: Path, *, max_bytes: int | None, max_docs: int | None, sample_review: int) -> dict[str, Any]:
    tok = ZhTokenizerV1()
    leaks = load_eval_leaks()
    wiki_raw = load_hash_set(WIKI_HASH_DIR / "wiki-raw.sha256") if (WIKI_HASH_DIR / "wiki-raw.sha256").is_file() else set()
    wiki_norm = load_hash_set(WIKI_HASH_DIR / "wiki-norm.sha256") if (WIKI_HASH_DIR / "wiki-norm.sha256").is_file() else set()

    n_docs = 0
    n_chars = 0
    n_tokens = 0
    n_unk = 0
    n_cjk = n_latin = n_digit = 0
    exact = set()
    n_exact_dup = 0
    n_short = 0
    n_jsonish = 0
    n_leak = 0
    n_wiki_raw = 0
    n_wiki_norm = 0
    length_hist = Counter()
    flag_hist = Counter()
    stored: list[dict[str, Any]] = []
    near_left: list[tuple[str, list[int]]] = []

    for text in iter_source_texts(path, max_bytes=max_bytes, max_docs=max_docs):
        raw = str(text or "")
        if not raw.strip():
            continue
        norm = normalize_document(raw)
        ids = tok.encode_document(norm)
        n_tok = len(ids)
        n_docs += 1
        n_chars += len(norm)
        n_tokens += n_tok
        n_unk += sum(1 for i in ids if i == UNK_ID)
        cc = char_class_stats(norm)
        n_cjk += cc["n_cjk"]
        n_latin += cc["n_latin"]
        n_digit += cc["n_digit"]
        h_raw = sha256_text(raw)
        h_norm = sha256_text(norm)
        if h_norm in exact:
            n_exact_dup += 1
        else:
            exact.add(h_norm)
        if len(norm) < 256:
            n_short += 1
        if JSONISH_RE.match(norm):
            n_jsonish += 1
        if document_leaks_eval(norm, leaks):
            n_leak += 1
        if h_raw in wiki_raw or h_norm in wiki_raw:
            n_wiki_raw += 1
        if h_norm in wiki_norm:
            n_wiki_norm += 1
        bin_edge = 256
        while bin_edge < len(norm) and bin_edge < 32768:
            bin_edge *= 2
        length_hist[f"le_{bin_edge}" if len(norm) <= 32768 else "gt_32768"] += 1
        flags = heuristic_flags(norm)
        for k, v in flags.items():
            if v:
                flag_hist[k] += 1
        if len(stored) < 4000:
            stored.append(
                {
                    "id": f"{source_id}:{n_docs}",
                    "n_chars": len(norm),
                    "preview": norm[:240],
                    "flags": [k for k, v in flags.items() if v],
                }
            )
        if len(near_left) < 64 and n_tok >= 8:
            near_left.append((f"{source_id}:{n_docs}", ids[:256]))
        if max_docs is not None and n_docs >= max_docs:
            break

    unk_rate = (n_unk / n_tokens) if n_tokens else 1.0
    wiki_overlap = (n_wiki_norm / n_docs) if n_docs else 0.0
    noise_rate = ((flag_hist["nav_or_ad"] + flag_hist["garbled"]) / n_docs) if n_docs else 1.0
    review = stratified_sample(stored, sample_review)
    for row in review:
        row["noise"] = bool(set(row["flags"]) & {"nav_or_ad", "garbled", "email", "phone"})

    card = {
        "source_id": source_id,
        "path": str(path),
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "n_unk_tokens": n_unk,
        "unk_token_rate": round(unk_rate, 6),
        "unk_gate_ok": unk_rate <= UNK_TOKEN_MAX,
        "chars_per_token": round((n_chars / n_tokens), 4) if n_tokens else None,
        "cjk_char_ratio": round(n_cjk / n_chars, 4) if n_chars else 0.0,
        "latin_char_ratio": round(n_latin / n_chars, 4) if n_chars else 0.0,
        "digit_char_ratio": round(n_digit / n_chars, 4) if n_chars else 0.0,
        "exact_dup_rate": round(n_exact_dup / n_docs, 4) if n_docs else 0.0,
        "short_frac_lt_256": round(n_short / n_docs, 4) if n_docs else 0.0,
        "jsonish_frac": round(n_jsonish / n_docs, 4) if n_docs else 0.0,
        "leak_docs": n_leak,
        "wiki_raw_overlap": round(n_wiki_raw / n_docs, 4) if n_docs else 0.0,
        "wiki_norm_overlap": round(wiki_overlap, 4),
        "complement_score": round(1.0 - wiki_overlap, 4),
        "heuristic_flag_frac": {k: round(v / n_docs, 4) for k, v in sorted(flag_hist.items())} if n_docs else {},
        "noise_rate_est": round(noise_rate, 4),
        "length_hist": dict(length_hist),
        "review_n": len(review),
        "review_noise_frac": round(sum(1 for r in review if r["noise"]) / len(review), 4) if review else None,
        "review_samples": review,
        "hard_gates": {
            "unk_ok": unk_rate <= UNK_TOKEN_MAX,
            "leak_ok": n_leak == 0,
            "wiki_overlap_ok": wiki_overlap < 0.85,
            "noise_ok": (review and (sum(1 for r in review if r["noise"]) / len(review) <= 0.15)) or (noise_rate <= 0.15),
        },
    }
    CARDS.mkdir(parents=True, exist_ok=True)
    out = CARDS / f"{source_id}.json"
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: card[k] for k in card if k != "review_samples"}, ensure_ascii=False, indent=2))
    return card


def summarize() -> dict[str, Any]:
    rows = []
    for path in sorted(CARDS.glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "source_id": card["source_id"],
                "n_docs": card["n_docs"],
                "n_tokens": card["n_tokens"],
                "unk_token_rate": card["unk_token_rate"],
                "chars_per_token": card["chars_per_token"],
                "cjk_char_ratio": card["cjk_char_ratio"],
                "wiki_norm_overlap": card["wiki_norm_overlap"],
                "complement_score": card["complement_score"],
                "noise_rate_est": card["noise_rate_est"],
                "review_noise_frac": card.get("review_noise_frac"),
                "jsonish_frac": card["jsonish_frac"],
                "hard_gates": card["hard_gates"],
            }
        )
    report = {"n_cards": len(rows), "cards": rows}
    (PROBE_ROOT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index-wiki")
    an = sub.add_parser("analyze")
    an.add_argument("--source-id", required=True)
    an.add_argument("--path", required=True)
    an.add_argument("--max-bytes", type=int, default=None)
    an.add_argument("--max-docs", type=int, default=None)
    an.add_argument("--sample-review", type=int, default=40)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "index-wiki":
        index_wiki()
    elif args.cmd == "analyze":
        analyze(args.source_id, Path(args.path), max_bytes=args.max_bytes, max_docs=args.max_docs, sample_review=args.sample_review)
    elif args.cmd == "summarize":
        summarize()


if __name__ == "__main__":
    main()
