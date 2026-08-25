"""Real-tokenizer packing for pretrain windows and assistant-only SFT masks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from grammar import dump_calls
from schema_render import (
    render_request,
    render_route_request,
    render_tools_block,
    resolve_toolset,
    ROUTE_SERIALIZER_ID,
)
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


def format_sft_user_text(row: dict[str, Any], *, lang: str | None = None) -> str:
    """Same scene prefix as eval/Qwen runners: 场景：…。用户：… / 用户：…"""
    from schema_render import format_user_text

    return format_user_text(row, lang=lang)


def _route_target(row: dict[str, Any], toolset: dict[str, Any]) -> dict[str, Any]:
    from candidates import ToolContext, entities_from_mode, load_entity_catalog, load_lexicon
    from route_compiler import compile_routes, gold_route_id
    from route_protocol import dump_internal, manifest_hash

    cat = load_entity_catalog()
    ctx = ToolContext(
        query=str(row.get("query") or ""),
        scene=row.get("scene") if isinstance(row.get("scene"), str) else None,
        toolset=toolset,
        entities=list(row.get("entities") or entities_from_mode(str(row.get("entity_mode") or "train"))),
        lexicon=dict(row.get("lexicon") or load_lexicon()),
        param_types=dict(row.get("param_types") or cat.get("param_types") or {}),
        conflict_tags=list(row.get("conflict_tags") or []),
    )
    manifest = compile_routes(ctx)
    answers = row.get("answers") or []
    gold_call = None
    if answers:
        gold_call = answers[0]
    elif isinstance(row.get("gold"), dict):
        calls = (row.get("gold") or {}).get("function_calls") or []
        gold_call = calls[0] if calls else None
    rid = None
    if gold_call:
        rid = gold_route_id(manifest, gold_call)
        if rid is None:
            return {"reject_reason": "gold_not_in_manifest", "manifest": manifest}
    elif row.get("gold_route_id") is not None and str(row.get("kind") or "") not in {
        "refuse",
        "missing",
        "ambiguous",
        "conflict",
        "offtopic",
        "nomatch",
        "injection",
        "range",
    }:
        rid = int(row["gold_route_id"])
        if manifest.by_id(rid) is None:
            return {"reject_reason": "gold_not_in_manifest", "manifest": manifest}
    return {
        "reject_reason": None,
        "manifest": manifest,
        "answer_text": dump_internal(rid),
        "manifest_hash": manifest_hash(manifest),
        "gold_route_id": rid,
    }


def encode_sft_row(
    tok: ZhTokenizerV1,
    row: dict[str, Any],
    seq_len: int,
    *,
    inject_schema: bool | None = None,
    toolset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = row.get("answers") or []
    is_route = str(row.get("serializer") or "") == ROUTE_SERIALIZER_ID or str(row.get("protocol") or "") == "mei-route-protocol-v1"
    answer_text = dump_calls(answers)
    manifest = None
    if inject_schema is None:
        inject_schema = bool(
            toolset is not None
            or row.get("schema_conditioned")
            or row.get("inject_schema")
            or is_route
        )
    if inject_schema:
        ts = resolve_toolset(row, toolset=toolset)
        tools_n = len(tok.encode(render_tools_block(ts)))
        if tools_n >= seq_len:
            return {
                "x": [],
                "y": [],
                "mask": [],
                "confidence": 0.0,
                "n_prompt": 0,
                "n_unmasked": 0,
                "reject_reason": "tools_overflow",
                "n_prompt_tokens": tools_n,
                "n_total": tools_n,
            }
        if is_route:
            routed = _route_target(row, ts)
            if routed.get("reject_reason"):
                return {
                    "x": [],
                    "y": [],
                    "mask": [],
                    "confidence": 0.0,
                    "n_prompt": 0,
                    "n_unmasked": 0,
                    "reject_reason": routed["reject_reason"],
                    "n_prompt_tokens": tools_n,
                    "n_total": tools_n,
                }
            manifest = routed["manifest"]
            answer_text = routed["answer_text"]
            from candidates import entities_from_mode

            query = render_route_request(
                row,
                ts,
                manifest_dict=manifest.as_dict(),
                manifest_hash=routed["manifest_hash"],
                entities=list(row.get("entities") or entities_from_mode(str(row.get("entity_mode") or "train"))),
            )
        else:
            query = render_request(row, ts)
    else:
        query = format_sft_user_text(row)
    packed = tok.encode_chat(query, answer_text)
    ids = list(packed["ids"])
    n_prompt = int(packed["n_prompt"])
    if len(ids) > seq_len:
        return {
            "x": [],
            "y": [],
            "mask": [],
            "confidence": 0.0,
            "n_prompt": n_prompt,
            "n_unmasked": 0,
            "reject_reason": "seq_overflow",
            "n_prompt_tokens": n_prompt,
            "n_total": len(ids),
        }
    pad_id = tok.pad_id
    if len(ids) < seq_len:
        ids = ids + [pad_id] * (seq_len - len(ids))
    x, y = ids[:-1], ids[1:]
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
        "reject_reason": None,
        "n_prompt_tokens": n_prompt,
        "n_total": n_prompt + len(packed.get("answer_ids") or []),
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
    right_prep: list[tuple[str, set[int], int]] = []
    for eid, b in right:
        if len(b) < 4:
            continue
        sb = set(b)
        right_prep.append((eid, sb, len(sb)))
    for sid, a in left:
        if len(a) < 4:
            continue
        sa = set(a)
        na = len(sa)
        if na == 0:
            continue
        for eid, sb, nb in right_prep:
            if nb == 0:
                continue
            if min(na, nb) / max(na, nb) < threshold:
                continue
            inter = len(sa & sb)
            score = inter / (na + nb - inter)
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


def build_leak_index(leak_strings: list[str]) -> dict[str, Any]:
    """Prefix index for document_leaks_eval. Same matches, cheaper on large banks."""
    exact: set[str] = set()
    prefix: dict[str, list[str]] = {}
    shorts: list[str] = []
    for s in leak_strings:
        if not s:
            continue
        exact.add(s)
        if s.isdigit():
            continue
        latin = all(ord(c) < 128 for c in s)
        if latin and len(s) < 12:
            continue
        if len(s) >= 8:
            prefix.setdefault(s[:8], []).append(s)
        elif 4 <= len(s) < 8:
            shorts.append(s)
    return {"exact": exact, "prefix": prefix, "shorts": shorts}


def document_leaks_eval(text: str, leak_strings: list[str], *, index: dict[str, Any] | None = None) -> str | None:
    """Return the leak string if this pretrain doc must be dropped."""
    t = normalize_document(text)
    if not t:
        return None
    idx = index or build_leak_index(leak_strings)
    if t in idx["exact"]:
        return t
    n = len(t)
    for i in range(0, max(0, n - 7)):
        pref = t[i : i + 8]
        for s in idx["prefix"].get(pref, ()):
            if t.startswith(s, i):
                return s
    if idx["shorts"]:
        padded = f" {t} "
        for s in idx["shorts"]:
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
    """List packed uint16 shards. A mix.json atlas wins over tokens/{split}-*.bin."""
    root = Path(root)
    mix_path = root / "mix.json"
    if mix_path.is_file():
        mix = json.loads(mix_path.read_text(encoding="utf-8"))
        rels = mix.get(f"{split}_shards") or []
        from repo_paths import ROOT as REPO_ROOT

        out: list[Path] = []
        for rel in rels:
            path = Path(rel)
            if not path.is_absolute():
                path = REPO_ROOT / rel
            if path.is_file():
                out.append(path.resolve())
        return out
    token_dir = root / "tokens"
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


def _resolve_repo_path(rel: str | Path) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    from repo_paths import ROOT as REPO_ROOT

    return (REPO_ROOT / rel).resolve()


def load_mix_document(root: Path) -> dict[str, Any]:
    mix_path = Path(root) / "mix.json"
    if not mix_path.is_file():
        return {}
    return json.loads(mix_path.read_text(encoding="utf-8"))


def list_mix_rels(root: Path, rels: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for rel in rels:
        path = _resolve_repo_path(rel)
        if path.is_file():
            out.append(path)
    return out


def list_source_shards(root: Path, source: str, split: str) -> list[Path]:
    mix = load_mix_document(root)
    src = (mix.get("sources") or {}).get(source) or {}
    rels = src.get(f"{split}_shards") or []
    return list_mix_rels(root, rels)


def list_valid_set(root: Path, name: str) -> list[Path]:
    mix = load_mix_document(root)
    sets = mix.get("valid_sets") or {}
    if name in sets:
        return list_mix_rels(root, sets.get(name) or [])
    if name == "wiki":
        return list_token_shards(root, "valid")
    return []


class PackedTokenSource:
    """Random-access packed LM windows over concatenated uint16 mmap shards."""

    def __init__(self, paths: list[Path], seq_len: int, pad_id: int = 0, *, skip_tokens: int = 0):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("no token shards")
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.skip_tokens = max(0, int(skip_tokens))
        self.maps = [np.memmap(p, dtype="<u2", mode="r") for p in self.paths]
        self.lengths = [int(m.shape[0]) for m in self.maps]
        self.prefix: list[int] = []
        acc = 0
        for n in self.lengths:
            acc += n
            self.prefix.append(acc)
        self.n_tokens = acc
        usable = max(0, self.n_tokens - self.skip_tokens - 1)
        self.n_windows = (usable + self.seq_len - 1) // self.seq_len if usable else 0

    @property
    def n_predictable_tokens(self) -> int:
        """Unique next-token targets in one mmap pass: every token except the first."""
        return max(0, int(self.n_tokens) - int(self.skip_tokens) - 1)

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
        start = self.skip_tokens + idx * self.seq_len
        chunk = self._gather(start, self.seq_len + 1)
        return window_from_chunk(chunk, self.seq_len, self.pad_id)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class WeightedPackedSources:
    """Deterministic window-level mix. Token-level shuffle is forbidden."""

    def __init__(
        self,
        sources: dict[str, PackedTokenSource],
        weights: dict[str, float],
        *,
        seed: int = 0,
        max_epochs: dict[str, float] | None = None,
    ):
        self.names = [
            name
            for name, src in sources.items()
            if len(src) > 0 and float(weights.get(name) or 0.0) > 0.0
        ]
        if not self.names:
            raise ValueError("no weighted sources")
        self.sources = {name: sources[name] for name in self.names}
        raw = np.array([max(0.0, float(weights.get(name) or 0.0)) for name in self.names], dtype=np.float64)
        if float(raw.sum()) <= 0:
            raise ValueError("weights sum to 0")
        self.probs = raw / raw.sum()
        self.seed = int(seed)
        self.max_epochs = {name: float((max_epochs or {}).get(name) or 1.0) for name in self.names}
        self.rng = np.random.default_rng(self.seed)
        self.perm: dict[str, np.ndarray] = {}
        for name in self.names:
            n_win = len(self.sources[name])
            self.perm[name] = self.rng.permutation(n_win) if n_win else np.zeros(0, dtype=np.int64)
        self.cursor: dict[str, int] = {name: 0 for name in self.names}
        self.window_counts: dict[str, int] = {name: 0 for name in self.names}
        self.draw_count = 0
        self.exhausted_names: set[str] = set()
        self.schedule: dict[str, Any] = {}
        self.schedule_path: Path | None = None

    def _cap(self, name: str) -> int:
        return int(math.floor(len(self.sources[name]) * self.max_epochs[name] + 1e-9))

    def _alive(self) -> list[str]:
        alive: list[str] = []
        for name in self.names:
            if name in self.exhausted_names:
                continue
            if self.cursor[name] >= self._cap(name):
                self.exhausted_names.add(name)
                continue
            alive.append(name)
        return alive

    def take_one(self) -> dict[str, Any] | None:
        alive = self._alive()
        if not alive:
            return None
        p = np.array([self.probs[self.names.index(name)] for name in alive], dtype=np.float64)
        p = p / p.sum()
        name = alive[int(self.rng.choice(len(alive), p=p))]
        src = self.sources[name]
        pos = int(self.cursor[name] % len(src))
        win_i = int(self.perm[name][pos])
        win = dict(src[win_i])
        win["source_id"] = name
        win["source_window"] = win_i
        self.cursor[name] += 1
        self.window_counts[name] += 1
        self.draw_count += 1
        return win

    def take_windows(self, n: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max(0, int(n))):
            win = self.take_one()
            if win is None:
                break
            out.append(win)
        return out

    @property
    def consumed_windows(self) -> int:
        return int(self.draw_count)

    def __len__(self) -> int:
        return int(sum(len(src) for src in self.sources.values()))

    @property
    def n_tokens(self) -> int:
        return int(sum(src.n_tokens for src in self.sources.values()))

    @property
    def n_predictable_tokens(self) -> int:
        return int(sum(src.n_predictable_tokens for src in self.sources.values()))

    def exposure_cap_tokens(self) -> int:
        total = 0
        for name, src in self.sources.items():
            total += int(src.n_predictable_tokens * self.max_epochs[name])
        return total

    def unique_predictable_tokens(self) -> int:
        return int(self.n_predictable_tokens)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        remaining = int(idx)
        if remaining < 0:
            remaining += len(self)
        for name in self.names:
            n_win = len(self.sources[name])
            if remaining < n_win:
                win = dict(self.sources[name][remaining])
                win["source_id"] = name
                return win
            remaining -= n_win
        raise IndexError(idx)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "names": list(self.names),
            "cursor": dict(self.cursor),
            "window_counts": dict(self.window_counts),
            "draw_count": self.draw_count,
            "exhausted": sorted(self.exhausted_names),
            "rng_state": _jsonable(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        names = list(state.get("names") or [])
        if names != list(self.names):
            raise ValueError(f"sampler source names mismatch: ckpt={names} expected={list(self.names)}")
        self.cursor = {name: int(state["cursor"][name]) for name in self.names}
        counts = state.get("window_counts") or {}
        self.window_counts = {name: int(counts.get(name) or 0) for name in self.names}
        self.draw_count = int(state.get("draw_count") or 0)
        self.exhausted_names = set(state.get("exhausted") or [])
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


def load_scheduled_train(root: Path, seq_len: int, pad_id: int) -> WeightedPackedSources | None:
    root = Path(root)
    sched_path = root / "schedule.json"
    if not sched_path.is_file():
        return None
    schedule = json.loads(sched_path.read_text(encoding="utf-8"))
    sources: dict[str, PackedTokenSource] = {}
    weights: dict[str, float] = {}
    max_epochs: dict[str, float] = {}
    for name, cfg in (schedule.get("sources") or {}).items():
        shards = list_source_shards(root, name, "train")
        if not shards:
            continue
        sources[name] = PackedTokenSource(
            shards,
            seq_len,
            pad_id,
            skip_tokens=int(cfg.get("skip_tokens") or 0),
        )
        weights[name] = float(cfg.get("weight") or 0.0)
        max_epochs[name] = float(cfg.get("max_epochs") or 1.0)
    if not sources:
        return None
    packed = WeightedPackedSources(
        sources,
        weights,
        seed=int(schedule.get("sampler_seed") or 0),
        max_epochs=max_epochs,
    )
    packed.schedule = schedule
    packed.schedule_path = sched_path
    return packed


def windows_from_ids(ids: list[int], seq_len: int, pad_id: int = 0) -> list[dict[str, list]]:
    return pack_windows([ids], seq_len, pad_id)


PackedTokenSource = PackedTokenSource
write_uint16_tokens = write_uint16_tokens
list_raw_pages = list_raw_pages
