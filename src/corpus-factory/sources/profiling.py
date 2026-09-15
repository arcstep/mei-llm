"""Reproducible, preparation-only source surveys. No admission or model calls.

Invoke through ``python -m mei_llm corpus source profile`` or
``python -m mei_llm corpus evaluate audit-coverage``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fnmatch
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import threading
import zipfile
from typing import Any
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
GIB = 1024 ** 3


def resolve_path(value: str | Path) -> Path:
    """Resolve relocated corpus inputs using the shared repository resolver."""
    import sys
    factory = str(ROOT / "src/model-factory")
    if factory not in sys.path:
        sys.path.insert(0, factory)
    from common.paths import resolve_repo_path
    return resolve_repo_path(value)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_survey_bytes(directory: Path, seen=None):
    seen=set() if seen is None else seen
    directory=directory.resolve()
    if directory in seen: raise ValueError('cyclic evidence adoption')
    seen.add(directory)
    manifest=json.loads((directory/'survey.json').read_text())
    for rel,expected in manifest['artifacts'].items():
        path=(directory/rel).resolve()
        if not path.is_relative_to(directory) or digest(path)!=expected:raise ValueError(f'survey artifact integrity failure: {rel}')
    binding=directory/'adopted-survey-binding.json'
    if 'adopted-survey-binding.json' in manifest['artifacts']:
        b=json.loads(binding.read_text());previous=Path(b['survey'])
        if digest(previous/'survey.json')!=b['manifest_sha256']:raise ValueError('adopted manifest changed')
        verify_survey_bytes(previous,seen)


def write_new(path: Path, value: Any) -> None:
    data = canonical(value)
    with path.open("xb") as f:
        f.write(data)


def seeded(seed: int, key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big"))


def select_shards(files: list[dict], seed: int, width: int = 64) -> list[dict]:
    """Two disjoint draws per stratum; each wave has marginal pi=1/N_h.

    The second draw is NOT statistically independent of the first. Small
    singleton strata occur only in wave one. Pool using combined pi=k_h/N_h.
    """
    if width < 1:
        raise ValueError("width must be positive")
    ordered = sorted(files, key=lambda r: r["path"])
    if len({r["path"] for r in ordered}) != len(ordered):
        raise ValueError("duplicate frame paths")
    n = len(ordered)
    result = []
    for i in range(min(width, n)):
        lo, hi = i * n // min(width, n), (i + 1) * n // min(width, n)
        group = ordered[lo:hi]
        chosen = seeded(seed, f"shard:{i}").sample(group, min(2, len(group)))
        for wave, row in enumerate(chosen, 1):
            result.append({**row, "stratum": i, "wave": wave,
                           "stratum_size": len(group), "shard_probability": len(chosen) / len(group),
                           "wave_probability": 1 / len(group)})
    return sorted(result, key=lambda row: (row["wave"], row["stratum"]))


class Budget:
    def __init__(self, directory: Path, max_bytes: int, reserve_bytes: int):
        self.directory, self.max_bytes, self.reserve_bytes = directory, max_bytes, reserve_bytes
        self.reserved = 0
        self.lock = threading.Lock()

    def reserve(self, size: int) -> None:
        if size < 0:
            raise ValueError("negative reservation")
        with self.lock:
            if self.reserved + size > self.max_bytes:
                raise RuntimeError("network budget exhausted")
            if shutil.disk_usage(self.directory).free < self.reserve_bytes:
                raise RuntimeError("disk free-space reserve reached")
            # Keep failed/uncertain requests charged conservatively.
            self.reserved += size


def fetch(url: str, budget: Budget, *, start: int | None = None, length: int = 8 << 20) -> tuple[bytes, dict]:
    budget.reserve(length)
    headers = {"User-Agent": "mei-source-profile/1", "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as r:
        response_headers = dict(r.headers.items())
        response_headers["X-Mei-Resolved-URL"] = r.geturl()
        if start is not None:
            expected = f"bytes {start}-{start + length - 1}/"
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError("server did not honor exact byte range; full download refused")
        data = r.read(length + 1)
        if len(data) > length or (start is not None and len(data) != length):
            raise RuntimeError("oversized or incomplete response")
        return data, response_headers


class RangeReader(io.RawIOBase):
    """Bounded HTTP range reader; never silently downloads whole remote files."""
    def __init__(self, url: str, budget: Budget):
        super().__init__()
        self.url, self.budget, self.position = url, budget, 0
        self.cache = OrderedDict()
        _, headers = fetch(url, budget, start=0, length=1)
        # Resolve CDN redirects once. Signed transport URLs stay in memory only.
        self.url = headers["X-Mei-Resolved-URL"]
        content_range = next(v for k, v in headers.items() if k.lower() == "content-range")
        self.size = int(content_range.rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 64 << 20:
            raise RuntimeError("single range exceeds 64 MiB survey limit")
        # Adjacent tiny Parquet columns otherwise each incur a remote RTT.
        # Bound the per-file cache to 16 MiB; never cache embeddings deliberately.
        block_size = 1 << 20
        if size <= block_size:
            chunks = []
            start, end = self.position, self.position + size
            while start < end:
                base = start // block_size * block_size
                if base not in self.cache:
                    block, _ = fetch(self.url, self.budget, start=base, length=min(block_size, self.size - base))
                    self.cache[base] = block
                    if len(self.cache) > 16:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(base)
                take = min(end - start, base + len(self.cache[base]) - start)
                chunks.append(self.cache[base][start - base:start - base + take])
                start += take
            data = b"".join(chunks)
        else:
            data, _ = fetch(self.url, self.budget, start=self.position, length=size)
        self.position += len(data)
        return data


def hf_frame(source: dict, budget: Budget, out: Path) -> dict:
    repo = source["repository"]
    revision = source.get("revision", "main")
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    raw, _ = fetch(url, budget)
    info = json.loads(raw)
    sha = info["sha"]
    # Archive the API bytes, including unselected paths; do not infer size from names.
    with (out / "upstream-api.json").open("xb") as f:
        f.write(raw)
    files = [{"path": r["rfilename"]} for r in info["siblings"]
             if any(fnmatch.fnmatch(r["rfilename"], p) for p in source["patterns"])]
    if not files:
        raise ValueError("no files matched source patterns")
    return {"kind": "hf_revision_listing", "repository": repo, "revision": sha,
            "upstream_response_sha256": hashlib.sha256(raw).hexdigest(),
            "complete_for_patterns": True, "patterns": source["patterns"], "files": files}


def github_frame(source: dict, budget: Budget, out: Path) -> dict:
    repo = source["github_repository"]
    if source.get("catalog_frame"):
        path = ROOT / source["catalog_frame"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["catalog_frame_sha256"]:
            raise ValueError("frozen catalog frame hash changed")
        previous = json.loads(raw)
        if previous["repository"] != repo or previous["kind"] != "github_tree_metadata":
            raise ValueError("catalog repository or frame kind differs")
        patterns = source["patterns"]
        with (out / "adopted-catalog-frame.json").open("xb") as f:
            f.write(raw)
        return {**previous, "patterns": patterns,
                "complete_for_patterns": bool(previous.get("complete_for_patterns")) and
                    (previous.get("patterns") == ["*"] or previous.get("patterns") == patterns),
                "parent_frame_sha256": hashlib.sha256(raw).hexdigest(),
                "files": [r for r in previous["files"] if any(fnmatch.fnmatch(r["path"], p) for p in patterns)]}
    revision = urllib.parse.quote(source.get("revision", "HEAD"), safe="")
    raw, _ = fetch(f"https://api.github.com/repos/{repo}/commits/{revision}", budget)
    commit = json.loads(raw)
    with (out / "upstream-commit.json").open("xb") as f:
        f.write(raw)
    tree_hash = commit["commit"]["tree"]["sha"]
    raw, _ = fetch(f"https://api.github.com/repos/{repo}/git/trees/{tree_hash}?recursive=1", budget)
    tree = json.loads(raw)
    with (out / "upstream-tree.json").open("xb") as f:
        f.write(raw)
    patterns = source.get("patterns", ["*"])
    files = [{"path": r["path"], "bytes": r.get("size"), "git_blob_sha": r["sha"]}
             for r in tree["tree"] if r["type"] == "blob"
             and any(fnmatch.fnmatch(r["path"], p) for p in patterns)]
    return {"kind": "github_tree_metadata", "repository": repo, "revision": commit["sha"],
            "tree_sha": tree_hash, "complete_for_patterns": not tree.get("truncated", False),
            "patterns": patterns, "files": files,
            "content_available_locally": False}


def text_of(row: dict) -> str:
    for key in ("text", "content", "code", "document"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""


def metadata_of(row: dict) -> dict:
    url = str(row.get("url") or "")
    try:
        host = urllib.parse.urlsplit(url).hostname or "unknown"
    except ValueError:
        host = "unknown"
    path = str(row.get("path") or row.get("file_path") or "")
    return {"host": host, "dump": str(row.get("dump") or "unknown"),
            "language": str(row.get("language") or "unknown"),
            "project": str(row.get("repo_name") or row.get("repository_name") or "unknown"),
            "extension": Path(path).suffix.lower() or "unknown",
            "dependency_path": "node_modules/" in path or "/vendor/" in path,
            "topic": "unknown", "quality_score": row.get("quality_score")}


def sample_parquet(handle: Any, shard: dict, seed: int, records: int = 200) -> dict:
    import pyarrow.parquet as pq
    if records < 1:
        raise ValueError("records must be positive")
    pf = pq.ParquetFile(handle, pre_buffer=True)
    total_groups = pf.metadata.num_row_groups
    nonempty = [g for g in range(total_groups) if pf.metadata.row_group(g).num_rows]
    g_count = min(8, records, len(nonempty))
    rng = seeded(seed, shard["path"])
    groups = sorted(rng.sample(nonempty, g_count))
    available = pf.schema_arrow.names
    allowed = {"text", "content", "code", "document", "id", "url", "dump", "date", "title",
               "language", "language_score", "quality_score", "minhash_cluster_size", "path",
               "file_path", "repo_name", "repository_name", "license"}
    columns = [c for c in available if c in allowed]
    if not any(c in columns for c in ("text", "content", "code", "document")):
        raise ValueError("no supported raw text column; relation adapter required")
    offsets = [0]
    for g in range(total_groups):
        offsets.append(offsets[-1] + pf.metadata.row_group(g).num_rows)
    samples = []
    for index, group in enumerate(groups):
        rows = pf.read_row_group(group, columns=columns, use_threads=False).to_pylist()
        take = min(len(rows), records // g_count + (index < records % g_count))
        for i in sorted(rng.sample(range(len(rows)), take)):
            row = rows[i]
            text = text_of(row)
            p = shard["shard_probability"] * g_count / len(nonempty) * take / len(rows)
            samples.append({"row_index": offsets[group] + i, "row_group": group,
                            "inclusion_probability": p, "weight": 1 / p,
                            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "utf8_bytes": len(text.encode()), "characters": len(text),
                            "metadata": metadata_of(row), "original": row,
                            "purpose": "population", "semantic_review": "pending"})
    return {"shard": shard, "rows": pf.metadata.num_rows, "row_groups": total_groups,
            "sampled_row_groups": groups, "columns": columns, "samples": samples}


def sample_jsonl(path: Path, shard: dict, seed: int, records: int = 200) -> dict:
    """Full-stream reservoir over records, preserving entire dialogue objects."""
    if records < 1:
        raise ValueError("records must be positive")
    rng = seeded(seed, shard["path"])
    selected = []
    count = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            slot = count - 1 if count <= records else rng.randrange(count)
            if slot < records:
                item = (line_number, record)
                if count <= records:
                    selected.append(item)
                else:
                    selected[slot] = item
    samples = []
    for line, original in sorted(selected):
        if isinstance(original, list) and all(isinstance(t, str) for t in original):
            row = {"turns": original}
            text = "\n".join(original)
        elif isinstance(original, dict):
            row = original
            text = text_of(row)
        else:
            raise ValueError("unsupported JSONL record shape; explicit relation adapter required")
        probability = shard["shard_probability"] * len(selected) / count
        samples.append({"row_index": line - 1, "source_line": line,
                        "inclusion_probability": probability, "weight": 1 / probability,
                        "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "utf8_bytes": len(text.encode()), "characters": len(text),
                        "metadata": {**metadata_of(row), "turn_count": len(row.get("turns", [])) or None},
                        "original": original, "purpose": "population", "semantic_review": "pending"})
    return {"shard": shard, "rows": count, "sampling": "full-stream record reservoir",
            "local_sha256": digest(path), "samples": samples}


def sample_relations(path: Path, shard: dict, source: dict, seed: int, records: int) -> dict:
    if source["relation_kind"] == "zip-inventory":
        with zipfile.ZipFile(path) as archive:
            members = [{"path": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size,
                        "crc32": i.CRC, "directory": i.is_dir()} for i in archive.infolist()]
        return {"shard": shard, "rows": 0, "samples": [], "archive_members": members,
                "local_sha256": digest(path), "scope": "archive directory only; no content survey"}
    from relations import relation_units, iter_json_object
    if source["relation_kind"] in {"crosswoz", "risawoz"} and source.get("archive_member"):
        rng = seeded(seed, shard["path"])
        chosen, count = [], 0
        member_hash = hashlib.sha256()
        with zipfile.ZipFile(path) as archive, archive.open(source["archive_member"]) as stream:
            for index, (key, value) in enumerate(iter_json_object(stream, member_hash, mapping=source["relation_kind"] == "crosswoz")):
                count += 1
                slot = index if count <= records else rng.randrange(count)
                if slot < records:
                    unit = next(relation_units({key: value} if source["relation_kind"] == "crosswoz" else [value], source["relation_kind"]))
                    if count <= records: chosen.append((index, unit))
                    else: chosen[slot] = (index, unit)
        samples = []
        for index, unit in sorted(chosen):
            text = json.dumps(unit["input"], ensure_ascii=False, sort_keys=True)
            probability = shard["shard_probability"] * len(chosen) / count
            samples.append({"row_index": index, "unit_id": unit["unit_id"], "unit_kind": unit["unit_kind"],
                "inclusion_probability": probability, "weight": 1/probability,
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "utf8_bytes": len(text.encode()), "characters": len(text), "metadata": metadata_of({}),
                "original": unit, "purpose": "population", "semantic_review": "pending",
                "serialization": "whole-dialogue survey view; not CPT or SFT release"})
        return {"shard": shard, "rows": count, "samples": samples, "local_sha256": digest(path),
                "member_sha256": member_hash.hexdigest(), "sampling": "full-stream whole-dialogue reservoir"}
    if source.get("archive_member"):
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo(source["archive_member"])
            if member.file_size > 128 << 20:
                raise ValueError("relation archive member exceeds survey memory bound")
            raw = archive.read(member)
    else:
        if path.stat().st_size > 128 << 20:
            raise ValueError("relation JSON requires a streaming adapter above 128 MiB")
        raw = path.read_bytes()
    units = list(relation_units(json.loads(raw), source["relation_kind"]))
    rng = seeded(seed, shard["path"])
    selected = sorted(rng.sample(range(len(units)), min(records, len(units))))
    samples = []
    for index in selected:
        unit = units[index]
        text = json.dumps(unit["input"], ensure_ascii=False, sort_keys=True)
        probability = shard["shard_probability"] * len(selected) / len(units)
        samples.append({"row_index": index, "unit_id": unit["unit_id"], "unit_kind": unit["unit_kind"],
                        "inclusion_probability": probability, "weight": 1 / probability,
                        "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "utf8_bytes": len(text.encode()), "characters": len(text),
                        "metadata": metadata_of({}), "original": unit, "purpose": "population",
                        "semantic_review": "pending", "serialization": "survey view; not CPT or SFT release"})
    return {"shard": shard, "rows": len(units), "samples": samples,
            "local_sha256": digest(path), "member_sha256": hashlib.sha256(raw).hexdigest()}


def summarize(shards: list[dict], *, single_wave: bool = False) -> dict:
    """HT totals/ratio descriptions over completed samples, never guessed net yield."""
    hist = {key: Counter() for key in ("host", "dump", "language", "project", "extension", "topic")}
    byte_hist = {key: Counter() for key in hist}
    hashes = Counter()
    seen_positions = set()
    count = 0
    weight_total = byte_total = 0.0
    length_hist = Counter()
    character_total = character_weight = 0.0
    for shard in shards:
        for row in shard["samples"]:
            if row.get("purpose") != "population":
                continue
            pos = (shard["shard"]["path"], row["row_index"])
            if pos in seen_positions:
                raise ValueError("duplicate sample position")
            seen_positions.add(pos)
            p = row["inclusion_probability"]
            if single_wave:
                # Row probabilities are stored for the union of both waves.
                # A wave alone uses its marginal shard probability instead.
                design = shard["shard"]
                p *= design["wave_probability"] / design["shard_probability"]
            if not 0 < p <= 1:
                raise ValueError("invalid inclusion probability")
            w = 1 / p
            count += 1
            weight_total += w
            byte_total += w * row["utf8_bytes"]
            characters = row.get("characters")
            if characters is None and "text" in row:
                characters = len(row["text"])
            length_bin = "unknown"
            if characters is not None:
                character_total += w * characters
                character_weight += w
                length_bin = next((f"le_{n}" for n in (128, 512, 2048, 8192, 32768) if characters <= n), "gt_32768")
            length_hist[length_bin] += w
            hashes[row["text_sha256"]] += 1
            for key in hist:
                value = str(row["metadata"].get(key, "unknown"))
                hist[key][value] += w
                byte_hist[key][value] += w * row["utf8_bytes"]
    def distribution(values, total):
        return {k: v / total for k, v in sorted(values.items())} if total else {}
    return {"sample_records": count, "represented_record_total": weight_total,
            "probability_basis": "single_wave_marginal" if single_wave else "combined_waves",
            "represented_utf8_bytes": byte_total,
            "weighted_mean_characters": character_total / character_weight if character_weight else None,
            "character_length_distribution": distribution(length_hist, weight_total),
            "document_distributions": {k: distribution(v, weight_total) for k, v in hist.items()},
            "byte_distributions": {k: distribution(v, byte_total) for k, v in byte_hist.items()},
            "observed_exact_duplicate_excess": sum(n - 1 for n in hashes.values()),
            "audited_net_tokens": None, "topic_labels_verified": False,
            "limitations": ["ratio estimates, not census counts", "no confidence interval computed",
                            "sample duplicate rate is not whole-source dedup loss",
                            "semantic suitability and naturalness require review",
                            "UTF-8 bytes are not tokenizer tokens"]}


def sample_text(path: Path, shard: dict, source: dict) -> dict:
    """A whole version-bound document; links/tests are not inferred from proximity."""
    raw = path.read_bytes()
    text = raw.decode(source.get("encoding", "utf-8"))
    original_text = text
    body_range = None
    if source.get('strip_gutenberg_wrapper'):
        start = re.search(r'^\*\*\* START OF .*?\*\*\*\s*$', text, re.M)
        end = re.search(r'^\*\*\* END OF .*?\*\*\*\s*$', text, re.M)
        if not start or not end or start.end() >= end.start():
            raise ValueError('Gutenberg body boundaries missing')
        body_range = [start.end(), end.start()]
        text = text[start.end():end.start()]
    p = shard["shard_probability"]
    return {"shard": shard, "rows": 1, "local_sha256": digest(path), "samples": [{
        "row_index": 0, "inclusion_probability": p, "weight": 1 / p,
        "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text), "utf8_bytes": len(text.encode()),
        "metadata": {**metadata_of({}), "project": source.get("github_repository", "unknown"),
                     "extension": Path(shard["path"]).suffix or "unknown",
                     "language": source.get("language", "unknown"),
                     "work_group": source.get('work_group'), 'title': source.get('title')},
        "original": {"path": shard["path"], "content": original_text, 'body_character_range': body_range}, "purpose": "population",
        "semantic_review": "pending", "serialization": "whole source document; not executable gold"}]}


def safe_id(value: str) -> str:
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
        raise ValueError("source_id must be a simple identifier")
    return value


def ledger_inventory(root: Path) -> dict:
    """Census provenance without reading/replacing token arrays or old receipts."""
    rows = []
    for manifest in sorted(root.glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        ledger = manifest.parent / "documents.jsonl"
        paths, extensions, splits = Counter(), Counter(), Counter()
        count = dependency = 0
        h = hashlib.sha256()
        if ledger.exists():
            with ledger.open("rb") as stream:
                for line in stream:
                    h.update(line)
                    record = json.loads(line)
                    paths[str(record.get("source_path") or "unknown")] += 1
                    key = str(record.get("record_key") or "")
                    extensions[Path(key).suffix.lower() or "unknown"] += 1
                    dependency += int("node_modules/" in key or "/vendor/" in key)
                    splits[str(record.get("split") or "unknown")] += 1
                    count += 1
        rows.append({"asset": manifest.parent.name, "source_id": data.get("source_id"),
                     "manifest_path": str(manifest), "manifest_sha256": digest(manifest),
                     "ledger_sha256": h.hexdigest() if ledger.exists() else None,
                     "manifest_documents": data.get("documents"), "ledger_records": count,
                     "declared_tokens": data.get("tokens"), "tokenizer_id": data.get("tokenizer_id"),
                     "source_path_counts": dict(paths), "extension_counts": dict(extensions),
                     "dependency_path_records": dependency, "split_counts": dict(splits),
                     "clearance_receipt": data.get("clearance_receipt"),
                     "license_reviewed_claim": data.get("license_reviewed"),
                     "approved_for_new_recipe": False})
        print(json.dumps({"inventory_asset": manifest.parent.name, "records": count}), flush=True)
    return {"schema": "mei-source-ledger-census-v1", "root": str(root), "assets": rows,
            "scope": "admitted ledger census; declarations are not new source review",
            "cross_asset_counts_additive": False, "audited_net_tokens": None}


def profile(config: dict, out: Path, *, network: bool = False, metadata_only: bool = False,
            resume_from: Path | None = None, frozen_evidence_only: bool = False) -> dict:
    if frozen_evidence_only and (not resume_from or network):
        raise ValueError("frozen evidence requires a previous run and forbids network")
    if not 1 <= int(config.get("shards_per_wave", 64)) <= 64:
        raise ValueError("survey plan permits 1..64 strata")
    max_records = 10000 if config.get('purpose') == 'candidate_preparation' else 200
    if not 1 <= int(config.get("records_per_shard", 200)) <= max_records:
        raise ValueError(f"this mode permits 1..{max_records} records per shard")
    ids = [safe_id(s["source_id"]) for s in config["sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source IDs")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "config.json", config)
    write_new(out / "execution.json", {"allow_network": network, "metadata_only": metadata_only,
              "frozen_evidence_only": frozen_evidence_only, "resume_from": str(resume_from) if resume_from else None})
    implementation = Path(__file__).read_bytes()
    with (out / "implementation.py.snapshot").open("xb") as f:
        f.write(implementation)
    if any(s.get("format") == "relation" for s in config["sources"]):
        with (out / "relations.py.snapshot").open("xb") as f:
            f.write(Path(__file__).with_name("relations.py").read_bytes())
    if any(s.get("format") == "remote-zip" for s in config["sources"]):
        with (out / "archive_surveys.py.snapshot").open("xb") as f:
            f.write(Path(__file__).with_name("archive_surveys.py").read_bytes())
    if resume_from:
        old_config = json.loads((resume_from / "config.json").read_text())
        selected_ids = {s["source_id"] for s in config["sources"]}
        old_subset = {**old_config, "sources": [s for s in old_config["sources"] if s["source_id"] in selected_ids]}
        if canonical(old_subset) != canonical(config):
            raise ValueError("resume requires identical frozen survey design; a source subset is allowed")
        with (out / "adopted-implementation.py.snapshot").open("xb") as f:
            f.write((resume_from / "implementation.py.snapshot").read_bytes())
        if (resume_from/'survey.json').exists():
            verify_survey_bytes(resume_from)
            write_new(out/'adopted-survey-binding.json',{'survey':str(resume_from.resolve()),'manifest_sha256':digest(resume_from/'survey.json')})
    budget = Budget(out, min(int(config.get("max_network_bytes", 100 * GIB)), 100 * GIB),
                    max(int(config.get("reserve_disk_bytes", 100 * GIB)), 100 * GIB))
    seed = int(config.get("seed", 20260913))
    reports = []
    if config.get("catalog_pages"):
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def __init__(self): super().__init__(); self.links = []
            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    href = dict(attrs).get("href")
                    if href: self.links.append(href)
        entries = []
        for page in config["catalog_pages"]:
            entry = {"id": safe_id(page["id"]), "url": page["url"], "status": "pending"}
            try:
                if not network: raise ValueError("catalog page access requires --allow-network")
                raw, headers = fetch(page["url"], budget, length=min(int(page.get("max_bytes", 8 << 20)), 32 << 20))
                filename = f"catalog-{entry['id']}.raw"
                with (out / filename).open("xb") as f: f.write(raw)
                parser = Links(); parser.feed(raw.decode("utf-8", errors="replace"))
                entry.update(status="captured", file=filename, sha256=hashlib.sha256(raw).hexdigest(),
                             links=sorted(set(urllib.parse.urljoin(page["url"], h) for h in parser.links)),
                             content_type=next((v for k,v in headers.items() if k.lower()=="content-type"), None))
            except Exception as exc: entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            entries.append(entry)
        write_new(out / "catalog-pages.json", {"entries": entries, "scope": "retrieved metadata; not a complete data manifest or content admission"})
    if config.get("ledger_root"):
        write_new(out / "ledger-inventory.json", ledger_inventory(ROOT / config["ledger_root"]))
    if config.get("local_diagnostics"):
        from local_diagnostics import diagnostics
        helper = Path(__file__).with_name("local_diagnostics.py")
        with (out / "local_diagnostics.py.snapshot").open("xb") as f:
            f.write(helper.read_bytes())
        write_new(out / "local-diagnostics.json", diagnostics(ROOT, config["local_diagnostics"]))
    for source in config["sources"]:
        source_id = safe_id(source["source_id"])
        directory = out / source_id
        directory.mkdir()
        report = {"source_id": source_id, "role": source.get("role"),
                  "process_status": "pending", "m1_eligible": False, "training_adoption_eligible": False,
                  "source": source, "errors": []}
        local = []
        for pattern in source.get("local_globs", []):
            local.extend(ROOT.glob(pattern))
        local = sorted(set(p.resolve() for p in local if p.is_file()))
        try:
            if frozen_evidence_only:
                previous = resume_from / source_id
                frame = json.loads((previous / "frame.json").read_text())
                for raw_path in previous.glob("upstream-*.json"):
                    with (directory / raw_path.name).open("xb") as f:
                        f.write(raw_path.read_bytes())
            elif source.get('frozen_frame'):
                frozen = ROOT / source['frozen_frame']
                if digest(frozen) != source.get('frozen_frame_sha256'):
                    raise ValueError('frozen source listing changed')
                frame = json.loads(frozen.read_text())
            elif source.get("repository") and network:
                frame = hf_frame(source, budget, directory)
            elif source.get("github_repository") and network:
                frame = github_frame(source, budget, directory)
            elif source.get("http_files") and network:
                frame = {"kind": "explicit_http_inventory", "complete_for_patterns": False,
                         "revision": source.get("revision"), "files": source["http_files"],
                         "scope": "explicit source list; upstream completeness requires supporting catalog review"}
                if source.get('catalog_path'):
                    catalog=ROOT/source['catalog_path'];raw=catalog.read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=source.get('catalog_sha256'):raise ValueError('HTTP supporting catalog hash changed')
                    with (directory/'supporting-catalog.raw').open('xb') as f:f.write(raw)
                    frame['supporting_catalog_sha256']=source['catalog_sha256']
            else:
                frame = {"kind": "local_inventory", "complete_for_patterns": False,
                         "files": [{"path": str(p), "bytes": p.stat().st_size} for p in local]}
            write_new(directory / "frame.json", frame)
            report["frame"] = {k: v for k, v in frame.items() if k != "files"}
            report["frame_files"] = len(frame["files"])
            report["local_files"] = len(local)
            if 'random_shard_fraction' in source or 'random_shard_count' in source:
                files = sorted(frame['files'], key=lambda row: row['path'])
                if len({r['path'] for r in files}) != len(files):
                    raise ValueError('duplicate source paths')
                if 'random_shard_count' in source:
                    count = source['random_shard_count']
                    if 'random_shard_fraction' in source or type(count) is not int or not 1 <= count <= len(files):
                        raise ValueError('specify one valid shard count or fraction')
                else:
                    fraction = float(source['random_shard_fraction'])
                    if not 0 < fraction <= 1: raise ValueError('invalid fraction')
                    count = math.ceil(len(files) * fraction)
                selected = sorted([{**r, 'shard_probability': count / len(files),
                    'selection_method': 'uniform_without_replacement'}
                    for r in seeded(seed, source_id).sample(files, count)], key=lambda r: r['path']) if files else []
            else:
                selected = select_shards(frame["files"], seed, int(config.get("shards_per_wave", 64)))
            write_new(directory / "sampling-plan.json", selected)
            report["planned_shards"] = len(selected)
            results = []
            if not metadata_only and (frame["kind"] != "github_tree_metadata" or source.get("format") in {"relation", "text"}):
                def work(shard):
                    name = hashlib.sha256(shard["path"].encode()).hexdigest()[:20]
                    try:
                        if resume_from and frame["kind"] in {"hf_revision_listing", "explicit_http_inventory", "github_tree_metadata"}:
                            previous = resume_from / source_id
                            old_frame_path = previous / "frame.json"
                            old_sample_path = previous / f"sample-{name}.json"
                            if old_frame_path.exists() and old_sample_path.exists():
                                old_frame = json.loads(old_frame_path.read_text())
                                if old_frame.get("kind") == frame["kind"] and old_frame.get("revision") == frame.get("revision") and old_frame.get("files") == frame["files"]:
                                    if frame['kind']!='hf_revision_listing' and not (resume_from/'survey.json').exists():raise ValueError('non-HF adoption requires a complete bound parent manifest')
                                    old_sample = json.loads(old_sample_path.read_text())
                                    if old_sample["shard"] != shard:
                                        raise ValueError("adopted shard sampling design differs")
                                    result = {**old_sample, "adopted_sample": {
                                        "path": str(old_sample_path), "sha256": digest(old_sample_path),
                                        "source_code_sha256": digest(resume_from / "implementation.py.snapshot")}}
                                    write_new(directory / f"sample-{name}.json", result)
                                    return result, None
                        if frozen_evidence_only:
                            raise ValueError("no complete frozen sample; source remains pending")
                        if frame['kind'] in {'explicit_http_inventory', 'github_tree_metadata', 'hf_revision_listing'} and not network:
                            raise ValueError('remote source body access requires --allow-network')
                        if frame["kind"] == "explicit_http_inventory":
                            if source.get("format") != "remote-zip": raise ValueError('HTTP inventory supports remote ZIP surveys only')
                            from archive_surveys import sample_zip
                            if source.get('archive_transport') == 'whole-bounded':
                                size=shard.get('bytes')
                                if size is None:
                                    with RangeReader(shard['url'],budget) as probe: size=probe.size
                                if not isinstance(size,int) or not 0<size<=2<<30: raise ValueError('whole survey archive needs known size <=2 GiB')
                                budget.reserve(size)
                                raw_path=directory/f'archive-{name}.zip'
                                h=hashlib.md5(); remaining=size
                                request=urllib.request.Request(shard['url'],headers={'User-Agent':'mei-source-profile/1','Accept-Encoding':'identity'})
                                with urllib.request.urlopen(request,timeout=25) as response, raw_path.open('xb') as f:
                                    while remaining:
                                        data=response.read(min(1<<20,remaining))
                                        if not data:raise ValueError('incomplete archive')
                                        f.write(data);h.update(data);remaining-=len(data)
                                    if response.read(1):raise ValueError('archive larger than frozen size')
                                checksum=shard.get('upstream_checksum')
                                if checksum and checksum!='md5:'+h.hexdigest():raise ValueError('archive upstream checksum mismatch')
                                with raw_path.open('rb') as handle:
                                    handle.size=size
                                    result=sample_zip(handle,shard,source,seed,int(config.get('records_per_shard',200)),metadata_of,directory/'members')
                                result['archive_sha256']=digest(raw_path)
                                result['integrity_scope']='full downloaded archive SHA256 and optional upstream MD5; member CRC/SHA256 and original bytes preserved'
                            else:
                                with RangeReader(shard['url'], budget) as handle:
                                    result = sample_zip(handle, shard, source, seed, int(config.get('records_per_shard',200)), metadata_of,directory/'members')
                        elif frame["kind"] == "github_tree_metadata":
                            size = shard.get("bytes")
                            if not isinstance(size, int) or not 0 < size <= 128 << 20:
                                raise ValueError("GitHub relation survey requires known file size <=128 MiB")
                            url = f"https://raw.githubusercontent.com/{source['github_repository']}/{frame['revision']}/{urllib.parse.quote(shard['path'], safe='/')}"
                            raw, _ = fetch(url, budget, length=size)
                            blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                            if len(raw) != size or blob_sha != shard["git_blob_sha"]:
                                raise ValueError("GitHub source bytes do not match frozen blob")
                            raw_path = directory / f"raw-{name}"
                            with raw_path.open("xb") as f:
                                f.write(raw)
                            result = (sample_text(raw_path, shard, source) if source.get("format") == "text" else
                                      sample_relations(raw_path, shard, source, seed, int(config.get("records_per_shard", 200))))
                            result["source_revision"] = frame["revision"]
                            result["git_blob_sha"] = blob_sha
                        elif frame["kind"] == "hf_revision_listing":
                            url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{frame['revision']}/{urllib.parse.quote(shard['path'], safe='/')}"
                            if source.get("format") in {"relation", "text", "jsonl"}:
                                limit = min(int(source.get("max_source_bytes", 64 << 20)), 128 << 20)
                                raw, _ = fetch(url, budget, length=limit)
                                raw_path = directory / f"raw-{name}"
                                with raw_path.open("xb") as f: f.write(raw)
                                if source.get("format") == "text":
                                    result = sample_text(raw_path, shard, source)
                                elif source.get("format") == "jsonl":
                                    result = sample_jsonl(raw_path, shard, seed, int(config.get("records_per_shard", 200)))
                                else:
                                    result = sample_relations(raw_path, shard, source, seed, int(config.get("records_per_shard", 200)))
                                result["source_revision"] = frame["revision"]
                            else:
                                with RangeReader(url, budget) as handle:
                                    result = sample_parquet(handle, shard, seed, int(config.get("records_per_shard", 200)))
                        else:
                            if source.get("format") == "jsonl":
                                result = sample_jsonl(Path(shard["path"]), shard, seed, int(config.get("records_per_shard", 200)))
                            elif source.get("format") == "relation":
                                result = sample_relations(Path(shard["path"]), shard, source, seed, int(config.get("records_per_shard", 200)))
                            elif source.get("format") == "text":
                                result = sample_text(Path(shard["path"]), shard, source)
                            else:
                                result = sample_parquet(shard["path"], shard, seed, int(config.get("records_per_shard", 200)))
                            result["local_sha256"] = digest(Path(shard["path"]))
                        for row in result["samples"]:
                            row["split"] = source.get("official_split", row.get("original", {}).get("split", "unknown") if isinstance(row.get("original"), dict) else "unknown")
                        if source.get('write_candidate_texts'):
                            if source.get('format') != 'text' or len(result['samples']) != 1:
                                raise ValueError('candidate text export requires one whole text document')
                            candidate = directory / f'candidate-text-{name}.txt'
                            with candidate.open('x') as f: f.write(result['samples'][0]['text'])
                            result['candidate_text'] = {'path': candidate.name, 'sha256': digest(candidate), 'release': False}
                        write_new(directory / f"sample-{name}.json", result)
                        return result, None
                    except Exception as exc:
                        failure = {"path": shard["path"], "error": f"{type(exc).__name__}: {exc}"}
                        write_new(directory / f"failure-{name}.json", failure)
                        return None, failure
                workers=int(config.get('download_workers',2))
                if workers not in (1,2):raise ValueError('download_workers must be 1 or 2')
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for result, error in executor.map(work, selected):
                        if result is not None:
                            results.append(result)
                        if error:
                            report["errors"].append(error)
                        print(json.dumps({"source": source_id, "completed": len(results), "failed": len(report["errors"])}), flush=True)
            report["sampled_shards"] = len(results)
            report["statistics"] = summarize(results)
            report["wave_statistics"] = {str(w): summarize([r for r in results if r["shard"].get("wave") == w], single_wave=True)
                                         for w in (1, 2) if any(r["shard"].get("wave") == w for r in results)}
            report["estimation_scope"] = ("complete_sampling_frame" if results and len(results) == len(selected)
                                          else "completed_samples_only_no_population_extrapolation")
            report["process_status"] = "surveyed_pending_review" if results else "metadata_only" if frame["files"] else "hold"
            if report["errors"]:
                report["process_status"] = "partial"
            report["m1_gaps"] = ["target-suitability review", "net-yield and cross-source overlap review"]
            if not report["statistics"]["sample_records"]:
                report["m1_gaps"].append("content survey missing")
            if not frame["complete_for_patterns"]:
                report["m1_gaps"].append("upstream population coverage not established")
        except Exception as exc:
            report["process_status"] = "hold"
            report["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
        write_new(directory / "profile.json", report)
        reports.append(report)
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    summary = {"schema": "mei-source-survey-v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "seed": seed, "sources": [{k: v for k, v in r.items() if k in
                   ("source_id", "process_status", "frame_files", "sampled_shards", "m1_eligible", "errors")} for r in reports],
               "network_reserved_bytes": budget.reserved, "m1_passed": False,
               "m2_passed": False, "training_adoption_eligible": False,
               "source_code_sha256": hashlib.sha256(implementation).hexdigest(), "artifacts": files}
    write_new(out / "survey.json", summary)
    return summary


def audit_coverage(paths: list[Path]) -> dict:
    profiles = []
    observed = defaultdict(list)
    unique_positions = set()
    wave_diagnostics = []
    dossiers = []
    for path in paths:
        verify_survey_bytes(path)
        manifest = json.loads((path / "survey.json").read_text())
        bound_samples = defaultdict(list)
        bound_profiles = []
        for rel, expected in manifest["artifacts"].items():
            p = (path / rel).resolve()
            if not p.is_relative_to(path.resolve()) or digest(p) != expected:
                raise ValueError(f"survey artifact integrity failure: {rel}")
            if p.name.startswith("sample-"):
                sample = json.loads(p.read_text())
                source = Path(rel).parts[0]
                bound_samples[source].append(sample)
                for row in sample["samples"]:
                    if not row.get("text"):
                        continue
                    observed[row["text_sha256"]].append({"survey": str(path), "source": source,
                        "path": sample["shard"]["path"], "row_index": row["row_index"],
                        "split": row.get("split", "unknown")})
                    unique_positions.add((source, sample["shard"]["path"], row["row_index"], row["text_sha256"]))
            if p.name == "profile.json":
                bound_profiles.append(p)
        for source, samples in sorted(bound_samples.items()):
            if not samples or not all("wave_probability" in s["shard"] for s in samples):
                continue
            wave_diagnostics.append({"survey": str(path), "source_id": source,
                "wave_statistics": {str(w): summarize([s for s in samples if s["shard"]["wave"] == w], single_wave=True)
                                    for w in (1, 2) if any(s["shard"]["wave"] == w for s in samples)},
                "scope": "observed wave support; wave 2 excludes singleton strata; no extrapolation with failures"})
        for p in sorted(bound_profiles):
            entry = json.loads(p.read_text())
            entry = {**entry, "survey_path": str(path), "statistics": summarize(bound_samples.get(entry["source_id"], []))}
            profiles.append(entry)
            stats = entry.get("statistics", {})
            complete = entry.get("estimation_scope") == "complete_sampling_frame"
            dossiers.append({"source_id": entry["source_id"], "survey": str(path),
                "evidence": {"profile_path": str(p), "profile_sha256": digest(p),
                             "survey_manifest_sha256": digest(path / "survey.json")},
                "population": {"frame": entry.get("frame"), "file_count": entry.get("frame_files"),
                               "ordering_verified": None, "scope": entry.get("estimation_scope")},
                "inventory": {"located_local_files": entry.get("local_files"),
                              "admission_or_reuse_decision": False},
                "intended_use": {"role": entry.get("role"), "question": entry.get("source", {}).get("survey_question"),
                                 "semantic_suitability": "pending_review"},
                "volume": {"published_records": None, "sampled_records": stats.get("sample_records"),
                           "sampling_frame_record_estimate": stats.get("represented_record_total") if complete else None,
                           "sampling_frame_utf8_byte_estimate": stats.get("represented_utf8_bytes") if complete else None,
                           "audited_net_tokens": None, "cross_source_dedup_yield": None},
                "bias": {"document_distributions": stats.get("document_distributions"),
                         "byte_distributions": stats.get("byte_distributions"),
                         "observed_sample_duplicate_excess": stats.get("observed_exact_duplicate_excess"),
                         "sampling_failures": entry.get("errors"), "topic_verified": False},
                "acquisition": {"next_action": "review_and_quantify_remaining_gaps" if complete else "extend_or_repair_source_survey",
                                "independent_increment_estimate": None, "resource_estimate": None,
                                "production_status": "hold_before_M1_M2"}})
    comparisons = []
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            if a["source_id"] != b["source_id"]:
                continue
            metrics = {}
            for axis in ("dump", "host", "language", "project", "extension"):
                x = a.get("statistics", {}).get("document_distributions", {}).get(axis, {})
                y = b.get("statistics", {}).get("document_distributions", {}).get(axis, {})
                known_x = sum(v for k,v in x.items() if k != "unknown")
                known_y = sum(v for k,v in y.items() if k != "unknown")
                metrics[axis] = sum(abs(x.get(k, 0) - y.get(k, 0)) for k in x.keys() | y.keys()) / 2 if known_x and known_y else None
            x = a["statistics"]["character_length_distribution"]
            y = b["statistics"]["character_length_distribution"]
            metrics["character_length_bin"] = sum(abs(x.get(k, 0) - y.get(k, 0)) for k in x.keys() | y.keys()) / 2 if x and y and set(x) != {"unknown"} and set(y) != {"unknown"} else None
            comparisons.append({"source_id": a["source_id"], "left_survey": a["survey_path"],
                                "right_survey": b["survey_path"], "total_variation_distance": metrics,
                                "interpretation": "descriptive; sample size/design uncertainty not quantified"})
    overlap = [rows for rows in observed.values() if len({r["source"] for r in rows}) > 1]
    split_leaks = [rows for rows in observed.values()
                   if "train" in {r["split"] for r in rows} and "test" in {r["split"] for r in rows}]
    return {"schema": "mei-source-coverage-audit-v1", "integrity_passed": True,
            "m1_passed": False, "m2_passed": False, "training_adoption_eligible": False,
            "sources": [{"source_id": p["source_id"], "status": p["process_status"],
                         "gaps": p.get("m1_gaps", ["source survey failed"])} for p in profiles],
            "comparisons": comparisons,
            "wave_diagnostics": wave_diagnostics,
            "source_dossiers": dossiers,
            "unique_observed_positions": len(unique_positions),
            "unique_observed_nonempty_text_hashes": len(observed),
            "sample_positions_are_independent_capacity": False,
            "observed_cross_source_exact_overlap_groups": len(overlap),
            "observed_train_test_exact_leak_groups": len(split_leaks),
            "overlap_examples": overlap[:20], "leak_examples": split_leaks[:20],
            "limitations": ["coverage diagnostics cannot grant human review or training admission",
                            "source topic and intended task coverage remain separate requirements",
                            "exact overlap of samples only; zero does not prove whole-corpus isolation"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("profile")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--resume-from", type=Path, help="Adopt complete shard evidence into a NEW run; retry missing shards")
    p.add_argument("--frozen-evidence-only", action="store_true", help="Seal only existing evidence into a NEW run, without fetching missing samples")
    a = commands.add_parser("audit-coverage")
    a.add_argument("--surveys", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    review_mode = a.add_mutually_exclusive_group()
    review_mode.add_argument('--pro-calibrate-from', type=Path, help='User-authorized Pro rubric calibration under cumulative 20M token budget')
    a.add_argument('--teacher-system-prompt', type=Path)
    a.add_argument('--teacher-offsets', type=int, nargs='+')
    review_mode.add_argument('--flash-compare-from', type=Path, help='Authorized 24-record blind replay; cloud budget <= 1 CNY')
    a.add_argument('--teacher-pricing', type=Path)
    a.add_argument('--teacher-provider-status', type=Path)
    a.add_argument('--teacher-continue-from', type=Path, help='Adopt a billed Flash prefix into a new ID; never retry requests')
    a.add_argument('--cloud-teacher-model', choices=['deepseek-v4-flash', 'deepseek-v4-pro'], default='deepseek-v4-flash', help='Explicitly authorized teacher; Flash <=1 CNY, Pro <=5 CNY')
    review_mode.add_argument("--local-teacher-review", action="store_true",
                   help="Annotate frozen survey snippets with the explicitly configured local Qwen; no cloud calls")
    review_mode.add_argument("--review-bundle", action="store_true", help="Create a blinded human review entry packet, never an approval")
    a.add_argument("--teacher-sources", nargs="+", help="Limit local annotation to these source IDs")
    a.add_argument('--teacher-per-source',type=int,default=32)
    a.add_argument('--teacher-quality',action='store_true',help='Bounded source keep/reject/uncertain probe with literal evidence; no admission')
    a.add_argument('--teacher-max-seconds',type=int,default=1800)
    args = parser.parse_args(argv)
    if args.action == "profile":
        result = profile(json.loads(args.config.read_text()), args.out,
                         network=args.allow_network, metadata_only=args.metadata_only, resume_from=args.resume_from,
                         frozen_evidence_only=args.frozen_evidence_only)
    else:
        result = audit_coverage(args.surveys)
        if args.pro_calibrate_from:
            if not args.teacher_system_prompt or not args.teacher_offsets:
                parser.error('Pro calibration requires a rubric and explicit offsets')
            from teacher_calibration import calibrate
            result = calibrate(args.pro_calibrate_from, args.out, args.teacher_system_prompt, args.teacher_offsets)
        elif args.flash_compare_from:
            if args.cloud_teacher_model == 'deepseek-v4-pro':
                parser.error('Pro now uses cumulative tokens: use --pro-calibrate-from, --teacher-system-prompt and --teacher-offsets')
            if not args.teacher_pricing or not args.teacher_provider_status:
                parser.error('Flash replay requires provider pricing and status snapshots')
            from semantic_survey import flash_compare
            result = flash_compare(args.flash_compare_from, args.out, args.teacher_pricing, args.teacher_provider_status, args.teacher_continue_from, args.cloud_teacher_model)
        elif args.local_teacher_review:
            from semantic_survey import review
            result = review(args.surveys, args.out, source_ids=args.teacher_sources, per_source=args.teacher_per_source, quality=args.teacher_quality, max_seconds=args.teacher_max_seconds)
        elif args.review_bundle:
            from review_bundle import build
            result = build(args.surveys, result, args.out)
        else:
            write_new(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"artifacts", "outcomes", "wave_diagnostics", "source_dossiers"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
