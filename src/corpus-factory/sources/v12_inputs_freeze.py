"""Recount, encode and freeze all v1.2 CPT inputs against an explicit tokenizer."""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

from profiling import resolve_path, ROOT, digest

SEED = 20260914
SPLIT_BUCKETS = ((50, "test"), (100, "calibration"), (200, "dev"))
PHASE_NAMES = {1: "phase-1", 2: "phase-2", 3: "phase-3"}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(_json_bytes(value))
    tmp.replace(path)


def _strict_row(raw: bytes, path: Path, row_number: int) -> tuple[dict, str, str]:
    try:
        line = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 at {path}:{row_number}: {error}") from error
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{row_number}: {error}") from error
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"missing exact text at {path}:{row_number}")
    text.encode("utf-8", errors="strict")
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    declared = row.get("text_sha256")
    if declared and declared != text_sha:
        raise ValueError(f"text hash mismatch at {path}:{row_number}")
    return row, text, text_sha


def _compact_title(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"(?:第)?[0-9０-９]+(?:\s*[-–—]\s*[0-9０-９]+)?(?:回|卷|章|部)\s*$", "", value)
    value = re.sub(r"(?:chapters?|parts?|volumes?|books?)?\s*[0-9０-９]+(?:\s*[-–—]\s*[0-9０-９]+)?\s*$", "", value)
    value = re.sub(r"[\s:：._-]+", "", value)
    return value or "untitled"


def group_key(row: dict, source: str, text_sha: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    origin = row.get("origin") if isinstance(row.get("origin"), dict) else {}
    if metadata.get("work_group"):
        return "work:" + str(metadata["work_group"])
    if metadata.get("repo_name"):
        return "repo:" + str(metadata["repo_name"]).casefold()
    if metadata.get("project") and str(metadata.get("project")).casefold() != "unknown":
        return "project:" + str(metadata["project"]).casefold()
    if source == "project-gutenberg" and metadata.get("title"):
        return "work:gutenberg:" + _compact_title(str(metadata["title"]))
    if source.startswith("openapi-"):
        path = str(origin.get("path") or row.get("group_id") or "")
        match = re.search(r"APIs/([^/]+)", path)
        return "api:" + (match.group(1).casefold() if match else path.casefold())
    if source.startswith("opensubtitles-") and row.get("group_id"):
        raw_subtitle = re.sub(r"^opensubtitles-[^:]+:OpenSubtitles/raw/[^/]+/", "", str(row["group_id"]), flags=re.I)
        return "opensubtitles-work:" + raw_subtitle
    raw = row.get("group_id")
    if raw:
        raw = str(raw)
        return raw if raw.startswith(source + ":") else source + ":" + raw
    if metadata.get("id"):
        return source + ":id:" + str(metadata["id"])
    if metadata.get("title"):
        return source + ":title:" + _compact_title(str(metadata["title"]))
    return source + ":text:" + text_sha


def split_phase(group: str, forced_train: set[str], forced_dev: set[str]) -> tuple[str, int]:
    if group in forced_dev:
        return "dev", 0
    if group in forced_train:
        return "train", _phase(group)
    bucket = int(hashlib.sha256(f"{SEED}:split:{group}".encode()).hexdigest()[:12], 16) % 10_000
    for ceiling, label in SPLIT_BUCKETS:
        if bucket < ceiling:
            return label, 0
    return "train", _phase(group)


def _phase(group: str) -> int:
    bucket = int(hashlib.sha256(f"{SEED}:phase:{group}".encode()).hexdigest()[:12], 16) % 14
    return 1 if bucket < 5 else (2 if bucket < 10 else 3)


def priority(group: str, text_sha: str) -> int:
    # SQLite INTEGER is signed 64-bit.
    return int(hashlib.sha256(f"{SEED}:order:{group}:{text_sha}".encode()).hexdigest()[:15], 16)


def language_of(source: str, row: dict, text: str) -> str:
    if source in {"wiki-en", "wiki-en-foundation-expansion-20260913-v1"}:
        return "en"
    if source in {"hk-traditional-foundation-expansion-20260913-v1", "opensubtitles-zh-TW", "chinese-author-novel-cc-by-4"}:
        return "zh_hant"
    if source in {"web-hq-zh", "wiki-zh", "lccc-dialogue", "naturalconv", "opensubtitles-zh-CN", "ted2020-zh-cn", "crosswoz", "risawoz"}:
        return "zh_hans"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("language") == "en":
        return "en"
    if metadata.get("language") in {"zh", "zh-cn", "zh_CN"}:
        return "zh_hans"
    if source == "literature-zh-translation":
        return "zh_hant"
    han = sum(1 for char in text[:8192] if "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in text[:8192] if char.isascii() and char.isalpha())
    if latin > max(40, han * 2):
        return "en"
    if han:
        return "zh_mixed"
    return "structured_mixed"


def _sample_groups(path: Path) -> set[str]:
    result = set()
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group = row.get("group")
            if group:
                result.add(str(group))
    return result


def _eval_hashes() -> set[str]:
    """Conservative exact/line leaf hashes from every existing frozen eval bank."""
    values: set[str] = set()
    keys = {"query", "input", "target", "text", "prompt", "utterance"}

    def visit(value, key=""):
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in keys:
            norm = " ".join(value.split()).strip()
            if len(norm) >= 8:
                values.add(hashlib.sha256(norm.encode()).hexdigest())

    for path in sorted((resolve_path(ROOT / "corpus/eval-lock")).glob("*/banks/*.jsonl")):
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.strip():
                    visit(json.loads(line))
    return values


def _known_eval_overlap(text: str, protected: set[str]) -> bool:
    candidates = [text]
    candidates.extend(text.splitlines())
    for value in candidates:
        norm = " ".join(value.split()).strip()
        if len(norm) >= 8 and hashlib.sha256(norm.encode()).hexdigest() in protected:
            return True
    return False


def _schema(db: sqlite3.Connection) -> None:
    db.executescript("""
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=NORMAL;
      CREATE TABLE IF NOT EXISTS input_files(
        file_index INTEGER PRIMARY KEY, path TEXT UNIQUE, expected_sha TEXT,
        expected_records INTEGER, actual_sha TEXT, records INTEGER, unique_records INTEGER,
        tokens INTEGER, bin_path TEXT, bin_sha TEXT, status TEXT);
      CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY, file_index INTEGER, row_number INTEGER, source TEXT,
        domain TEXT, language TEXT, group_key TEXT, split TEXT, phase INTEGER,
        priority INTEGER, text_sha TEXT UNIQUE, token_offset INTEGER, token_length INTEGER,
        known_eval_overlap INTEGER DEFAULT 0);
      CREATE TABLE IF NOT EXISTS selected(record_id INTEGER PRIMARY KEY, phase INTEGER);
      CREATE TABLE IF NOT EXISTS selected_offsets(
        record_id INTEGER PRIMARY KEY, source TEXT, phase INTEGER,
        output_offset INTEGER, token_length INTEGER);
      CREATE INDEX IF NOT EXISTS records_source_phase_order ON records(source,split,phase,priority);
      CREATE INDEX IF NOT EXISTS records_domain_phase ON records(domain,split,phase,source);
      CREATE INDEX IF NOT EXISTS records_selection_order
        ON records(split,phase,domain,source,language,priority,id);
    """)


def _progress(db: sqlite3.Connection, out: Path, started: float, status: str) -> dict:
    cells = [dict(zip(("domain", "language", "records", "tokens"), row)) for row in db.execute(
        "SELECT domain,language,count(*),sum(token_length) FROM records GROUP BY domain,language ORDER BY domain,language"
    )]
    totals = db.execute("SELECT count(*),coalesce(sum(token_length),0) FROM records").fetchone()
    files = db.execute("SELECT count(*) FROM input_files WHERE status='complete'").fetchone()[0]
    value = {"schema": "mei-v12-input-progress-v1", "status": status,
             "files_complete": files, "unique_records": totals[0], "encoded_tokens": totals[1],
             "cells": cells, "elapsed_seconds": time.time() - started}
    _atomic_json(out / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def encode_inventory(config: dict, out: Path, db: sqlite3.Connection, tokenizer) -> None:
    plan = json.loads((resolve_path(ROOT / config["corpus_plan"])).read_text(encoding="utf-8"))
    domain_for = {source: item["domain"] for item in plan["domains"] for source in item["source_ids"]}
    reserved_sources = set(plan.get("reserved_source_ids", []))
    train_groups = _sample_groups(resolve_path(ROOT / config["tokenizer_train_sample"]))
    dev_groups = _sample_groups(resolve_path(ROOT / config["tokenizer_dev_sample"]))
    overlap = train_groups & dev_groups
    if overlap:
        raise ValueError(f"tokenizer train/dev association overlap: {sorted(overlap)[:5]}")
    eval_hashes = _eval_hashes()
    (out / "encoded-candidates").mkdir(exist_ok=True)
    started = time.time()
    next_report = (db.execute("SELECT coalesce(sum(token_length),0) FROM records").fetchone()[0] // 10_000_000 + 1) * 10_000_000
    for file_index, spec in enumerate(plan["candidate_files"]):
        existing = db.execute("SELECT status,bin_path,bin_sha FROM input_files WHERE file_index=?", (file_index,)).fetchone()
        if existing and existing[0] == "complete":
            bin_path = resolve_path(ROOT / existing[1])
            if not bin_path.is_file() or digest(bin_path) != existing[2]:
                raise ValueError(f"completed encoded candidate changed: {bin_path}")
            continue
        rel = spec["file"]
        path = resolve_path(ROOT / rel)
        tmp = out / "encoded-candidates" / f"file-{file_index:04d}.bin.tmp"
        final = out / "encoded-candidates" / f"file-{file_index:04d}.bin"
        tmp.unlink(missing_ok=True); final.unlink(missing_ok=True)
        raw_sha = hashlib.sha256(); bin_sha = hashlib.sha256()
        records = unique = tokens = offset = 0
        db.execute("BEGIN")
        try:
            db.execute("DELETE FROM records WHERE file_index=?", (file_index,))
            with path.open("rb") as source_handle, tmp.open("xb") as token_handle:
                for row_number, raw in enumerate(source_handle):
                    raw_sha.update(raw)
                    if not raw.strip():
                        continue
                    records += 1
                    row, text, text_sha = _strict_row(raw, path, row_number)
                    source = str(row.get("source_id") or "")
                    if source in reserved_sources:
                        continue
                    domain = domain_for.get(source)
                    if not domain:
                        raise ValueError(f"unmapped source {source!r} at {path}:{row_number}")
                    group = group_key(row, source, text_sha)
                    split, phase = split_phase(group, train_groups, dev_groups)
                    known_overlap = int(_known_eval_overlap(text, eval_hashes))
                    if known_overlap:
                        split, phase = "existing_eval_overlap", 0
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO records(file_index,row_number,source,domain,language,group_key,split,phase,priority,text_sha,token_offset,token_length,known_eval_overlap) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (file_index,row_number,source,domain,language_of(source,row,text),group,split,phase,priority(group,text_sha),text_sha,offset,0,known_overlap))
                    if cursor.rowcount == 0:
                        continue
                    ids = tokenizer.encode_document(text)
                    if not ids or max(ids) >= 24_000 or min(ids) < 0:
                        raise ValueError(f"invalid token IDs at {path}:{row_number}")
                    payload = array("H", ids).tobytes()
                    if sys.byteorder != "little":
                        raise RuntimeError("uint16 corpus writer requires little-endian host")
                    token_handle.write(payload); bin_sha.update(payload)
                    length = len(ids)
                    db.execute("UPDATE records SET token_length=? WHERE text_sha=?", (length,text_sha))
                    unique += 1; tokens += length; offset += length
                    total_now = db.execute("SELECT coalesce(sum(tokens),0) FROM input_files WHERE status='complete'").fetchone()[0] + tokens
                    if total_now >= next_report:
                        if shutil.disk_usage(out).free < 100 * 1024**3:
                            raise RuntimeError("100 GiB free disk reserve reached during encoding")
                        db.commit(); db.execute("BEGIN")
                        _progress(db, out, started, "encoding")
                        next_report = (total_now // 10_000_000 + 1) * 10_000_000
            if raw_sha.hexdigest() != spec["sha256"]:
                raise ValueError(f"input file hash changed: {rel}")
            if records != int(spec["records"]):
                raise ValueError(f"input record count changed: {rel}: {records} != {spec['records']}")
            tmp.replace(final)
            final_rel = str(final.relative_to(ROOT))
            db.execute("INSERT OR REPLACE INTO input_files VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (file_index,rel,spec["sha256"],spec["records"],raw_sha.hexdigest(),records,unique,tokens,final_rel,bin_sha.hexdigest(),"complete"))
            db.commit()
        except Exception:
            db.rollback(); tmp.unlink(missing_ok=True); final.unlink(missing_ok=True)
            raise
        _progress(db, out, started, "encoding")


def _allocate(capacities: dict[str, int], target: int) -> dict[str, int]:
    capacities = {key: max(0, int(value)) for key, value in capacities.items() if value > 0}
    target = min(int(target), sum(capacities.values()))
    if not capacities or target <= 0:
        return {key: 0 for key in capacities}
    total = sum(capacities.values())
    result = {key: min(value, int(target * value / total)) for key, value in capacities.items()}
    remaining = target - sum(result.values())
    order = sorted(capacities, key=lambda key: (-(target * capacities[key] / total - result[key]), key))
    index = 0
    while remaining:
        key = order[index % len(order)]; index += 1
        if result[key] < capacities[key]:
            result[key] += 1; remaining -= 1
    return result


def select_records(config: dict, db: sqlite3.Connection) -> dict:
    plan = json.loads((resolve_path(ROOT / config["corpus_plan"])).read_text(encoding="utf-8"))
    db.execute("DELETE FROM selected")
    allocation = []
    for phase_spec in plan["phases"]:
        phase = int(phase_spec["phase"])
        for domain, requested in phase_spec["domain_target_tokens"].items():
            rows = db.execute("SELECT source,language,sum(token_length) FROM records WHERE split='train' AND phase=? AND domain=? GROUP BY source,language", (phase,domain)).fetchall()
            cells = {(source,language): int(tokens) for source,language,tokens in rows}
            cell_targets: dict[tuple[str,str],int] = {}
            language_weights = plan.get("language_weights", {}).get(domain)
            if language_weights is not None:
                if not language_weights or any(v <= 0 for v in language_weights.values()):
                    raise ValueError("language weights must be positive")
                lang_targets = _allocate({k: int(v) * requested for k, v in language_weights.items()}, requested)
                for language, lang_target in lang_targets.items():
                    subset = {source: tokens for (source,lang),tokens in cells.items() if lang == language}
                    for source,target in _allocate(subset,lang_target).items(): cell_targets[(source,language)] = target
            elif domain == "foundation":
                lang_targets = {"zh_hans": round(requested*.8), "zh_hant": round(requested*.1)}
                lang_targets["en"] = requested - sum(lang_targets.values())
                for language, lang_target in lang_targets.items():
                    subset = {source: tokens for (source,lang),tokens in cells.items() if lang == language}
                    for source,target in _allocate(subset,lang_target).items(): cell_targets[(source,language)] = target
            elif domain == "oral":
                lang_targets = {"zh_hans": round(requested*.9), "zh_hant": requested-round(requested*.9)}
                for language, lang_target in lang_targets.items():
                    subset = {source: tokens for (source,lang),tokens in cells.items() if lang == language}
                    for source,target in _allocate(subset,lang_target).items(): cell_targets[(source,language)] = target
            else:
                flat = {source + "\0" + language: tokens for (source,language),tokens in cells.items()}
                for key,target in _allocate(flat,requested).items():
                    source,language=key.split("\0",1); cell_targets[(source,language)] = target
            for (source,language), target in sorted(cell_targets.items()):
                chosen=[]; actual=0
                for record_id,length in db.execute("SELECT id,token_length FROM records WHERE split='train' AND phase=? AND domain=? AND source=? AND language=? ORDER BY priority,id", (phase,domain,source,language)):
                    if actual >= target: break
                    chosen.append((record_id,phase)); actual += int(length)
                db.executemany("INSERT INTO selected(record_id,phase) VALUES(?,?)", chosen)
                allocation.append({"phase":phase,"domain":domain,"source":source,"language":language,"target_tokens":target,"selected_tokens":actual,"records":len(chosen)})
    db.commit()
    return {"cells": allocation}


def _copy_tokens(source_handle, source_path: Path, source_offset: int, length: int, target_handle, target_sha) -> None:
    source_handle.seek(source_offset * 2)
    payload = source_handle.read(length * 2)
    if len(payload) != length * 2:
        raise ValueError(f"encoded candidate truncated: {source_path}")
    target_handle.write(payload); target_sha.update(payload)


def rounded_stage_quotas(source_total: dict, targets: list[int], seq_len: int = 2048) -> dict:
    """Allocate whole windows, preserving unique capacity and reporting real exposure.

    The existing sampler counts complete windows. Round targets up to windows
    explicitly instead of advertising exact quotas that it would overrun.
    """
    if len(targets) != 3 or any(type(t) is not int or t <= 0 for t in targets):
        raise ValueError("three positive integer stage targets required")
    available = {s: max(0, (int(n)-1)//seq_len) for s,n in source_total.items()}
    wanted = [(t+seq_len-1)//seq_len for t in targets]
    if sum(wanted) > sum(available.values()):
        raise ValueError("insufficient unique window capacity; repetition forbidden")
    result = {s: {} for s in available}
    for phase, target in enumerate(wanted, 1):
        allocated = _allocate(available, target)
        for source, n in allocated.items():
            result[source][phase] = n*seq_len
            available[source] -= n
    return result


def materialize_release(config: dict, out: Path, db: sqlite3.Connection, selection: dict, tokenizer_manifest: Path) -> dict:
    release = resolve_path(ROOT / config["release_dir"])
    staging = release.with_name("." + release.name + ".staging")
    if release.exists():
        raise FileExistsError(release)
    if staging.exists(): shutil.rmtree(staging)
    (staging / "tokens").mkdir(parents=True)
    file_bins = {row[0]: resolve_path(ROOT / row[1]) for row in db.execute("SELECT file_index,bin_path FROM input_files WHERE status='complete'")}
    sources = [row[0] for row in db.execute("SELECT DISTINCT source FROM records ORDER BY source")]
    token_files=[]; source_phase=defaultdict(Counter); source_total=Counter()
    db.execute("DELETE FROM selected_offsets")
    with ExitStack() as stack:
        source_handles = {index: stack.enter_context(path.open("rb")) for index, path in file_bins.items()}
        offset_rows=[]
        for source in sources:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", source)
            path = staging / "tokens" / f"train-{safe}.bin"; sha=hashlib.sha256(); offset=0; records=0
            with path.open("xb") as target:
                rows=db.execute("SELECT r.id,r.file_index,r.row_number,r.token_offset,r.token_length,r.text_sha,r.group_key,r.domain,r.language,s.phase FROM records r JOIN selected s ON s.record_id=r.id WHERE r.source=? ORDER BY s.phase,r.priority,r.id",(source,))
                for record_id,file_index,row_number,source_offset,length,text_sha,group,domain,language,phase in rows:
                    _copy_tokens(source_handles[file_index],file_bins[file_index],source_offset,length,target,sha)
                    offset_rows.append((record_id,source,phase,offset,length))
                    if len(offset_rows) >= 10_000:
                        db.executemany("INSERT INTO selected_offsets VALUES(?,?,?,?,?)",offset_rows); offset_rows.clear()
                    offset += length; records += 1; source_phase[(source,phase)]["tokens"]+=length; source_phase[(source,phase)]["records"]+=1; source_total[source]+=length
            if records:
                token_files.append({"source":source,"path":str(path.relative_to(staging)),"sha256":sha.hexdigest(),"tokens":offset,"records":records})
            else: path.unlink()
        if offset_rows:
            db.executemany("INSERT INTO selected_offsets VALUES(?,?,?,?,?)",offset_rows)
        db.commit()
        # Freeze all non-training splits as base-LM development/calibration/test reserves.
        reserve_files=[]
        for split in ("dev","calibration","test","existing_eval_overlap"):
            for source in sources:
                safe=re.sub(r"[^A-Za-z0-9_.-]+","-",source); path=staging/"tokens"/f"{split}-{safe}.bin"; sha=hashlib.sha256(); offset=records=0
                with path.open("xb") as target:
                    for file_index,source_offset,length in db.execute("SELECT file_index,token_offset,token_length FROM records WHERE source=? AND split=? ORDER BY priority,id",(source,split)):
                        _copy_tokens(source_handles[file_index],file_bins[file_index],source_offset,length,target,sha); offset+=length; records+=1
                if records: reserve_files.append({"split":split,"source":source,"path":str(path.relative_to(staging)),"sha256":sha.hexdigest(),"tokens":offset,"records":records})
                else: path.unlink()
    # A compact relational record index retains raw file/line, group, split and token references.
    index_path=staging/"record-index.sqlite"
    target_db=sqlite3.connect(index_path)
    db.backup(target_db); target_db.close()
    index_sha=digest(index_path)
    # Align the first two continuation cursors. The last stage consumes every
    # remaining predictable token once, so a stage transition never skips or wraps.
    aligned={}
    for source in sources:
        q1=(int(source_phase[(source,1)]["tokens"])//2048)*2048
        q2=(int(source_phase[(source,2)]["tokens"])//2048)*2048
        q3=max(0,int(source_total[source])-1-q1-q2)
        aligned[source]={1:q1,2:q2,3:q3}
    if config.get("stage_target_tokens"):
        aligned = rounded_stage_quotas(source_total, config["stage_target_tokens"])
    phases=[]; cumulative=0
    for phase in (1,2,3):
        quotas={source:aligned[source][phase] for source in sources if aligned[source][phase]}
        actual=sum(quotas.values()); cumulative+=actual
        phases.append({"id":PHASE_NAMES[phase],"index":phase-1,"seq_len":2048,"stage_tokens":actual,"stop_at_tokens":cumulative,
                       "sources":{source:{"token_quota":amount} for source,amount in sorted(quotas.items())}})
    release_rel=config["release_dir"]
    repo_paths={item["source"]:str(Path(release_rel)/item["path"]) for item in token_files}
    schedule={"schema":"mei-quota-curriculum-v2","sampler":"quota_plan","seed":SEED,"sampler_seed":SEED,"seq_len":2048,"exposure_tokens":cumulative,
              "sources":{item["source"]:{"paths":[repo_paths[item["source"]]],"token_quota":source_total[item["source"]]} for item in token_files},"curriculum":phases,
              "continuation":{"preserve":["rng_state","source_order","token_cursors","source_tokens_drawn"],"stage_transition_resets_data_cursor":False}}
    _atomic_json(staging/"schedule.json",schedule)
    mix_sources={item["source"]:{"train_shards":[repo_paths[item["source"]]]} for item in token_files}
    for item in reserve_files:
        mix_sources.setdefault(item["source"],{}).setdefault(item["split"]+"_shards",[]).append(str(Path(release_rel)/item["path"]))
    mix={"schema":"mei-cpt-uint16-mix-v2","sources":mix_sources,
         "train_shards":[repo_paths[x["source"]] for x in token_files],
         "valid_shards":[str(Path(release_rel)/x["path"]) for x in reserve_files if x["split"]=="dev"],
         "valid_sets":{"base_dev":[str(Path(release_rel)/x["path"]) for x in reserve_files if x["split"]=="dev"]},
         "tokenizer_manifest":"TOKENIZER.json","record_index":"record-index.sqlite"}
    _atomic_json(staging/"mix.json",mix)
    tokenizer_release=json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
    split_counts=[dict(zip(("split","domain","language","records","tokens"),row)) for row in db.execute("SELECT split,domain,language,count(*),sum(token_length) FROM records GROUP BY split,domain,language ORDER BY split,domain,language")]
    selected_counts=[dict(zip(("phase","domain","language","records","tokens"),row)) for row in db.execute("SELECT s.phase,r.domain,r.language,count(*),sum(r.token_length) FROM records r JOIN selected s ON s.record_id=r.id GROUP BY s.phase,r.domain,r.language ORDER BY s.phase,r.domain,r.language")]
    books={"candidate_batch_old_tokenizer":json.loads((resolve_path(ROOT/config["corpus_plan"])).read_text()).get("candidate_total_tokens"),
           "eligible_unique_new_tokenizer":db.execute("SELECT coalesce(sum(token_length),0) FROM records WHERE split!='existing_eval_overlap'").fetchone()[0],
           "selected_net_new_tokenizer":sum(source_total.values()),"planned_exposure":cumulative}
    manifest={"schema":"mei-v12-cpt-input-release-v1","release_id":config["release_id"],"status":"validating","tokenizer":{
        "manifest":"TOKENIZER.json","tokenizer_id":tokenizer_release["tokenizer_id"],"model_sha256":tokenizer_release["model_sha256"],"encoding_profile_id":tokenizer_release["encoding_profile_id"],"vocab_size":tokenizer_release["vocab_size"]},
        "split_seed":SEED,"split_policy":{"train":.98,"dev":.01,"calibration":.005,"test":.005,"group_before_window":True},"books":books,
        "candidate_files":db.execute("SELECT count(*) FROM input_files WHERE status='complete'").fetchone()[0],"unique_records":db.execute("SELECT count(*) FROM records").fetchone()[0],
        "exact_duplicates_removed":sum(x[0] for x in db.execute("SELECT expected_records-unique_records FROM input_files")),"known_eval_overlaps_isolated":db.execute("SELECT count(*) FROM records WHERE known_eval_overlap=1").fetchone()[0],
        "split_counts":split_counts,"selected_counts":selected_counts,"source_phase_counts":[{"source":s,"phase":p,**dict(c)} for (s,p),c in sorted(source_phase.items())],
        "token_files":token_files,"reserve_files":reserve_files,"record_index":{"path":"record-index.sqlite","sha256":index_sha},"schedule":{"path":"schedule.json","sha256":digest(staging/"schedule.json")},
        "format":{"dtype":"little-endian uint16","document_boundaries":"BOS/EOS in every indexed record","loss_mask":"all document tokens; runtime windows derive padding masks","long_content":"retained and packed across 2048-token windows; never rejected for length"},
        "limitations":["source-level public-data rights metadata is preserved; this freeze is not a commercial rights clearance","known eval isolation checks exact normalized records and lines plus association groups; semantic paraphrase leakage remains an evaluation audit task","CPT reserves are base-language diagnostics, not substitutes for separately frozen SFT/task eval"],
        "current_mutated":False,"a10_started":False,"cpt_started":False}
    if config.get("stage_target_tokens"):
        manifest["stage_partition"] = {
            "requested_new_exposure":config["stage_target_tokens"],
            "actual_new_exposure":[p["stage_tokens"] for p in phases],
            "rule":"whole 2048-token windows; stable source cursors; no repeated exposure",
            "record_phase_field":"original physical selection bucket, not the new exposure stage",
            "unexposed_selected_tokens":sum(source_total.values())-cumulative,
        }
    tokenizer_model = tokenizer_manifest.parent / tokenizer_release["model_file"]
    tokenizer_model_target = staging / tokenizer_release["model_file"]
    shutil.copyfile(tokenizer_model, tokenizer_model_target)
    if digest(tokenizer_model_target) != tokenizer_release["model_sha256"]:
        raise ValueError("bundled tokenizer model hash mismatch")
    if tokenizer_release.get("vocab_file"):
        tokenizer_vocab = tokenizer_manifest.parent / tokenizer_release["vocab_file"]
        tokenizer_vocab_target = staging / tokenizer_release["vocab_file"]
        shutil.copyfile(tokenizer_vocab, tokenizer_vocab_target)
        if digest(tokenizer_vocab_target) != tokenizer_release.get("vocab_sha256"):
            raise ValueError("bundled tokenizer vocab hash mismatch")
    shutil.copyfile(tokenizer_manifest,staging/"TOKENIZER.json")
    _atomic_json(staging/"RELEASE.json",manifest)
    release.parent.mkdir(parents=True,exist_ok=True); staging.replace(release)
    return manifest


def validate_sampler(config: dict, release: Path) -> dict:
    sys.path[:0] = [
        str(ROOT/"src/model-factory"),
        str(ROOT/"src/platform/_shared/runtime"),
        str(ROOT/"src/architecture/mei-1.2-51m"),
    ]
    from common.data import PackedTokenSource, QuotaPackedSources, build_quota_plan
    schedule=json.loads((release/"schedule.json").read_text())
    plans=[]
    for stage in schedule["curriculum"]:
        quotas={name:cfg["token_quota"] for name,cfg in stage["sources"].items()}
        stage_seed=SEED+int(stage["index"])
        p1=build_quota_plan(sorted(quotas),quotas,2048,stage_seed)
        p2=build_quota_plan(sorted(quotas),quotas,2048,stage_seed)
        if p1!=p2: raise ValueError("full quota plan is not reproducible")
        plans.append({"stage":stage["id"],"windows":len(p1),"plan_sha256":hashlib.sha256("\n".join(p1).encode()).hexdigest(),"source_windows":dict(Counter(p1))})
    first=schedule["curriculum"][0]; names=sorted(first["sources"])
    sample_names=names[:min(5,len(names))]
    sources={name:PackedTokenSource([resolve_path(ROOT/schedule["sources"][name]["paths"][0])],2048,0) for name in sample_names}
    quotas={name:min(int(first["sources"][name]["token_quota"]),2048*1600) for name in sample_names}
    a=QuotaPackedSources(sources,quotas,seed=SEED,seq_len=2048,stage_id="replay")
    before=a.take_windows(1024); state=a.state_dict(); expected=a.take_windows(1024)
    b=QuotaPackedSources(sources,quotas,seed=SEED,seq_len=2048,stage_id="replay"); b.load_state_dict(state); actual=b.take_windows(1024)
    def signature(rows):
        h=hashlib.sha256()
        for row in rows:
            h.update(str((row["source_id"],row["source_token_cursor"],row["n_predictable"])).encode()); h.update(array("H",row["x"]).tobytes())
        return h.hexdigest()
    if signature(expected)!=signature(actual): raise ValueError("sampler resume changed token/source sequence")
    report={"schema":"mei-v12-sampler-validation-v1","status":"passed","full_plan_replay":plans,"sample_sources":sample_names,"prefix_windows":len(before),"resumed_windows":len(actual),
            "expected_sha256":signature(expected),"actual_sha256":signature(actual),"state_fields":sorted(state),"source_exhaustion_policy":"fail; no wraparound"}
    _atomic_json(release/"SAMPLER-VALIDATION.json",report); return report


def run(config: dict, out: Path) -> dict:
    out = (resolve_path(ROOT / out)).resolve() if not out.is_absolute() else out.resolve()
    if int(config.get("seed",0)) != SEED: raise ValueError("split seed must be 20260914")
    if shutil.disk_usage(ROOT).free < 100*1024**3: raise ValueError("100 GiB disk reserve reached")
    tokenizer_manifest=(resolve_path(ROOT/config["tokenizer_manifest"])).resolve()
    sys.path.insert(0,str(ROOT/"src/corpus-factory/sources"))
    from source_manager import load_tokenizer
    tokenizer=load_tokenizer(tokenizer_manifest)
    if tokenizer.vocab_size!=24_000 or tokenizer.encoding_profile_id!="mei-lossless-identity-v1": raise ValueError("wrong explicit tokenizer")
    out.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(_json_bytes(config)).hexdigest(); lock=out/"RUN.json"
    if lock.exists():
        if json.loads(lock.read_text()).get("config_sha256")!=config_hash: raise ValueError("resume config changed")
    else:
        _atomic_json(lock,{"schema":"mei-v12-input-run-v1","config_sha256":config_hash,"status":"active"})
        shutil.copyfile(__file__,out/"implementation.py.snapshot")
    cache = config.get("reuse_encoded_run")
    if cache and not (out/"index.sqlite").exists():
        previous = resolve_path(ROOT/cache)
        previous_config = json.loads(resolve_path(ROOT/config["reuse_encoded_config"]).read_text())
        previous_lock = json.loads((previous/"RUN.json").read_text())
        if hashlib.sha256(_json_bytes(previous_config)).hexdigest() != previous_lock["config_sha256"]:
            raise ValueError("cached encoding config binding mismatch")
        if digest(resolve_path(ROOT/previous_config["tokenizer_manifest"])) != digest(tokenizer_manifest):
            raise ValueError("cached encoding uses another tokenizer")
        old = sqlite3.connect(f"file:{previous/'index.sqlite'}?mode=ro", uri=True)
        bins = old.execute("SELECT bin_path,bin_sha,tokens,status FROM input_files").fetchall()
        if not bins or any(row[3] != "complete" for row in bins):
            raise ValueError("cached encoding is incomplete")
        for name, expected, tokens, _ in bins:
            path = resolve_path(ROOT/name)
            if path.stat().st_size != tokens*2 or digest(path) != expected:
                raise ValueError("cached token bytes mismatch: " + name)
        temp_index = out/"index.sqlite.partial"
        if temp_index.exists():
            raise FileExistsError(temp_index)
        target = sqlite3.connect(temp_index)
        old.backup(target); target.close(); old.close()
        temp_index.replace(out/"index.sqlite")
        _atomic_json(out/"CACHED-ENCODING.json",{"run":cache,"config_sha256":previous_lock["config_sha256"],
            "tokenizer_manifest_sha256":digest(tokenizer_manifest),"verified_files":len(bins),
            "index_sha256":digest(out/"index.sqlite"),"original_run_mutated":False})
    db=sqlite3.connect(out/"index.sqlite",timeout=60); _schema(db)
    if not cache:
        encode_inventory(config,out,db,tokenizer)
    release=resolve_path(ROOT/config["release_dir"])
    if release.exists():
        manifest=json.loads((release/"RELEASE.json").read_text(encoding="utf-8"))
        if manifest.get("release_id") != config["release_id"] or manifest.get("status") != "validating":
            raise FileExistsError(release)
        if manifest.get("tokenizer",{}).get("model_sha256") != tokenizer.model_sha256:
            raise ValueError("existing validating release uses another tokenizer")
    else:
        if cache:
            if not (out/"CACHED-ENCODING.json").exists():
                raise ValueError("cached input reuse receipt missing")
            selection={"reuse_selected_records":True,"run":cache}
            if db.execute("SELECT count(*) FROM selected").fetchone()[0] == 0:
                raise ValueError("no selected cached records")
        else:
            selection=select_records(config,db)
        _atomic_json(out/"selection.json",selection)
        manifest=materialize_release(config,out,db,selection,tokenizer_manifest)
    sampler=validate_sampler(config,release)
    manifest=json.loads((release/"RELEASE.json").read_text()); manifest["sampler_validation"]={"path":"SAMPLER-VALIDATION.json","sha256":digest(release/"SAMPLER-VALIDATION.json"),"status":sampler["status"]}; manifest["status"]="inputs_ready"
    _atomic_json(release/"RELEASE.json",manifest)
    _atomic_json(lock,{"schema":"mei-v12-input-run-v1","config_sha256":config_hash,"status":"complete","release":config["release_dir"],"release_sha256":digest(release/"RELEASE.json")})
    _progress(db,out,time.time(),"inputs_ready"); db.close(); return manifest
