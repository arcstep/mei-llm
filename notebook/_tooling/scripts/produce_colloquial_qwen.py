#!/usr/bin/env python3
"""Realtime qwen-plus colloquial producer with SQLite durable queue.

Writes only to an independent qwen corpus dir. Never resumes into the offline
v1 accepted.jsonl. formal_cpt_eligible is never set here.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo_paths import (
    CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1,
    CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1,
    CORPORA_ROOT,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_release import freeze_release, load_leaks, tokenize_kept  # noqa: E402
from colloquial_synth_lib import (  # noqa: E402
    filter_reasons,
    generate_one,
    iter_frames,
    load_contract,
    provenance_row,
    sha256_text,
    spend_cny,
)
from tokenizer import ZhTokenizerV1  # noqa: E402
from zh_pretrain_ingest import (  # noqa: E402
    NearDupIndex,
    colloquial_keep,
    dump_json,
    hash_bucket,
    pii_or_nav,
    simhash64,
)

GENERATOR = "qwen-plus"
ID_PREFIX = "csynth-qwen-v1"
OFFLINE_DIR_NAMES = {
    "zh-pretrain-colloquial-synth-v1",
    "zh-pretrain-colloquial-synth-v2",
    "zh-pretrain-colloquial-synth-smoke",
}


class TokenBucket:
    def __init__(self, qps: float):
        self.interval = 1.0 / max(float(qps), 0.05)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def take(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.next_t = now + self.interval


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
          frame_id TEXT PRIMARY KEY,
          idx INTEGER NOT NULL UNIQUE,
          frame_json TEXT NOT NULL,
          status TEXT NOT NULL,
          text TEXT,
          n_tokens INTEGER,
          n_unk INTEGER,
          split TEXT,
          prompt_tokens INTEGER DEFAULT 0,
          completion_tokens INTEGER DEFAULT 0,
          spend_cny REAL DEFAULT 0,
          request_sha256 TEXT,
          response_sha256 TEXT,
          model_snapshot TEXT,
          error TEXT,
          reasons TEXT,
          retries INTEGER DEFAULT 0,
          flushed INTEGER DEFAULT 0,
          updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_flushed ON jobs(flushed, status)")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    return conn


def meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return str(row["v"]) if row else default


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", (key, value))


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os_open(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"producer already running: {path}") from exc
    return fd


def os_open(path: Path):
    return os.open(str(path), os.O_CREAT | os.O_RDWR)


def refuse_offline_dir(out: Path, generator: str) -> None:
    if out.resolve() == CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1.resolve() or out.name in OFFLINE_DIR_NAMES:
        raise RuntimeError(
            f"refuse production into offline/engineering dir {out}."
        )


def refuse_qwen_v1_sidecar(out: Path, *, generator: str, model: str, frozen: str) -> None:
    if out.resolve() != CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1.resolve():
        return
    if generator != "qwen-plus" or model != frozen:
        raise RuntimeError(
            f"refuse sidecar generator={generator} model={model} into qwen-v1; use a separate corpus dir"
        )


def enqueue_frames(conn: sqlite3.Connection, lock: threading.Lock, frames: list[dict]) -> int:
    n = 0
    now = time.time()
    with lock:
        for frame in frames:
            try:
                conn.execute(
                    "INSERT INTO jobs(frame_id, idx, frame_json, status, updated_at) VALUES(?,?,?,?,?)",
                    (frame["frame_id"], int(frame["index"]), json.dumps(frame, ensure_ascii=False), "pending", now),
                )
                n += 1
            except sqlite3.IntegrityError:
                continue
    return n


def claim_job(conn: sqlite3.Connection, lock: threading.Lock) -> dict | None:
    with lock:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT frame_id, idx, frame_json FROM jobs WHERE status='pending' ORDER BY idx LIMIT 1"
        ).fetchone()
        if not row:
            conn.execute("COMMIT")
            return None
        cur = conn.execute(
            "UPDATE jobs SET status='running', updated_at=? WHERE frame_id=? AND status='pending'",
            (time.time(), row["frame_id"]),
        )
        conn.execute("COMMIT")
        if cur.rowcount != 1:
            return None
        return {"frame_id": row["frame_id"], "idx": int(row["idx"]), "frame": json.loads(row["frame_json"])}


def sums(conn: sqlite3.Connection, lock: threading.Lock | None = None) -> dict:
    sql = """
        SELECT
          COALESCE(SUM(CASE WHEN status='succeeded' AND split='train' THEN n_tokens ELSE 0 END), 0) AS unique_train,
          COALESCE(SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END), 0) AS n_keep,
          COALESCE(SUM(spend_cny), 0) AS spend,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END), 0) AS n_pending,
          COALESCE(SUM(CASE WHEN status='running' THEN 1 ELSE 0 END), 0) AS n_running,
          COALESCE(MAX(idx), -1) AS max_idx
        FROM jobs
        """
    if lock:
        with lock:
            row = conn.execute(sql).fetchone()
    else:
        row = conn.execute(sql).fetchone()
    return {k: row[k] for k in row.keys()}


def recover_running(conn: sqlite3.Connection, lock: threading.Lock | None = None, stale_s: float = 180.0) -> int:
    cutoff = time.time() - stale_s
    params = (time.time(), cutoff)
    sql = "UPDATE jobs SET status='pending', updated_at=? WHERE status='running' AND updated_at<?"
    if lock:
        with lock:
            cur = conn.execute(sql, params)
    else:
        cur = conn.execute(sql, params)
    return int(cur.rowcount or 0)


def flush_jsonl(
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    accepted,
    rejected,
    write_lock: threading.Lock,
    contract: dict,
) -> int:
    with db_lock:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE flushed=0 AND status IN ('succeeded','rejected') ORDER BY idx"
        ).fetchall()
    n = 0
    with write_lock:
        for row in rows:
            payload = json.loads(row["frame_json"])
            rec = provenance_row(
                frame=payload,
                text=row["text"] or "",
                generator=GENERATOR,
                contract=contract,
                request_sha256=str(row["request_sha256"] or ""),
                response_sha256=str(row["response_sha256"] or ""),
                usage={
                    "prompt_tokens": int(row["prompt_tokens"] or 0),
                    "completion_tokens": int(row["completion_tokens"] or 0),
                },
                filter_ok=row["status"] == "succeeded",
                reasons=json.loads(row["reasons"] or "[]"),
                n_tokens=int(row["n_tokens"] or 0),
                split=str(row["split"] or "train"),
            )
            rec["n_unk"] = row["n_unk"]
            rec["error"] = row["error"]
            rec["retries"] = row["retries"]
            rec["spend_cny"] = row["spend_cny"]
            rec["model_snapshot"] = row["model_snapshot"] or rec.get("model_snapshot")
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if row["status"] == "succeeded":
                accepted.write(line)
            else:
                rejected.write(line)
            n += 1
        accepted.flush()
        rejected.flush()
    if rows:
        with db_lock:
            conn.executemany(
                "UPDATE jobs SET flushed=1 WHERE frame_id=?",
                [(r["frame_id"],) for r in rows],
            )
    return n


def process_job(
    *,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    job: dict,
    contract: dict,
    tok: ZhTokenizerV1,
    leaks: list[str],
    seen_sha: set[str],
    seen_lock: threading.Lock,
    near: NearDupIndex,
    spend_state: dict,
    spend_lock: threading.Lock,
    stats: dict,
    stats_lock: threading.Lock,
) -> None:
    frame = job["frame"]
    gen = generate_one(frame, generator=GENERATOR, contract=contract)
    usage = gen.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cost = spend_cny(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, contract=contract)
    with spend_lock:
        spend_state["cny"] += cost
        spend_state["prompt_tokens"] += prompt_tokens
        spend_state["completion_tokens"] += completion_tokens
    now = time.time()
    if not gen.get("ok"):
        err = str(gen.get("error") or "qwen_error")
        requeue = any(x in err for x in ("qwen_http_429", "qwen_http_5", "qwen_timeout", "qwen_http_408"))
        with stats_lock:
            stats["n_gen_fail"] += 1
        with db_lock:
            if requeue:
                conn.execute(
                    "UPDATE jobs SET status='pending', error=?, retries=?, updated_at=? WHERE frame_id=?",
                    (err, int(gen.get("retries") or 0), now, job["frame_id"]),
                )
                time.sleep(1.5)
                return
            else:
                conn.execute(
                    """
                    UPDATE jobs SET status='rejected', error=?, retries=?, prompt_tokens=?, completion_tokens=?,
                      spend_cny=?, reasons=?, updated_at=? WHERE frame_id=?
                    """,
                    (
                        err,
                        int(gen.get("retries") or 0),
                        prompt_tokens,
                        completion_tokens,
                        cost,
                        json.dumps(["gen_fail"], ensure_ascii=False),
                        now,
                        job["frame_id"],
                    ),
                )
        return
    text = str(gen.get("text") or "")
    reasons = filter_reasons(text, leaks=leaks, pii_fn=pii_or_nav)
    if not colloquial_keep(text, domain="dialogue"):
        reasons.append("not_spoken")
    digest = sha256_text(text)
    with seen_lock:
        if digest in seen_sha:
            reasons.append("exact_dup")
            with stats_lock:
                stats["n_dup"] += 1
        sim = simhash64(text)
        if near.near(sim):
            reasons.append("near_dup")
            with stats_lock:
                stats["n_near"] += 1
        elif not reasons:
            seen_sha.add(digest)
            near.add(sim)
    ids = tok.encode_document(text)
    n_tok = len(ids)
    n_unk = sum(1 for t in ids if t == tok.unk_id)
    if n_tok and (n_unk / n_tok) > 0.02:
        reasons.append("unk")
    split = hash_bucket(digest)
    snap = str(gen.get("model_snapshot") or contract["generator"]["frozen_snapshot"])
    want = contract["generator"]["frozen_snapshot"]
    if snap and want not in snap and snap != want:
        reasons.append("snapshot_mismatch")
    ok = not reasons
    row = provenance_row(
        frame=frame,
        text=text,
        generator=GENERATOR,
        contract=contract,
        request_sha256=str(gen.get("request_sha256") or ""),
        response_sha256=digest,
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        filter_ok=ok,
        reasons=reasons,
        n_tokens=n_tok,
        split=split,
    )
    status = "succeeded" if ok else "rejected"
    if not ok:
        with stats_lock:
            stats["n_filter"] += 1
    else:
        with stats_lock:
            stats["n_keep"] += 1
    with db_lock:
        conn.execute(
            """
            UPDATE jobs SET status=?, text=?, n_tokens=?, n_unk=?, split=?, prompt_tokens=?, completion_tokens=?,
              spend_cny=?, request_sha256=?, response_sha256=?, model_snapshot=?, error=?, reasons=?,
              retries=?, updated_at=? WHERE frame_id=?
            """,
            (
                status,
                text,
                n_tok,
                n_unk,
                split,
                prompt_tokens,
                completion_tokens,
                cost,
                row["request_sha256"],
                digest,
                snap,
                None if ok else ",".join(reasons),
                json.dumps(reasons, ensure_ascii=False),
                int(gen.get("retries") or 0),
                now,
                job["frame_id"],
            ),
        )


def freeze_now(
    out: Path,
    *,
    contract: dict,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    tok: ZhTokenizerV1,
    stats: dict,
    release_kind: str,
    spend_state: dict,
) -> dict:
    kept = []
    with db_lock:
        rows = conn.execute("SELECT * FROM jobs WHERE status='succeeded' ORDER BY idx").fetchall()
    for row in rows:
        kept.append(
            {
                "doc_id": row["frame_id"],
                "frame": json.loads(row["frame_json"]),
                "text": row["text"],
                "n_tokens": int(row["n_tokens"] or 0),
                "n_unk": int(row["n_unk"] or 0),
                "split": row["split"] or "train",
                "generator": GENERATOR,
                "filter": {"ok": True, "reasons": []},
            }
        )
    if kept and not any(r.get("split") == "valid" for r in kept):
        kept[-1]["split"] = "valid"
        with db_lock:
            conn.execute("UPDATE jobs SET split='valid' WHERE frame_id=?", (kept[-1]["doc_id"],))
            rows = conn.execute("SELECT * FROM jobs WHERE status='succeeded' ORDER BY idx").fetchall()
    tok_stats = tokenize_kept(out, kept, tok)
    stats = dict(stats)
    stats["n_unk"] = tok_stats["n_unk"]
    stats["release_kind"] = release_kind
    stats["spend_cny"] = spend_state["cny"]
    stats["prompt_tokens"] = spend_state["prompt_tokens"]
    stats["completion_tokens"] = spend_state["completion_tokens"]
    stats["model_snapshot"] = contract["generator"]["frozen_snapshot"]
    stats["mixed_generator"] = False
    raw_path = out / "raw" / "accepted.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = json.loads(row["frame_json"])
            rec = provenance_row(
                frame=payload,
                text=row["text"] or "",
                generator=GENERATOR,
                contract=contract,
                request_sha256=str(row["request_sha256"] or ""),
                response_sha256=str(row["response_sha256"] or sha256_text(str(row["text"] or ""))),
                usage={
                    "prompt_tokens": int(row["prompt_tokens"] or 0),
                    "completion_tokens": int(row["completion_tokens"] or 0),
                },
                filter_ok=True,
                reasons=[],
                n_tokens=int(row["n_tokens"] or 0),
                split=str(row["split"] or "train"),
            )
            rec["n_unk"] = row["n_unk"]
            rec["spend_cny"] = row["spend_cny"]
            rec["retries"] = row["retries"]
            rec["model_snapshot"] = row["model_snapshot"] or rec.get("model_snapshot")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with db_lock:
        conn.execute("UPDATE jobs SET flushed=1 WHERE status='succeeded'")
    release = freeze_release(out, contract=contract, generator=GENERATOR, kept=kept, stats=stats)
    dump_json(out / "reviews" / f"{release_kind}.json", {"ok": True, "release": release, "stats": stats})
    return release


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-unique-tokens", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-spend-cny", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--release-kind", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--qps", type=float, default=None)
    ap.add_argument("--enqueue-batch", type=int, default=2000)
    ap.add_argument("--model", default=None, help="DashScope model id; default is frozen qwen-plus snapshot")
    ap.add_argument("--generator", default=None, help="Provenance generator name; default qwen-plus or model id")
    ap.add_argument("--id-prefix", default=None)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--input-cny-per-million", type=float, default=None)
    ap.add_argument("--output-cny-per-million", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()
    global GENERATOR, ID_PREFIX
    contract = copy.deepcopy(load_contract())
    frozen = str(contract["generator"]["frozen_snapshot"])
    model = str(args.model or frozen)
    generator = str(args.generator or ("qwen-plus" if model == frozen else model))
    id_prefix = str(args.id_prefix or ("csynth-qwen-v1" if generator == "qwen-plus" else f"csynth-{generator}-v1"))
    GENERATOR = generator
    ID_PREFIX = id_prefix
    contract["generator"]["production"] = generator
    contract["generator"]["frozen_snapshot"] = model
    if args.input_cny_per_million is not None:
        contract.setdefault("budget", {})["input_cny_per_million"] = float(args.input_cny_per_million)
    if args.output_cny_per_million is not None:
        contract.setdefault("budget", {})["output_cny_per_million_non_thinking"] = float(args.output_cny_per_million)
    if args.temperature is not None:
        contract.setdefault("sampling", {})["temperature"] = float(args.temperature)
    if args.top_p is not None:
        contract.setdefault("sampling", {})["top_p"] = float(args.top_p)
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.smoke and args.out_dir == CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1 and generator == "qwen-plus":
        out = CORPORA_ROOT / "zh-pretrain-colloquial-synth-qwen-smoke"
    refuse_offline_dir(out, GENERATOR)
    refuse_qwen_v1_sidecar(out, generator=GENERATOR, model=model, frozen=frozen)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "state").mkdir(parents=True, exist_ok=True)
    (out / "reviews").mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw" / "accepted.jsonl"
    rej_path = out / "raw" / "rejected.jsonl"
    db_path = out / "state" / "queue.sqlite"
    nonempty = db_path.is_file() or (raw_path.is_file() and raw_path.stat().st_size > 0)
    if nonempty and not args.resume and not args.smoke:
        print("existing qwen corpus requires explicit --resume", file=sys.stderr)
        return 5
    lock_fd = acquire_lock(out / "state" / "produce.lock")
    conn = connect(db_path)
    existing_gen = meta_get(conn, "generator")
    existing_snap = meta_get(conn, "model_snapshot")
    want_snap = contract["generator"]["frozen_snapshot"]
    if existing_gen and existing_gen != GENERATOR:
        print(f"generator mismatch existing={existing_gen} requested={GENERATOR}", file=sys.stderr)
        return 5
    if existing_snap and existing_snap != want_snap:
        print(f"snapshot mismatch existing={existing_snap} requested={want_snap}", file=sys.stderr)
        return 5
    meta_set(conn, "generator", GENERATOR)
    meta_set(conn, "model_snapshot", want_snap)
    db_lock = threading.Lock()
    recover_running(conn, db_lock, stale_s=0)
    with db_lock:
        conn.execute(
            "UPDATE jobs SET status='pending', flushed=0, updated_at=? "
            "WHERE status='rejected' AND IFNULL(error,'') LIKE '%qwen_http_429%'",
            (time.time(),),
        )
    target = args.target_unique_tokens
    if target is None:
        target = 8_000 if args.smoke else int(contract["targets"]["pilot_unique_tokens"])
    max_docs = args.max_docs
    if max_docs is None:
        max_docs = 160 if args.smoke else max(20_000, int(target / 40) * 2)
    max_spend = args.max_spend_cny
    if max_spend is None:
        max_spend = float((contract.get("budget") or {}).get("production_max_spend_cny") or 200.0)
        if args.smoke:
            max_spend = min(max_spend, 15.0)
    warn_ratio = float((contract.get("budget") or {}).get("warn_spend_ratio") or 0.8)
    workers = args.workers or (4 if args.smoke else 12)
    qps = args.qps or (4.0 if args.smoke else 8.0)
    release_kind = args.release_kind or ("smoke" if args.smoke else "pilot")
    tok = ZhTokenizerV1()
    leaks = load_leaks(full=True)
    limiter = TokenBucket(qps)
    write_lock = threading.Lock()
    seen_lock = threading.Lock()
    stats_lock = threading.Lock()
    spend_lock = threading.Lock()
    cur = sums(conn, db_lock)
    spend_state = {
        "cny": float(cur["spend"] or 0),
        "prompt_tokens": int(cur["prompt_tokens"] or 0),
        "completion_tokens": int(cur["completion_tokens"] or 0),
    }
    stats = {
        "n_in": 0,
        "n_keep": int(cur["n_keep"] or 0),
        "n_dup": 0,
        "n_near": 0,
        "n_filter": 0,
        "n_gen_fail": 0,
        "budget_stop": False,
        "budget_warn": False,
        "release_kind": release_kind,
        "generator": GENERATOR,
    }
    seen_sha: set[str] = set()
    near = NearDupIndex(max_hamming=3)
    with db_lock:
        succeeded_rows = conn.execute(
            "SELECT response_sha256, text FROM jobs WHERE status='succeeded' AND response_sha256 IS NOT NULL"
        ).fetchall()
    for row in succeeded_rows:
        seen_sha.add(str(row["response_sha256"]))
        if row["text"]:
            near.add(simhash64(str(row["text"])))
    stop = threading.Event()
    origin = int(args.start_index or 0)
    max_idx = int(cur["max_idx"] or origin - 1)
    next_index = max(max_idx + 1, origin)
    index_limit = origin + max_docs
    accepted = raw_path.open("a", encoding="utf-8")
    rejected = rej_path.open("a", encoding="utf-8")

    def enqueue_more() -> None:
        nonlocal next_index
        if next_index >= index_limit:
            return
        n = min(args.enqueue_batch, index_limit - next_index)
        frames = list(iter_frames(n, seed=args.seed, axes=contract["axes"], start=next_index, id_prefix=ID_PREFIX))
        added = enqueue_frames(conn, db_lock, frames)
        next_index += n
        with stats_lock:
            stats["n_in"] += added

    def worker_loop() -> None:
        while not stop.is_set():
            if spend_state["cny"] >= max_spend:
                stop.set()
                break
            job = claim_job(conn, db_lock)
            if job is None:
                time.sleep(0.05)
                continue
            limiter.take()
            process_job(
                conn=conn,
                db_lock=db_lock,
                job=job,
                contract=contract,
                tok=tok,
                leaks=leaks,
                seen_sha=seen_sha,
                seen_lock=seen_lock,
                near=near,
                spend_state=spend_state,
                spend_lock=spend_lock,
                stats=stats,
                stats_lock=stats_lock,
            )

    enqueue_more()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qwen")
    for _ in range(workers):
        pool.submit(worker_loop)
    budget_ok = True
    warned = False
    cursor_path = out / "state" / "cursor.json"
    try:
        while True:
            cur = sums(conn, db_lock)
            unique = int(cur["unique_train"] or 0)
            spend = float(spend_state["cny"])
            if spend >= max_spend * warn_ratio and not warned:
                stats["budget_warn"] = True
                warned = True
                print(json.dumps({"warn": "budget_80pct", "spend_cny": spend, "max_spend_cny": max_spend}), flush=True)
            if spend >= max_spend:
                stats["budget_stop"] = True
                budget_ok = False
                stop.set()
                break
            if unique >= target and (not args.smoke or int(cur["n_keep"] or 0) >= int(contract["targets"]["smoke_docs"])):
                stop.set()
                break
            if args.smoke and int(cur["n_keep"] or 0) >= int(contract["targets"]["smoke_docs"]):
                stop.set()
                break
            if int(cur["n_pending"] or 0) + int(cur["n_running"] or 0) < workers * 2:
                enqueue_more()
            if next_index >= index_limit and int(cur["n_pending"] or 0) == 0 and int(cur["n_running"] or 0) == 0:
                stop.set()
                break
            flush_jsonl(conn, db_lock, accepted, rejected, write_lock, contract)
            dump_json(
                cursor_path,
                {
                    "next_index": next_index,
                    "n_docs": int(cur["n_keep"] or 0),
                    "n_unique_train_tokens": unique,
                    "spend_cny": spend,
                    "prompt_tokens": spend_state["prompt_tokens"],
                    "completion_tokens": spend_state["completion_tokens"],
                    "generator": GENERATOR,
                    "model_snapshot": want_snap,
                    "n_pending": int(cur["n_pending"] or 0),
                    "n_running": int(cur["n_running"] or 0),
                },
            )
            now = time.time()
            if now - float(stats.get("_last_print") or 0) >= 20:
                stats["_last_print"] = now
                print(
                    json.dumps(
                        {
                            "progress": True,
                            "n_docs": int(cur["n_keep"] or 0),
                            "n_unique_train_tokens": unique,
                            "target": target,
                            "spend_cny": round(spend, 4),
                            "n_pending": int(cur["n_pending"] or 0),
                            "n_running": int(cur["n_running"] or 0),
                        }
                    ),
                    flush=True,
                )
            time.sleep(0.4)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=False)
        recover_running(conn, db_lock, stale_s=0)
        flush_jsonl(conn, db_lock, accepted, rejected, write_lock, contract)
        accepted.close()
        rejected.close()
    cur = sums(conn, db_lock)
    stats["budget_ok"] = budget_ok
    stats["target_unique_tokens"] = target
    release = freeze_now(
        out,
        contract=contract,
        conn=conn,
        db_lock=db_lock,
        tok=tok,
        stats=stats,
        release_kind=release_kind,
        spend_state=spend_state,
    )
    conn.close()
    os.close(lock_fd)
    print(json.dumps(release, ensure_ascii=False, indent=2))
    unique = int(release.get("n_unique_train_tokens") or 0)
    if args.smoke:
        n_docs = int(release.get("n_docs") or 0)
        snap_ok = release.get("model_snapshot") == want_snap
        return 0 if n_docs >= 8 and snap_ok and float(release.get("unk_rate") or 1) <= float(contract["hard_gates"]["unk_rate_max"]) else 1
    if not budget_ok:
        return 4
    if unique < target:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
