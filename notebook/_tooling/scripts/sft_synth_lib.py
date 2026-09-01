#!/usr/bin/env python3
"""Recoverable multi-model SFT synthesis fleet.

Queue, rate limit, budget, resume, and mei-eval provider adapter.
Teachers may only rewrite the visible query. Gold is never sampled from the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from repo_paths import FLEET_SFT_SYNTH, ROOT

MEI_EVAL_SRC = ROOT.parent / "tools" / "mei-eval" / "python" / "src"
PROMPT_VERSION = "sft-teacher-v2-query-only.1"
TEACHER_SYSTEM = """你是 MeiLang 闭集工具调用的问法改写教师。
只改写用户说法，不要回答任务，不要输出工具名、参数 JSON、reason_code 或 answers。
必须遵守：
- 不得改变是否应当执行或拒绝
- 不得把开/关/取消极性改反
- 不得改变受保护槽的中文名词；数字、时间、温度必须保留
- 若原始说法以「当前环境有变化」开头，改写后也必须以该前缀开头，且保持环境句，禁止改成「请开/帮我把」类用户口令
- 不得补问用户
- 不要输出 EVAL 编号
只输出一个 JSON 对象：{"query":"..."}，不要 markdown。"""


class BudgetExceeded(RuntimeError):
    pass


class ResumeMismatch(RuntimeError):
    pass


class SpendNotAllowed(RuntimeError):
    pass


@dataclass(frozen=True)
class Lane:
    name: str
    role: str
    provider: str
    model_snapshot: str
    start_workers: int
    start_qps: float
    max_workers: int
    max_qps: float
    input_cny_per_million: float
    output_cny_per_million: float
    lane_budget_cny: float
    enabled: bool = True

    @property
    def price_fingerprint(self) -> str:
        payload = f"{self.model_snapshot}|{self.input_cny_per_million}|{self.output_cny_per_million}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fleet(path: Path | None = None) -> dict:
    p = path or FLEET_SFT_SYNTH
    return json.loads(p.read_text(encoding="utf-8"))


def lanes_of(fleet: dict, *, include_disabled: bool = False) -> dict[str, Lane]:
    out: dict[str, Lane] = {}
    for name, raw in (fleet.get("lanes") or {}).items():
        lane = Lane(
            name=name,
            role=str(raw.get("role") or ""),
            provider=str(raw.get("provider") or ""),
            model_snapshot=str(raw.get("model_snapshot") or name),
            start_workers=int(raw.get("start_workers") or 1),
            start_qps=float(raw.get("start_qps") or 1.0),
            max_workers=int(raw.get("max_workers") or raw.get("start_workers") or 1),
            max_qps=float(raw.get("max_qps") or raw.get("start_qps") or 1.0),
            input_cny_per_million=float(raw.get("input_cny_per_million") or 0.0),
            output_cny_per_million=float(raw.get("output_cny_per_million") or 0.0),
            lane_budget_cny=float(raw.get("lane_budget_cny") or 0.0),
            enabled=bool(raw.get("enabled", True)),
        )
        if lane.enabled or include_disabled:
            out[name] = lane
    return out


def fleet_price_fingerprint(fleet: dict) -> str:
    rows = []
    for name, lane in sorted(lanes_of(fleet, include_disabled=True).items()):
        rows.append(
            {
                "name": name,
                "model": lane.model_snapshot,
                "in": lane.input_cny_per_million,
                "out": lane.output_cny_per_million,
            }
        )
    return sha256_obj(rows)


def spend_cny(*, prompt_tokens: int, completion_tokens: int, lane: Lane) -> float:
    return (int(prompt_tokens) / 1_000_000.0) * lane.input_cny_per_million + (
        int(completion_tokens) / 1_000_000.0
    ) * lane.output_cny_per_million


def budget_status(spend: float, limit: float, *, warn_frac: float = 0.8, hard_frac: float = 1.0) -> str:
    if limit <= 0:
        return "unlimited" if spend <= 0 else "stop"
    if spend >= limit * hard_frac:
        return "stop"
    if spend >= limit * warn_frac:
        return "warn"
    return "ok"


def export_queue_rows(conn: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    for row in conn.execute("SELECT * FROM items ORDER BY idx"):
        status = str(row["status"] or "")
        if status == "accepted" and row["result_json"]:
            accepted.append(json.loads(row["result_json"]))
        elif status in {"rejected", "error"}:
            rejected.append(
                {
                    "case_id": row["case_id"],
                    "reason": row["reject_reason"],
                    "query": row["query"],
                    "status": status,
                }
            )
    return accepted, rejected


def fleet_contract(fleet: dict) -> dict:
    return {
        "fleet": "notebook/sft/mei-1.0-51m/recipes/sft-synth-fleet-v1.json",
        "fleet_id": fleet.get("id") or "sft-synth-fleet-v1",
        "fleet_hash": sha256_obj(fleet),
        "prompt_version": fleet.get("prompt_version") or PROMPT_VERSION,
        "serializer_version": fleet.get("serializer_version") or "mei-tool-call-serializer-v2",
        "global_budget_cny": float(fleet.get("global_budget_cny") or 0.0),
        "allow_spend": False,
        "metric_unit": "accepted_unique_row",
        "canary_budget_cny": float(fleet.get("canary_budget_cny") or 0.0),
        "shared_ledger": fleet.get("shared_ledger"),
    }


def load_budget_plan(fleet: dict) -> dict:
    rel = fleet.get("budget_plan")
    if not rel:
        return {}
    path = ROOT / str(rel)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cost_per_1k_accepted(spend_cny_total: float, n_accepted_unique: int) -> float:
    if n_accepted_unique <= 0:
        return 0.0
    return (spend_cny_total / n_accepted_unique) * 1000.0


class TokenBucket:
    def __init__(self, qps: float):
        self.lock = threading.Lock()
        self.next_t = 0.0
        self.set_qps(qps)

    def set_qps(self, qps: float) -> None:
        self.interval = 1.0 / max(float(qps), 0.05)

    def take(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.next_t = now + self.interval


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"producer already running: {path}") from exc
    return fd


def connect_queue(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
          case_id TEXT PRIMARY KEY,
          idx INTEGER NOT NULL UNIQUE,
          case_json TEXT NOT NULL,
          lane TEXT NOT NULL,
          status TEXT NOT NULL,
          query TEXT,
          reject_reason TEXT,
          prompt_tokens INTEGER DEFAULT 0,
          completion_tokens INTEGER DEFAULT 0,
          spend_cny REAL DEFAULT 0,
          request_sha256 TEXT,
          response_sha256 TEXT,
          model_snapshot TEXT,
          prompt_version TEXT,
          latency_ms REAL DEFAULT 0,
          retries INTEGER DEFAULT 0,
          http_status INTEGER,
          updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS items_status ON items(status)")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(items)").fetchall()}
    if "result_json" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN result_json TEXT")
    if "task" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN task TEXT")
    return conn


def meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return str(row["v"]) if row else default


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", (key, value))


def bind_resume_contract(
    conn: sqlite3.Connection,
    *,
    fleet_id: str,
    lane: str,
    model_snapshot: str,
    prompt_version: str,
    price_fingerprint: str,
    serializer_version: str | None = None,
    toolset_hash: str | None = None,
) -> None:
    wanted = {
        "fleet_id": fleet_id,
        "lane": lane,
        "model_snapshot": model_snapshot,
        "prompt_version": prompt_version,
        "price_fingerprint": price_fingerprint,
    }
    if serializer_version:
        wanted["serializer_version"] = serializer_version
    if toolset_hash:
        wanted["toolset_hash"] = toolset_hash
    for key, value in wanted.items():
        got = meta_get(conn, key)
        if got is not None and got != value:
            raise ResumeMismatch(f"resume mismatch {key}: stored={got} wanted={value}")
        if got is None:
            meta_set(conn, key, value)


def enqueue_cases(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    cases: list[dict],
    *,
    lane: str,
) -> int:
    n = 0
    now = time.time()
    with lock:
        for i, case in enumerate(cases):
            raw_id = str(case.get("case_id") or f"case-{i:06d}")
            payload = dict(case)
            payload["idx"] = i
            inserted = False
            for attempt in range(8):
                case_id = raw_id if attempt == 0 else f"{raw_id}::{i:06d}::{attempt}"
                payload["case_id"] = case_id
                try:
                    conn.execute(
                        "INSERT INTO items(case_id, idx, case_json, lane, status, updated_at) VALUES(?,?,?,?,?,?)",
                        (case_id, i, json.dumps(payload, ensure_ascii=False), lane, "pending", now),
                    )
                    n += 1
                    inserted = True
                    break
                except sqlite3.IntegrityError:
                    continue
            if not inserted:
                continue
    return n


def reclaim_running(conn: sqlite3.Connection, lock: threading.Lock) -> int:
    with lock:
        cur = conn.execute(
            "UPDATE items SET status='pending', updated_at=? WHERE status='running'",
            (time.time(),),
        )
        return int(cur.rowcount or 0)


def claim_item(conn: sqlite3.Connection, lock: threading.Lock) -> dict | None:
    with lock:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT case_id, idx, case_json, lane FROM items WHERE status='pending' ORDER BY idx LIMIT 1"
        ).fetchone()
        if not row:
            conn.execute("COMMIT")
            return None
        cur = conn.execute(
            "UPDATE items SET status='running', updated_at=? WHERE case_id=? AND status='pending'",
            (time.time(), row["case_id"]),
        )
        conn.execute("COMMIT")
        if cur.rowcount != 1:
            return None
        return {
            "case_id": row["case_id"],
            "idx": int(row["idx"]),
            "lane": row["lane"],
            "case": json.loads(row["case_json"]),
        }


def finish_item(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    case_id: str,
    *,
    status: str,
    query: str | None = None,
    reject_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    spend: float = 0.0,
    request_sha256: str | None = None,
    response_sha256: str | None = None,
    model_snapshot: str | None = None,
    prompt_version: str | None = None,
    latency_ms: float = 0.0,
    retries: int = 0,
    http_status: int | None = None,
    result_json: str | None = None,
    task: str | None = None,
) -> None:
    with lock:
        conn.execute(
            """
            UPDATE items SET
              status=?, query=?, reject_reason=?, prompt_tokens=?, completion_tokens=?,
              spend_cny=?, request_sha256=?, response_sha256=?, model_snapshot=?,
              prompt_version=?, latency_ms=?, retries=?, http_status=?, updated_at=?,
              result_json=COALESCE(?, result_json),
              task=COALESCE(?, task)
            WHERE case_id=?
            """,
            (
                status,
                query,
                reject_reason,
                int(prompt_tokens),
                int(completion_tokens),
                float(spend),
                request_sha256,
                response_sha256,
                model_snapshot,
                prompt_version,
                float(latency_ms),
                int(retries),
                http_status,
                time.time(),
                result_json,
                task,
                case_id,
            ),
        )


def queue_sums(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END), 0) AS n_accepted,
          COALESCE(SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END), 0) AS n_rejected,
          COALESCE(SUM(CASE WHEN status='error' THEN 1 ELSE 0 END), 0) AS n_error,
          COALESCE(SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END), 0) AS n_pending,
          COALESCE(SUM(CASE WHEN status='running' THEN 1 ELSE 0 END), 0) AS n_running,
          COALESCE(SUM(spend_cny), 0) AS spend,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens
        FROM items
        """
    ).fetchone()
    return {k: row[k] for k in row.keys()}


def accepted_unique_count(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT query FROM items WHERE status='accepted' AND query IS NOT NULL").fetchall()
    return len({sha256_text(str(r["query"]).strip()) for r in rows})


def maybe_ramp(lane: Lane, stats: dict, ramp: dict | None = None) -> tuple[int, float]:
    cfg = ramp or {}
    min_n = int(cfg.get("min_n") or 20)
    n = int(stats.get("n") or 0)
    workers = int(stats.get("workers") or lane.start_workers)
    qps = float(stats.get("qps") or lane.start_qps)
    if n < min_n:
        return workers, qps
    rate_429 = float(stats.get("rate_429") or 0.0)
    retry_error = float(stats.get("retry_error_rate") or 0.0)
    p95_ok = bool(stats.get("p95_stable", True))
    if (
        rate_429 <= float(cfg.get("max_429_rate") or 0.01)
        and retry_error <= float(cfg.get("max_retry_error_rate") or 0.02)
        and p95_ok
    ):
        workers = min(workers * 2, lane.max_workers)
        qps = min(qps * 2, lane.max_qps)
        return workers, qps
    return max(lane.start_workers, workers // 2), max(lane.start_qps, qps / 2.0)


def strip_fence(raw: str) -> str:
    text = (raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_teacher_query(text: str) -> tuple[str | None, str | None]:
    raw = strip_fence(text)
    if not raw:
        return None, "empty"
    lowered = raw.lower()
    for leak in ("reason_code", "function_calls", "gold_route_id", '"answers"', "<routes>"):
        if leak in lowered or leak in raw:
            return None, "gold_leak"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        query = raw.strip().strip('"')
        return (query, None) if query else (None, "empty")
    if isinstance(obj, dict):
        if any(k in obj for k in ("answers", "reason_code", "function_calls", "gold_args", "act")):
            return None, "gold_leak"
        query = str(obj.get("query") or obj.get("text") or "").strip()
        return (query, None) if query else (None, "empty")
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, str) and first.strip():
            return first.strip(), None
        if isinstance(first, dict):
            query = str(first.get("query") or "").strip()
            return (query, None) if query else (None, "empty")
    return None, "unparsed"


def teacher_prompt(case: dict) -> str:
    slots = case.get("zh_slots") or {}
    slot_line = "、".join(f"{k}={v}" for k, v in slots.items() if v) or "(无)"
    stem = str(case.get("stem") or case.get("query") or "")
    env = str(case.get("park_layer") or "") == "L2" or str(case.get("intent_id") or "").startswith("env-")
    style = (
        "这是环境摘要，不是用户口令。改写后必须仍以「当前环境有变化」开头，保持环境句。"
        if env
        else "请改写成另一句自然中文用户指令，保持开/关/取消极性不变。"
    )
    return (
        f"任务族：{case.get('task') or ''} / {case.get('family') or ''} / {case.get('park_layer') or ''}\n"
        f"必须原样出现的中文槽：{slot_line}\n"
        f"原始说法：{stem}\n"
        f"{style}"
    )


def template_rewrite(case: dict) -> tuple[str, dict[str, Any], float]:
    query = str(case.get("query") or case.get("stem") or "").strip()
    return query, {"prompt_tokens": 0, "completion_tokens": 0, "dry_run": True}, 0.0


def require_spend_allowed(lane: Lane, *, allow_spend: bool) -> None:
    if lane.provider == "offline" or lane.model_snapshot == "template":
        return
    if not allow_spend:
        raise SpendNotAllowed(
            f"lane {lane.name} needs --allow-spend and an explicit budget; refusing paid API"
        )
    if lane.lane_budget_cny <= 0:
        raise SpendNotAllowed(f"lane {lane.name} has lane_budget_cny=0; refusing paid API")


def mei_eval_rewrite(
    case: dict,
    lane: Lane,
    *,
    allow_spend: bool,
    temperature: float = 0.7,
) -> tuple[str, dict[str, Any], float]:
    require_spend_allowed(lane, allow_spend=allow_spend)
    import sys

    sys.path.insert(0, str(MEI_EVAL_SRC))
    from mei_eval.chat import chat_complete, openai_client
    from mei_eval.endpoint import resolve_endpoint

    cfg = resolve_endpoint(provider=lane.provider, model=lane.model_snapshot)
    client = openai_client(cfg.base_url, cfg.api_key, timeout=60.0, max_retries=0)
    text, usage, latency_ms = chat_complete(
        client,
        model=lane.model_snapshot,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": teacher_prompt(case)},
        ],
        temperature=temperature,
        max_tokens=256,
        enable_thinking=False,
    )
    return text, usage, latency_ms


def rewrite_query(
    case: dict,
    lane: Lane,
    *,
    allow_spend: bool = False,
    teacher: Callable[..., tuple[str, dict[str, Any], float]] | None = None,
) -> dict:
    started = time.perf_counter()
    if teacher is not None:
        raw, usage, latency_ms = teacher(case, lane)
    elif lane.provider == "offline" or lane.model_snapshot == "template":
        raw, usage, latency_ms = template_rewrite(case)
    else:
        raw, usage, latency_ms = mei_eval_rewrite(case, lane, allow_spend=allow_spend)
    query, err = parse_teacher_query(raw) if not (usage or {}).get("dry_run") else (raw, None)
    if (usage or {}).get("dry_run"):
        query, err = (raw.strip() if raw else None), (None if raw and raw.strip() else "empty")
    request_sha = sha256_text(teacher_prompt(case) + "\n" + TEACHER_SYSTEM)
    response_sha = sha256_text(raw or "")
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    return {
        "query": query,
        "reject_reason": err,
        "raw": raw,
        "usage": usage,
        "latency_ms": latency_ms if latency_ms else (time.perf_counter() - started) * 1000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "spend_cny": spend_cny(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, lane=lane),
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "model_snapshot": lane.model_snapshot,
        "prompt_version": PROMPT_VERSION,
    }


def route_lane(case: dict, fleet: dict, *, ignore_prefer_template: bool = False) -> str:
    """Cheap paraphrases → volume; high-risk / slot-dense / first-fail → contract."""
    if case.get("high_risk") or case.get("kind") in {"missing", "illegal_pair", "scene_conflict", "ambiguous"}:
        return "qwen-plus-2025-12-01"
    slots = case.get("zh_slots") or {}
    if len(slots) >= 2:
        return "qwen-plus-2025-12-01"
    if case.get("diversity"):
        return "qwen3.7-plus"
    if (not ignore_prefer_template) and (case.get("prefer_template") or not any(lanes_of(fleet))):
        return "template"
    return "deepseek-v4-flash-0731"


PAID_SHARE = {
    "deepseek-v4-flash-0731": 0.60,
    "qwen-plus-2025-12-01": 0.25,
    "qwen3.7-plus": 0.15,
}


def assign_pareto_lanes(cases: list[dict]) -> dict[str, list[dict]]:
    """Split cases by frozen canary share. Ignore prefer_template. High-risk prefers Qwen Plus."""
    n = len(cases)
    names = list(PAID_SHARE)
    caps = {k: int(n * v) for k, v in PAID_SHARE.items()}
    caps[names[0]] += n - sum(caps.values())
    buckets: dict[str, list[dict]] = {k: [] for k in names}
    plus_name = "qwen-plus-2025-12-01"
    priority: list[dict] = []
    rest: list[dict] = []
    for case in cases:
        kind = str(case.get("kind") or "")
        if case.get("high_risk") or kind in {"missing", "illegal_pair", "scene_conflict", "ambiguous"}:
            priority.append(case)
        else:
            rest.append(case)
    for case in priority:
        if len(buckets[plus_name]) < caps[plus_name]:
            buckets[plus_name].append(case)
        else:
            rest.append(case)
    order = names
    idx = 0
    for case in rest:
        placed = False
        for _ in range(len(order)):
            name = order[idx % len(order)]
            idx += 1
            if len(buckets[name]) < caps[name]:
                buckets[name].append(case)
                placed = True
                break
        if not placed:
            name = min(names, key=lambda k: len(buckets[k]) / max(1, caps[k]))
            buckets[name].append(case)
    return buckets


def bakeoff_metrics(rows_by_lane: dict[str, list[dict]]) -> dict:
    report = {"lanes": {}, "metric_unit": "accepted_unique_row"}
    for lane, rows in rows_by_lane.items():
        n = len(rows)
        accepted = [r for r in rows if r.get("status") == "accepted"]
        unique = {sha256_text(str(r.get("query") or "")) for r in accepted}
        spend = sum(float(r.get("spend_cny") or 0) for r in rows)
        gold_keep = sum(1 for r in accepted if r.get("gold_kept"))
        slot_keep = sum(1 for r in accepted if r.get("slots_kept"))
        compiler_ok = sum(1 for r in accepted if r.get("compiler_ok"))
        latencies = sorted(float(r.get("latency_ms") or 0) for r in rows)
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        reasons: dict[str, int] = {}
        for row in rows:
            if row.get("status") == "accepted":
                continue
            key = str(row.get("reject_reason") or row.get("status") or "unknown")
            reasons[key] = reasons.get(key, 0) + 1
        report["lanes"][lane] = {
            "n": n,
            "n_accepted": len(accepted),
            "n_accepted_unique": len(unique),
            "gold_keep_rate": (gold_keep / n) if n else 0.0,
            "protected_slot_keep_rate": (slot_keep / n) if n else 0.0,
            "compiler_pass_rate": (compiler_ok / n) if n else 0.0,
            "spend_cny": spend,
            "cny_per_1k_accepted_unique": cost_per_1k_accepted(spend, len(unique)),
            "p95_latency_ms": p95,
            "reject_reasons": reasons,
        }
    return report
