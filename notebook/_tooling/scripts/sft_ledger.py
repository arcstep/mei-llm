"""Shared atomic spend ledger for the SFT teacher fleet.

All jobs and lanes debit the same 1000 CNY envelope. A request reserves its
worst-case cost before the HTTP call; crash recovery cannot lose the reserve.
Missing usage is billed at the reserved cap and the provider is disabled.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from sft_synth_lib import Lane, budget_status, spend_cny

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
  task TEXT NOT NULL,
  case_id TEXT NOT NULL,
  lane TEXT NOT NULL,
  model_snapshot TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  reserved_cny REAL NOT NULL,
  actual_cny REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  phase TEXT NOT NULL DEFAULT '',
  provider TEXT,
  http_status INTEGER,
  usage_missing INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL,
  PRIMARY KEY (task, case_id, lane, model_snapshot, prompt_version)
);
CREATE TABLE IF NOT EXISTS provider_state (
  provider TEXT PRIMARY KEY,
  disabled INTEGER NOT NULL DEFAULT 0,
  reason TEXT
);
CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
"""


class BudgetHardStop(RuntimeError):
    pass


class ProviderDisabled(RuntimeError):
    pass


def default_reserve_cny(lane: Lane, *, max_prompt: int = 2048, max_completion: int = 256) -> float:
    if lane.provider == "offline" or lane.model_snapshot == "template":
        return 0.0
    return spend_cny(prompt_tokens=max_prompt, completion_tokens=max_completion, lane=lane)


class SharedLedger:
    def __init__(
        self,
        path: Path,
        *,
        global_limit: float,
        warn_frac: float = 0.8,
        hard_frac: float = 1.0,
        phase_limit: float | None = None,
        phase: str = "canary",
        lane_caps: dict[str, float] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.global_limit = float(global_limit)
        self.warn_frac = float(warn_frac)
        self.hard_frac = float(hard_frac)
        self.phase = phase
        self.phase_limit = None if phase_limit is None else float(phase_limit)
        self.lane_caps = {k: float(v) for k, v in (lane_caps or {}).items()}
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), timeout=60, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(LEDGER_SCHEMA)
        cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(reservations)").fetchall()}
        if "phase" not in cols:
            self.conn.execute("ALTER TABLE reservations ADD COLUMN phase TEXT NOT NULL DEFAULT ''")
        self.conn.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('global_limit', ?)", (str(self.global_limit),))

    def _sums(self) -> dict[str, float]:
        row = self.conn.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_cny ELSE actual_cny END), 0) AS committed,
              COALESCE(SUM(actual_cny), 0) AS actual
            FROM reservations
            """
        ).fetchone()
        return {"committed": float(row["committed"]), "actual": float(row["actual"])}

    def _lane_committed(self, lane: str) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_cny ELSE actual_cny END), 0)
            FROM reservations WHERE lane=?
            """,
            (lane,),
        ).fetchone()
        return float(row[0])

    def _phase_committed(self) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_cny ELSE actual_cny END), 0)
            FROM reservations WHERE phase=?
            """,
            (self.phase,),
        ).fetchone()
        return float(row[0])

    def provider_ok(self, provider: str) -> tuple[bool, str | None]:
        row = self.conn.execute("SELECT disabled, reason FROM provider_state WHERE provider=?", (provider,)).fetchone()
        if row and int(row["disabled"]):
            return False, str(row["reason"] or "disabled")
        return True, None

    def disable_provider(self, provider: str, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO provider_state(provider, disabled, reason) VALUES(?,?,?) "
            "ON CONFLICT(provider) DO UPDATE SET disabled=1, reason=excluded.reason",
            (provider, 1, reason),
        )

    def enable_provider(self, provider: str) -> None:
        self.conn.execute(
            "INSERT INTO provider_state(provider, disabled, reason) VALUES(?,?,?) "
            "ON CONFLICT(provider) DO UPDATE SET disabled=0, reason=''",
            (provider, 0, ""),
        )

    def snapshot(self) -> dict[str, Any]:
        sums = self._sums()
        g_status = budget_status(sums["committed"], self.global_limit, warn_frac=self.warn_frac, hard_frac=self.hard_frac)
        return {
            "path": str(self.path),
            "phase": self.phase,
            "committed_cny": sums["committed"],
            "actual_cny": sums["actual"],
            "global_limit": self.global_limit,
            "phase_limit": self.phase_limit,
            "global_status": g_status,
            "warn_cny": self.global_limit * self.warn_frac,
        }

    def reserve(
        self,
        *,
        task: str,
        case_id: str,
        lane: Lane,
        prompt_version: str,
        reserved_cny: float | None = None,
    ) -> float:
        amount = float(reserved_cny if reserved_cny is not None else default_reserve_cny(lane))
        key = (task, case_id, lane.name, lane.model_snapshot, prompt_version)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                ok, why = self.provider_ok(lane.provider)
                if not ok:
                    self.conn.execute("ROLLBACK")
                    raise ProviderDisabled(f"{lane.provider}: {why}")
                existing = self.conn.execute(
                    "SELECT status, reserved_cny, actual_cny FROM reservations "
                    "WHERE task=? AND case_id=? AND lane=? AND model_snapshot=? AND prompt_version=?",
                    key,
                ).fetchone()
                if existing and existing["status"] in {"settled", "missing_usage"}:
                    self.conn.execute("ROLLBACK")
                    return float(existing["actual_cny"] or 0)
                sums = self._sums()
                extra = 0.0 if existing and existing["status"] == "reserved" else amount
                committed = sums["committed"] + extra
                g_status = budget_status(committed, self.global_limit, warn_frac=self.warn_frac, hard_frac=self.hard_frac)
                if g_status == "stop" and extra > 0:
                    self.conn.execute("ROLLBACK")
                    raise BudgetHardStop(f"global hard-stop committed={committed:.4f} limit={self.global_limit}")
                phase_next = self._phase_committed() + extra
                if self.phase_limit is not None and phase_next > self.phase_limit + 1e-12 and extra > 0:
                    self.conn.execute("ROLLBACK")
                    raise BudgetHardStop(
                        f"phase {self.phase} hard-stop committed={phase_next:.4f} limit={self.phase_limit}"
                    )
                lane_cap = self.lane_caps.get(lane.name)
                if lane_cap is not None and lane_cap > 0:
                    lane_next = self._lane_committed(lane.name) + extra
                    if lane_next > lane_cap + 1e-12:
                        self.conn.execute("ROLLBACK")
                        raise BudgetHardStop(f"lane {lane.name} hard-stop committed={lane_next:.4f} cap={lane_cap}")
                now = time.time()
                self.conn.execute(
                    """
                    INSERT INTO reservations(
                      task, case_id, lane, model_snapshot, prompt_version,
                      reserved_cny, actual_cny, status, phase, provider, updated_at
                    ) VALUES(?,?,?,?,?,?,0,'reserved',?,?,?)
                    ON CONFLICT(task, case_id, lane, model_snapshot, prompt_version) DO UPDATE SET
                      reserved_cny=excluded.reserved_cny,
                      status='reserved',
                      phase=excluded.phase,
                      updated_at=excluded.updated_at
                    """,
                    (*key, amount, self.phase, lane.provider, now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        return amount

    def settle(
        self,
        *,
        task: str,
        case_id: str,
        lane: Lane,
        prompt_version: str,
        actual_cny: float | None,
        reserved_cny: float,
        usage: dict | None = None,
        http_status: int | None = None,
    ) -> float:
        usage = usage or {}
        missing = actual_cny is None or (
            lane.provider != "offline"
            and lane.model_snapshot != "template"
            and not usage.get("dry_run")
            and int(usage.get("prompt_tokens") or 0) == 0
            and int(usage.get("completion_tokens") or 0) == 0
            and float(actual_cny or 0) == 0.0
        )
        billed = float(reserved_cny if missing else actual_cny or 0.0)
        status = "missing_usage" if missing and lane.provider != "offline" else "settled"
        key = (task, case_id, lane.name, lane.model_snapshot, prompt_version)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                """
                UPDATE reservations SET actual_cny=?, status=?, usage_missing=?, http_status=?, updated_at=?
                WHERE task=? AND case_id=? AND lane=? AND model_snapshot=? AND prompt_version=?
                """,
                (billed, status, 1 if missing else 0, http_status, time.time(), *key),
            )
            if missing and lane.provider not in {"offline", ""}:
                self.disable_provider(lane.provider, "usage_missing_billed_at_reserve")
            self.conn.execute("COMMIT")
        return billed

    def void(self, *, task: str, case_id: str, lane: Lane, prompt_version: str) -> None:
        key = (task, case_id, lane.name, lane.model_snapshot, prompt_version)
        with self._lock:
            self.conn.execute(
                """
                UPDATE reservations SET status='void', reserved_cny=0, actual_cny=0, updated_at=?
                WHERE task=? AND case_id=? AND lane=? AND model_snapshot=? AND prompt_version=?
                  AND status='reserved'
                """,
                (time.time(), *key),
            )
