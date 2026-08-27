#!/usr/bin/env python3
"""Fleet status: plus + sidecar unique sum toward one 30M cap."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from colloquial_fleet import POOLED_UNIQUE_TARGET
from repo_paths import CORPORA_ROOT, EXPERIMENTS_RUNS

BJ = timezone(timedelta(hours=8))
SNAP = EXPERIMENTS_RUNS / "colloquial-fleet" / "snapshot.json"

LANES = [
    {
        "id": "plus-formal",
        "dir": "zh-pretrain-colloquial-synth-qwen-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": True,
        "spend_cap": 200.0,
    },
    {
        "id": "dsflash",
        "dir": "zh-pretrain-colloquial-synth-dsflash-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": False,
        "spend_cap": 200.0,
    },
    {
        "id": "kimi-k3",
        "dir": "zh-pretrain-colloquial-synth-kimi-k3-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": False,
        "spend_cap": 200.0,
    },
    {
        "id": "qwen3.7-plus",
        "dir": "zh-pretrain-colloquial-synth-qwen37plus-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": False,
        "spend_cap": 200.0,
    },
    {
        "id": "glm-5.2",
        "dir": "zh-pretrain-colloquial-synth-glm52-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": False,
        "spend_cap": 200.0,
    },
    {
        "id": "qwen3.6-plus",
        "dir": "zh-pretrain-colloquial-synth-qwen36plus-v1",
        "target": POOLED_UNIQUE_TARGET,
        "formal": False,
        "spend_cap": 200.0,
    },
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def eta_text(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds <= 0:
        return "done"
    t = datetime.now(tz=BJ) + timedelta(seconds=seconds)
    h = seconds / 3600.0
    return f"{h:.1f}h → {t.strftime('%m-%d %H:%M')}"


def main() -> int:
    now = time.time()
    prev = load_json(SNAP)
    prev_lanes = {row["id"]: row for row in (prev.get("lanes") or [])}
    dt = now - float(prev.get("ts") or 0) if prev.get("ts") else None
    rows = []
    for spec in LANES:
        cur = load_json(CORPORA_ROOT / spec["dir"] / "state" / "cursor.json")
        unique = int(cur.get("n_unique_train_tokens") or 0)
        docs = int(cur.get("n_docs") or 0)
        spend = float(cur.get("spend_cny") or 0)
        running = int(cur.get("n_running") or 0)
        alive = bool(cur) and running > 0
        old = prev_lanes.get(spec["id"]) or {}
        unique_s = docs_s = None
        if dt and dt > 5 and old:
            unique_s = (unique - int(old.get("unique") or 0)) / dt
            docs_s = (docs - int(old.get("docs") or 0)) / dt
        rows.append(
            {
                "id": spec["id"],
                "formal": spec["formal"],
                "alive": alive,
                "model": cur.get("model_snapshot") or cur.get("generator"),
                "docs": docs,
                "unique": unique,
                "spend_cny": round(spend, 4),
                "spend_cap": spec["spend_cap"],
                "n_running": running,
                "docs_per_s": None if docs_s is None else round(docs_s, 3),
                "unique_per_s": None if unique_s is None else round(unique_s, 1),
                "unique_per_h": None if unique_s is None else round(unique_s * 3600),
                "tok_per_doc": round(unique / docs, 1) if docs else 0,
            }
        )
    formal = next(r for r in rows if r["formal"])
    sidecar_unique = sum(r["unique"] for r in rows if not r["formal"])
    pooled_unique = sum(r["unique"] for r in rows)
    pooled_rate = sum((r["unique_per_s"] or 0) for r in rows if r["alive"])
    pooled_rem = max(0, POOLED_UNIQUE_TARGET - pooled_unique)
    pooled_eta_s = (pooled_rem / pooled_rate) if pooled_rate > 0 else None
    sidecar_rate = sum((r["unique_per_s"] or 0) for r in rows if not r["formal"] and r["alive"])
    report = {
        "ts": now,
        "ts_bj": datetime.now(tz=BJ).strftime("%Y-%m-%d %H:%M:%S"),
        "window_s": None if dt is None else round(dt, 1),
        "pooled_target": POOLED_UNIQUE_TARGET,
        "pooled_unique": pooled_unique,
        "pooled_remain": pooled_rem,
        "pooled_pct": round(100.0 * pooled_unique / POOLED_UNIQUE_TARGET, 2),
        "pooled_unique_per_h": round(pooled_rate * 3600) if pooled_rate else None,
        "pooled_eta": eta_text(pooled_eta_s),
        "formal_plus": formal,
        "sidecar_unique_sum": sidecar_unique,
        "sidecar_unique_per_h": round(sidecar_rate * 3600) if sidecar_rate else None,
        "note": "30M is pooled unique across plus+sidecars; sidecars stop at the cap. Mixing into formal CPT still needs a contract change.",
        "lanes": rows,
    }
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    SNAP.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
