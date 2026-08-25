#!/usr/bin/env python3
"""Official Needle 2 contrast runner for the VRM closed-set bank.

Uses cactus-needle complete() (do not run() — that executes tools and wipes
function_calls). Slot-filling NL is scored as _clarify, not [].
Not a needle-zh proxy. Scores stay in experiments/runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_needle_toolcall_v0 import load_jsonl, load_toolset, score
from repo_paths import BANK_NEEDLE_VRM_AGENT, BANK_NEEDLE_VRM_AGENT_EN, EXPERIMENTS_RUNS, ROOT
from run_eval_needle_qwen_v0 import CLARIFY, CLARIFY_RE, UNPARSED, summarize_latency, user_text

DEFAULT_NEEDLE_PYTHON = (
    ROOT.parent / "workspaces/ws-needles/notebook/hello-needle/.venv/bin/python"
)

SYSTEM_CONTRACT_ZH = """你是闭集家居代理的路由器。
只允许调用系统给出的工具，或什么都不调用。
缺必填槽、离题、场景已使动作无效、订餐非法搭配：不要调用任何工具。
禁止向用户提问或请其补充槽位。
"""

SYSTEM_CONTRACT_EN = """You are a closed-set home-agent router.
You may only call the provided tools, or call nothing.
If a required slot is missing, the request is off-topic, the scene already makes the action invalid, or the shop/dish pair is illegal: do not call any tool.
Do not ask the user to fill slots.
"""

EXECUTE_FAMILIES = frozenset({"gesture", "home", "order", "paraphrase"})
REFUSE_FAMILIES = frozenset({"missing", "scene_conflict", "illegal_pair", "offtopic"})


def to_needle_tools(toolset: dict) -> list[dict]:
    out = []
    for tool in toolset.get("tools") or []:
        out.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def contract_system(lang: str) -> str:
    if str(lang).lower().startswith("en"):
        return SYSTEM_CONTRACT_EN.strip()
    return SYSTEM_CONTRACT_ZH.strip()


def envelope_text(resp: dict) -> str:
    parts = []
    for key in ("reasoning", "reason", "type", "error", "message", "text"):
        val = resp.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def norm_calls(resp: dict) -> list[dict]:
    raw = resp.get("function_calls") or resp.get("function_calls") or []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("name") or ""
        args = item.get("arguments")
        if args is None:
            args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {}
        if name:
            out.append({"name": str(name), "arguments": args})
    return out


def classify(resp: dict) -> tuple[list[dict], str]:
    calls = norm_calls(resp)
    blob = envelope_text(resp)
    if (not calls) and CLARIFY_RE.search(blob):
        return list(CLARIFY), "clarify"
    if calls:
        return calls, "ok"
    return [], "ok"


def family_slice(scored: dict, families: frozenset[str]) -> dict:
    n = n_pass = 0
    for fam, slot in (scored.get("by_family") or {}).items():
        if fam not in families:
            continue
        n += int(slot.get("n") or 0)
        n_pass += int(slot.get("n_pass") or 0)
    return {
        "n": n,
        "n_pass": n_pass,
        "exact_match": round(n_pass / n, 4) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Official Needle 2 closed-set contrast")
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_VRM_AGENT_EN)
    ap.add_argument("--system-mode", choices=["contract", "native"], default="contract")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    try:
        import needle
    except ImportError:
        print(
            "needle is not importable in this Python.\n"
            f"Re-run with: {DEFAULT_NEEDLE_PYTHON} {Path(__file__).name} ...",
            file=sys.stderr,
        )
        return 2

    bank = args.bank if args.bank.is_absolute() else (ROOT / args.bank)
    rows = load_jsonl(bank)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("empty bank", file=sys.stderr)
        return 1

    toolsets: dict[str, dict] = {}
    for row in rows:
        tid = str(row.get("toolset_id") or "")
        if tid and tid not in toolsets:
            toolsets[tid] = load_toolset(tid)

    lang = str(rows[0].get("lang") or "zh")
    system = contract_system(lang) if args.system_mode == "contract" else ""
    primary_tid = str(rows[0].get("toolset_id") or "")
    tools = to_needle_tools(toolsets[primary_tid])

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"official-{getattr(needle, '__version__', 'unknown')}-{lang}-{args.system_mode}"
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / f"{tag}-needle-vrm-{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = needle.Needle(tools=tools, system=system or None)

    preds_path = out_dir / "predictions.jsonl"
    summary: dict = {
        "created_utc": tag,
        "spine": "needle2-official-contrast",
        "note": "Official Needle 2 is a contrast, not a needle-zh proxy. Missing-slot gold is [].",
        "bank": str(bank.resolve().relative_to(ROOT)),
        "needle_version": getattr(needle, "__version__", None),
        "system_mode": args.system_mode,
        "n": len(rows),
        "n_ok_parse": 0,
        "n_clarify": 0,
        "n_unparsed": 0,
        "n_error": 0,
        "run_wall_s": None,
        "latency": None,
    }

    run_t0 = time.perf_counter()
    with preds_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            user = user_text(row)
            rec = {
                "item_id": row["item_id"],
                "family": row.get("family"),
                "toolset_id": row.get("toolset_id"),
                "user_prompt": user,
                "raw_text": "",
                "parse_status": None,
                "function_calls": [],
                "error": None,
                "latency": None,
                "needle_type": None,
                "confidence": None,
            }
            t0 = time.perf_counter()
            try:
                if hasattr(agent, "reset"):
                    agent.reset()
                resp = agent.complete(user)
                wall_ms = round((time.perf_counter() - t0) * 1000, 1)
                if not isinstance(resp, dict):
                    resp = {"_raw": resp}
                rec["raw_text"] = json.dumps(resp, ensure_ascii=False)
                rec["needle_type"] = resp.get("type")
                rec["confidence"] = resp.get("confidence")
                rec["latency"] = {
                    "wall_ms": wall_ms,
                    "prefill_tps": resp.get("prefill_tps"),
                    "decode_tps": resp.get("decode_tps"),
                    "peak_ram_mb": resp.get("peak_ram_mb"),
                }
                calls, status = classify(resp)
                rec["function_calls"] = calls
                rec["parse_status"] = status
                key = {"ok": "n_ok_parse", "clarify": "n_clarify", "unparsed": "n_unparsed"}[status]
                summary[key] += 1
            except Exception as exc:  # noqa: BLE001 — engine/native failures must be recorded
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["function_calls"] = list(UNPARSED)
                rec["parse_status"] = "error"
                rec["latency"] = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
                summary["n_error"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    summary["run_wall_s"] = round(time.perf_counter() - run_t0, 3)
    (out_dir / "system_used.md").write_text(
        (system or "(native default system)")
        + "\n\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    preds = load_jsonl(preds_path)
    scored = score(rows, preds)
    summary["score"] = {
        "n": scored["n"],
        "n_pass": scored["n_pass"],
        "exact_match": scored["exact_match"],
        "by_family": scored["by_family"],
        "missing_pred": scored["missing_pred"],
        "execute_families": family_slice(scored, EXECUTE_FAMILIES),
        "refuse_families": family_slice(scored, REFUSE_FAMILIES),
    }
    summary["latency"] = summarize_latency(preds)
    call_walls = []
    empty_walls = []
    for pred in preds:
        wall = (pred.get("latency") or {}).get("wall_ms")
        if wall is None:
            continue
        calls = pred.get("function_calls") or []
        names = [str(c.get("name") or "") for c in calls if isinstance(c, dict)]
        if calls and not any(n.startswith("_") for n in names):
            call_walls.append(float(wall))
        elif not calls:
            empty_walls.append(float(wall))
    if call_walls:
        summary["latency"]["call_wall_ms_p50"] = sorted(call_walls)[len(call_walls) // 2]
        summary["latency"]["call_n"] = len(call_walls)
    if empty_walls:
        summary["latency"]["empty_wall_ms_p50"] = sorted(empty_walls)[len(empty_walls) // 2]
        summary["latency"]["empty_n"] = len(empty_walls)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "score.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {preds_path}")
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
