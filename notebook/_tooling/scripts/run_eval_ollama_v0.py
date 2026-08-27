#!/usr/bin/env python3
"""Ollama eval runner v0 — SECONDARY spine (end-side quantized only).

Primary baseline: scripts/run_eval_qwen_cloud_v0.py (DashScope cloud).
Do not treat Ollama q-scores as method/dev-assist ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import BANK_MEI_EXPERT, ROOT

MEI_PROJECTS_ROOT = ROOT.parent
DRAFT_DOCS = MEI_PROJECTS_ROOT / "docs/draft/mei-llm"
DEFAULT_BANK = BANK_MEI_EXPERT
DEFAULT_CTX = MEI_PROJECTS_ROOT / "docs/mei-llm/02-eval/0205-ctx-stable-v0.md"
DEFAULT_HOST = "http://127.0.0.1:11434"

_EVAL_SRC = MEI_PROJECTS_ROOT / "tools/mei-eval/python/src"
if _EVAL_SRC.is_dir():
    sys.path.insert(0, str(_EVAL_SRC))
try:
    from mei_eval.env import load_mei_env
except ImportError:  # pragma: no cover
    load_mei_env = None  # type: ignore[misc, assignment]

DEFAULT_SLOTS = {
    "app_name": "demo-app",
    "org_name": "示例单位",
    "entity_a": "甲实体",
    "entity_b": "乙实体",
    "metric_name": "示例指标",
}

REFUSAL_HINTS = (
    "拒绝",
    "不能",
    "不可",
    "不应",
    "不要",
    "禁止",
    "先补",
    "须先",
    "需要先",
    "无法同意",
    "不同意",
    "不会按",
    "请先",
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_ctx(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    # drop leading YAML-ish meta quotes block? keep full draft body
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n…（ctx truncated for edge）"
    return text


def fill_slots(prompt: str, slots: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return slots.get(key, m.group(0))

    return re.sub(r"\{\{([^}]+)\}\}", repl, prompt)


def heuristic_judge(item: dict, answer: str) -> dict:
    """Trend-only. Never treat as acceptance KPI."""
    ans = answer or ""
    pol = item.get("polarity")
    hits_refuse = sum(1 for h in REFUSAL_HINTS if h in ans)
    pass_hits = 0
    for c in item.get("pass_criteria") or []:
        # crude: any 2+ consecutive CJK/latin token from criterion appears
        toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", c)
        if any(t in ans for t in toks[:6]):
            pass_hits += 1
    forbid_hits = 0
    for f in item.get("forbid") or []:
        toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", f)
        # if answer echoes forbidden behavior phrasing without refusal — weak signal
        if any(t in ans for t in toks[:4]) and hits_refuse == 0:
            forbid_hits += 1

    if pol == "pol.neg":
        ok = hits_refuse >= 1 and forbid_hits == 0
        reason = f"refuse_hints={hits_refuse} forbid_echo={forbid_hits}"
    else:
        need = max(1, min(2, len(item.get("pass_criteria") or []) // 2 or 1))
        ok = pass_hits >= need and hits_refuse == 0
        reason = f"pass_hits={pass_hits} need>={need} refuse_hints={hits_refuse}"
    return {"judge": "heuristic", "pass_guess": ok, "detail": reason}


def ollama_chat(host: str, model: str, system: str, user: str, timeout: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content") or ""


def parse_slots(extra: list[str] | None) -> dict[str, str]:
    slots = dict(DEFAULT_SLOTS)
    for item in extra or []:
        if "=" not in item:
            raise SystemExit(f"bad --slot {item!r}, want key=value")
        k, v = item.split("=", 1)
        slots[k.strip()] = v.strip()
    return slots


def main() -> int:
    ap = argparse.ArgumentParser(description="mei-llm eval Ollama runner v0")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default="", help="Ollama model name; empty with --dry-run")
    ap.add_argument("--face", default="", help="filter face.dev / face.edge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge", choices=["none", "heuristic"], default="none")
    ap.add_argument("--ctx-max-chars", type=int, default=0, help="0=full ctx")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--slot", action="append", default=[], help="key=value override")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if load_mei_env is not None:
        load_mei_env(start=ROOT)

    if not args.dry_run and not args.model:
        print("need --model or --dry-run", file=sys.stderr)
        return 2

    items = load_jsonl(args.bank)
    if args.face:
        items = [i for i in items if i.get("face") == args.face]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    slots = parse_slots(args.slot)
    ctx_max = args.ctx_max_chars or None
    if args.face == "face.edge" and not args.ctx_max_chars:
        ctx_max = 3500
    system = load_ctx(args.ctx, ctx_max)

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "dry" if args.dry_run else args.model.replace(":", "_").replace("/", "_")
    out_dir = args.out_dir or (ROOT / "notebook/archive/runs" / f"{tag}-{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.jsonl"
    summary = {
        "created_utc": tag,
        "bank": str(args.bank.name),
        "model": args.model or None,
        "dry_run": args.dry_run,
        "judge": args.judge,
        "face_filter": args.face or None,
        "n": len(items),
        "note": "pending bank; not acceptance KPI unless review approved",
        "heuristic_pass": 0,
        "heuristic_fail": 0,
        "errors": 0,
    }

    with preds_path.open("w", encoding="utf-8") as fout:
        for item in items:
            user = fill_slots(item.get("prompt") or "", slots)
            rec = {
                "item_id": item["item_id"],
                "face": item.get("face"),
                "chapter": item.get("chapter"),
                "topic": item.get("topic"),
                "polarity": item.get("polarity"),
                "review_status": item.get("review_status"),
                "user_prompt": user,
                "answer": None,
                "error": None,
                "judgment": None,
            }
            if args.dry_run:
                rec["answer"] = ""
                rec["assembled_system_chars"] = len(system)
            else:
                try:
                    rec["answer"] = ollama_chat(
                        args.host, args.model, system, user, args.timeout
                    )
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                    rec["error"] = str(e)
                    summary["errors"] += 1

            if args.judge == "heuristic" and rec.get("answer") and not rec.get("error"):
                rec["judgment"] = heuristic_judge(item, rec["answer"])
                if rec["judgment"]["pass_guess"]:
                    summary["heuristic_pass"] += 1
                else:
                    summary["heuristic_fail"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "system_used.md").write_text(system, encoding="utf-8")
    from _run_report import write_report

    write_report(out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {preds_path}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
