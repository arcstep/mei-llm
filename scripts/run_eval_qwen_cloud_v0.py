#!/usr/bin/env python3
"""Qwen cloud eval runner v0 — OpenAI-compatible (DashScope / mei-projects .env).

Reads QWEN_BASE_URL / QWEN_API_KEY / QWEN_COMPLETION_MODEL from env, optionally
auto-loading mei-projects/.env (same keys as Host). Falls back to DASHSCOPE_*.
Does NOT write predictions into train. Never prints API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # mei-llm/
MEI_PROJECTS_ROOT = ROOT.parent  # mei-llm → mei-projects
DRAFT_DOCS = MEI_PROJECTS_ROOT / "docs/draft/mei-llm"
DEFAULT_BANK = ROOT / "data/eval/eval-bank-v0.pending.jsonl"
DEFAULT_CTX = DRAFT_DOCS / "2026-07-31-ctx-stable-v0.md"
FALLBACK_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

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


def _load_dotenv_file(path: Path) -> bool:
    """Minimal KEY=VALUE loader; does not override existing os.environ."""
    if not path.is_file():
        return False
    loaded = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            loaded = True
    return loaded


def resolve_qwen_config(env_file: Path | None = None) -> tuple[str, str, list[str], str]:
    """Return (base_url, api_key, model_candidates, env_source_note)."""
    source = "process-env"
    if env_file is not None:
        if _load_dotenv_file(env_file):
            source = str(env_file)
    elif load_mei_env is not None:
        env_path = load_mei_env(start=ROOT)
        source = str(env_path) if env_path.is_file() else "process-env"
    else:
        for p in (MEI_PROJECTS_ROOT / ".env", Path.cwd() / ".env"):
            if _load_dotenv_file(p):
                source = str(p)
                break

    base = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or FALLBACK_BASE
    ).strip()
    key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    models_raw = (
        os.environ.get("QWEN_COMPLETION_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or ""
    ).strip()
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    return base, key, models, source


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_ctx(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n…（ctx truncated for edge）"
    return text


def fill_slots(prompt: str, slots: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return slots.get(key, m.group(0))

    return re.sub(r"\{\{([^}]+)\}\}", repl, prompt)


def heuristic_judge(item: dict, answer: str) -> dict:
    ans = answer or ""
    pol = item.get("polarity")
    hits_refuse = sum(1 for h in REFUSAL_HINTS if h in ans)
    pass_hits = 0
    for c in item.get("pass_criteria") or []:
        toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", c)
        if any(t in ans for t in toks[:6]):
            pass_hits += 1
    forbid_hits = 0
    for f in item.get("forbid") or []:
        toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", f)
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


def openai_compat_chat(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    temperature: float,
) -> tuple[str, dict]:
    import ssl

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    # macOS system Python often lacks CA bundle; prefer certifi when present.
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
    usage = data.get("usage") or {}
    return content, usage


def parse_slots(extra: list[str] | None) -> dict[str, str]:
    slots = dict(DEFAULT_SLOTS)
    for item in extra or []:
        if "=" not in item:
            raise SystemExit(f"bad --slot {item!r}, want key=value")
        k, v = item.split("=", 1)
        slots[k.strip()] = v.strip()
    return slots


def main() -> int:
    ap = argparse.ArgumentParser(
        description="mei-llm eval Qwen cloud runner v0 (QWEN_* / OpenAI-compatible)"
    )
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    ap.add_argument(
        "--base-url",
        default="",
        help="override; default QWEN_BASE_URL from mei-projects/.env",
    )
    ap.add_argument(
        "--model",
        default="",
        help="override; default first of QWEN_COMPLETION_MODEL",
    )
    ap.add_argument("--face", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge", choices=["none", "heuristic"], default="none")
    ap.add_argument("--ctx-max-chars", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--slot", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="explicit .env path (default: mei-projects/.env)",
    )
    args = ap.parse_args()

    if args.env_file:
        base_url, api_key, model_list, env_source = resolve_qwen_config(args.env_file)
    else:
        base_url, api_key, model_list, env_source = resolve_qwen_config()
    if args.base_url:
        base_url = args.base_url.strip()

    model = (args.model or (model_list[0] if model_list else "")).strip()

    if not args.dry_run:
        if not model:
            print(
                "need --model or QWEN_COMPLETION_MODEL in .env (or --dry-run)",
                file=sys.stderr,
            )
            return 2
        if not api_key:
            print(
                "missing QWEN_API_KEY (mei-projects/.env or export; do not commit)",
                file=sys.stderr,
            )
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
    mode = "dry" if args.dry_run else ("cloud-" + model.replace(":", "_").replace("/", "_"))
    out_dir = args.out_dir or (ROOT / "experiments/runs" / f"{tag}-{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.jsonl"
    summary = {
        "created_utc": tag,
        "provider": "openai-compatible",
        "spine": "qwen-cloud-primary",
        "env_source": env_source,
        "bank": str(args.bank.name),
        "base_url": base_url,
        "model": model or None,
        "model_candidates": model_list,
        "dry_run": args.dry_run,
        "judge": args.judge,
        "face_filter": args.face or None,
        "n": len(items),
        "note": "cloud baseline; pending bank ≠ acceptance until human review",
        "heuristic_pass": 0,
        "heuristic_fail": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
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
                "provider": "openai-compatible",
                "model": model or None,
                "user_prompt": user,
                "answer": None,
                "usage": None,
                "error": None,
                "judgment": None,
            }
            if args.dry_run:
                rec["answer"] = ""
                rec["assembled_system_chars"] = len(system)
            else:
                try:
                    answer, usage = openai_compat_chat(
                        base_url,
                        api_key,
                        model,
                        system,
                        user,
                        args.timeout,
                        args.temperature,
                    )
                    rec["answer"] = answer
                    rec["usage"] = usage
                    summary["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    summary["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    json.JSONDecodeError,
                ) as e:
                    err = str(e)
                    if isinstance(e, urllib.error.HTTPError):
                        try:
                            err = e.read().decode("utf-8", errors="replace")[:500]
                        except Exception:
                            pass
                    rec["error"] = err
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
