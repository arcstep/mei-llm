#!/usr/bin/env python3
"""qwen-max teacher: rewrite query/scene wording only. Never emit answers/gold.

Uses mei_eval load_mei_env + resolve_endpoint(provider=QWEN, model=qwen-max) +
chat_complete. Explicit --model required. Checkpoints under notebook/archive/runs/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from repo_paths import BANK_NEEDLE_VRM_AGENT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "tools/mei-eval/python/src"))

from needle_home_sft_lib import (  # noqa: E402
    PROMPT_VERSION,
    dump_jsonl,
    load_jsonl,
    protected_slots_ok,
    query_banned,
    seen_key,
    sha256_file,
    sha256_text,
)
from tokenizer import ZhTokenizerV1  # noqa: E402
from data import token_jaccard  # noqa: E402

SYSTEM = """你是 needle-zh 家居助手的问法改写教师。
只改写用户说法，不要回答任务，不要输出工具名或 JSON 调用。
必须遵守：
- 不得改变是否应当执行
- 不得改变受保护槽的中文名词（店名、菜名、房间灯、门、地点、指向对象）
- 不得补问用户、不得写「请补充」
- 不要输出 EVAL 编号
只输出 JSON 数组，三项中文字符串，不要 markdown。"""


def load_style_hints(limit: int = 12) -> list[str]:
    cards_dir = ROOT / "notebook/sft/style-v0/cards"
    hints: list[str] = []
    for name in ("crosswoz-style.jsonl", "kdconv-style.jsonl"):
        for row in load_jsonl(cards_dir / name)[:200]:
            tags = ",".join(row.get("tags") or [])
            ex = str(row.get("example_redacted") or "")[:24]
            if tags:
                hints.append(f"{tags}｜{ex}")
    random.Random(0).shuffle(hints)
    return hints[:limit]


def load_utterance_hashes() -> set[str]:
    path = ROOT / "notebook/sft/style-v0/hashes/utterance.sha256"
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def strip_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_variants(text: str) -> list[str]:
    raw = strip_fence(text)
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    if not isinstance(obj, list):
        return []
    out: list[str] = []
    for item in obj:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("query"), str):
            out.append(str(item["query"]).strip())
    return out[:3]


def user_prompt(row: dict, hints: list[str]) -> str:
    intent = {
        "family": row.get("family"),
        "kind": row.get("kind"),
        "name": row.get("gold_name"),
        "scene": row.get("scene"),
        "stem_query": row.get("query"),
        "style_tags": row.get("style_tags"),
    }
    hint = "；".join(hints[:6]) if hints else "口语、命令、省略均可"
    return (
        "把下面意图改写成 3 句不同的中文用户说法。\n"
        f"意图JSON：{json.dumps(intent, ensure_ascii=False)}\n"
        f"口吻参考（不要抄原句）：{hint}\n"
        "输出：JSON 字符串数组，长度 3。"
    )


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "timeout" in name or "connection" in name or "internal" in name:
        return True
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "500", "502", "503", "504", "timeout", "temporar"))


def chat_with_retry(client, model: str, messages: list, *, retries: int, sleep_s: float):
    from mei_eval.chat import chat_complete

    delay = sleep_s
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return chat_complete(
                client,
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
                enable_thinking=False,
            )
        except Exception as exc:  # noqa: BLE001 — cloud retry
            last_exc = exc
            if attempt >= retries or not _retryable(exc):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise last_exc  # pragma: no cover


def overlay_pack(pack: Path, accepted: dict[str, str], teacher_model: str) -> int:
    rows = load_jsonl(pack)
    used = {seen_key(r) for r in rows}
    n = 0
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid not in accepted:
            continue
        new_q = accepted[sid]
        trial = dict(row)
        trial["query"] = new_q
        sk = seen_key(trial)
        if sk in used and new_q != row.get("query"):
            continue
        used.discard(seen_key(row))
        used.add(sk)
        row["query"] = new_q
        row["teacher_model"] = teacher_model
        row["source_role"] = "qwen-max-variant"
        n += 1
    dump_jsonl(pack, rows)
    man_path = pack.with_name(pack.name.replace(".jsonl", ".manifest.json"))
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["sha256"] = sha256_file(pack)
        man["teacher_model"] = f"template+{teacher_model}"
        man["qwen_overlay_n"] = n
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return n


def slots_from_original(row: dict) -> dict:
    q = str(row.get("query") or "")
    slots: dict[str, str] = {}
    for shop, dish in (("兰州拉面", "牛肉面"), ("麦当劳", "巨无霸")):
        if shop in q:
            slots["shop"] = shop
        if dish in q:
            slots["dish"] = dish
    for zh in ("厨房灯", "客厅灯"):
        if zh in q:
            slots["light"] = zh
    for zh in ("前门", "后门"):
        if zh in q:
            slots["door"] = zh
    for zh in ("厨房", "客厅", "门口", "垃圾桶"):
        if zh in q:
            slots["place"] = zh
    for zh in ("左边", "右边"):
        if zh in q:
            slots["target"] = zh
    if "指" in q and "我" in q:
        slots["target"] = slots.get("target") or "我"
    return slots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="must be qwen-max")
    ap.add_argument("--provider", default="QWEN")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--canary", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--apply-run", type=Path, default=None, help="overlay from an existing teacher run dir; no new API calls")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    if args.model.strip() != "qwen-max":
        print("refusing: pass --model qwen-max explicitly", file=sys.stderr)
        return 2
    pack = args.pack or (TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs/home-sft-2k.jsonl")
    if args.apply_run:
        accepted_path = args.apply_run / "accepted.jsonl"
        accepted = {str(r["sample_id"]): str(r["query"]) for r in load_jsonl(accepted_path)}
        n = overlay_pack(pack, accepted, "qwen-max") if args.overlay else 0
        print(json.dumps({"apply_run": str(args.apply_run), "overlay_n": n, "n_accepted": len(accepted)}, indent=2))
        return 0
    rows = load_jsonl(pack)
    if args.canary:
        rows = rows[: args.canary]
    elif args.limit:
        rows = rows[: args.limit]
    if args.max_calls is not None:
        rows = rows[: args.max_calls]

    from mei_eval.env import load_mei_env
    from mei_eval.endpoint import resolve_endpoint
    from mei_eval.chat import openai_client

    env_path = load_mei_env()
    ep = resolve_endpoint(provider=args.provider, model="qwen-max")
    if ep.model != "qwen-max":
        print(json.dumps({"error": "resolved model is not qwen-max", "got": ep.model}), file=sys.stderr)
        return 2
    client = openai_client(ep.base_url, api_key=ep.api_key, timeout=args.timeout, max_retries=0)

    run_dir = EXPERIMENTS_RUNS / f"needle-zh-sft-teacher-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint.jsonl"
    done_ids = {str(r.get("sample_id")) for r in load_jsonl(ckpt)}
    hints = load_style_hints()
    style_hashes = load_utterance_hashes()
    tok = ZhTokenizerV1()
    eval_queries = {str(r.get("query") or "").strip() for r in load_jsonl(BANK_NEEDLE_VRM_AGENT)}
    eval_ids = [tok.encode(q) for q in eval_queries if q]
    usage_prompt = 0
    usage_comp = 0
    accepted: dict[str, str] = {}
    n_fail = 0
    n_calls = 0

    def eval_leak(q: str) -> bool:
        if q in eval_queries:
            return True
        ids = tok.encode(q)
        if len(ids) < 2:
            return False
        return any(token_jaccard(ids, e) >= 0.9 for e in eval_ids if len(e) >= 2)

    with ckpt.open("a", encoding="utf-8") as handle:
        for row in rows:
            sid = str(row.get("sample_id") or "")
            if sid in done_ids:
                continue
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(row, hints)},
            ]
            prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False))
            rec = {
                "sample_id": sid,
                "prompt_hash": prompt_hash,
                "model": "qwen-max",
                "prompt_version": PROMPT_VERSION,
            }
            try:
                text, usage, ms = chat_with_retry(
                    client, "qwen-max", messages, retries=args.retries, sleep_s=args.sleep
                )
                n_calls += 1
                rec["latency_ms"] = round(ms, 1)
                rec["response_hash"] = sha256_text(text)
                rec["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                rec["raw_text"] = text
                usage_prompt += int(usage.get("prompt_tokens") or 0)
                usage_comp += int(usage.get("completion_tokens") or 0)
                variants = parse_variants(text)
                kept: list[str] = []
                intent = {
                    "kind": row.get("kind"),
                    "name": row.get("gold_name"),
                    "zh_slots": slots_from_original(row),
                    "family": row.get("family"),
                }
                for q in variants:
                    reason = query_banned(q)
                    if reason:
                        rec.setdefault("reject", []).append({"q": q, "reason": reason})
                        continue
                    if sha256_text(q) in style_hashes or eval_leak(q):
                        rec.setdefault("reject", []).append({"q": q, "reason": "near_dup"})
                        continue
                    if not protected_slots_ok(intent, q):
                        rec.setdefault("reject", []).append({"q": q, "reason": "slot_drop"})
                        continue
                    kept.append(q)
                rec["kept"] = kept
                if kept:
                    accepted[sid] = kept[0]
                    rec["accept"] = kept[0]
                else:
                    n_fail += 1
                    rec["accept"] = None
            except Exception as exc:  # noqa: BLE001
                rec["error"] = type(exc).__name__
                rec["error_text"] = str(exc)[:200]
                n_fail += 1
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            time.sleep(args.sleep)

    summary = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "env_source": Path(env_path).name,
        "model": "qwen-max",
        "provider": ep.provider,
        "base_url_host": ep.base_url.split("/")[2] if "://" in ep.base_url else "redacted",
        "n_in": len(rows),
        "n_calls": n_calls,
        "n_accepted": len(accepted),
        "n_fail": n_fail,
        "prompt_tokens": usage_prompt,
        "completion_tokens": usage_comp,
        "prompt_version": PROMPT_VERSION,
        "canary": bool(args.canary),
        "overlay": bool(args.overlay),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_jsonl(run_dir / "accepted.jsonl", [{"sample_id": k, "query": v} for k, v in accepted.items()])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.overlay and accepted:
        overlay_pack(pack, accepted, "qwen-max")
    if args.canary and rows and len(accepted) < max(1, int(0.7 * len(rows))):
        print("canary accept rate below 70%", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
