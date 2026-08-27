#!/usr/bin/env python3
"""Optional qwen-max teacher for holdout v2 query variants. Gold is never rewritten.

Template bank is the deliverable; this script only overlays wording.
Pass --model qwen-max explicitly. Raw responses stay under notebook/archive/runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_VRM_AGENT_V2,
    EXPERIMENTS_RUNS,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "tools/mei-eval/python/src"))

from generate_needle_home_sft_variants import (  # noqa: E402
    chat_with_retry,
    load_style_hints,
    parse_variants,
)
from needle_home_sft_lib import dump_jsonl, load_jsonl, sha256_text  # noqa: E402
from needle_vrm_holdout_v2_lib import (  # noqa: E402
    PROMPT_VERSION,
    collect_blocked_queries,
    intent_from_row,
    protected_slots_ok,
    query_banned,
)
from tokenizer import ZhTokenizerV1  # noqa: E402
from data import token_jaccard  # noqa: E402

SYSTEM = """你是 needle-zh 家居评测题的问法改写教师。
只改写用户说法，不要回答任务，不要输出工具名或 JSON 调用或金答。
必须遵守：
- 不得改变是否应当执行，不得改变调用顺序
- 不得改变受保护槽的中文名词
- 不得补问、不得写「请补充」、不得输出 EVAL 编号
只输出 JSON 数组，三项中文字符串，不要 markdown。"""


def user_prompt(row: dict, hints: list[str]) -> str:
    payload = {
        "family": row.get("family"),
        "kind": row.get("kind"),
        "stem": row.get("stem"),
        "scene": row.get("scene"),
        "style_tags": row.get("style_tags"),
        "slots": row.get("zh_slots"),
    }
    hint = "；".join(hints[:6]) if hints else "口语、命令、省略均可"
    return (
        "把下面评测意图改写成 3 句不同的中文用户说法，保持槽位与执行结论。\n"
        f"意图JSON：{json.dumps(payload, ensure_ascii=False)}\n"
        f"口吻参考（不要抄原句）：{hint}\n"
        "输出：JSON 字符串数组，长度 3。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="QWEN")
    ap.add_argument("--bank", type=Path, default=BANK_NEEDLE_VRM_AGENT_V2)
    ap.add_argument("--canary", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()
    if args.model.strip() != "qwen-max":
        print("refusing: pass --model qwen-max explicitly", file=sys.stderr)
        return 2
    rows = load_jsonl(args.bank)
    if args.canary:
        rows = [r for r in rows if r.get("split") == "dev"][: args.canary]
    elif args.limit:
        rows = rows[: args.limit]

    from mei_eval.env import load_mei_env
    from mei_eval.endpoint import resolve_endpoint
    from mei_eval.chat import openai_client

    env_path = load_mei_env()
    ep = resolve_endpoint(provider=args.provider, model="qwen-max")
    if ep.model != "qwen-max":
        print(json.dumps({"error": "resolved model is not qwen-max", "got": ep.model}), file=sys.stderr)
        return 2
    client = openai_client(ep.base_url, api_key=ep.api_key, timeout=args.timeout, max_retries=0)
    run_dir = EXPERIMENTS_RUNS / f"needle-vrm-v2-teacher-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = ZhTokenizerV1()
    blocked = collect_blocked_queries()
    blocked_ids = [tok.encode(q) for q in blocked if q]
    hints = load_style_hints()
    accepted: dict[str, str] = {}
    n_fail = 0
    n_calls = 0
    ckpt = run_dir / "checkpoint.jsonl"
    with ckpt.open("a", encoding="utf-8") as handle:
        for row in rows:
            sid = str(row.get("item_id") or "")
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(row, hints)},
            ]
            rec: dict = {"item_id": sid, "prompt_version": PROMPT_VERSION, "model": "qwen-max"}
            try:
                text, usage, ms = chat_with_retry(
                    client, "qwen-max", messages, retries=args.retries, sleep_s=args.sleep
                )
                n_calls += 1
                rec["latency_ms"] = round(ms, 1)
                rec["raw_text"] = text
                rec["usage"] = usage
                kept: list[str] = []
                intent = intent_from_row(row)
                for q in parse_variants(text):
                    if query_banned(q):
                        continue
                    if q in blocked:
                        continue
                    ids = tok.encode(q)
                    if len(ids) >= 4 and any(
                        token_jaccard(ids, e) >= 0.9 for e in blocked_ids if len(e) >= 4
                    ):
                        continue
                    if not protected_slots_ok(intent, q):
                        continue
                    kept.append(q)
                rec["kept"] = kept
                if kept:
                    accepted[sid] = kept[0]
                    rec["accept"] = kept[0]
                else:
                    n_fail += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = type(exc).__name__
                rec["error_text"] = str(exc)[:200]
                n_fail += 1
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            time.sleep(args.sleep)
    dump_jsonl(run_dir / "accepted.jsonl", [{"item_id": k, "query": v} for k, v in accepted.items()])
    summary = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "env_source": Path(env_path).name,
        "model": "qwen-max",
        "n_in": len(rows),
        "n_calls": n_calls,
        "n_accepted": len(accepted),
        "n_fail": n_fail,
        "canary": bool(args.canary),
        "note": "gold unchanged; overlay only rewrites query",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.overlay and accepted:
        updated = []
        for row in load_jsonl(args.bank):
            sid = str(row.get("item_id") or "")
            if sid in accepted:
                row["query"] = accepted[sid]
                row["teacher_model"] = "qwen-max"
            updated.append(row)
        dump_jsonl(args.bank, updated)
    if args.canary and rows and len(accepted) < max(1, int(0.7 * len(rows))):
        print("canary accept rate below 70%", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
