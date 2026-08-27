#!/usr/bin/env python3
"""MLX local eval runner v0 — phase spine Qwen/Qwen3.5-0.8B (non-quantized).

Primary stage baseline. Do not treat Ollama q or cloud plus as phase KPI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import BANK_MEI_EXPERT, ROOT

MEI_PROJECTS_ROOT = ROOT.parent
DRAFT_DOCS = MEI_PROJECTS_ROOT / "docs/draft/mei-llm"
DEFAULT_BANK = BANK_MEI_EXPERT
DEFAULT_CTX = MEI_PROJECTS_ROOT / "docs/mei-llm/02-eval/0205-ctx-stable-v0.md"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

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
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_ctx(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n…（ctx truncated）"
    return text


def fill_slots(prompt: str, slots: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        return slots.get(m.group(1), m.group(0))

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


def parse_slots(extra: list[str] | None) -> dict[str, str]:
    slots = dict(DEFAULT_SLOTS)
    for item in extra or []:
        if "=" not in item:
            raise SystemExit(f"bad --slot {item!r}")
        k, v = item.split("=", 1)
        slots[k.strip()] = v.strip()
    return slots


class MlxChat:
    def __init__(self, model_id: str, adapter_path: str | None) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)

    def generate(self, system: str, user: str, max_tokens: int, temp: float) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt: str
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = (
                f"<|im_start|>system\n{system}\n"
                f"<|im_start|>user\n{user}\n"
                f"<|im_start|>assistant\n"
            )
        sampler = make_sampler(temp=temp)
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        text = re.sub(
            r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="mei-llm MLX eval runner v0 (Qwen3.5-0.8B)")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="", help="optional LoRA adapter path")
    ap.add_argument("--face", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge", choices=["none", "heuristic"], default="none")
    ap.add_argument("--ctx-max-chars", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--slot", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

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
    label = "dry" if args.dry_run else "mlx-" + args.model.replace("/", "_").replace(":", "_")
    if args.adapter:
        label += "-lora"
    out_dir = args.out_dir or (ROOT / "notebook/archive/runs" / f"{tag}-{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    chat: MlxChat | None = None
    if not args.dry_run:
        print(f"loading {args.model} …", file=sys.stderr)
        chat = MlxChat(args.model, args.adapter or None)

    summary = {
        "created_utc": tag,
        "provider": "mlx_lm",
        "spine": "qwen35-0p8b-phase",
        "bank": args.bank.name,
        "model": args.model,
        "adapter": args.adapter or None,
        "dry_run": args.dry_run,
        "judge": args.judge,
        "face_filter": args.face or None,
        "n": len(items),
        "note": "phase baseline; pending ≠ acceptance until human review",
        "heuristic_pass": 0,
        "heuristic_fail": 0,
        "errors": 0,
    }

    preds_path = out_dir / "predictions.jsonl"
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
                "provider": "mlx_lm",
                "model": args.model,
                "adapter": args.adapter or None,
                "user_prompt": user,
                "answer": None,
                "error": None,
                "judgment": None,
            }
            if args.dry_run:
                rec["answer"] = ""
                rec["assembled_system_chars"] = len(system)
            else:
                assert chat is not None
                try:
                    rec["answer"] = chat.generate(
                        system, user, args.max_tokens, args.temperature
                    )
                except Exception as e:  # noqa: BLE001 — keep run going
                    rec["error"] = f"{type(e).__name__}: {e}"
                    summary["errors"] += 1

            if args.judge == "heuristic" and rec.get("answer") and not rec.get("error"):
                rec["judgment"] = heuristic_judge(item, rec["answer"])
                if rec["judgment"]["pass_guess"]:
                    summary["heuristic_pass"] += 1
                else:
                    summary["heuristic_fail"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

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
