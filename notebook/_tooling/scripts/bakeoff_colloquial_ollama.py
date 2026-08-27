#!/usr/bin/env python3
"""Same-frame bake-off: local Ollama Qwen vs frozen cloud qwen-plus samples.

Does not write into the production qwen-v1 accepted.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1, CORPORA_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_synth_lib import (  # noqa: E402
    filter_reasons,
    load_contract,
    parse_turns_payload,
    sanitize_spoken_text,
    system_prompt,
    user_prompt_for_frame,
)
from zh_pretrain_ingest import colloquial_keep, dump_json, pii_or_nav  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODELS = (
    ("qwen3.5-9b", "qwen3.5:9b-mlx", 4),
    ("qwen3.6-27b", "qwen3.6:27b-mlx", 2),
    ("qwen3.6-35b", "qwen3.6:35b-mlx", 1),
)


def load_plus_rows(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("filter") or {}).get("ok") and row.get("frame"):
                rows.append(row)
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for row in rows:
        frame = row["frame"]
        key = f"{frame.get('scene')}|{(frame.get('styles') or [''])[0]}"
        by.setdefault(key, []).append(row)
    picked = []
    keys = list(by)
    rng.shuffle(keys)
    for key in keys:
        if len(picked) >= n:
            break
        bucket = by[key]
        picked.append(bucket[rng.randrange(len(bucket))])
    if len(picked) < n:
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def ollama_chat(model: str, frame: dict, contract: dict, timeout_s: int = 300) -> dict:
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "messages": [
            {"role": "system", "content": system_prompt(contract["prompt_version"])},
            {"role": "user", "content": user_prompt_for_frame(frame, min_turns=6)},
        ],
        "options": {
            "temperature": float(contract["sampling"]["temperature"]),
            "top_p": float(contract["sampling"]["top_p"]),
            "num_predict": int(contract["sampling"]["max_tokens"]),
        },
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=data, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}", "elapsed_s": time.time() - t0}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "elapsed_s": time.time() - t0}
    content = str(((raw.get("message") or {}).get("content")) or "")
    parsed = parse_turns_payload(content)
    if parsed is None:
        return {"ok": False, "error": "invalid_json", "raw": content[:800], "elapsed_s": time.time() - t0}
    text = sanitize_spoken_text(parsed["text"])
    reasons = filter_reasons(text, leaks=[], pii_fn=pii_or_nav)
    if not colloquial_keep(text, domain="dialogue"):
        reasons.append("not_spoken")
    return {
        "ok": not reasons,
        "text": text,
        "reasons": reasons,
        "elapsed_s": time.time() - t0,
        "eval_count": int((raw.get("eval_count") or 0)),
        "prompt_eval_count": int((raw.get("prompt_eval_count") or 0)),
        "eval_duration_ns": int((raw.get("eval_duration") or 0)),
    }


def score_auto(text: str, frame: dict | None = None, plus: str | None = None) -> dict:
    import re

    spoken = bool(re.search(r"(吗|呢|吧|啊|哎|哦|恩|呀|啥|咱们|那个)", text or ""))
    news = bool(re.search(r"(据悉|近年来|综上所述|进行了|具有重要意义)", text or ""))
    tool = bool(re.search(r"(\{\"route_id\"|EVAL-|function_calls)", text or ""))
    n_turns = (text or "").count("甲：") + (text or "").count("乙：")
    styles = list((frame or {}).get("styles") or [])
    style_hits = []
    if "code_mix" in styles:
        style_hits.append(bool(re.search(r"[A-Za-z]{2,}", text or "")))
    if "negation" in styles:
        style_hits.append(bool(re.search(r"(不|没|别|甭)", text or "")))
    if "ellipsis" in styles:
        style_hits.append("…" in (text or "") or "..." in (text or "") or "那个" in (text or ""))
    if "repair" in styles:
        style_hits.append(bool(re.search(r"(不是|我是说|等下|等等|重新)", text or "")))
    style_rate = (sum(style_hits) / len(style_hits)) if style_hits else 1.0
    plus_len = len(plus or "")
    local_len = len(text or "")
    len_ratio = (local_len / plus_len) if plus_len else 0.0
    thin = local_len < 80 or (plus_len and len_ratio < 0.45)
    auto_pass = spoken and not news and not tool and n_turns >= 6 and not thin and style_rate >= 0.5
    return {
        "spoken_marker": spoken,
        "news_register": news,
        "toolish": tool,
        "n_turn_lines": n_turns,
        "style_rate": round(style_rate, 3),
        "char_len": local_len,
        "len_ratio_vs_plus": round(len_ratio, 3),
        "thin": thin,
        "auto_pass": auto_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--plus-corpus", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    ap.add_argument("--out-dir", type=Path, default=CORPORA_ROOT / "zh-pretrain-colloquial-synth-ollama-bakeoff")
    ap.add_argument("--models", default="qwen3.5-9b,qwen3.6-27b,qwen3.6-35b")
    args = ap.parse_args()
    contract = load_contract()
    plus_dir = args.plus_corpus if args.plus_corpus.is_absolute() else ROOT / args.plus_corpus
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews").mkdir(parents=True, exist_ok=True)
    plus_rows = load_plus_rows(plus_dir / "raw" / "accepted.jsonl", args.n, args.seed)
    if not plus_rows:
        print("need plus samples, got 0", file=sys.stderr)
        return 2
    want = {x.strip() for x in args.models.split(",") if x.strip()}
    models = [m for m in MODELS if m[0] in want]
    pairs = []
    summary = {"n": len(plus_rows), "models": {}}
    for key, model, workers in models:
        t0 = time.time()
        got_map: dict[str, dict] = {}

        def one(row: dict, model=model) -> tuple[str, dict]:
            frame = row["frame"]
            fid = str(row.get("doc_id") or frame.get("frame_id"))
            return fid, ollama_chat(model, frame, contract)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, row) for row in plus_rows]
            for fut in as_completed(futs):
                fid, got = fut.result()
                got_map[fid] = got
        elapsed = time.time() - t0
        n_ok = sum(1 for g in got_map.values() if g.get("ok"))
        n_json = sum(1 for g in got_map.values() if g.get("text") or g.get("ok"))
        n_auto = 0
        toks_s = []
        len_ratios = []
        style_rates = []
        n_thin = 0
        n_plus_auto = 0
        for row in plus_rows:
            fid = str(row.get("doc_id") or row["frame"]["frame_id"])
            g = got_map.get(fid) or {}
            auto = score_auto(str(g.get("text") or ""), row.get("frame"), row.get("text"))
            plus_auto = score_auto(str(row.get("text") or ""), row.get("frame"), row.get("text"))
            if plus_auto["auto_pass"]:
                n_plus_auto += 1
            if auto["auto_pass"] and g.get("ok"):
                n_auto += 1
            if auto.get("thin"):
                n_thin += 1
            len_ratios.append(float(auto.get("len_ratio_vs_plus") or 0))
            style_rates.append(float(auto.get("style_rate") or 0))
            dur = float(g.get("eval_duration_ns") or 0) / 1e9
            ntok = int(g.get("eval_count") or 0)
            if dur > 0 and ntok:
                toks_s.append(ntok / dur)
        summary["plus_auto_pass_rate"] = round(n_plus_auto / len(plus_rows), 4)
        summary["models"][key] = {
            "ollama_model": model,
            "workers": workers,
            "elapsed_s": round(elapsed, 1),
            "docs_per_s": round(len(plus_rows) / elapsed, 3) if elapsed else 0,
            "ok_filter_rate": round(n_ok / len(plus_rows), 4),
            "auto_pass_rate": round(n_auto / len(plus_rows), 4),
            "json_ok_rate": round(n_json / len(plus_rows), 4),
            "thin_rate": round(n_thin / len(plus_rows), 4),
            "mean_len_ratio_vs_plus": round(sum(len_ratios) / len(len_ratios), 3) if len_ratios else 0,
            "mean_style_rate": round(sum(style_rates) / len(style_rates), 3) if style_rates else 0,
            "mean_decode_tok_s": round(sum(toks_s) / len(toks_s), 1) if toks_s else 0,
        }
        print(json.dumps({"model": key, **summary["models"][key]}, ensure_ascii=False), flush=True)
        if not pairs:
            for row in plus_rows:
                fid = str(row.get("doc_id") or row["frame"]["frame_id"])
                pairs.append(
                    {
                        "frame_id": fid,
                        "frame": row["frame"],
                        "plus": row.get("text"),
                        "local": {},
                    }
                )
        for pair in pairs:
            g = got_map.get(pair["frame_id"]) or {}
            pair["local"][key] = {
                "ok": g.get("ok"),
                "text": g.get("text"),
                "reasons": g.get("reasons"),
                "error": g.get("error"),
                "elapsed_s": g.get("elapsed_s"),
                "auto": score_auto(str(g.get("text") or ""), pair["frame"], pair.get("plus")),
            }
    pairs_path = out / "reviews" / "bakeoff-pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    dump_json(out / "reviews" / "bakeoff-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
