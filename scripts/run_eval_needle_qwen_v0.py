#!/usr/bin/env python3
"""Closed-set needle tool-router baseline (Qwen). Not a needle-zh proxy.

Writes predictions JSONL for eval_needle_toolcall_v0.py. Natural-language
clarifications are NOT coerced to [] (that would false-pass missing/offtopic).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval_needle_toolcall_v0 import load_jsonl, load_toolset, score
from repo_paths import BANK_NEEDLE_VRM_AGENT, EXPERIMENTS_RUNS, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from schema_render import compact_tools  # noqa: E402

UNPARSED = [{"name": "_unparsed", "arguments": {}}]
CLARIFY = [{"name": "_clarify", "arguments": {}}]

CLARIFY_RE = re.compile(
    r"请补充|请提供|请问(?:哪|什么|要开|要关)|哪一扇|哪扇门|哪盏灯|哪个开关|"
    r"告诉我.{0,8}(?:门|灯|店|菜|房间)|你想(?:开|关|去|点)|"
    r"还需要(?:什么|知道)|缺少(?:必填|参数)|需要知道|"
    r"which (?:door|light|one|switch|place|shop|dish)|please specify|"
    r"need to know which|front or back|what (?:door|light)|which one",
    re.I,
)

SYSTEM = """你是闭集家居代理的路由器，不是聊天助手。
只允许调用系统给出的工具，或输出空列表。
缺必填槽、离题、场景已使动作无效（门已开再开、无进行中订单却取消）、订餐非法搭配：必须输出 []。
禁止提问、禁止请用户补充、禁止解释、禁止自然语言。
只输出 JSON 数组，形如 [{"name":"nod","arguments":{}}] 或 []。
"""

SYSTEM_EN = """You are a closed-set home-agent router, not a chatbot.
You may only call the given tools, or output an empty list.
If a required slot is missing, the request is off-topic, the scene already makes the action invalid (door already open, no in-progress order to cancel), or the shop/dish pair is illegal: you MUST output [].
Do not ask questions, do not ask the user to fill slots, do not explain, do not use natural language.
Output only a JSON array, like [{"name":"nod","arguments":{}}] or [].
"""

_MLX = {"model": None, "tokenizer": None}


def _lang(row: dict) -> str:
    return str(row.get("lang") or "zh").lower()


def user_text(row: dict) -> str:
    query = str(row.get("query") or "").strip()
    scene = row.get("scene")
    if _lang(row).startswith("en"):
        if isinstance(scene, str) and scene.strip():
            return f"Scene: {scene.strip()}. User: {query}"
        return f"User: {query}"
    if isinstance(scene, str) and scene.strip():
        return f"场景：{scene.strip()}。用户：{query}"
    return f"用户：{query}"


def system_for(toolset: dict, lang: str = "zh") -> str:
    tools = json.dumps(compact_tools(toolset), ensure_ascii=False)
    if str(lang).lower().startswith("en"):
        return SYSTEM_EN + "\nTool schema:\n" + tools
    return SYSTEM + "\n工具 schema：\n" + tools


def _strip_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _as_call(obj: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    name_field = obj.get("name") or obj.get("function") or ""
    args: object
    if isinstance(name_field, dict):
        args = name_field.get("arguments")
        if args is None:
            args = name_field.get("arguments")
        name = name_field.get("name") or ""
    else:
        name = name_field
        args = obj.get("arguments")
        if args is None:
            args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not name:
        return None
    if not isinstance(args, dict):
        args = {}
    return {"name": str(name), "arguments": args}


def _parse_json_value(text: str):
    t = _strip_fence(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = t.find(opener)
        end = t.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_function_calls(raw: str) -> tuple[list[dict], str]:
    text = raw or ""
    data = _parse_json_value(text)
    calls: list[dict] | None = None
    if data == []:
        calls = []
    elif isinstance(data, list):
        parsed = [_as_call(x) for x in data]
        if parsed and all(c is not None for c in parsed):
            calls = [c for c in parsed if c]
    elif isinstance(data, dict):
        wrapped = data.get("function_calls")
        if isinstance(wrapped, list):
            if wrapped == []:
                calls = []
            else:
                parsed = [_as_call(x) for x in wrapped]
                if parsed and all(c is not None for c in parsed):
                    calls = [c for c in parsed if c]
        else:
            one = _as_call(data)
            if one:
                calls = [one]

    clarify = bool(CLARIFY_RE.search(text))
    if clarify and (calls is None or calls == []):
        return list(CLARIFY), "clarify"
    if calls is None:
        return list(UNPARSED), "unparsed"
    return calls, "ok"


def _ns_ms(ns: object) -> float | None:
    if ns is None:
        return None
    try:
        return round(float(ns) / 1e6, 3)
    except (TypeError, ValueError):
        return None


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return round(ys[idx], 1)


def summarize_latency(rows: list[dict]) -> dict:
    walls = [float(r["latency"]["wall_ms"]) for r in rows if (r.get("latency") or {}).get("wall_ms") is not None]
    evals = [float(r["latency"]["eval_ms"]) for r in rows if (r.get("latency") or {}).get("eval_ms") is not None]
    tok_s = [float(r["latency"]["output_tok_s"]) for r in rows if (r.get("latency") or {}).get("output_tok_s")]
    loads = [float(r["latency"]["load_ms"]) for r in rows if (r.get("latency") or {}).get("load_ms") is not None]
    steady_walls = walls[1:] if len(walls) > 1 else walls

    def pack(name: str, xs: list[float]) -> dict:
        if not xs:
            return {name: None}
        return {
            f"{name}_mean": round(statistics.fmean(xs), 1),
            f"{name}_p50": _quantile(xs, 0.50),
            f"{name}_p95": _quantile(xs, 0.95),
            f"{name}_max": round(max(xs), 1),
        }

    out = {
        "n_timed": len(walls),
        "cold_first_wall_ms": round(walls[0], 1) if walls else None,
        "note": "wall_ms is client RTT; eval_ms is Ollama decode; first item often includes model load",
    }
    out.update(pack("wall_ms", walls))
    out.update(pack("steady_wall_ms", steady_walls))
    out.update(pack("eval_ms", evals))
    if tok_s:
        out["output_tok_s_mean"] = round(statistics.fmean(tok_s), 2)
    if loads:
        out["load_ms_max"] = round(max(loads), 1)
    return out


def ollama_chat(
    host: str, model: str, system: str, user: str, timeout: int, temperature: float
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": 256},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000
    msg = data.get("message") or {}
    text = (msg.get("content") or msg.get("thinking") or "").strip()
    eval_count = data.get("eval_count")
    eval_ns = data.get("eval_duration")
    out_tok_s = None
    if eval_count and eval_ns:
        try:
            out_tok_s = round(float(eval_count) / (float(eval_ns) / 1e9), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            out_tok_s = None
    latency = {
        "wall_ms": round(wall_ms, 1),
        "total_ms": _ns_ms(data.get("total_duration")),
        "load_ms": _ns_ms(data.get("load_duration")),
        "prompt_eval_ms": _ns_ms(data.get("prompt_eval_duration")),
        "eval_ms": _ns_ms(eval_ns),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": eval_count,
        "output_tok_s": out_tok_s,
    }
    return text, latency


def load_mlx(model_id: str) -> None:
    from mlx_lm import load

    _MLX["model"], _MLX["tokenizer"] = load(model_id)


def mlx_chat(system: str, user: str, max_tokens: int, temperature: float) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model = _MLX["model"]
    tokenizer = _MLX["tokenizer"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        prompt = (
            f"<|im_start|>system\n{system}\n"
            f"<|im_start|>user\n{user}\n"
            f"<|im_start|>assistant\n"
        )
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
    )
    return re.sub(
        r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
        "",
        text,
        flags=re.DOTALL | re.I,
    ).strip()


def self_test() -> int:
    cases = [
        ("[]", [], "ok"),
        ('[{"name":"nod","arguments":{}}]', [{"name": "nod", "arguments": {}}], "ok"),
        ("请补充要开哪扇门", list(CLARIFY), "clarify"),
        ("今天股市怎么样，我再想想", list(UNPARSED), "unparsed"),
        ("```json\n[]\n```", [], "ok"),
        ('{"name":"wave","arguments":{}}', [{"name": "wave", "arguments": {}}], "ok"),
    ]
    failed = 0
    for raw, want, status in cases:
        got, st = parse_function_calls(raw)
        if got != want or st != status:
            print(f"FAIL {raw!r}: got {got} {st} want {want} {status}", file=sys.stderr)
            failed += 1
    print(json.dumps({"ok": failed == 0, "failed": failed}, ensure_ascii=False))
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen closed-set needle tool-router baseline")
    ap.add_argument(
        "--bank",
        type=Path,
        default=BANK_NEEDLE_VRM_AGENT,
        help="Explicit bank path. Default is frozen v1 holdout; do not infer from directory order. "
        "For current 2K KPI pass eval/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl",
    )
    ap.add_argument("--backend", choices=["ollama", "openai", "mlx"], default="ollama")
    ap.add_argument("--model", default="qwen3.5:0.8b-mlx")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--split", default=None, help="Optional split filter (dev|eval). Default: whole bank.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    bank = args.bank if args.bank.is_absolute() else (ROOT / args.bank)
    rows = load_jsonl(bank)
    if args.split:
        rows = [r for r in rows if str(r.get("split") or "") == args.split]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    toolsets: dict[str, dict] = {}
    for row in rows:
        tid = str(row.get("toolset_id") or "")
        if tid and tid not in toolsets:
            toolsets[tid] = load_toolset(tid)

    if args.backend == "mlx" and not args.dry_run:
        print(f"loading mlx {args.model} …", file=sys.stderr)
        load_mlx(args.model)

    openai_chat = None
    openai_base = openai_key = openai_model = ""
    if args.backend == "openai" and not args.dry_run:
        from run_eval_qwen_cloud_v0 import openai_compat_chat, resolve_qwen_config

        openai_base, openai_key, models, _source = resolve_qwen_config()
        if not openai_key:
            print("missing QWEN_API_KEY / DASHSCOPE_API_KEY", file=sys.stderr)
            return 2
        openai_model = args.model if args.model and ":" not in args.model else (
            models[0] if models else args.model
        )
        openai_chat = openai_compat_chat

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "dry" if args.dry_run else f"{args.backend}-{args.model.replace('/', '_').replace(':', '_')}"
    out_dir = args.out_dir or (EXPERIMENTS_RUNS / f"{tag}-needle-vrm-{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.jsonl"
    summary: dict = {
        "created_utc": tag,
        "spine": "needle-qwen-behavior-baseline",
        "note": "Qwen is a behavior baseline, not a needle-zh proxy",
        "bank": str(bank.resolve().relative_to(ROOT)),
        "backend": args.backend,
        "model": openai_model or args.model,
        "dry_run": args.dry_run,
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
            tid = str(row.get("toolset_id") or "")
            system = system_for(toolsets[tid], _lang(row))
            user = user_text(row)
            rec = {
                "item_id": row["item_id"],
                "family": row.get("family"),
                "toolset_id": tid,
                "user_prompt": user,
                "raw_text": "",
                "parse_status": None,
                "function_calls": [],
                "error": None,
                "latency": None,
            }
            if args.dry_run:
                rec["parse_status"] = "dry"
                rec["assembled_system_chars"] = len(system)
            else:
                try:
                    t0 = time.perf_counter()
                    latency: dict = {}
                    if args.backend == "ollama":
                        raw, latency = ollama_chat(
                            args.host, args.model, system, user, args.timeout, args.temperature
                        )
                    elif args.backend == "openai":
                        assert openai_chat is not None
                        raw, _usage = openai_chat(
                            openai_base,
                            openai_key,
                            openai_model,
                            system,
                            user,
                            args.timeout,
                            args.temperature,
                        )
                        latency = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
                    else:
                        raw = mlx_chat(system, user, 256, args.temperature)
                        latency = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
                    rec["latency"] = latency
                    rec["raw_text"] = raw
                    calls, status = parse_function_calls(raw)
                    rec["function_calls"] = calls
                    rec["parse_status"] = status
                    key = {"ok": "n_ok_parse", "clarify": "n_clarify", "unparsed": "n_unparsed"}[status]
                    summary[key] += 1
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, RuntimeError) as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["function_calls"] = list(UNPARSED)
                    rec["parse_status"] = "error"
                    rec["latency"] = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
                    summary["n_error"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    summary["run_wall_s"] = round(time.perf_counter() - run_t0, 3)
    (out_dir / "system_used.md").write_text(
        system_for(next(iter(toolsets.values())), _lang(rows[0]) if rows else "zh") + "\n",
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
    }
    if not args.dry_run:
        summary["latency"] = summarize_latency(preds)
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
