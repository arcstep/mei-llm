#!/usr/bin/env python3
"""Validate a needle-zh home SFT pack against schema-program gold and mixture gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from repo_paths import BANK_NEEDLE_VRM_AGENT, EVAL_BANKS_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import format_sft_user_text, token_jaccard  # noqa: E402
from needle_home_sft_lib import (  # noqa: E402
    ASK_RE,
    EVAL_RE,
    FAKE_VARIANT_RE,
    MIXTURE_PATH,
    PII_RE,
    TOOLSET_ID,
    TOOL_NAME_RE,
    assistant_truncated,
    gold_from_intent,
    load_jsonl,
    load_mixture,
    load_toolset,
    query_banned,
    seen_key,
    sha256_file,
    tools_of,
)
from tokenizer import ZhTokenizerV1  # noqa: E402

REQUIRED = (
    "sample_id",
    "split",
    "lang",
    "family",
    "toolset_id",
    "query",
    "answers",
    "act",
    "confidence_label",
    "intent_id",
)
REFUSE = {"missing", "scene_conflict", "illegal_pair", "offtopic"}
EXECUTE_FAM = {"gesture", "home", "order", "paraphrase"}


def _eq_calls(a, b) -> bool:
    return json.dumps(a or [], ensure_ascii=False, sort_keys=True) == json.dumps(
        b or [], ensure_ascii=False, sort_keys=True
    )


def validate(rows: list[dict], mixture: dict, *, seq_len: int) -> dict:
    errors: list[str] = []
    catalog = tools_of(load_toolset(TOOLSET_ID))
    tok = ZhTokenizerV1()
    seen_ids: set[str] = set()
    seen_qs: set[str] = set()
    tools_hit: set[str] = set()
    fam = Counter()
    n_empty = 0
    n_exec = 0
    for row in rows:
        sid = str(row.get("sample_id") or "missing")
        for key in REQUIRED:
            if key not in row:
                errors.append(f"{sid}: missing {key}")
        if sid in seen_ids:
            errors.append(f"{sid}: duplicate sample_id")
        seen_ids.add(sid)
        if row.get("split") not in {"train", "valid"}:
            errors.append(f"{sid}: bad split")
        if row.get("act") is not None:
            errors.append(f"{sid}: act must be null")
        if row.get("toolset_id") != TOOLSET_ID:
            errors.append(f"{sid}: toolset_id mismatch")
        query = str(row.get("query") or "")
        banned = query_banned(query)
        if banned:
            errors.append(f"{sid}: query banned ({banned})")
        blob = f"{query}\n{row.get('scene') or ''}"
        if ASK_RE.search(blob) or EVAL_RE.search(blob) or PII_RE.search(blob):
            errors.append(f"{sid}: ASK/EVAL/PII in query/scene")
        if FAKE_VARIANT_RE.search(query):
            errors.append(f"{sid}: fake variant wording")
        if TOOL_NAME_RE.search(query):
            errors.append(f"{sid}: tool name leaked into query")
        sk = seen_key(row)
        if sk in seen_qs:
            errors.append(f"{sid}: duplicate query+scene")
        seen_qs.add(sk)
        family = str(row.get("family") or "")
        fam[family] += 1
        if family == "scene_conflict" and not str(row.get("scene") or "").strip():
            errors.append(f"{sid}: scene_conflict missing scene")
        user_text = format_sft_user_text(row)
        if str(row.get("scene") or "").strip():
            if not user_text.startswith("场景："):
                errors.append(f"{sid}: scene not encoded in user text")
            if str(row.get("query") or "") not in user_text:
                errors.append(f"{sid}: query missing from encoded user text")
        elif not user_text.startswith("用户："):
            errors.append(f"{sid}: missing 用户： prefix")
        intent = {
            "kind": row.get("kind") or ("execute" if row.get("answers") else family),
            "name": row.get("gold_name"),
            "args": row.get("gold_args") or {},
            "family": family,
            "scene": row.get("scene"),
        }
        recomputed = gold_from_intent(intent, mixture)
        if not _eq_calls(recomputed, row.get("answers")):
            errors.append(f"{sid}: gold != schema-program")
        answers = row.get("answers")
        if not isinstance(answers, list):
            errors.append(f"{sid}: answers not list")
            continue
        empty = not answers
        if empty:
            n_empty += 1
        else:
            n_exec += 1
        want_conf = 0 if empty else 1
        conf = row.get("confidence_label")
        if conf is None or int(conf) != want_conf:
            errors.append(f"{sid}: confidence_label mismatch")
        if family in REFUSE and answers:
            errors.append(f"{sid}: refuse family must be []")
        if family in EXECUTE_FAM and not answers:
            errors.append(f"{sid}: execute family must have a call")
        for call in answers:
            if not isinstance(call, dict):
                errors.append(f"{sid}: call not object")
                continue
            name = str(call.get("name") or "")
            args = call.get("arguments")
            if name not in catalog:
                errors.append(f"{sid}: unknown tool {name}")
                continue
            tools_hit.add(name)
            spec = catalog[name]
            params = spec.get("parameters") or {}
            required = list(params.get("required") or [])
            props = params.get("properties") or {}
            if not isinstance(args, dict):
                errors.append(f"{sid}: arguments not object")
                continue
            for req in required:
                if req not in args:
                    errors.append(f"{sid}: missing arg {req}")
            for key, val in args.items():
                prop = props.get(key) or {}
                enum = prop.get("enum")
                if enum is not None and val not in enum:
                    errors.append(f"{sid}: {key}={val!r} not in enum")
                if isinstance(val, str) and val.strip() == query.strip():
                    errors.append(f"{sid}: slot equals full query")
        if assistant_truncated(tok, row, seq_len):
            errors.append(f"{sid}: assistant gold truncated at seq_len={seq_len}")
    n = max(len(rows), 1)
    empty_frac = n_empty / n
    exec_frac = n_exec / n
    offtopic_frac = fam.get("offtopic", 0) / n
    tlo, thi = mixture["targets"]["execute_frac"]
    rlo, rhi = mixture["targets"]["refuse_frac"]
    olo, ohi = mixture["targets"]["offtopic_frac"]
    if not (tlo - 0.03 <= exec_frac <= thi + 0.03):
        errors.append(f"execute_frac {exec_frac:.3f} outside {tlo}-{thi}")
    if not (rlo - 0.03 <= empty_frac <= rhi + 0.03):
        errors.append(f"refuse_frac {empty_frac:.3f} outside {rlo}-{rhi}")
    if not (olo - 0.03 <= offtopic_frac <= ohi + 0.03):
        errors.append(f"offtopic_frac {offtopic_frac:.3f} outside {olo}-{ohi}")
    missing_tools = sorted(set(mixture["required_tools"]) - tools_hit)
    if missing_tools:
        errors.append(f"missing tools: {missing_tools}")
    valid_frac = sum(1 for r in rows if r.get("split") == "valid") / n
    vf = float(mixture.get("valid_frac") or 0.09)
    if abs(valid_frac - vf) > 0.04:
        errors.append(f"valid_frac {valid_frac:.3f} far from {vf}")
    tpl = Counter(str(r.get("template_id") or "") for r in rows)
    if tpl:
        top_n, top_c = tpl.most_common(1)[0]
        if top_c / n > 0.20 and top_n:
            errors.append(f"template {top_n} share {top_c / n:.3f} > 0.20")
    tok_ids = [
        (str(r.get("split")), tok.encode(format_sft_user_text(r)))
        for r in rows
    ]
    train_ids = [ids for split, ids in tok_ids if split == "train" and len(ids) >= 2]
    valid_ids = [ids for split, ids in tok_ids if split == "valid" and len(ids) >= 2]
    cross = 0
    for v in valid_ids:
        if any(token_jaccard(v, t) >= 0.9 for t in train_ids):
            cross += 1
            if cross >= 3:
                break
    if cross:
        errors.append(f"train/valid token Jaccard>=0.9 hits={cross}")
    eval_rows = load_jsonl(BANK_NEEDLE_VRM_AGENT)
    eval_queries = {str(r.get("query") or "").strip() for r in eval_rows}
    eval_ids = [tok.encode(q) for q in eval_queries if q]
    eval_hits = 0
    for row in rows:
        q = str(row.get("query") or "").strip()
        if q in eval_queries:
            eval_hits += 1
            continue
        ids = tok.encode(q)
        if len(ids) >= 2 and any(token_jaccard(ids, e) >= 0.9 for e in eval_ids if len(e) >= 2):
            eval_hits += 1
    if eval_hits:
        errors.append(f"eval near-dup hits={eval_hits}")
    return {
        "ok": not errors,
        "n": len(rows),
        "empty_answers": n_empty,
        "execute_frac": round(exec_frac, 4),
        "refuse_frac": round(empty_frac, 4),
        "offtopic_frac": round(offtopic_frac, 4),
        "valid_frac": round(valid_frac, 4),
        "family": dict(fam),
        "tools_hit": sorted(tools_hit),
        "missing_tools": missing_tools,
        "errors": errors[:80],
        "n_errors": len(errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, default=MIXTURE_PATH)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()
    if not args.pack.is_file():
        print(f"missing {args.pack}", file=sys.stderr)
        return 1
    mixture = load_mixture(args.mixture)
    rows = load_jsonl(args.pack)
    lock = EVAL_BANKS_ROOT / "needle-vrm-agent-v0" / "holdout-v1.lock.json"
    freeze = json.loads(lock.read_text(encoding="utf-8")) if lock.is_file() else {}
    report = validate(rows, mixture, seq_len=int(mixture.get("seq_len_train") or 256))
    report["pack"] = str(args.pack.relative_to(ROOT)) if str(args.pack).startswith(str(ROOT)) else str(args.pack)
    report["sha256"] = sha256_file(args.pack)
    report["eval_sha256"] = freeze.get("sha256")
    report["eval_lock_ok"] = freeze.get("sha256") == mixture.get("eval_sha256")
    if not report["eval_lock_ok"]:
        report["ok"] = False
        report["errors"] = list(report["errors"]) + ["eval lock hash mismatch"]
        report["n_errors"] = len(report["errors"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write_report:
        out = args.pack.with_name(args.pack.name.replace(".jsonl", ".validate.json"))
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
