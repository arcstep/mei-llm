#!/usr/bin/env python3
"""sft-v2-eval-lock-v2 layered hard gates on a disk 51M quantized product package.

Thresholds must already be pre-registered. This script never edits them after seeing scores.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mlx.core as mx

from identity_51m import (
    JOBS_DIR,
    Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_DIR,
    ROOT,
    SFT_QAT_PACKAGE_DIR,
    fail,
    load_json,
    write_json,
)
from tokenizer import ASSISTANT_PREFIX, TURN_END

LOCK_DIR = ROOT / "notebook/evaluation/banks/sft-v2-eval-lock-v2"
THRESHOLDS = JOBS_DIR / "sft-v2-51m-thresholds-preregister.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (text or "").lower()) if t}


def lexical_rank(query: str, catalog: list[dict], k: int = 5) -> list[dict]:
    q = toks(query)
    scored = []
    for tool in catalog:
        blob = " ".join(
            [
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
            ]
        )
        tb = toks(blob)
        union = q | tb
        score = (len(q & tb) / len(union)) if union else 0.0
        scored.append((score, tool))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [t for _, t in scored[:k]]


def calls_equal(a, b) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    left, right = a[0], b[0]
    if str(left.get("name")) != str(right.get("name")):
        return False
    return json.dumps(left.get("arguments") or {}, sort_keys=True, ensure_ascii=False) == json.dumps(
        right.get("arguments") or {}, sort_keys=True, ensure_ascii=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--limit-retrieval", type=int, default=1000)
    parser.add_argument("--limit-fullcall", type=int, default=1200)
    args = parser.parse_args()
    th = load_json(THRESHOLDS)
    if not th.get("registered_before_eval"):
        return fail("missing pre-registered 51M lock-v2 thresholds")
    layers = th["layers"]
    pkg_dir = args.package_dir
    if pkg_dir is None:
        if (SFT_QAT_PACKAGE_DIR / "weights.q4").is_file():
            pkg_dir = SFT_QAT_PACKAGE_DIR
        elif (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file():
            pkg_dir = QAT_Q4_PACKAGE_DIR
        else:
            pkg_dir = Q4_PACKAGE_DIR

    import sys

    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from mei_sdk.package import load_package
    from mei_sdk.protocol import render_request
    from mei_sdk.runtime_51m import apply_confidence_gate, load_51m_runtime, validate_call

    universe = {t["name"]: t for t in load_json(LOCK_DIR / "tool-universe-v1.json").get("tools") or []}
    pkg = load_package(pkg_dir)
    runtime, loaded = load_51m_runtime(pkg)
    tok = runtime.tokenizer
    tool_vecs: dict[str, mx.array] = {}

    def tool_vec(tool: dict):
        name = str(tool.get("name") or "")
        if name not in tool_vecs:
            blob = json.dumps(
                {"name": name, "description": tool.get("description") or "", "parameters": tool.get("parameters") or {}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tool_vecs[name] = runtime.embed_text(blob)
        return tool_vecs[name]

    def learned_top(query: str, catalog: list[dict], k: int = 5) -> list[dict]:
        if len(catalog) <= k:
            return list(catalog)
        q = runtime.embed_text(query)
        scored = []
        for tool in catalog:
            sim = float(mx.sum(q * tool_vec(tool)).item())
            scored.append((sim, tool))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [t for _, t in scored[:k]]

    def catalog_of(names: list[str]) -> list[dict]:
        out = []
        for name in names:
            if name in universe:
                out.append(universe[name])
        return out

    def generate(query: str, tools: list[dict], *, facts: str = "", max_new: int = 64) -> dict:
        rendered = render_request({"query": query, "system_facts": facts}, tools)
        ids = tok.encode(rendered["prompt"] + ASSISTANT_PREFIX, add_bos=True, add_eos=False)
        decoded = runtime.greedy(ids, tools=tools, max_new=max_new, decode_mode="constrained")
        text = decoded.get("text") or ""
        validated = validate_call(text, tools=tools, query=query, system_facts=facts)
        gated = apply_confidence_gate(validated, decoded.get("confidence"))
        return {"text": text, "validated": validated, "gated": gated, "confidence": decoded.get("confidence")}

    ret_rows = load_jsonl(LOCK_DIR / "eval-retrieval-test.jsonl")[: args.limit_retrieval]
    learned_hits = 0
    lexical_hits = 0
    for row in ret_rows:
        catalog = catalog_of(row.get("catalog_tool_names") or [])
        gold = str(row.get("gold_tool") or "")
        learned = [str(t.get("name")) for t in learned_top(str(row.get("query") or ""), catalog, 5)]
        lexical = [str(t.get("name")) for t in lexical_rank(str(row.get("query") or ""), catalog, 5)]
        learned_hits += int(gold in learned)
        lexical_hits += int(gold in lexical)
    n_ret = max(len(ret_rows), 1)
    recall_learned = learned_hits / n_ret
    recall_lex = lexical_hits / n_ret

    fc_rows = load_jsonl(LOCK_DIR / "eval-fullcall-test.jsonl")[: args.limit_fullcall]
    oracle_exact = 0
    oracle_er = 0
    learned_exact = 0
    unsupported = 0
    unprov = 0
    conf_pairs = []
    for i, row in enumerate(fc_rows):
        query = str(row.get("query") or "")
        facts = str(row.get("system_facts") or "")
        gold = row.get("answers") or []
        kind = str(row.get("kind") or "execute")
        oracle_tools = row.get("oracle_top5") or catalog_of((row.get("catalog_tool_names") or [])[:5])
        o = generate(query, oracle_tools, facts=facts)
        parsed = o["validated"].get("function_calls") or []
        if kind == "execute":
            hit = calls_equal(parsed, gold) and not o["validated"].get("refuse")
        else:
            hit = bool(o["validated"].get("refuse")) or parsed == []
        oracle_exact += int(hit)
        want_exec = kind == "execute"
        got_exec = o["gated"].get("execution") == "execute"
        oracle_er += int(want_exec == got_exec or (not want_exec and o["gated"].get("refuse")))
        unsupported += int(o["validated"].get("unsupported_accepted") or 0)
        unprov += int(o["validated"].get("unprovenanced_argument_accepted") or 0)
        catalog = catalog_of(row.get("catalog_tool_names") or [])
        learned_tools = learned_top(query, catalog, 5) if catalog else oracle_tools
        l = generate(query, learned_tools, facts=facts)
        lparsed = l["validated"].get("function_calls") or []
        if kind == "execute":
            lhit = calls_equal(lparsed, gold) and not l["validated"].get("refuse")
        else:
            lhit = bool(l["validated"].get("refuse")) or lparsed == []
        learned_exact += int(lhit)
        if l.get("confidence") is not None:
            conf_pairs.append((float(l["confidence"]), int(lhit)))
        if (i + 1) % 50 == 0:
            print(json.dumps({"fullcall": i + 1, "oracle_exact_so_far": oracle_exact / (i + 1)}), flush=True)

    n_fc = max(len(fc_rows), 1)
    oracle_rate = oracle_exact / n_fc
    learned_rate = learned_exact / n_fc
    ece = 0.0
    brier = 0.0
    if conf_pairs:
        brier = sum((p - y) ** 2 for p, y in conf_pairs) / len(conf_pairs)
        bins = 5
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            bucket = [p for p in conf_pairs if lo <= p[0] < hi or (b == bins - 1 and p[0] == 1)]
            if not bucket:
                continue
            conf = sum(p[0] for p in bucket) / len(bucket)
            freq = sum(p[1] for p in bucket) / len(bucket)
            ece += (len(bucket) / len(conf_pairs)) * abs(conf - freq)

    gates = {
        "recall_at_5_learned": {
            "value": recall_learned,
            "floor": layers["recall_at_5_learned_min"],
            "ok": recall_learned >= float(layers["recall_at_5_learned_min"]),
        },
        "recall_beats_lexical": {
            "learned": recall_learned,
            "lexical": recall_lex,
            "ok": (not layers.get("recall_at_5_must_exceed_lexical")) or recall_learned > recall_lex,
        },
        "oracle_top5_fullcall_exact": {
            "value": oracle_rate,
            "floor": layers["oracle_top5_fullcall_exact_min"],
            "ok": oracle_rate >= float(layers["oracle_top5_fullcall_exact_min"]),
        },
        "oracle_execute_refuse": {
            "value": oracle_er / n_fc,
            "floor": layers["oracle_top5_execute_refuse_ok_min"],
            "ok": (oracle_er / n_fc) >= float(layers["oracle_top5_execute_refuse_ok_min"]),
        },
        "learned_top5_e2e_exact": {
            "value": learned_rate,
            "floor": layers["learned_top5_e2e_exact_min"],
            "ok": learned_rate >= float(layers["learned_top5_e2e_exact_min"]),
        },
        "unsupported_accepted": {
            "value": unsupported,
            "max": layers["unsupported_accepted_max"],
            "ok": unsupported <= int(layers["unsupported_accepted_max"]),
        },
        "unprovenanced_argument_accepted": {
            "value": unprov,
            "max": layers["unprovenanced_argument_accepted_max"],
            "ok": unprov <= int(layers["unprovenanced_argument_accepted_max"]),
        },
    }
    hard_ok = all(row["ok"] for row in gates.values())
    report = {
        "kind": "toolcall-qat-lock-v2",
        "lock": "sft-v2-eval-lock-v2",
        "package_dir": str(pkg_dir),
        "thresholds": str(THRESHOLDS.relative_to(ROOT)),
        "n_retrieval": len(ret_rows),
        "n_fullcall": len(fc_rows),
        "universe_n": len(universe),
        "recall_at_5_learned": recall_learned,
        "recall_at_5_lexical": recall_lex,
        "oracle_top5_fullcall_exact": oracle_rate,
        "learned_top5_e2e_exact": learned_rate,
        "unsupported_accepted": unsupported,
        "unprovenanced_argument_accepted": unprov,
        "confidence_ece": ece,
        "confidence_brier": brier,
        "layers": gates,
        "hard_ok": hard_ok,
        "toy_scores_rejected": True,
        "n_loaded": loaded.get("n_loaded"),
        "qat_mandatory": True,
        "not_a_claim": "Lock-v2 scores are not a CURRENT freeze.",
    }
    path = args.jobs_dir / "toolcall-qat-lock-v2.json"
    blocked = write_json(path, report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if hard_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
