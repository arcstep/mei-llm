#!/usr/bin/env python3
"""Head-to-head: Qwen3.5 9B vs mei-1.0-58m on mei-tool-schema-v1.

Only these two systems. Extra axes beyond headline exact_match.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from eval_mei_tool_schema_v1 import score_item
from eval_needle_toolcall_v0 import load_jsonl, norm_calls
from repo_paths import EVAL_BANKS_ROOT, EXPERIMENTS_RUNS, ROOT

BANK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"
MATRIX = EXPERIMENTS_RUNS / "mei-1.0-58m-matrix-cpt300m"
QWEN = MATRIX / "qwen35-9b/predictions.jsonl"
MEI = MATRIX / "mei-1.0-58m-tool-cpt300m-sft10k-v1/predictions.jsonl"
OUT = MATRIX / "compare-qwen9b-mei58m.json"

SLICE_LABEL = {
    "S0": "S0 seen execute",
    "S1": "S1 unseen execute",
    "S2": "S2 mutation",
    "S3": "S3 rename",
    "S4": "S4 crosstalk",
    "S5": "S5 type",
    "S6": "S6 refuse",
}


def pct(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def lat_pack(xs: list[float]) -> dict:
    if not xs:
        return {}
    o = sorted(xs)
    n = len(o)

    def q(p: float) -> float:
        return o[min(n - 1, int(p * (n - 1)))]

    return {
        "n": n,
        "min_ms": round(o[0], 1),
        "p50_ms": round(q(0.5), 1),
        "p90_ms": round(q(0.9), 1),
        "mean_ms": round(statistics.fmean(o), 1),
        "max_ms": round(o[-1], 1),
    }


def name_of(calls: list) -> list[str]:
    return [str(c.get("name") or "") for c in calls]


def analyze_one(rows: list[dict], preds: dict[str, dict]) -> dict:
    scored = []
    lats = []
    parse = defaultdict(int)
    n_gold_ex = n_gold_rf = 0
    n_pred_ex = n_pred_rf = 0
    n_ex_tp = n_ex_fp = n_ex_fn = n_ex_tn = 0
    n_name = n_slot = n_wrong_tool = n_right_tool_bad_slot = 0
    n_pred_nonempty = 0
    n_legal_nonempty = 0
    n_empty_args = 0
    by_slice: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "n_exact": 0,
            "n_legal": 0,
            "n_unsafe": 0,
            "n_gold_execute": 0,
            "n_pred_execute": 0,
            "n_name_match": 0,
            "n_missed_execute": 0,
            "n_false_execute": 0,
        }
    )
    by_family: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_exact": 0, "n_legal": 0, "n_pred_execute": 0})
    by_toolset: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_exact": 0, "n_legal": 0, "n_pred_execute": 0})
    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_exact": 0, "n_pred_execute": 0})

    for row in rows:
        iid = str(row.get("item_id"))
        pr = preds.get(iid) or {}
        raw = str(pr.get("text") or "")
        gold = norm_calls((row.get("gold") or {}).get("function_calls") or row.get("answers"))
        sc = score_item(row, pr.get("function_calls") or [], raw_text=raw)
        scored.append(sc)
        if pr.get("latency_ms") is not None:
            lats.append(float(pr["latency_ms"]))
        parse[str(pr.get("parse_status") or ("ok" if sc["parsed_ok"] else "unparsed"))] += 1

        gold_ex = not sc["gold_empty"]
        pred_ex = not sc["pred_empty"]
        n_gold_ex += int(gold_ex)
        n_gold_rf += int(not gold_ex)
        n_pred_ex += int(pred_ex)
        n_pred_rf += int(not pred_ex)
        if gold_ex and pred_ex:
            n_ex_tp += 1
        elif (not gold_ex) and pred_ex:
            n_ex_fp += 1
        elif gold_ex and (not pred_ex):
            n_ex_fn += 1
        else:
            n_ex_tn += 1

        if pred_ex:
            n_pred_nonempty += 1
            n_legal_nonempty += int(sc["legal"])
            pred_calls = norm_calls(pr.get("function_calls") or [])
            if any(not (c.get("arguments") or {}) for c in pred_calls):
                n_empty_args += 1
            if name_of(pred_calls) == name_of(gold) and gold:
                n_name += 1
                if sc["exact"]:
                    n_slot += 1
                else:
                    n_right_tool_bad_slot += 1
            elif gold:
                n_wrong_tool += 1

        sl = by_slice[str(sc["slice"])]
        sl["n"] += 1
        sl["n_exact"] += int(sc["exact"])
        sl["n_legal"] += int(sc["legal"])
        sl["n_unsafe"] += int(sc["unsafe"])
        sl["n_gold_execute"] += int(gold_ex)
        sl["n_pred_execute"] += int(pred_ex)
        sl["n_missed_execute"] += int(gold_ex and not pred_ex)
        sl["n_false_execute"] += int((not gold_ex) and pred_ex)
        if pred_ex and gold_ex and name_of(norm_calls(pr.get("function_calls") or [])) == name_of(gold):
            sl["n_name_match"] += 1

        fam = str(sc.get("family") or "?")
        by_family[fam]["n"] += 1
        by_family[fam]["n_exact"] += int(sc["exact"])
        by_family[fam]["n_legal"] += int(sc["legal"])
        by_family[fam]["n_pred_execute"] += int(pred_ex)

        ts = str(sc.get("toolset_id") or "?")
        by_toolset[ts]["n"] += 1
        by_toolset[ts]["n_exact"] += int(sc["exact"])
        by_toolset[ts]["n_legal"] += int(sc["legal"])
        by_toolset[ts]["n_pred_execute"] += int(pred_ex)

        kind = str(sc.get("kind") or "?")
        by_kind[kind]["n"] += 1
        by_kind[kind]["n_exact"] += int(sc["exact"])
        by_kind[kind]["n_pred_execute"] += int(pred_ex)

    n = len(scored)
    n_exact = sum(int(s["exact"]) for s in scored)
    n_legal = sum(int(s["legal"]) for s in scored)
    n_unsafe = sum(int(s["unsafe"]) for s in scored)
    n_parsed = sum(int(s["parsed_ok"]) for s in scored)

    def finish_group(table: dict) -> dict:
        out = {}
        for k, v in table.items():
            rec = dict(v)
            rec["exact"] = pct(v["n_exact"], v["n"])
            if "n_legal" in v:
                rec["legal"] = pct(v["n_legal"], v["n"])
            out[k] = rec
        return out

    return {
        "n": n,
        "headline": {
            "exact_match": pct(n_exact, n),
            "n_exact": n_exact,
            "legal_rate": pct(n_legal, n),
            "parsed_ok": pct(n_parsed, n),
            "unsafe_rate": pct(n_unsafe, n),
        },
        "decision": {
            "n_gold_execute": n_gold_ex,
            "n_gold_refuse": n_gold_rf,
            "n_pred_execute": n_pred_ex,
            "n_pred_refuse": n_pred_rf,
            "execute_recall": pct(n_ex_tp, n_gold_ex),
            "execute_precision": pct(n_ex_tp, n_pred_ex),
            "refuse_recall": pct(n_ex_tn, n_gold_rf),
            "refuse_precision": pct(n_ex_tn, n_pred_rf),
            "missed_execute": n_ex_fn,
            "missed_execute_rate": pct(n_ex_fn, n_gold_ex),
            "false_execute": n_ex_fp,
            "false_execute_rate": pct(n_ex_fp, n),
            "confusion": {
                "tp_execute": n_ex_tp,
                "fp_execute": n_ex_fp,
                "fn_execute": n_ex_fn,
                "tn_refuse": n_ex_tn,
            },
        },
        "routing": {
            "n_pred_execute": n_pred_nonempty,
            "legal_among_execute": pct(n_legal_nonempty, n_pred_nonempty),
            "n_empty_arguments": n_empty_args,
            "name_match_on_gold_execute": pct(n_name, n_gold_ex),
            "slot_exact_on_gold_execute": pct(n_slot, n_gold_ex),
            "n_name_match": n_name,
            "n_slot_exact": n_slot,
            "n_right_tool_bad_slot": n_right_tool_bad_slot,
            "n_wrong_tool_when_gold_execute": n_wrong_tool,
        },
        "parse_status": dict(parse),
        "latency": lat_pack(lats),
        "by_slice": finish_group(by_slice),
        "by_family": finish_group(by_family),
        "by_toolset": finish_group(by_toolset),
        "by_kind": finish_group(by_kind),
        "scored": scored,
    }


def pairwise(rows, qwen_preds, mei_preds, q_scored, m_scored) -> dict:
    qmap = {s["item_id"]: s for s in q_scored}
    mmap = {s["item_id"]: s for s in m_scored}
    same_dec = same_calls = both_ok = both_bad = mei_only = qwen_only = 0
    disagreements = []
    for row in rows:
        iid = str(row.get("item_id"))
        qs, ms = qmap[iid], mmap[iid]
        q_empty, m_empty = qs["pred_empty"], ms["pred_empty"]
        same_dec += int(q_empty == m_empty)
        qp = norm_calls((qwen_preds.get(iid) or {}).get("function_calls") or [])
        mp = norm_calls((mei_preds.get(iid) or {}).get("function_calls") or [])
        same_calls += int(qp == mp)
        both_ok += int(qs["exact"] and ms["exact"])
        both_bad += int((not qs["exact"]) and (not ms["exact"]))
        mei_only += int(ms["exact"] and not qs["exact"])
        qwen_only += int(qs["exact"] and not ms["exact"])
        if qs["exact"] != ms["exact"] or q_empty != m_empty:
            disagreements.append(
                {
                    "item_id": iid,
                    "slice": qs["slice"],
                    "family": qs.get("family"),
                    "kind": qs.get("kind"),
                    "gold_refuse": qs["gold_empty"],
                    "qwen_exact": qs["exact"],
                    "mei_exact": ms["exact"],
                    "qwen_execute": not q_empty,
                    "mei_execute": not m_empty,
                    "qwen_unsafe": qs["unsafe"],
                    "mei_unsafe": ms["unsafe"],
                    "qwen_legal": qs["legal"],
                    "mei_legal": ms["legal"],
                }
            )
    n = len(rows)
    return {
        "n": n,
        "same_decision": pct(same_dec, n),
        "n_same_decision": same_dec,
        "same_calls": pct(same_calls, n),
        "n_same_calls": same_calls,
        "both_exact": both_ok,
        "both_wrong": both_bad,
        "mei_only_exact": mei_only,
        "qwen_only_exact": qwen_only,
        "disagreements": disagreements,
    }


def rate_table(q: dict, m: dict, keys: list[tuple[str, str]]) -> list[dict]:
    out = []
    for path, label in keys:
        def grab(blob, p):
            cur = blob
            for part in p.split("."):
                cur = cur[part]
            return cur

        qv, mv = grab(q, path), grab(m, path)
        out.append({"metric": label, "qwen9b": qv, "mei58m": mv, "delta": round(mv - qv, 4) if isinstance(qv, (int, float)) and isinstance(mv, (int, float)) else None})
    return out


def main() -> int:
    rows = load_jsonl(BANK)
    qwen_preds = {str(r.get("item_id")): r for r in load_jsonl(QWEN)}
    mei_preds = {str(r.get("item_id")): r for r in load_jsonl(MEI)}
    q = analyze_one(rows, qwen_preds)
    m = analyze_one(rows, mei_preds)
    pair = pairwise(rows, qwen_preds, mei_preds, q["scored"], m["scored"])
    q_pub = {k: v for k, v in q.items() if k != "scored"}
    m_pub = {k: v for k, v in m.items() if k != "scored"}
    report = {
        "protocol": "mei-tool-schema-v1",
        "n": len(rows),
        "systems": {
            "qwen35-9b": {
                "id": "qwen3.5:9b-mlx",
                "backend": "ollama",
                "role": "publish_bar",
                **q_pub,
            },
            "mei-1.0-58m": {
                "id": "mei-1.0-58m-tool-cpt300m-sft10k-v1",
                "role": "candidate",
                **m_pub,
            },
        },
        "pairwise": {k: v for k, v in pair.items() if k != "disagreements"},
        "disagreements": pair["disagreements"],
        "slice_labels": SLICE_LABEL,
        "latency_note": "9B is Ollama full-answer RTT; 58m is greedy_constrained student wall clock. Not the same clock.",
        "decision_note": "execute = nonempty function_calls after parse. Independent of slot exact.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "n": len(rows), "pairwise": report["pairwise"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
