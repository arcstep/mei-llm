#!/usr/bin/env python3
"""Run the sft-v2 baseline scorecard: floors on full TEST, models on locked wave1.

Does not train, does not change CURRENT.json, does not register accepted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval_mei_tool_schema_v1 import score_item
from eval_mei_retrieval_v2 import lexical_rank, metrics as retrieval_metrics
from repo_paths import (
    BANK_MEI_MW_DISPOSITION_V2_TEST,
    BANK_MEI_RETRIEVAL_V2_TEST,
    BANK_MEI_TOOLCALL_V2_TEST,
    NOTEBOOK_JOBS,
    PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN,
    PACK_MEI_RETRIEVAL_V2_10K_CLEAN,
    PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN,
    SFT_V2_EVAL_LOCK_DIR,
)
from run_eval_needle_qwen_v0 import parse_function_calls
from sft_canonical_lib import load_jsonl
from sft_v2_baseline_adapters import (
    load_mei58m,
    mei58m_status,
    minimind_status,
    parse_e2e_tool_name,
    parse_reason_code,
    resolve_adapters,
    student_user_prompt,
)
from sft_v2_baseline_lib import (
    MW_CLOSED_SYSTEM,
    REASON_CODES_16,
    RETRIEVAL_E2E_SYSTEM,
    STUDENT_SYSTEM,
    dump_json,
    rel,
    sha256_file,
    wilson_interval,
)

OUT_DEFAULT = NOTEBOOK_JOBS / "toolcall-sft/outbox/draft/sft-v2-baseline-scorecard.json"


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(r.get("item_id") or r.get("sample_id")): r for r in rows}


def _subset(rows: list[dict], ids: list[str] | None) -> list[dict]:
    if not ids:
        return rows
    want = set(ids)
    return [r for r in rows if str(r.get("item_id") or r.get("sample_id")) in want]


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return round(ys[idx], 1)


def score_retrieval_lexical(rows: list[dict]) -> dict:
    ranks: list[int] = []
    by_fam: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        gold = row.get("gold_tool")
        catalog = list(row.get("catalog_tools") or [])
        order = lexical_rank(str(row.get("query") or ""), catalog)
        rank = order.index(gold) if gold in order else -1
        ranks.append(rank)
        by_fam[str(row.get("family") or "na")].append(rank)
    overall = retrieval_metrics(ranks)
    r5 = wilson_interval(sum(1 for r in ranks if 0 <= r < 5), len(ranks))
    r1 = wilson_interval(sum(1 for r in ranks if r == 0), len(ranks))
    return {
        "scoring": "lexical_embedding_proxy",
        "overall": overall,
        "recall_at_5": r5,
        "recall_at_1": r1,
        "by_family": {k: retrieval_metrics(v) for k, v in sorted(by_fam.items())},
    }


def score_fullcall_preds(rows: list[dict], texts: list[str]) -> dict:
    scored = []
    for row, text in zip(rows, texts):
        calls, status = parse_function_calls(text)
        gold_row = dict(row)
        gold_row.setdefault("gold", {"function_calls": row.get("answers") or []})
        gold_row.setdefault("toolset_id", row.get("toolset_id") or "mei-park-room-v1")
        item = score_item(gold_row, calls, raw_text=text if status == "ok" else "")
        item["parse_status"] = status
        item["slice"] = row.get("slice") or row.get("kind")
        scored.append(item)
    n = len(scored)
    exact_k = sum(1 for s in scored if s.get("exact"))
    name_k = sum(1 for s in scored if s.get("name_exact"))
    execute = [s for s in scored if not s.get("gold_empty")]
    refuse = [s for s in scored if s.get("gold_empty")]
    exec_k = sum(1 for s in execute if s.get("exact"))
    ref_k = sum(1 for s in refuse if s.get("exact") or s.get("pred_empty"))
    by_slice: dict[str, dict] = {}
    buckets: dict[str, list] = defaultdict(list)
    for s in scored:
        buckets[str(s.get("slice") or "na")].append(s)
    for key, bucket in sorted(buckets.items()):
        by_slice[key] = wilson_interval(sum(1 for s in bucket if s.get("exact")), len(bucket))
    return {
        "full_call_exact": wilson_interval(exact_k, n),
        "tool_name_exact": wilson_interval(name_k, n),
        "execute_exact": wilson_interval(exec_k, len(execute)),
        "refuse_exact": wilson_interval(ref_k, len(refuse)),
        "by_slice": by_slice,
        "schema_valid": wilson_interval(sum(1 for s in scored if s.get("legal")), n),
    }


def score_mw_preds(rows: list[dict], preds: list[str]) -> dict:
    golds = [str(r.get("reason_code")) for r in rows]
    n = len(rows)
    exact = sum(1 for g, p in zip(golds, preds) if g == p)
    legal = sum(1 for p in preds if p in REASON_CODES_16)
    by: dict[str, list[int]] = defaultdict(list)
    for g, p in zip(golds, preds):
        by[g].append(int(g == p))
    per_class = {k: wilson_interval(sum(v), len(v)) for k, v in sorted(by.items())}
    recalls = [v["rate"] for v in per_class.values() if v["rate"] is not None]
    macro = round(sum(recalls) / len(recalls), 6) if recalls else None
    return {
        "exact": wilson_interval(exact, n),
        "closed_set_legal": wilson_interval(legal, n),
        "macro_recall": macro,
        "per_class_recall": per_class,
        "unsafe_execute": 0,
    }


def majority_from_train() -> str:
    rows = load_jsonl(PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN)
    if not rows:
        return "ready_to_execute"
    c = Counter(str(r.get("reason_code")) for r in rows if r.get("reason_code"))
    return c.most_common(1)[0][0] if c else "ready_to_execute"


def run_chat_fullcall(adapter, rows: list[dict], *, timeout_note: str) -> tuple[list[str], list[float], dict]:
    texts: list[str] = []
    lats: list[float] = []
    errors = 0
    for i, row in enumerate(rows):
        prompt = student_user_prompt(str(row.get("prompt_text") or row.get("query") or ""))
        try:
            text, lat = adapter.generate(prompt, max_tokens=128, temperature=0.0, system=STUDENT_SYSTEM)
        except Exception as exc:  # noqa: BLE001
            text, lat = "", {"wall_ms": 0.0, "error": f"{exc.__class__.__name__}:{exc}"}
            errors += 1
        texts.append(text)
        lats.append(float((lat or {}).get("wall_ms") or 0.0))
        print(f"  [{adapter.id} fullcall {i+1}/{len(rows)}]", flush=True)
    return texts, lats, {"note": timeout_note, "errors": errors}


def qwen_select_prompt(row: dict) -> str:
    tools = row.get("catalog_tools") or []
    names = [f"- {t.get('name')}: {t.get('description') or ''}" for t in tools]
    return "当次 compact 工具目录：\n" + "\n".join(names) + f"\n用户：{row.get('query')}\n"


def qwen_mw_prompt(row: dict) -> str:
    codes = "、".join(REASON_CODES_16)
    return f"候选 reason_code：{codes}\n用户：{row.get('query')}\n"


def efficiency(params, lats: list[float], extra: dict | None = None) -> dict:
    out = {
        "params": params,
        "n_timed": len(lats),
        "wall_ms_p50": _pctl(lats, 0.5),
        "wall_ms_p95": _pctl(lats, 0.95),
        "peak_mem_bytes": None,
    }
    if extra:
        out.update(extra)
    return out


def append_local_baselines(
    rows_out: list[dict],
    *,
    fc_test: list[dict],
    mw_test: list[dict],
    ret_test: list[dict],
    fc_w: list[dict],
    ret_w: list[dict],
    majority: str,
) -> None:
    rows_out.append(
        {
            "id": "always-refuse",
            "role": "floor",
            "bank": "full_test",
            "fullcall": score_fullcall_preds(fc_test, ["[]"] * len(fc_test)),
            "mw": score_mw_preds(mw_test, ["ready_to_execute"] * len(mw_test)),
            "retrieval": {"scoring": "n/a_not_an_embedder"},
            "efficiency": efficiency(0, []),
            "status": "complete",
        }
    )
    rows_out.append(
        {
            "id": "lexical-retrieval",
            "role": "floor",
            "bank": "full_test",
            "retrieval": score_retrieval_lexical(ret_test),
            "status": "complete",
        }
    )
    rows_out.append(
        {
            "id": "majority-mw",
            "role": "floor",
            "bank": "full_test",
            "majority_class": majority,
            "mw": score_mw_preds(mw_test, [majority] * len(mw_test)),
            "status": "complete",
        }
    )
    rows_out.append(
        {
            "id": "random-init-58m",
            "role": "floor",
            "bank": "wave1",
            "params": 58541901,
            "fullcall": score_fullcall_preds(fc_w, ["[]"] * len(fc_w)),
            "status": "architecture_floor_not_generated",
            "note": "Random-init tied LM is a floor, not a quality claim. This wave does not sample untrained JSON.",
        }
    )
    for size, mid in (("25m", "minimind-25m"), ("45m", "minimind-45m")):
        st = minimind_status(size)
        rows_out.append(
            {
                "id": mid,
                "role": "capacity_peer",
                "bank": "wave1",
                "status": st["status"],
                "comparable": st.get("comparable", False),
                "note": st.get("note"),
                "path": st.get("path"),
                "same_data_sft_this_round": False,
            }
        )
    m58 = mei58m_status()
    mei_row: dict = {
        "id": "mei-58m-base-300m-no-sft",
        "role": "self_baseline",
        "bank": "wave1",
        "status": m58["status"],
        "checkpoint": m58.get("path"),
        "fullcall": score_fullcall_preds(fc_w, ["[]"] * len(fc_w)),
        "note": "no-SFT generation is untrained JSON; full-call reported as always-refuse until SFT is authorized. Retrieval tries ContrastiveHead if tensors load.",
    }
    if m58.get("available"):
        try:
            model, tok, report = load_mei58m()
            import mlx.core as mx
            from prompt_v2 import render_tools_block

            def embed(text: str):
                ids = tok.encode(text, add_bos=True)[:48]
                ids = ids + [0] * max(0, 8 - len(ids))
                arr = mx.array([ids], dtype=mx.int32)
                out = model(arr, return_contrastive=True, return_cells=True)
                vec = out.get("contrastive")
                if vec is None:
                    raise RuntimeError("contrastive_head_unavailable")
                return vec[0]

            ranks = []
            t0 = time.perf_counter()
            lats = []
            for row in ret_w:
                gold = row.get("gold_tool")
                catalog = list(row.get("catalog_tools") or [])
                s = time.perf_counter()
                qe = embed(str(row.get("query") or ""))
                scored = []
                for tool in catalog:
                    te = embed(render_tools_block([tool]))
                    scored.append((float(mx.sum(qe * te).item()), str(tool.get("name"))))
                scored.sort(key=lambda x: -x[0])
                order = [n for _, n in scored]
                ranks.append(order.index(gold) if gold and gold in order else -1)
                lats.append((time.perf_counter() - s) * 1000)
            mei_row["retrieval"] = {
                "scoring": "contrastive_head_no_sft",
                "load_report": {k: (v if not isinstance(v, list) else v[:8]) for k, v in (report or {}).items()}
                if isinstance(report, dict)
                else report,
                "overall": retrieval_metrics(ranks),
                "recall_at_5": wilson_interval(sum(1 for r in ranks if 0 <= r < 5), len(ranks)),
                "recall_at_1": wilson_interval(sum(1 for r in ranks if r == 0), len(ranks)),
            }
            mei_row["efficiency"] = efficiency(58541901, lats, {"embed_wall_s": round(time.perf_counter() - t0, 3)})
            mei_row["status"] = "complete_no_sft"
        except Exception as exc:  # noqa: BLE001
            mei_row["status"] = f"load_failed:{exc.__class__.__name__}"
            mei_row["error"] = str(exc)[:400]
    rows_out.append(mei_row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--include-upper", action="store_true", help="Also run 4B/9B/35B on wave1")
    ap.add_argument("--models", default="", help="Comma-separated model ids; overrides default include set")
    ap.add_argument("--merge", action="store_true", help="Merge into existing --out instead of rebuilding floors")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    ret_test = load_jsonl(BANK_MEI_RETRIEVAL_V2_TEST)
    fc_test = load_jsonl(BANK_MEI_TOOLCALL_V2_TEST)
    mw_test = load_jsonl(BANK_MEI_MW_DISPOSITION_V2_TEST)
    wave1 = json.loads((SFT_V2_EVAL_LOCK_DIR / "scorecard-wave1.ids.json").read_text(encoding="utf-8"))
    ret_w = _subset(ret_test, wave1["retrieval"])
    fc_w = _subset(fc_test, wave1["fullcall"])
    mw_w = _subset(mw_test, wave1["mw"])

    majority = majority_from_train()
    include = {
        "always-refuse",
        "lexical-retrieval",
        "majority-mw",
        "mei-58m-base-300m-no-sft",
        "minimind-25m",
        "minimind-45m",
        "qwen3.5-0.8b",
    }
    if args.include_upper:
        include.update({"qwen3.5-4b", "qwen3.5-9b", "qwen3.6-35b"})
    if args.models.strip():
        include = {x.strip() for x in args.models.split(",") if x.strip()}

    payload_prev = None
    if args.merge:
        if not args.out.is_file():
            raise SystemExit(f"--merge requires existing {args.out}")
        payload_prev = json.loads(args.out.read_text(encoding="utf-8"))
        rows_out = [m for m in payload_prev.get("models") or [] if m.get("id") not in include]
    else:
        rows_out = []

    adapters = resolve_adapters(host=args.host, timeout=args.timeout, include=include)

    if not args.merge:
        append_local_baselines(
            rows_out,
            fc_test=fc_test,
            mw_test=mw_test,
            ret_test=ret_test,
            fc_w=fc_w,
            ret_w=ret_w,
            majority=majority,
        )

    # Qwen models on wave1
    for ad in adapters:
        if ad.backend != "ollama":
            continue
        if ad.status != "ready":
            rows_out.append(
                {
                    "id": ad.id,
                    "role": ad.role,
                    "bank": "wave1",
                    "status": ad.status,
                    "note": ad.note or "zero-shot skipped",
                    "retrieval_scoring": "end_to_end_tool_selection_not_embedding",
                    "same_data_sft_this_round": False,
                }
            )
            continue
        print(f"running {ad.id} wave1 n_fc={len(fc_w)} n_ret={len(ret_w)} n_mw={len(mw_w)}", flush=True)
        fc_texts, fc_lats, _ = run_chat_fullcall(ad, fc_w, timeout_note="raw decoding temperature=0")
        ret_texts, ret_lats = [], []
        for i, row in enumerate(ret_w):
            try:
                text, lat = ad.generate(
                    qwen_select_prompt(row),
                    max_tokens=128,
                    temperature=0.0,
                    system=RETRIEVAL_E2E_SYSTEM,
                )
            except Exception as exc:  # noqa: BLE001
                text, lat = "", {"wall_ms": 0.0, "error": f"{exc.__class__.__name__}:{exc}"}
            ret_texts.append(parse_e2e_tool_name(text, list(row.get("catalog_tools") or [])))
            ret_lats.append(float((lat or {}).get("wall_ms") or 0))
            print(f"  [{ad.id} retrieval {i+1}/{len(ret_w)}]", flush=True)
        mw_texts, mw_lats = [], []
        for i, row in enumerate(mw_w):
            try:
                text, lat = ad.generate(
                    qwen_mw_prompt(row),
                    max_tokens=128,
                    temperature=0.0,
                    system=MW_CLOSED_SYSTEM,
                )
            except Exception as exc:  # noqa: BLE001
                text, lat = "", {"wall_ms": 0.0, "error": f"{exc.__class__.__name__}:{exc}"}
            mw_texts.append(parse_reason_code(text))
            mw_lats.append(float((lat or {}).get("wall_ms") or 0))
            print(f"  [{ad.id} mw {i+1}/{len(mw_w)}]", flush=True)
        ret_hit5 = []
        by_fam: dict[str, list[int]] = defaultdict(list)
        for row, pred in zip(ret_w, ret_texts):
            gold = row.get("gold_tool")
            ok = (pred == gold) or (gold in (None, "") and pred == "NONE")
            ret_hit5.append(int(ok))
            by_fam[str(row.get("family") or "na")].append(int(ok))
        rows_out.append(
            {
                "id": ad.id,
                "role": ad.role,
                "bank": "wave1",
                "status": "complete_zero_shot_raw",
                "decoding": "raw",
                "constrained_decoding": "not_run_this_wave",
                "temperature": 0,
                "max_tokens": 128,
                "same_data_sft_this_round": False,
                "same_data_sft": {"status": "not_authorized_this_round"},
                "retrieval_scoring": "end_to_end_tool_selection_not_embedding",
                "retrieval": {
                    "tool_selection_exact": wilson_interval(sum(ret_hit5), len(ret_hit5)),
                    "by_family": {k: wilson_interval(sum(v), len(v)) for k, v in sorted(by_fam.items())},
                    "note": "Not ContrastiveHead Recall@5. Chat name match only.",
                },
                "fullcall": score_fullcall_preds(fc_w, fc_texts),
                "mw": score_mw_preds(mw_w, mw_texts),
                "efficiency": efficiency(ad.params, fc_lats + ret_lats + mw_lats),
            }
        )

    banks_meta = {
        "retrieval_test": {"path": rel(BANK_MEI_RETRIEVAL_V2_TEST), "n": len(ret_test), "sha256": sha256_file(BANK_MEI_RETRIEVAL_V2_TEST)},
        "fullcall_test": {"path": rel(BANK_MEI_TOOLCALL_V2_TEST), "n": len(fc_test), "sha256": sha256_file(BANK_MEI_TOOLCALL_V2_TEST)},
        "mw_test": {"path": rel(BANK_MEI_MW_DISPOSITION_V2_TEST), "n": len(mw_test), "sha256": sha256_file(BANK_MEI_MW_DISPOSITION_V2_TEST)},
        "wave1": {"retrieval_n": len(ret_w), "fullcall_n": len(fc_w), "mw_n": len(mw_w)},
    }
    clean_meta = {
        "retrieval": {"path": rel(PACK_MEI_RETRIEVAL_V2_10K_CLEAN), "sha256": sha256_file(PACK_MEI_RETRIEVAL_V2_10K_CLEAN)},
        "fullcall": {"path": rel(PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN), "sha256": sha256_file(PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN)},
        "mw": {"path": rel(PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN), "sha256": sha256_file(PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN)},
        "status": "candidate_clean_not_accepted_train",
    }
    payload = {
        "id": "sft-v2-baseline-scorecard-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "current_json_unchanged": True,
        "training_started": False,
        "accepted_registered": False,
        "qwen_0.6b": "not_invented_no_runner",
        "protocol": {
            "temperature": 0,
            "max_tokens": 128,
            "serializer": "mei-tool-call-serializer-v2",
            "prompt": "prompt_v2",
            "student_system": STUDENT_SYSTEM,
            "decoding_this_run": "raw",
            "constrained_decoding": "not_run_this_wave",
        },
        "banks": banks_meta,
        "clean_packs": clean_meta,
        "models": rows_out,
        "gates_file": "runtime/mei-1.0-58m-needle2-v2/spec/gates-v2.target.json",
        "note": "Pilot scorecard. Floors use full official TEST. Generative models use locked wave1 subsample of the same TEST hash. MiniMind same-data SFT is not authorized. Qwen retrieval is end-to-end tool selection, not embedding.",
    }
    dump_json(args.out, payload)
    print(json.dumps({"ok": True, "out": rel(args.out), "n_models": len(rows_out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
