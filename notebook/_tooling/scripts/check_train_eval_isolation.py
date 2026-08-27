#!/usr/bin/env python3
"""Fail if train seed intersects eval banks, embeds EVAL-* text,
or lacks task/evidence release provenance when claimed as domain-train samples.

Default: mei-expert seed vs mei-expert bank (legacy smoke).
Pass --all to check every task seed against every eval/*.jsonl bank.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from repo_paths import (
    BANK_MEI_EXPERT,
    BANK_NEEDLE_PRETRAIN_PROBES,
    HELDOUT_MEI_EXPERT,
    ISOLATION_SCOPES,
    LANGUAGE_WORK_V0,
    ROOT,
    SEED_MEI_EXPERT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    SCRIPTS_ROOT,
    all_eval_jsonl,
    all_train_seeds,
    cpt_corpus_dirs,
)

EVAL_RE = re.compile(r"\bEVAL-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+\b")
GOLD_RE = re.compile(r"(acceptance/|gold/|holdout-answer)", re.I)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def record_id(row: dict) -> str:
    for key in ("item_id", "sample_id", "sample_id", "id"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


_PUNCT = "?!,.;:、。！？，；：…—-~·\"'“”‘’（）()[]【】《》<>"


def canon_query(text: str) -> str:
    """Whitespace-normalize and strip leading/trailing punctuation for near-dup checks."""
    q = " ".join((text or "").split())
    return q.strip().strip(_PUNCT)


def record_query(row: dict) -> str:
    """Needle-style items use `query`. Do not exact-match expert `prompt` vs chat messages."""
    val = row.get("query")
    if isinstance(val, str) and val.strip():
        return " ".join(val.split())
    return ""


def eval_ids_of(rows: list[dict]) -> set[str]:
    return {record_id(r) for r in rows if record_id(r)}


def check_seed_against_eval(
    seed_path: Path,
    eval_rows: list[dict],
    eval_id_set: set[str],
    eval_queries: set[str],
) -> dict:
    seed = load_jsonl(seed_path)
    seed_ids = {record_id(r) for r in seed if record_id(r)}
    overlap = sorted(eval_id_set & seed_ids)
    leaks: list[str] = []
    query_hits: list[str] = []
    eval_canon = {canon_query(q) for q in eval_queries if canon_query(q)}
    for r in seed:
        sid = record_id(r)
        blob = json.dumps(r, ensure_ascii=False)
        if EVAL_RE.search(blob):
            leaks.append(sid)
        if GOLD_RE.search(blob):
            leaks.append(f"{sid}:gold_path")
        q = record_query(r)
        cq = canon_query(q)
        if q and (q in eval_queries or cq in eval_canon):
            query_hits.append(f"{sid}:{q[:48]}")
    return {
        "seed": str(seed_path.relative_to(ROOT)),
        "seed_n": len(seed),
        "id_overlap": overlap,
        "eval_text_leaks_in_seed": leaks,
        "query_overlap": query_hits,
        "ok": not overlap and not leaks and not query_hits,
        "rows": seed,
    }


def _scan_pretrain_leaks(eval_rows: list[dict], corpus_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
    from data import (  # noqa: E402
        build_leak_index,
        document_leaks_eval,
        iter_jsonl,
        leak_strings_from_rows,
        list_pretrain_shards,
        list_raw_pages,
        normalize_document,
        token_near_dups as token_near_dups_pretrain,
    )
    from tokenizer import ZhTokenizerV1  # noqa: E402

    leak_rows = list(eval_rows)
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        leak_rows.extend(load_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
    leaks = leak_strings_from_rows(leak_rows)
    leak_index = build_leak_index(leaks)
    tok = ZhTokenizerV1()
    eval_tok = [
        (
            str(r.get("probe_id") or record_id(r) or record_query(r)[:24] or "eval"),
            tok.encode(record_query(r) or str(r.get("input") or r.get("target") or "")),
        )
        for r in leak_rows
        if record_query(r) or str(r.get("input") or r.get("target") or "").strip()
    ]
    pretrain_leaks: list[dict] = []
    pretrain_sample: list[tuple[str, list[int]]] = []
    for corpus in corpus_dirs:
        wiki_shards = list_pretrain_shards(corpus, smoke=False)
        smoke_shards = list_pretrain_shards(corpus, smoke=True)
        raw_pages = list_raw_pages(corpus)
        extra_raw = []
        raw_dir = corpus / "raw"
        if raw_dir.is_dir():
            extra_raw = [
                p
                for p in sorted(raw_dir.glob("*.jsonl"))
                if p.is_file() and p.name != "rejected.jsonl"
            ]
        shards = extra_raw or raw_pages or wiki_shards or smoke_shards
        token_shas: set[str] = set()
        token_dir = corpus / "tokens"
        if token_dir.is_dir():
            for idx_path in sorted(token_dir.glob("*.idx.jsonl")):
                for row in iter_jsonl(idx_path):
                    h = str(row.get("sha256") or "")
                    if h:
                        token_shas.add(h)
        for sp in shards:
            if not sp.is_file():
                continue
            for i, row in enumerate(iter_jsonl(sp)):
                text = normalize_document(str(row.get("text") or ""))
                hit = document_leaks_eval(text, leaks, index=leak_index)
                if hit is not None:
                    h = str(row.get("sha256") or "")
                    consumed = (not token_shas) or (h in token_shas)
                    if consumed:
                        pretrain_leaks.append(
                            {"shard": str(sp.relative_to(ROOT)), "row": i, "leak": hit[:80]}
                        )
                        if len(pretrain_leaks) >= 16:
                            break
                if len(pretrain_sample) < 512 and text:
                    pretrain_sample.append((f"{sp.name}:{i}", tok.encode(text[:512])))
            if len(pretrain_leaks) >= 16:
                break
        if len(pretrain_leaks) >= 16:
            break
    pretrain_token_hits: list[dict] = []
    if pretrain_sample and eval_tok:
        pretrain_token_hits = token_near_dups_pretrain(
            pretrain_sample, eval_tok, threshold=0.9, limit=16
        )
    return pretrain_leaks, pretrain_token_hits


def run_scope(scope: str, args: argparse.Namespace) -> tuple[int, dict]:
    bank_paths = all_eval_jsonl(scope)
    seed_paths = all_train_seeds(scope)
    eval_rows: list[dict] = []
    for bp in bank_paths:
        eval_rows.extend(load_jsonl(bp))
    eval_id_set = eval_ids_of(eval_rows)
    eval_queries = {q for q in (record_query(r) for r in eval_rows) if q}

    pair_reports = []
    all_seed_rows: list[dict] = []
    ok_pairs = True
    for sp in seed_paths:
        if not sp.is_file():
            continue
        report = check_seed_against_eval(sp, eval_rows, eval_id_set, eval_queries)
        all_seed_rows.extend(report.pop("rows"))
        ok_pairs = ok_pairs and report["ok"]
        pair_reports.append(report)

    domain_errors: list[str] = []
    domain_rows = load_jsonl(args.domain_train) if args.domain_train else []
    for r in domain_rows:
        sid = record_id(r)
        if not r.get("task_catalog_release"):
            domain_errors.append(f"{sid}:missing_task_catalog_release")
        if not r.get("evidence_release"):
            domain_errors.append(f"{sid}:missing_evidence_release")
        if not r.get("source_refs"):
            domain_errors.append(f"{sid}:missing_source_refs")
        if not r.get("transform_recipe"):
            domain_errors.append(f"{sid}:missing_transform_recipe")
        blob = json.dumps(r, ensure_ascii=False)
        if GOLD_RE.search(blob):
            domain_errors.append(f"{sid}:gold_leak")
        if EVAL_RE.search(blob):
            domain_errors.append(f"{sid}:eval_text")

    heldout_leaks: list[str] = []
    heldout_path = args.heldout_task_ids
    if heldout_path is None and HELDOUT_MEI_EXPERT.is_file() and scope in {"legacy", "all"}:
        heldout_path = HELDOUT_MEI_EXPERT
    if heldout_path and heldout_path.is_file():
        heldout_ids = json.loads(heldout_path.read_text(encoding="utf-8"))
        corpus = all_seed_rows + domain_rows
        for r in corpus:
            blob = json.dumps(r, ensure_ascii=False)
            for hid in heldout_ids:
                if hid and hid in blob:
                    heldout_leaks.append(f"{record_id(r)}:{hid}")

    token_hits: list[dict] = []
    scan_heavy = scope in {"legacy", "cpt-v2"}
    if scan_heavy and (all_seed_rows or scope == "cpt-v2"):
        sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
        from data import token_near_dups
        from tokenizer import ZhTokenizerV1

        tok = ZhTokenizerV1()
        eval_ids = [
            (record_id(r) or record_query(r)[:24], tok.encode(record_query(r)))
            for r in eval_rows
            if record_query(r)
        ]
        seed_ids = [
            (record_id(r) or record_query(r)[:24], tok.encode(record_query(r)))
            for r in all_seed_rows
            if record_query(r)
        ]
        if seed_ids:
            token_hits = token_near_dups(seed_ids, eval_ids, threshold=0.9, limit=32)

    pretrain_leaks: list[dict] = []
    pretrain_token_hits: list[dict] = []
    if scope in {"legacy", "cpt-v2"}:
        dirs = [LANGUAGE_WORK_V0] if scope == "legacy" else cpt_corpus_dirs("cpt-v2")
        pretrain_leaks, pretrain_token_hits = _scan_pretrain_leaks(eval_rows, dirs)

    grounded_leaks: list[str] = []
    if scope == "legacy":
        sys.path.insert(0, str(SCRIPTS_ROOT))
        from mei_tool_grounded_lib import normalized_body as _nb

        g_eval = [
            r
            for r in eval_rows
            if str(r.get("protocol") or "") == "mei-route-protocol-v1"
            or str(r.get("generator_version") or "").startswith("mei-tool-grounded")
        ]
        g_train = [r for r in all_seed_rows if str(r.get("protocol") or "") == "mei-route-protocol-v1"]
        eval_pairs = {(_nb(str(r.get("query") or "")), str(r.get("toolset_id") or "")) for r in g_eval}
        eval_cf = {str(r.get("counterfactual_id") or "") for r in g_eval if r.get("counterfactual_id")}
        for r in g_train:
            pair = (_nb(str(r.get("query") or "")), str(r.get("toolset_id") or ""))
            if pair in eval_pairs:
                grounded_leaks.append(f"{record_id(r)}:body_toolset")
            cf = str(r.get("counterfactual_id") or "")
            if cf and cf in eval_cf and pair[0] in {p[0] for p in eval_pairs}:
                grounded_leaks.append(f"{record_id(r)}:counterfactual")

    sft_v2_extra: dict = {}
    if scope in {"sft-v2", "sft-v2-lock-v2"}:
        sys.path.insert(0, str(SCRIPTS_ROOT))
        from park_toolcall_lib import char_trigrams, holdout_lock_payload, near_dup_holdout

        lock = holdout_lock_payload()
        eval_cf2 = {str(r.get("cf_group") or r.get("counterfactual_group") or "") for r in eval_rows}
        eval_tmpl = {str(r.get("template_id") or "") for r in eval_rows if r.get("template_id")}
        eval_tri = [(record_query(er), char_trigrams(record_query(er))) for er in eval_rows if record_query(er)]
        inv: dict[str, list[int]] = {}
        for i, (_eq, tset) in enumerate(eval_tri):
            for tri in tset:
                inv.setdefault(tri, []).append(i)
        hits: list[str] = []
        for r in all_seed_rows:
            if r.get("engineering_smoke"):
                continue
            q = record_query(r)
            leak = near_dup_holdout(q) if q else None
            if leak:
                hits.append(f"{record_id(r)}:{leak}")
            cf = str(r.get("cf_group") or r.get("counterfactual_group") or "")
            if cf and cf in eval_cf2:
                hits.append(f"{record_id(r)}:cf_group")
            tmpl = str(r.get("template_id") or "")
            if tmpl and tmpl in eval_tmpl:
                hits.append(f"{record_id(r)}:template")
            if q:
                tq = char_trigrams(q)
                cands: set[int] = set()
                for tri in tq:
                    cands.update(inv.get(tri) or [])
                for j in cands:
                    eq, te = eval_tri[j]
                    union = tq | te
                    if union and len(tq & te) / len(union) >= 0.9:
                        hits.append(f"{record_id(r)}:trigram")
                        break
        sft_v2_extra = {
            "holdout_lock_fp": lock.get("toolset_fingerprint"),
            "hits": hits[:40],
            "n_hits": len(hits),
        }
        if hits:
            grounded_leaks.extend(hits[:16])

    ok = (
        ok_pairs
        and not domain_errors
        and not heldout_leaks
        and not token_hits
        and not pretrain_leaks
        and not pretrain_token_hits
        and not grounded_leaks
    )
    out = {
        "ok": ok,
        "mode": scope,
        "scope": scope,
        "eval_n": len(eval_id_set),
        "banks": [str(p.relative_to(ROOT)) for p in bank_paths],
        "seeds": [str(p.relative_to(ROOT)) for p in seed_paths if p.is_file()],
        "pairs": pair_reports,
        "domain_train_n": len(domain_rows),
        "domain_provenance_errors": domain_errors,
        "heldout_leaks": heldout_leaks,
        "token_near_dups": token_hits,
        "pretrain_leaks": pretrain_leaks,
        "pretrain_token_near_dups": pretrain_token_hits,
        "grounded_leaks": grounded_leaks,
        "sft_v2_extra": sft_v2_extra,
        "publish_gate": scope in {"cpt-v2", "sft-v2"},
    }
    return (0 if ok else 1), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_MEI_EXPERT)
    ap.add_argument("--seed", type=Path, default=SEED_MEI_EXPERT)
    ap.add_argument(
        "--all",
        action="store_true",
        help="Legacy diagnostic: every historical seed vs every eval bank (may be red)",
    )
    ap.add_argument(
        "--scope",
        choices=ISOLATION_SCOPES,
        default=None,
        help="legacy=historical diagnostic; cpt-v2=CPT corpora vs frozen EVAL/probes; "
        "sft-v2=future v2 packs; all=emit all three reports",
    )
    ap.add_argument(
        "--domain-train",
        type=Path,
        default=None,
        help="Optional Evidence-derived supervision JSONL (must carry provenance)",
    )
    ap.add_argument(
        "--heldout-task-ids",
        type=Path,
        default=None,
        help="Optional JSON list of heldout task_ids that must not appear in train text",
    )
    args = ap.parse_args()
    scope = args.scope or ("all" if args.all else None)
    if scope == "all":
        reports = {}
        publish_ok = True
        for sub in ("legacy", "cpt-v2", "sft-v2"):
            code, payload = run_scope(sub, args)
            reports[sub] = payload
            if sub != "legacy" and code != 0:
                publish_ok = False
        out = {
            "ok": publish_ok,
            "mode": "all-scopes",
            "publish_scopes_ok": publish_ok,
            "legacy_diagnostic_ok": bool(reports["legacy"].get("ok")),
            "scopes": reports,
            "note": "legacy query overlap is diagnostic only and does not block cpt-v2/sft-v2",
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if publish_ok else 1
    if scope:
        code, payload = run_scope(scope, args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return code

    bank_paths = [args.bank]
    seed_paths = [args.seed]

    eval_rows: list[dict] = []
    for bp in bank_paths:
        eval_rows.extend(load_jsonl(bp))
    eval_id_set = eval_ids_of(eval_rows)
    eval_queries = {q for q in (record_query(r) for r in eval_rows) if q}

    pair_reports = []
    all_seed_rows: list[dict] = []
    ok_pairs = True
    for sp in seed_paths:
        report = check_seed_against_eval(sp, eval_rows, eval_id_set, eval_queries)
        all_seed_rows.extend(report.pop("rows"))
        ok_pairs = ok_pairs and report["ok"]
        pair_reports.append(report)

    domain_errors: list[str] = []
    domain_rows = load_jsonl(args.domain_train) if args.domain_train else []
    for r in domain_rows:
        sid = record_id(r)
        if not r.get("task_catalog_release"):
            domain_errors.append(f"{sid}:missing_task_catalog_release")
        if not r.get("evidence_release"):
            domain_errors.append(f"{sid}:missing_evidence_release")
        if not r.get("source_refs"):
            domain_errors.append(f"{sid}:missing_source_refs")
        if not r.get("transform_recipe"):
            domain_errors.append(f"{sid}:missing_transform_recipe")
        blob = json.dumps(r, ensure_ascii=False)
        if GOLD_RE.search(blob):
            domain_errors.append(f"{sid}:gold_leak")
        if EVAL_RE.search(blob):
            domain_errors.append(f"{sid}:eval_text")

    heldout_leaks: list[str] = []
    heldout_path = args.heldout_task_ids
    if heldout_path is None and HELDOUT_MEI_EXPERT.is_file() and args.all:
        heldout_path = HELDOUT_MEI_EXPERT
    if heldout_path and heldout_path.is_file():
        heldout_ids = json.loads(heldout_path.read_text(encoding="utf-8"))
        corpus = all_seed_rows + domain_rows
        for r in corpus:
            blob = json.dumps(r, ensure_ascii=False)
            for hid in heldout_ids:
                if hid and hid in blob:
                    heldout_leaks.append(f"{record_id(r)}:{hid}")

    token_hits: list[dict] = []
    if args.all:
        sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
        from data import token_near_dups
        from tokenizer import ZhTokenizerV1

        tok = ZhTokenizerV1()
        eval_ids = [
            (record_id(r) or record_query(r)[:24], tok.encode(record_query(r)))
            for r in eval_rows
            if record_query(r)
        ]
        seed_ids = [
            (record_id(r) or record_query(r)[:24], tok.encode(record_query(r)))
            for r in all_seed_rows
            if record_query(r)
        ]
        token_hits = token_near_dups(seed_ids, eval_ids, threshold=0.9, limit=32)

    pretrain_leaks: list[dict] = []
    pretrain_token_hits: list[dict] = []
    grounded_leaks: list[str] = []
    if args.all:
        sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
        from data import (  # noqa: E402
            document_leaks_eval,
            build_leak_index,
            iter_jsonl,
            leak_strings_from_rows,
            list_pretrain_shards,
            list_raw_pages,
            normalize_document,
            token_near_dups as token_near_dups_pretrain,
        )
        from tokenizer import ZhTokenizerV1  # noqa: E402

        leak_rows = list(eval_rows)
        if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
            leak_rows.extend(load_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
        leaks = leak_strings_from_rows(leak_rows)
        leak_index = build_leak_index(leaks)
        tok = ZhTokenizerV1()
        eval_tok = [
            (
                str(r.get("probe_id") or record_id(r) or record_query(r)[:24] or "eval"),
                tok.encode(record_query(r) or str(r.get("input") or r.get("target") or "")),
            )
            for r in leak_rows
            if record_query(r) or str(r.get("input") or r.get("target") or "").strip()
        ]
        wiki_shards = list_pretrain_shards(LANGUAGE_WORK_V0, smoke=False)
        smoke_shards = list_pretrain_shards(LANGUAGE_WORK_V0, smoke=True)
        raw_pages = list_raw_pages(LANGUAGE_WORK_V0)
        shards = raw_pages or wiki_shards or smoke_shards
        token_shas: set[str] = set()
        token_dir = LANGUAGE_WORK_V0 / "tokens"
        if token_dir.is_dir():
            for idx_path in sorted(token_dir.glob("*.idx.jsonl")):
                for row in iter_jsonl(idx_path):
                    h = str(row.get("sha256") or "")
                    if h:
                        token_shas.add(h)
        pretrain_sample: list[tuple[str, list[int]]] = []
        for sp in shards:
            if not sp.is_file():
                continue
            for i, row in enumerate(iter_jsonl(sp)):
                text = normalize_document(str(row.get("text") or ""))
                hit = document_leaks_eval(text, leaks, index=leak_index)
                if hit is not None:
                    h = str(row.get("sha256") or "")
                    consumed = (not token_shas) or (h in token_shas)
                    if consumed:
                        pretrain_leaks.append(
                            {"shard": str(sp.relative_to(ROOT)), "row": i, "leak": hit[:80]}
                        )
                        if len(pretrain_leaks) >= 16:
                            break
                if len(pretrain_sample) < 512 and text:
                    pretrain_sample.append((f"{sp.name}:{i}", tok.encode(text[:512])))
            if len(pretrain_leaks) >= 16:
                break
        if pretrain_sample and eval_tok:
            pretrain_token_hits = token_near_dups_pretrain(
                pretrain_sample, eval_tok, threshold=0.9, limit=16
            )

    grounded_leaks: list[str] = []
    if args.all:
        sys.path.insert(0, str(SCRIPTS_ROOT))
        from mei_tool_grounded_lib import normalized_body as _nb

        g_eval = [
            r
            for r in eval_rows
            if str(r.get("protocol") or "") == "mei-route-protocol-v1"
            or str(r.get("generator_version") or "").startswith("mei-tool-grounded")
        ]
        g_train = [r for r in all_seed_rows if str(r.get("protocol") or "") == "mei-route-protocol-v1"]
        eval_pairs = {(_nb(str(r.get("query") or "")), str(r.get("toolset_id") or "")) for r in g_eval}
        eval_cf = {str(r.get("counterfactual_id") or "") for r in g_eval if r.get("counterfactual_id")}
        for r in g_train:
            pair = (_nb(str(r.get("query") or "")), str(r.get("toolset_id") or ""))
            if pair in eval_pairs:
                grounded_leaks.append(f"{record_id(r)}:body_toolset")
            cf = str(r.get("counterfactual_id") or "")
            if cf and cf in eval_cf and pair[0] in {p[0] for p in eval_pairs}:
                grounded_leaks.append(f"{record_id(r)}:counterfactual")

    ok = (
        ok_pairs
        and not domain_errors
        and not heldout_leaks
        and not token_hits
        and not pretrain_leaks
        and not pretrain_token_hits
        and not grounded_leaks
    )
    out = {
        "ok": ok,
        "mode": "all" if args.all else "single",
        "eval_n": len(eval_id_set),
        "banks": [str(p.relative_to(ROOT)) for p in bank_paths],
        "pairs": pair_reports,
        "domain_train_n": len(domain_rows),
        "domain_provenance_errors": domain_errors,
        "heldout_leaks": heldout_leaks,
        "token_near_dups": token_hits,
        "pretrain_leaks": pretrain_leaks,
        "pretrain_token_near_dups": pretrain_token_hits,
        "grounded_leaks": grounded_leaks,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
