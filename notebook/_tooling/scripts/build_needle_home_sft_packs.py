#!/usr/bin/env python3
"""Build isolated home-vertical SFT packs. Does not rewrite EVAL or holdout lock.

Gold is schema-program. Default teacher is high-quality templates. Optional
overlay of qwen-max variants is applied by generate_needle_home_sft_variants.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from repo_paths import BANK_NEEDLE_VRM_AGENT, CORPORA_ROOT, EVAL_BANKS_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import format_sft_user_text, token_jaccard
from needle_home_sft_lib import (  # noqa: E402
    GENERATOR_VERSION,
    MIXTURE_PATH,
    TOOLSET_ID,
    assistant_truncated,
    dump_jsonl,
    load_jsonl,
    load_mixture,
    protected_slots_ok,
    query_banned,
    render_template_query,
    row_from_intent,
    sample_intent,
    seen_key,
    sha256_file,
    sha256_text,
)
from tokenizer import ZhTokenizerV1  # noqa: E402


def _eval_queries_and_ids(tok: ZhTokenizerV1) -> tuple[set[str], list[set[int]]]:
    rows = load_jsonl(BANK_NEEDLE_VRM_AGENT)
    queries = {str(r.get("query") or "").strip() for r in rows if r.get("query")}
    ids = [set(tok.encode(q)) for q in queries if q]
    return queries, ids


def _token_leak(tok: ZhTokenizerV1, query: str, eval_ids: list[set[int]], threshold: float = 0.9) -> bool:
    ids = set(tok.encode(query))
    n = len(ids)
    if n < 2:
        return False
    for other in eval_ids:
        if len(other) < 2:
            continue
        inter = len(ids & other)
        if inter / (n + len(other) - inter) >= threshold:
            return True
    return False


def _style_hashes() -> set[str]:
    path = CORPORA_ROOT / "sft-style-v0/hashes/utterance.sha256"
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _accept_row(
    *,
    intent: dict,
    query: str,
    tid: str,
    index: int,
    mixture: dict,
    tok: ZhTokenizerV1,
    blocked: set[str],
    eval_ids: list[list[int]],
    style_hashes: set[str],
    seen: set[str],
    seq_len: int,
) -> dict | None:
    reason = query_banned(query)
    if reason:
        return None
    if query in blocked or _token_leak(tok, query, eval_ids):
        return None
    if sha256_text(query) in style_hashes:
        return None
    if not protected_slots_ok(intent, query):
        return None
    row = row_from_intent(
        intent,
        query,
        mixture=mixture,
        index=index,
        teacher_model="template",
        source_role="schema-intent",
        template_id=tid,
    )
    sk = seen_key(row)
    if sk in seen:
        return None
    row["_seen"] = sk
    return row


def _tools_in(row: dict) -> set[str]:
    return {str(c.get("name")) for c in (row.get("answers") or []) if c.get("name")}


def select_balanced(cands: list[dict], n: int, mixture: dict, rng: random.Random) -> list[dict]:
    required = list(mixture["required_tools"])
    weights = mixture["family_weights"]
    fam_cap = {fam: max(1, int(n * float(w) + 0.5) + 12) for fam, w in weights.items()}
    pool = list(cands)
    rng.shuffle(pool)
    selected: list[dict] = []
    seen: set[str] = set()
    fam_n: Counter[str] = Counter()
    tools_hit: set[str] = set()

    def take(row: dict, *, ignore_cap: bool = False) -> bool:
        sk = row.get("_seen") or seen_key(row)
        if sk in seen:
            return False
        fam = str(row.get("family") or "other")
        if not ignore_cap and fam_n[fam] >= fam_cap.get(fam, n):
            return False
        seen.add(sk)
        fam_n[fam] += 1
        tools_hit.update(_tools_in(row))
        selected.append(row)
        return True

    missing = set(required)
    for row in pool:
        if not missing:
            break
        hit = _tools_in(row) & missing
        if hit and take(row, ignore_cap=True):
            missing -= hit
    for row in pool:
        if len(selected) >= n:
            break
        take(row)
    if len(selected) < n:
        for row in pool:
            if len(selected) >= n:
                break
            fam = str(row.get("family") or "other")
            if fam_n[fam] > fam_cap.get(fam, n) * 2:
                continue
            take(row, ignore_cap=True)
    if len(selected) < n:
        for row in pool:
            if len(selected) >= n:
                break
            take(row, ignore_cap=True)
    tlo, thi = (float(x) for x in mixture["targets"]["execute_frac"])
    olo, ohi = (float(x) for x in mixture["targets"]["offtopic_frac"])
    exec_n = sum(1 for r in selected if r.get("answers"))
    if exec_n / max(len(selected), 1) < tlo:
        have = {row.get("_seen") or seen_key(row) for row in selected}
        refuse_idx = [i for i, r in enumerate(selected) if not r.get("answers")]
        exec_cands = [
            r
            for r in pool
            if r.get("answers") and (r.get("_seen") or seen_key(r)) not in have
        ]
        rng.shuffle(exec_cands)
        need = int(tlo * n) - exec_n
        for extra, idx in zip(exec_cands, refuse_idx):
            if need <= 0:
                break
            selected[idx] = extra
            need -= 1
            have.add(extra.get("_seen") or seen_key(extra))
    off_idx = [i for i, r in enumerate(selected) if r.get("family") == "offtopic"]
    off_n = len(off_idx)
    if off_n / max(len(selected), 1) > ohi:
        have = {row.get("_seen") or seen_key(row) for row in selected}
        alt = [
            r
            for r in pool
            if r.get("family") in {"missing", "illegal_pair", "scene_conflict"}
            and (r.get("_seen") or seen_key(r)) not in have
        ]
        rng.shuffle(alt)
        need = off_n - int(ohi * n)
        for extra, idx in zip(alt, off_idx):
            if need <= 0:
                break
            selected[idx] = extra
            need -= 1
            have.add(extra.get("_seen") or seen_key(extra))
    out = selected[:n]
    for i, row in enumerate(out, start=1):
        row["sample_id"] = f"TRAIN-VRM-{i:06d}"
        row.pop("_seen", None)
    return out


def build_candidates(n: int, *, seed: int, mixture: dict) -> list[dict]:
    rng = random.Random(seed)
    tok = ZhTokenizerV1()
    blocked, eval_ids = _eval_queries_and_ids(tok)
    style_hashes = _style_hashes()
    overgen = int(mixture.get("candidate_overgen") or 3)
    want = n * overgen
    seq_len = int(mixture.get("seq_len_train") or 256)
    rows: list[dict] = []
    seen: set[str] = set()
    guard = 0
    i = 0
    stale = 0
    while len(rows) < want and guard < want * 12:
        guard += 1
        intent = sample_intent(rng, mixture)
        query, tid = render_template_query(intent, rng)
        row = _accept_row(
            intent=intent,
            query=query,
            tid=tid,
            index=i + 1,
            mixture=mixture,
            tok=tok,
            blocked=blocked,
            eval_ids=eval_ids,
            style_hashes=style_hashes,
            seen=seen,
            seq_len=seq_len,
        )
        if row is None:
            stale += 1
            if stale >= 3000 and len(rows) >= n:
                break
            continue
        stale = 0
        seen.add(row["_seen"])
        i += 1
        rows.append(row)
    required = set(mixture["required_tools"])
    hit = set()
    for row in rows:
        hit |= _tools_in(row)
    missing = required - hit
    extra = 0
    while missing and extra < 8000:
        extra += 1
        intent = sample_intent(rng, mixture)
        name = intent.get("name")
        if name not in missing:
            continue
        query, tid = render_template_query(intent, rng)
        row = _accept_row(
            intent=intent,
            query=query,
            tid=tid,
            index=i + 1,
            mixture=mixture,
            tok=tok,
            blocked=blocked,
            eval_ids=eval_ids,
            style_hashes=style_hashes,
            seen=seen,
            seq_len=seq_len,
        )
        if row is None:
            continue
        seen.add(row["_seen"])
        i += 1
        rows.append(row)
        missing -= _tools_in(row)
    if len(rows) < n:
        raise RuntimeError(f"only built {len(rows)}/{n} unique SFT candidates")
    return rows


def repair_split_near_dups(rows: list[dict], tok: ZhTokenizerV1) -> int:
    train_ids = [
        tok.encode(format_sft_user_text(r))
        for r in rows
        if r.get("split") == "train"
    ]
    train_ids = [ids for ids in train_ids if len(ids) >= 2]
    move: set[str] = set()
    for row in rows:
        if row.get("split") != "valid":
            continue
        ids = tok.encode(format_sft_user_text(row))
        if len(ids) < 2:
            continue
        if any(token_jaccard(ids, t) >= 0.9 for t in train_ids):
            move.add(str(row.get("intent_id") or row["sample_id"]))
    n = 0
    for row in rows:
        key = str(row.get("intent_id") or row["sample_id"])
        if key in move and row.get("split") == "valid":
            row["split"] = "train"
            n += 1
    return n


def assign_splits(rows: list[dict], valid_frac: float) -> None:
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(str(row.get("intent_id") or row["sample_id"]), []).append(row)
    keys = sorted(by, key=sha256_text)
    want = max(1, int(round(len(rows) * valid_frac)))
    n = 0
    valid_keys: set[str] = set()
    for key in keys:
        if n >= want:
            break
        valid_keys.add(key)
        n += len(by[key])
    for row in rows:
        key = str(row.get("intent_id") or row["sample_id"])
        row["split"] = "valid" if key in valid_keys else "train"


def stratified_review_sample(rows: list[dict], n: int, seed: int = 20260825) -> list[dict]:
    rng = random.Random(seed)
    by_fam: dict[str, list[dict]] = {}
    for row in rows:
        by_fam.setdefault(str(row.get("family") or "other"), []).append(row)
    families = sorted(by_fam)
    out: list[dict] = []
    if not families:
        return out
    per = max(1, n // len(families))
    for fam in families:
        pool = list(by_fam[fam])
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    if len(out) < n:
        leftover = [r for r in rows if r not in out]
        rng.shuffle(leftover)
        out.extend(leftover[: n - len(out)])
    return out[:n]


def family_report(rows: list[dict]) -> dict:
    fam = Counter(str(r.get("family") or "") for r in rows)
    tools = Counter()
    styles = Counter()
    for r in rows:
        for c in r.get("answers") or []:
            if c.get("name"):
                tools[str(c["name"])] += 1
        for tag in r.get("style_tags") or []:
            styles[str(tag)] += 1
    n = max(len(rows), 1)
    n_exec = sum(1 for r in rows if r.get("answers"))
    return {
        "family": dict(fam),
        "tools": dict(tools),
        "style_tags": dict(styles),
        "execute_frac": round(n_exec / n, 4),
        "refuse_frac": round((len(rows) - n_exec) / n, 4),
        "always_refuse_baseline_note": "empty answers are the refuse class, not a score",
        "n_train": sum(1 for r in rows if r.get("split") == "train"),
        "n_valid": sum(1 for r in rows if r.get("split") == "valid"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="2k", choices=["2k", "10k", "20k", "50k"])
    ap.add_argument("--mixture", type=Path, default=MIXTURE_PATH)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--review-n", type=int, default=200)
    args = ap.parse_args()
    mixture = load_mixture(args.mixture)
    n = int(mixture["tiers"][args.tier])
    lock = EVAL_BANKS_ROOT / "needle-vrm-agent-v0" / "holdout-v1.lock.json"
    freeze = json.loads(lock.read_text(encoding="utf-8"))
    want_sha = str(mixture.get("eval_sha256") or "")
    if want_sha and freeze.get("sha256") != want_sha:
        print(
            json.dumps(
                {
                    "error": "eval lock hash mismatch; refuse to build SFT",
                    "lock": freeze.get("sha256"),
                    "mixture": want_sha,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    cands = build_candidates(n, seed=args.seed, mixture=mixture)
    rng = random.Random(args.seed + 7)
    rows = select_balanced(cands, n, mixture, rng)
    assign_splits(rows, float(mixture.get("valid_frac") or 0.09))
    tok = ZhTokenizerV1()
    seq_len = int(mixture.get("seq_len_train") or 256)
    rows = [r for r in rows if not assistant_truncated(tok, r, seq_len)]
    if len(rows) < n:
        extra = [r for r in cands if r not in rows and not assistant_truncated(tok, r, seq_len)]
        rows.extend(extra[: n - len(rows)])
        for i, row in enumerate(rows, start=1):
            row["sample_id"] = f"TRAIN-VRM-{i:06d}"
            row.pop("_seen", None)
        assign_splits(rows, float(mixture.get("valid_frac") or 0.09))
    repair_split_near_dups(rows, tok)
    if len(rows) < n:
        raise RuntimeError(f"only selected {len(rows)}/{n} after truncation filter")
    packs = TASKS_ROOT / TASK_NEEDLE_ZH / "train/packs"
    pack_path = packs / f"home-sft-{args.tier}.jsonl"
    dump_jsonl(pack_path, rows)
    dump_jsonl(packs / f"home-sft-{args.tier}.candidates.jsonl", cands)
    review = stratified_review_sample(rows, args.review_n)
    dump_jsonl(packs / f"home-sft-{args.tier}.review-sample.jsonl", review)
    dist = family_report(rows)
    man = {
        "pack": pack_path.name,
        "n": len(rows),
        "n_candidates": len(cands),
        "candidate_overgen": int(mixture.get("candidate_overgen") or 3),
        "n_train": dist["n_train"],
        "n_valid": dist["n_valid"],
        "sha256": sha256_file(pack_path),
        "gold": "schema-program",
        "teacher": "query-variants-only",
        "teacher_model": "template",
        "mixture": str(args.mixture.relative_to(ROOT)),
        "generator_version": GENERATOR_VERSION,
        "eval_lock": str(lock.relative_to(ROOT)),
        "eval_sha256": freeze["sha256"],
        "token_near_dup_blocked": True,
        "style_near_dup_blocked": bool(_style_hashes()),
        "seed": args.seed,
        "toolset_id": TOOLSET_ID,
        "review_sample": f"home-sft-{args.tier}.review-sample.jsonl",
        "review_n": len(review),
        "distribution": dist,
    }
    (packs / f"home-sft-{args.tier}.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
