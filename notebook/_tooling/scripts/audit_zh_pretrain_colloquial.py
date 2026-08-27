#!/usr/bin/env python3
"""Audit synthetic colloquial releases: full-file hard gates plus stratified soft stats.

Generator/snapshot prove provenance only; this script never auto-passes quality
just because the generator name is qwen-plus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from repo_paths import BANK_NEEDLE_PRETRAIN_PROBES, CORPORA_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH, all_eval_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

SYNTH_SPOKEN_RE = re.compile(
    r"(吗|呢|吧|啊|哎|哦|恩|呀|怎么|为什么|有没有|我觉得|其实|哈哈|这个|那个|咱们|啥|真的|行啊|成了|得了)"
)
from colloquial_synth_lib import TOOLISH_RE, load_contract, unique_by_first_frame  # noqa: E402
from data import document_leaks_eval, iter_jsonl, leak_strings_from_rows  # noqa: E402
from zh_pretrain_ingest import (  # noqa: E402
    CJK_RE,
    NearDupIndex,
    SPOKEN_RE,
    dump_json,
    file_sha256,
    pii_or_nav,
    simhash64,
)

DEFAULT = CORPORA_ROOT / "zh-pretrain-colloquial-synth-qwen-v1"
STYLE_KEYS = (
    "ellipsis",
    "repair",
    "interrupt",
    "short_reply",
    "filler",
    "number_date_unit",
    "code_mix",
    "emotion",
    "negation",
    "deixis",
)


def load_accepted(corpus: Path) -> list[dict]:
    path = corpus / "raw" / "accepted.jsonl"
    if not path.is_file():
        return []
    return [r for r in iter_jsonl(path) if (r.get("filter") or {}).get("ok")]


def load_all_leaks() -> list[str]:
    rows = []
    for path in all_eval_jsonl():
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        for line in BANK_NEEDLE_PRETRAIN_PROBES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return leak_strings_from_rows(rows)


def first_utt(text: str) -> str:
    for line in str(text).splitlines():
        if "：" in line:
            return line.split("：", 1)[1].strip()[:16]
        if ":" in line:
            return line.split(":", 1)[1].strip()[:16]
    return str(text)[:16]


def last_utt(text: str) -> str:
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[-1]
    if "：" in line:
        return line.split("：", 1)[1].strip()[:16]
    return line[:16]


def stratified_sample(rows: list[dict], n: int, axes: dict) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    picked: list[dict] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if 0 <= idx < len(rows) and idx not in seen:
            seen.add(idx)
            picked.append(rows[idx])

    third = max(1, len(rows) // 3)
    for base in (0, third, 2 * third):
        span = min(third, len(rows) - base)
        take = max(8, n // 6)
        step = max(1, span // take)
        for i in range(base, min(base + span, len(rows)), step):
            add(i)
            if len(picked) >= n:
                return picked
    scenes = list(axes.get("scenes") or [])
    styles = list(axes.get("styles") or [])
    relations = list(axes.get("relations") or [])
    moods = list(axes.get("moods") or [])
    turns = list(axes.get("turns") or [])
    buckets = [
        ("scene", scenes),
        ("styles", styles),
        ("relation", relations),
        ("mood", moods),
        ("n_turns", turns),
    ]
    by_key: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        frame = row.get("frame") or {}
        for field, values in buckets:
            if field == "styles":
                for st in frame.get("styles") or []:
                    by_key.setdefault(f"style:{st}", []).append(i)
            else:
                by_key.setdefault(f"{field}:{frame.get(field)}", []).append(i)
    for field, values in buckets:
        keys = [f"style:{v}" for v in values] if field == "styles" else [f"{field}:{v}" for v in values]
        for key in keys:
            idxs = by_key.get(key) or []
            if not idxs:
                continue
            add(idxs[0])
            add(idxs[len(idxs) // 2])
            add(idxs[-1])
            if len(picked) >= n:
                return picked
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT)
    ap.add_argument("--sample", type=int, default=4096)
    ap.add_argument("--expected-generator", default=None)
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    contract = load_contract()
    rows = load_accepted(corpus)
    axes = contract.get("axes") or {}
    sample = stratified_sample(rows, args.sample, axes)
    leaks = load_all_leaks()
    n_pii = n_leak = n_illegal = n_tool = n_spoken = n_cjk = 0
    n_wrong_gen = 0
    near = NearDupIndex(max_hamming=3)
    n_near = 0
    sha: set[str] = set()
    n_exact = 0
    templates = Counter()
    scenes = Counter()
    styles = Counter()
    relations = Counter()
    moods = Counter()
    turns = Counter()
    openings = Counter()
    closings = Counter()
    expected_gen = args.expected_generator
    release = json.loads((corpus / "RELEASE.json").read_text(encoding="utf-8")) if (corpus / "RELEASE.json").is_file() else {}
    if expected_gen is None:
        expected_gen = str(release.get("generator") or "")
    want_snap = contract["generator"]["frozen_snapshot"]
    n_wrong_snap = 0
    for row in rows:
        gen = str(row.get("generator") or "")
        if expected_gen and gen != expected_gen:
            n_wrong_gen += 1
        snap = str(row.get("model_snapshot") or "")
        if expected_gen == "qwen-plus" and snap and want_snap not in snap and snap != want_snap:
            n_wrong_snap += 1
        text = str(row.get("text") or "")
        if pii_or_nav(text) == "pii":
            n_pii += 1
        if document_leaks_eval(text, leaks) is not None:
            n_leak += 1
        if text.count("甲：") + text.count("乙：") < 2:
            n_illegal += 1
        if TOOLISH_RE.search(text):
            n_tool += 1
    counts = unique_by_first_frame(rows)
    for row in sample:
        text = str(row.get("text") or "")
        frame = row.get("frame") or {}
        scenes[str(frame.get("scene") or "")] += 1
        relations[str(frame.get("relation") or "")] += 1
        moods[str(frame.get("mood") or "")] += 1
        turns[str(frame.get("n_turns") or "")] += 1
        for st in frame.get("styles") or []:
            styles[str(st)] += 1
        templates[f"{frame.get('scene')}|{frame.get('speech_act')}|{(frame.get('styles') or [''])[0]}"] += 1
        openings[first_utt(text)] += 1
        closings[last_utt(text)] += 1
        digest = str(row.get("response_sha256") or "")
        if digest in sha:
            n_exact += 1
        sha.add(digest)
        sim = simhash64(text)
        if near.near(sim):
            n_near += 1
        else:
            near.add(sim)
        if len(SYNTH_SPOKEN_RE.findall(text)) >= 1 or len(SPOKEN_RE.findall(text)) >= 1:
            n_spoken += 1
        if CJK_RE.search(text):
            n_cjk += 1
    n = max(len(sample), 1)
    top_share = (templates.most_common(1)[0][1] / n) if templates else 1.0
    open_share = (openings.most_common(1)[0][1] / n) if openings else 1.0
    close_share = (closings.most_common(1)[0][1] / n) if closings else 1.0
    ledger = json.loads((corpus / "unique-ledger.json").read_text(encoding="utf-8")) if (corpus / "unique-ledger.json").is_file() else {}
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8")) if (corpus / "manifest.json").is_file() else {}
    hash_mismatch = 0
    if (corpus / "hashes.json").is_file() and (corpus / "RELEASE.json").is_file():
        stored = json.loads((corpus / "hashes.json").read_text(encoding="utf-8"))
        if stored.get("RELEASE.json") != file_sha256(corpus / "RELEASE.json"):
            hash_mismatch += 1
        if stored.get("unique-ledger.json") != file_sha256(corpus / "unique-ledger.json"):
            hash_mismatch += 1
    ledger_mismatch = int(
        int(release.get("n_unique_train_tokens") or -1) != int(ledger.get("unique_train_tokens") or -2)
        or int(manifest.get("n_unique_train_tokens") or -1) != int(ledger.get("unique_train_tokens") or -2)
        or int(release.get("n_unique_train_tokens") or -1) != int(counts["unique_train_tokens"])
    )
    unk_rate = float(release.get("unk_rate") or 1.0)
    gates = contract["hard_gates"]
    spoken_rate = n_spoken / n
    dup_rate = (n_exact + n_near) / n
    missing_scenes = [k for k in (axes.get("scenes") or []) if scenes.get(k, 0) == 0]
    missing_styles = [k for k in STYLE_KEYS if styles.get(k, 0) == 0]
    hard = {
        "pii_accepted": n_pii,
        "eval_leak_accepted": n_leak,
        "illegal_structure_accepted": n_illegal + n_tool,
        "hash_ledger_mismatch": hash_mismatch + ledger_mismatch,
        "unk_rate": unk_rate,
        "duplicate_frame_id": int(counts["n_duplicate_frame_ids"]),
        "wrong_generator": n_wrong_gen + n_wrong_snap,
    }
    hard_ok = (
        hard["pii_accepted"] <= gates["pii_accepted_max"]
        and hard["eval_leak_accepted"] <= gates["eval_leak_accepted_max"]
        and hard["illegal_structure_accepted"] <= gates["illegal_structure_accepted_max"]
        and hard["hash_ledger_mismatch"] <= gates["hash_ledger_mismatch_max"]
        and unk_rate <= float(gates["unk_rate_max"])
        and hard["duplicate_frame_id"] <= int(gates.get("duplicate_frame_id_max") or 0)
        and hard["wrong_generator"] <= int(gates.get("wrong_generator_max") or 0)
    )
    soft_ok = (
        spoken_rate >= float(gates.get("spoken_rate_min") or 0.85)
        and dup_rate <= float(gates.get("exact_near_dup_rate_max") or 0.02)
        and not missing_scenes
        and not missing_styles
        and open_share <= 0.15
        and close_share <= 0.15
        and top_share <= 0.08
    )
    quality_ok = hard_ok and soft_ok and len(rows) > 0
    report = {
        "corpus": str(corpus.relative_to(ROOT)) if corpus.is_relative_to(ROOT) else str(corpus),
        "n_accepted": len(rows),
        "n_sample": len(sample),
        "hard": hard,
        "hard_ok": hard_ok,
        "soft": {
            "spoken_rate": round(spoken_rate, 4),
            "cjk_rate": round(n_cjk / n, 4),
            "exact_dup_in_sample": n_exact,
            "near_dup_in_sample": n_near,
            "dup_rate": round(dup_rate, 4),
            "top_template_share": round(top_share, 4),
            "opening_template_share": round(open_share, 4),
            "closing_template_share": round(close_share, 4),
            "n_scenes": len([k for k in scenes if k]),
            "n_styles": len(styles),
            "n_relations": len([k for k in relations if k]),
            "n_moods": len([k for k in moods if k]),
            "n_turns": len([k for k in turns if k]),
            "missing_scenes": missing_scenes,
            "missing_styles": missing_styles,
        },
        "unique_by_first_frame": counts,
        "generator": release.get("generator"),
        "model_snapshot": release.get("model_snapshot"),
        "provenance_ok_claimed": release.get("provenance_ok"),
        "formal_cpt_eligible_claimed": release.get("formal_cpt_eligible"),
        "quality_ok": quality_ok,
        "ok": quality_ok,
    }
    out = corpus / "reviews" / "audit.json"
    dump_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
