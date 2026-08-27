#!/usr/bin/env python3
"""Collect plus+sidecar accepted.jsonl into an independent pooled pack and audit quality.

Never writes qwen-v1 or the offline v1 contrast dir. formal_cpt_eligible stays false.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from colloquial_fleet import ALL_DIRS, FORMAL_DIR, POOLED_UNIQUE_TARGET
from repo_paths import COLLOQUIAL_DRAFT, CORPORA_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from audit_zh_pretrain_colloquial import (  # noqa: E402
    STYLE_KEYS,
    SYNTH_SPOKEN_RE,
    first_utt,
    last_utt,
    load_all_leaks,
)
from colloquial_release import freeze_release, tokenize_kept  # noqa: E402
from colloquial_synth_lib import TOOLISH_RE, load_contract, unique_by_first_frame  # noqa: E402
from data import document_leaks_eval  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402
from zh_pretrain_ingest import CJK_RE, NearDupIndex, SPOKEN_RE, dump_json, pii_or_nav, simhash64  # noqa: E402

OUT_NAME = "zh-pretrain-colloquial-synth-pooled-v1"
FORBIDDEN = {
    FORMAL_DIR,
    "zh-pretrain-colloquial-synth-v1",
    "zh-pretrain-colloquial-synth-v2",
}


class Reservoir:
    def __init__(self, n: int, seed: int) -> None:
        self.n = n
        self.rng = random.Random(seed)
        self.items: list[dict] = []
        self.seen = 0

    def add(self, row: dict) -> None:
        self.seen += 1
        if len(self.items) < self.n:
            self.items.append(row)
            return
        j = self.rng.randrange(self.seen)
        if j < self.n:
            self.items[j] = row


def empty_lane() -> dict:
    return {
        "n_docs": 0,
        "unique_train": 0,
        "n_unk": 0,
        "n_tokens": 0,
        "spend_cny": 0.0,
        "n_pii": 0,
        "n_leak": 0,
        "n_illegal": 0,
        "n_tool": 0,
        "n_spoken": 0,
        "n_cjk": 0,
        "scenes": Counter(),
        "styles": Counter(),
        "relations": Counter(),
        "moods": Counter(),
        "turns": Counter(),
        "templates": Counter(),
        "openings": Counter(),
        "closings": Counter(),
        "snapshots": Counter(),
    }


def copy_and_scan(out: Path, leaks: list[str]) -> tuple[dict, list[dict]]:
    raw = out / "raw" / "accepted.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    by_gen: dict[str, dict] = defaultdict(empty_lane)
    overall = empty_lane()
    samples = Reservoir(4096, 7)
    per_gen_sample: dict[str, Reservoir] = defaultdict(lambda: Reservoir(1024, 11))
    seen_all: set[str] = set()
    seen_gen: dict[str, set[str]] = defaultdict(set)
    parents = []
    n_copied = 0
    with raw.open("w", encoding="utf-8") as dest:
        for name in ALL_DIRS:
            src = CORPORA_ROOT / name / "raw" / "accepted.jsonl"
            info = {"dir": name, "src_exists": src.is_file(), "copied": 0, "skipped": 0}
            if not src.is_file():
                parents.append(info)
                continue
            with src.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not (row.get("filter") or {}).get("ok"):
                        info["skipped"] += 1
                        continue
                    dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_copied += 1
                    info["copied"] += 1
                    gen = str(row.get("generator") or "unknown")
                    text = str(row.get("text") or "")
                    frame = row.get("frame") or {}
                    n_tok = int(row.get("n_tokens") or 0)
                    n_unk = int(row.get("n_unk") or 0)
                    spend = float(row.get("spend_cny") or 0)
                    split = str(row.get("split") or "train")
                    fid = str(row.get("doc_id") or frame.get("frame_id") or "")
                    hit_pii = pii_or_nav(text) == "pii"
                    hit_leak = document_leaks_eval(text, leaks) is not None
                    illegal = text.count("甲：") + text.count("乙：") < 2
                    tool = TOOLISH_RE.search(text) is not None
                    spoken = SYNTH_SPOKEN_RE.search(text) is not None or SPOKEN_RE.search(text) is not None
                    cjk = CJK_RE.search(text) is not None
                    styles = list(frame.get("styles") or [])
                    tmpl = f"{frame.get('scene')}|{frame.get('speech_act')}|{(styles or [''])[0]}"
                    opening = first_utt(text)
                    closing = last_utt(text)
                    buckets = ((overall, seen_all), (by_gen[gen], seen_gen[gen]))
                    for b, seen in buckets:
                        b["n_docs"] += 1
                        b["n_unk"] += n_unk
                        b["n_tokens"] += n_tok
                        b["spend_cny"] += spend
                        if fid and fid not in seen:
                            seen.add(fid)
                            if split == "train":
                                b["unique_train"] += n_tok
                        b["snapshots"][str(row.get("model_snapshot") or "")] += 1
                        b["scenes"][str(frame.get("scene") or "")] += 1
                        b["relations"][str(frame.get("relation") or "")] += 1
                        b["moods"][str(frame.get("mood") or "")] += 1
                        b["turns"][str(frame.get("n_turns") or "")] += 1
                        for st in styles:
                            b["styles"][str(st)] += 1
                        b["templates"][tmpl] += 1
                        b["openings"][opening] += 1
                        b["closings"][closing] += 1
                        if hit_pii:
                            b["n_pii"] += 1
                        if hit_leak:
                            b["n_leak"] += 1
                        if illegal:
                            b["n_illegal"] += 1
                        if tool:
                            b["n_tool"] += 1
                        if spoken:
                            b["n_spoken"] += 1
                        if cjk:
                            b["n_cjk"] += 1
                    slim = {
                        "generator": gen,
                        "text": text,
                        "frame": frame,
                        "response_sha256": row.get("response_sha256"),
                        "n_tokens": n_tok,
                        "doc_id": fid,
                    }
                    samples.add(slim)
                    per_gen_sample[gen].add(slim)
            parents.append(info)
    overall["n_copied"] = n_copied
    return {
        "parents": parents,
        "overall": overall,
        "by_generator": dict(by_gen),
        "sample": samples.items,
        "per_gen_sample": {k: v.items for k, v in per_gen_sample.items()},
    }, []


def counter_share(counter: Counter, n: int) -> float:
    if n <= 0 or not counter:
        return 1.0
    return counter.most_common(1)[0][1] / n


def sample_dup_stats(rows: list[dict]) -> dict:
    sha: set[str] = set()
    n_exact = 0
    n_near = 0
    near = NearDupIndex(max_hamming=3)
    for row in rows:
        digest = str(row.get("response_sha256") or "")
        if digest and digest in sha:
            n_exact += 1
        if digest:
            sha.add(digest)
        text = str(row.get("text") or "")
        sim = simhash64(text)
        if near.near(sim):
            n_near += 1
        else:
            near.add(sim)
    n = max(len(rows), 1)
    return {
        "n_sample": len(rows),
        "exact_dup": n_exact,
        "near_dup": n_near,
        "dup_rate": round((n_exact + n_near) / n, 4),
    }


def summarize_lane(lane: dict, contract: dict, sample_rows: list[dict] | None = None) -> dict:
    n = max(int(lane["n_docs"]), 1)
    gates = contract["hard_gates"]
    axes = contract.get("axes") or {}
    spoken_rate = lane["n_spoken"] / n
    unk_rate = (lane["n_unk"] / lane["n_tokens"]) if lane["n_tokens"] else 1.0
    missing_scenes = [k for k in (axes.get("scenes") or []) if lane["scenes"].get(k, 0) == 0]
    missing_styles = [k for k in STYLE_KEYS if lane["styles"].get(k, 0) == 0]
    dups = sample_dup_stats(sample_rows or [])
    hard_ok = (
        lane["n_pii"] <= gates["pii_accepted_max"]
        and lane["n_leak"] <= gates["eval_leak_accepted_max"]
        and (lane["n_illegal"] + lane["n_tool"]) <= gates["illegal_structure_accepted_max"]
        and unk_rate <= float(gates["unk_rate_max"])
    )
    soft_ok = (
        spoken_rate >= float(gates.get("spoken_rate_min") or 0.85)
        and dups["dup_rate"] <= float(gates.get("exact_near_dup_rate_max") or 0.02)
        and not missing_scenes
        and not missing_styles
        and counter_share(lane["openings"], n) <= 0.15
        and counter_share(lane["closings"], n) <= 0.15
        and counter_share(lane["templates"], n) <= 0.08
    )
    examples = []
    for row in (sample_rows or [])[:2]:
        examples.append((str(row.get("text") or "").replace("\n", " | "))[:180])
    return {
        "n_docs": lane["n_docs"],
        "unique_train_tokens_sum": lane["unique_train"],
        "spend_cny": round(lane["spend_cny"], 4),
        "unk_rate": round(unk_rate, 6),
        "spoken_rate": round(spoken_rate, 4),
        "cjk_rate": round(lane["n_cjk"] / n, 4),
        "mean_tokens": round(lane["n_tokens"] / n, 1),
        "hard": {
            "pii": lane["n_pii"],
            "eval_leak": lane["n_leak"],
            "illegal_or_tool": lane["n_illegal"] + lane["n_tool"],
        },
        "hard_ok": hard_ok,
        "soft_ok": soft_ok,
        "quality_ok": hard_ok and soft_ok and lane["n_docs"] > 0,
        "dup_sample": dups,
        "n_scenes": len([k for k in lane["scenes"] if k]),
        "n_styles": len(lane["styles"]),
        "missing_scenes": missing_scenes,
        "missing_styles": missing_styles,
        "top_template_share": round(counter_share(lane["templates"], n), 4),
        "opening_share": round(counter_share(lane["openings"], n), 4),
        "closing_share": round(counter_share(lane["closings"], n), 4),
        "snapshots": dict(lane["snapshots"]),
        "examples": examples,
    }


def freeze_pack(out: Path, contract: dict) -> dict:
    kept = []
    src = out / "raw" / "accepted.jsonl"
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not (row.get("filter") or {}).get("ok"):
                continue
            kept.append(
                {
                    "doc_id": row.get("doc_id") or (row.get("frame") or {}).get("frame_id"),
                    "frame": row.get("frame") or {},
                    "text": row.get("text") or "",
                    "n_tokens": int(row.get("n_tokens") or 0),
                    "split": row.get("split") or "train",
                    "generator": row.get("generator"),
                }
            )
    tok = ZhTokenizerV1()
    tok_stats = tokenize_kept(out, kept, tok)
    counts = unique_by_first_frame(kept)
    stats = {
        "n_unk": tok_stats["n_unk"],
        "release_kind": "pooled-mixed",
        "mixed_generator": True,
        "spend_cny": 0.0,
        "model_snapshot": "mixed-bailian-fleet",
        "n_docs": counts["n_unique_frame_ids"],
    }
    spend = 0.0
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                spend += float(json.loads(line).get("spend_cny") or 0)
    stats["spend_cny"] = spend
    return freeze_release(
        out,
        contract=contract,
        generator="mixed-bailian-fleet",
        kept=kept,
        stats=stats,
    )


def jsonable(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=COLLOQUIAL_DRAFT)
    ap.add_argument("--skip-freeze", action="store_true")
    args = ap.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.name in FORBIDDEN:
        print("refuse to write pooled pack into a reserved corpus dir", file=sys.stderr)
        return 5
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews").mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    leaks = load_all_leaks()
    scan, _unused = copy_and_scan(out, leaks)
    by_gen_sum = {}
    for gen, lane in scan["by_generator"].items():
        by_gen_sum[gen] = summarize_lane(lane, contract, scan["per_gen_sample"].get(gen) or [])
    overall_sum = summarize_lane(scan["overall"], contract, scan["sample"])
    unique_stream = {
        "n_docs": scan["overall"]["n_docs"],
        "unique_train_tokens": scan["overall"]["unique_train"],
        "n_unique_frame_ids": scan["overall"]["n_docs"],
    }
    report = {
        "id": out.name,
        "formal_cpt_eligible": False,
        "mixed_generator": True,
        "pooled_target": POOLED_UNIQUE_TARGET,
        "parents": scan["parents"],
        "unique_by_first_frame": unique_stream,
        "cursor_sum_note": "cursor unique was per-dir first-frame; pooled unique below is first frame_id across concatenated jsonl",
        "overall": overall_sum,
        "by_generator": by_gen_sum,
        "note": "Quality uses produce-time n_unk. Mixed pack cannot admit under the frozen qwen-plus contract.",
    }
    dump_json(out / "reviews" / "quality.json", jsonable(report))
    dump_json(
        out / "parents.json",
        {
            "parents": scan["parents"],
            "reserved_untouched": list(FORBIDDEN),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.skip_freeze:
        return 0
    release = freeze_pack(out, contract)
    dump_json(out / "reviews" / "freeze.json", {"ok": True, "release": {k: release.get(k) for k in (
        "id", "generator", "model_snapshot", "n_unique_train_tokens", "unk_rate",
        "formal_cpt_eligible", "eligibility_reason", "n_docs", "spend_cny",
    )}})
    print(json.dumps({"freeze": True, "n_unique_train_tokens": release.get("n_unique_train_tokens"), "unk_rate": release.get("unk_rate"), "formal_cpt_eligible": release.get("formal_cpt_eligible")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
