#!/usr/bin/env python3
"""Build a 100% synthetic colloquial CPT role release.

Smoke and offline engineering pilots do not open formal CPT. Production
realization is produce_colloquial_qwen.py (frozen qwen-plus snapshot).
This entry keeps the offline renderer and dispatches qwen to the durable producer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import (
    CORPORA_ROOT,
    CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_release import freeze_release, load_jsonl, load_leaks, tokenize_kept  # noqa: E402
from colloquial_synth_lib import (  # noqa: E402
    filter_reasons,
    generate_one,
    iter_frames,
    load_contract,
    provenance_row,
    sha256_text,
    spend_cny,
    unique_by_first_frame,
)
from tokenizer import ZhTokenizerV1  # noqa: E402
from zh_pretrain_ingest import (  # noqa: E402
    NearDupIndex,
    colloquial_keep,
    dump_json,
    hash_bucket,
    pii_or_nav,
    simhash64,
)

DEFAULT_OUT = CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1
QWEN_PRODUCER = SCRIPTS_ROOT / "produce_colloquial_qwen.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--generator", choices=["auto", "qwen-plus", "offline-frame-renderer"], default="auto")
    ap.add_argument("--target-unique-tokens", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-spend-cny", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--release-kind", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--qps", type=float, default=None)
    args = ap.parse_args()
    contract = load_contract()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    generator = args.generator
    if generator == "auto":
        generator = "offline-frame-renderer" if args.smoke else "qwen-plus"
    if generator == "qwen-plus":
        cmd = [sys.executable, str(QWEN_PRODUCER), "--out-dir", str(out)]
        if args.smoke:
            cmd.append("--smoke")
        if args.target_unique_tokens is not None:
            cmd.extend(["--target-unique-tokens", str(args.target_unique_tokens)])
        if args.max_docs is not None:
            cmd.extend(["--max-docs", str(args.max_docs)])
        if args.seed:
            cmd.extend(["--seed", str(args.seed)])
        if args.max_spend_cny is not None:
            cmd.extend(["--max-spend-cny", str(args.max_spend_cny)])
        if args.resume:
            cmd.append("--resume")
        if args.release_kind:
            cmd.extend(["--release-kind", args.release_kind])
        if args.workers:
            cmd.extend(["--workers", str(args.workers)])
        if args.qps:
            cmd.extend(["--qps", str(args.qps)])
        return subprocess.run(cmd, cwd=str(ROOT)).returncode
    if args.smoke and args.out_dir == DEFAULT_OUT:
        out = CORPORA_ROOT / "zh-pretrain-colloquial-synth-smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "state").mkdir(parents=True, exist_ok=True)
    (out / "reviews").mkdir(parents=True, exist_ok=True)
    target = args.target_unique_tokens
    if target is None:
        target = 8_000 if args.smoke else int(contract["targets"]["pilot_unique_tokens"])
    max_docs = args.max_docs
    if max_docs is None:
        max_docs = int(contract["targets"]["smoke_docs"]) if args.smoke else max(10_000, target // 40)
    release_kind = args.release_kind or ("smoke" if args.smoke else "pilot")
    tok = ZhTokenizerV1()
    leaks = load_leaks(full=True)
    raw_path = out / "raw" / "accepted.jsonl"
    rej_path = out / "raw" / "rejected.jsonl"
    cursor_path = out / "state" / "cursor.json"
    nonempty = raw_path.is_file() and raw_path.stat().st_size > 0
    if nonempty and not args.resume and not args.smoke:
        print("existing corpus requires explicit --resume", file=sys.stderr)
        return 5
    seen_sha: set[str] = set()
    seen_frames: set[str] = set()
    near = NearDupIndex(max_hamming=3)
    start = 0
    n_unique = 0
    n_docs = 0
    spend = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    existing = load_jsonl(raw_path) if raw_path.is_file() and (args.resume or args.smoke) else []
    if existing:
        gens = {str(r.get("generator") or "") for r in existing}
        if gens - {generator, ""}:
            print(f"refuse mixed generator resume: {sorted(gens)} vs {generator}", file=sys.stderr)
            return 5
        for row in existing:
            fid = str(row.get("doc_id") or (row.get("frame") or {}).get("frame_id") or "")
            if fid:
                seen_frames.add(fid)
            seen_sha.add(str(row.get("response_sha256") or sha256_text(row.get("text") or "")))
            near.add(simhash64(str(row.get("text") or "")))
            start = max(start, int((row.get("frame") or {}).get("index") or 0) + 1)
        counts = unique_by_first_frame(existing)
        n_unique = int(counts["unique_train_tokens"])
        n_docs = int(counts["n_unique_frame_ids"])
        if cursor_path.is_file():
            cur = json.loads(cursor_path.read_text(encoding="utf-8"))
            spend = float(cur.get("spend_cny") or 0)
            prompt_tokens = int(cur.get("prompt_tokens") or 0)
            completion_tokens = int(cur.get("completion_tokens") or 0)
            cursor_gen = str(cur.get("generator") or generator)
            if cursor_gen != generator:
                print(f"cursor generator {cursor_gen} != {generator}", file=sys.stderr)
                return 5
    max_spend = args.max_spend_cny
    if max_spend is None:
        max_spend = float(contract["budget"]["default_max_spend_cny"])
    stats = {
        "n_in": 0,
        "n_keep": n_docs,
        "n_dup": 0,
        "n_near": 0,
        "n_filter": 0,
        "n_gen_fail": 0,
        "budget_stop": False,
        "release_kind": release_kind,
        "generator": generator,
        "mixed_generator": False,
    }
    budget_ok = True
    with raw_path.open("a", encoding="utf-8") as accepted, rej_path.open("a", encoding="utf-8") as rejected:
        for frame in iter_frames(max_docs, seed=args.seed, axes=contract["axes"], start=start):
            if n_unique >= target and n_docs >= (int(contract["targets"]["smoke_docs"]) if args.smoke else 1):
                break
            if generator == "qwen-plus" and spend >= max_spend:
                stats["budget_stop"] = True
                budget_ok = False
                break
            if frame["frame_id"] in seen_frames:
                continue
            stats["n_in"] += 1
            gen = generate_one(frame, generator=generator, contract=contract)
            if not gen.get("ok"):
                stats["n_gen_fail"] += 1
                rejected.write(json.dumps({"frame": frame, "error": gen.get("error")}, ensure_ascii=False) + "\n")
                continue
            usage = gen.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            prompt_tokens += pt
            completion_tokens += ct
            spend += spend_cny(prompt_tokens=pt, completion_tokens=ct, contract=contract)
            text = str(gen.get("text") or "")
            reasons = filter_reasons(text, leaks=leaks, pii_fn=pii_or_nav)
            if not colloquial_keep(text, domain="dialogue"):
                reasons.append("not_spoken")
            digest = sha256_text(text)
            if digest in seen_sha:
                reasons.append("exact_dup")
                stats["n_dup"] += 1
            sim = simhash64(text)
            if near.near(sim):
                reasons.append("near_dup")
                stats["n_near"] += 1
            ids = tok.encode_document(text)
            n_tok = len(ids)
            n_unk = sum(1 for t in ids if t == tok.unk_id)
            if n_tok and (n_unk / n_tok) > 0.02:
                reasons.append("unk")
            split = hash_bucket(digest)
            ok = not reasons
            row = provenance_row(
                frame=frame,
                text=text,
                generator=generator,
                contract=contract,
                request_sha256=str(gen.get("request_sha256") or ""),
                response_sha256=digest,
                usage=usage,
                filter_ok=ok,
                reasons=reasons,
                n_tokens=n_tok,
                split=split,
            )
            row["n_unk"] = n_unk
            if not ok:
                stats["n_filter"] += 1
                rejected.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            seen_sha.add(digest)
            seen_frames.add(frame["frame_id"])
            near.add(sim)
            accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_docs += 1
            if split == "train":
                n_unique += n_tok
            stats["n_keep"] = n_docs
            dump_json(
                cursor_path,
                {
                    "next_index": int(frame["index"]) + 1,
                    "n_docs": n_docs,
                    "n_unique_train_tokens": n_unique,
                    "spend_cny": spend,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generator": generator,
                },
            )
    kept = [r for r in load_jsonl(raw_path) if (r.get("filter") or {}).get("ok")]
    if kept and not any(r.get("split") == "valid" for r in kept):
        kept[-1]["split"] = "valid"
    tok_stats = tokenize_kept(out, kept, tok)
    stats["n_unk"] = tok_stats["n_unk"]
    stats["spend_cny"] = spend
    stats["prompt_tokens"] = prompt_tokens
    stats["completion_tokens"] = completion_tokens
    stats["target_unique_tokens"] = target
    stats["budget_ok"] = budget_ok
    stats["model_snapshot"] = "offline-v1"
    release = freeze_release(out, contract=contract, generator=generator, kept=kept, stats=stats)
    dump_json(out / "reviews" / f"{release_kind}.json", {"ok": True, "release": release, "stats": stats})
    print(json.dumps(release, ensure_ascii=False, indent=2))
    unique = int(release.get("n_unique_train_tokens") or 0)
    if args.smoke:
        return 0 if release["n_docs"] >= 8 and release["unk_rate"] <= float(contract["hard_gates"]["unk_rate_max"]) else 1
    if not budget_ok:
        return 4
    if unique < target:
        return 1
    return 0 if unique > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
